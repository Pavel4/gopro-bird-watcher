"""
Тесты для detector/feeder_analytics.py

Тестируем: FeederAnalytics — CSV, species,
визиты, статистика, потокобезопасность.
"""

import csv
import os
import time
import threading
from unittest.mock import MagicMock

import pytest

from feeder_analytics import (
    FeederAnalytics,
    VISITS_HEADERS,
    FOOD_LOG_HEADERS,
    DAILY_STATS_HEADERS,
)


class TestInit:
    def test_creates_csv_files(self, tmp_dir):
        """Инициализация создаёт CSV-файлы."""
        a_dir = os.path.join(tmp_dir, "analytics")
        analytics = FeederAnalytics(
            analytics_dir=a_dir,
        )
        assert os.path.exists(
            os.path.join(a_dir, "visits.csv")
        )
        assert os.path.exists(
            os.path.join(a_dir, "food_log.csv")
        )
        assert os.path.exists(
            os.path.join(a_dir, "daily_stats.csv")
        )

    def test_csv_headers_include_species(
        self, tmp_dir,
    ):
        """visits.csv содержит колонку species."""
        assert "species" in VISITS_HEADERS

        a_dir = os.path.join(tmp_dir, "analytics")
        FeederAnalytics(analytics_dir=a_dir)

        with open(
            os.path.join(a_dir, "visits.csv"),
            "r", encoding="utf-8",
        ) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert "species" in headers

    def test_default_food_type(self, tmp_dir):
        """Тип корма по умолчанию = 'mixed'."""
        a_dir = os.path.join(tmp_dir, "analytics")
        analytics = FeederAnalytics(
            analytics_dir=a_dir,
        )
        assert analytics.current_food_type == "mixed"

    def test_custom_food_type(self, tmp_dir):
        """Можно задать свой тип корма."""
        a_dir = os.path.join(tmp_dir, "analytics")
        analytics = FeederAnalytics(
            analytics_dir=a_dir,
            default_food_type="sunflower_seeds",
        )
        assert (
            analytics.current_food_type
            == "sunflower_seeds"
        )


class TestVisitLifecycle:
    def test_basic_visit(
        self, analytics_instance, tmp_dir,
    ):
        """
        visit_started -> visit_ended
        записывает строку в visits.csv.
        """
        a = analytics_instance
        a.visit_started(5.0)
        time.sleep(0.05)
        a.visit_ended()

        visits_csv = a.visits_csv
        with open(
            visits_csv, "r", encoding="utf-8",
        ) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["visit_id"] == "1"
        assert rows[0]["food_type"] == "seeds"
        assert float(rows[0]["duration_sec"]) > 0

    def test_visit_with_update(
        self, analytics_instance,
    ):
        """visit_update обновляет max area."""
        a = analytics_instance
        a.visit_started(2.0)
        a.visit_update(5.0)
        a.visit_update(8.0)
        a.visit_update(3.0)
        a.visit_ended()

        with open(
            a.visits_csv, "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert float(
            rows[0]["max_motion_area"]
        ) == pytest.approx(8.0, abs=0.1)

    def test_set_species(
        self, analytics_instance,
    ):
        """set_species записывает вид в CSV."""
        a = analytics_instance
        a.visit_started(5.0)
        a.set_species("Большая синица")
        a.visit_ended()

        with open(
            a.visits_csv, "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["species"] == "Большая синица"

    def test_visit_without_species(
        self, analytics_instance,
    ):
        """Без set_species -> species='unknown'."""
        a = analytics_instance
        a.visit_started(3.0)
        a.visit_ended()

        with open(
            a.visits_csv, "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert rows[0]["species"] == "unknown"

    def test_multiple_visits(
        self, analytics_instance,
    ):
        """Несколько визитов подряд."""
        a = analytics_instance

        for i in range(5):
            a.visit_started(float(i + 1))
            a.set_species(f"species_{i}")
            a.visit_ended()

        with open(
            a.visits_csv, "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 5
        # visit_id должны быть последовательны
        ids = [int(r["visit_id"]) for r in rows]
        assert ids == [1, 2, 3, 4, 5]

    def test_set_last_video(
        self, analytics_instance,
    ):
        """set_last_video привязывает файл."""
        a = analytics_instance
        a.visit_started(5.0)
        a.set_last_video("motion_001.mp4")
        a.visit_ended()

        with open(
            a.visits_csv, "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert (
            rows[0]["video_file"]
            == "motion_001.mp4"
        )


class TestFoodManagement:
    def test_set_food_type(
        self, analytics_instance,
    ):
        """set_food_type меняет тип корма."""
        a = analytics_instance
        a.set_food_type("walnuts", source="user")
        assert a.current_food_type == "walnuts"

    def test_food_logged_to_csv(
        self, analytics_instance,
    ):
        """Смена корма пишется в food_log.csv."""
        a = analytics_instance
        a.set_food_type("fat_ball", source="bot")

        with open(
            a.food_log_csv, "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) >= 1
        last = rows[-1]
        assert last["food_type"] == "fat_ball"
        assert last["source"] == "bot"


class TestSpeciesStats:
    def _create_visits(self, analytics):
        """Создать несколько визитов для тестов."""
        species = [
            "Большая синица",
            "Большая синица",
            "Лазоревка",
            "Большая синица",
            "Поползень",
        ]
        for sp in species:
            analytics.visit_started(5.0)
            analytics.set_species(sp)
            analytics.visit_ended()

    def test_get_species_stats(
        self, analytics_instance,
    ):
        """get_species_stats() агрегация."""
        self._create_visits(analytics_instance)
        stats = analytics_instance.get_species_stats()

        assert "Большая синица" in stats
        assert stats["Большая синица"]["visits"] == 3
        assert "Лазоревка" in stats
        assert stats["Лазоревка"]["visits"] == 1
        assert "Поползень" in stats
        assert stats["Поползень"]["visits"] == 1

    def test_get_species_food_preference(
        self, analytics_instance,
    ):
        """get_species_food_preference()."""
        a = analytics_instance
        # Визиты с разным кормом
        a.visit_started(5.0)
        a.set_species("Синица")
        a.visit_ended()

        a.set_food_type("walnuts", source="test")
        a.visit_started(5.0)
        a.set_species("Синица")
        a.visit_ended()

        prefs = a.get_species_food_preference()
        assert "Синица" in prefs
        assert "seeds" in prefs["Синица"]
        assert "walnuts" in prefs["Синица"]

    def test_format_species_stats_html(
        self, analytics_instance,
    ):
        """format_species_stats() возвращает HTML."""
        self._create_visits(analytics_instance)
        msg = (
            analytics_instance
            .format_species_stats()
        )
        assert "<b>" in msg
        assert "Большая синица" in msg
        assert "визитов" in msg

    def test_format_species_stats_empty(
        self, tmp_dir,
    ):
        """format_species_stats() без данных."""
        a_dir = os.path.join(tmp_dir, "analytics2")
        analytics = FeederAnalytics(
            analytics_dir=a_dir,
        )
        msg = analytics.format_species_stats()
        assert "Данных пока нет" in msg


class TestThreadSafety:
    def test_concurrent_visits(self, tmp_dir):
        """
        Параллельные вызовы из 2 потоков
        не ломают CSV.
        """
        a_dir = os.path.join(
            tmp_dir, "analytics_thread"
        )
        analytics = FeederAnalytics(
            analytics_dir=a_dir,
        )
        errors = []

        def worker(thread_id):
            try:
                for i in range(10):
                    analytics.visit_started(
                        float(thread_id * 10 + i)
                    )
                    analytics.set_species(
                        f"species_{thread_id}"
                    )
                    analytics.visit_ended()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(
            target=worker, args=(1,)
        )
        t2 = threading.Thread(
            target=worker, args=(2,)
        )
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0

        # Все визиты должны быть записаны
        # (не обязательно 20, т.к. visit_started
        # игнорирует если визит уже идёт)
        with open(
            analytics.visits_csv,
            "r", encoding="utf-8",
        ) as f:
            rows = list(csv.DictReader(f))

        assert len(rows) > 0
        # visit_id должны быть уникальны
        ids = [int(r["visit_id"]) for r in rows]
        assert len(ids) == len(set(ids))
