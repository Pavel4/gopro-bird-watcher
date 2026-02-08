#!/usr/bin/env python3
"""
Feeder Analytics для GoPro Bird Watcher
Сбор и анализ данных о визитах птиц на кормушку.

CSV-файлы:
- visits.csv      — каждый визит отдельной строкой
- food_log.csv    — лог смены корма
- daily_stats.csv — агрегированная статистика по дням
"""

import os
import csv
import time
import logging
from datetime import datetime, timezone, timedelta
from threading import Lock
from typing import Optional
from collections import defaultdict

# Московское время (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))

# Заголовки CSV-файлов
VISITS_HEADERS = [
    "visit_id",
    "date",
    "time_start",
    "time_end",
    "duration_sec",
    "max_motion_area",
    "avg_motion_area",
    "food_type",
    "video_file",
    "hour",
    "weekday",
    "species",
    "behavior",
]

FOOD_LOG_HEADERS = [
    "timestamp",
    "food_type",
    "source",
]

DAILY_STATS_HEADERS = [
    "date",
    "total_visits",
    "total_duration_sec",
    "avg_duration_sec",
    "max_duration_sec",
    "min_duration_sec",
    "peak_hour",
    "food_type",
    "first_visit",
    "last_visit",
]

WEEKDAY_NAMES_RU = [
    "Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"
]


class FeederAnalytics:
    """
    Аналитика кормушки: отслеживание визитов,
    типов корма и агрегированная статистика.
    """

    def __init__(
        self,
        analytics_dir: str = "./analytics",
        default_food_type: str = "mixed",
        logger: logging.Logger = None
    ):
        """
        Args:
            analytics_dir: Директория для CSV-файлов
            default_food_type: Тип корма по умолчанию
            logger: Логгер
        """
        self.analytics_dir = analytics_dir
        self.current_food_type = default_food_type
        self.logger = (
            logger or logging.getLogger(__name__)
        )

        # Потокобезопасность
        self._lock = Lock()

        # Состояние текущего визита
        self._visit_active = False
        self._visit_start_time = None
        self._visit_motion_areas = []
        self._visit_max_area = 0.0
        self._last_video_file = None
        self._current_species = None
        self._current_behavior = None

        # Счётчик визитов (загружается из CSV)
        self._next_visit_id = 1

        # Пути к CSV-файлам
        os.makedirs(self.analytics_dir, exist_ok=True)
        self.visits_csv = os.path.join(
            self.analytics_dir, "visits.csv"
        )
        self.food_log_csv = os.path.join(
            self.analytics_dir, "food_log.csv"
        )
        self.daily_stats_csv = os.path.join(
            self.analytics_dir, "daily_stats.csv"
        )

        # Инициализируем CSV-файлы
        self._init_csv(
            self.visits_csv, VISITS_HEADERS
        )
        self._init_csv(
            self.food_log_csv, FOOD_LOG_HEADERS
        )
        self._init_csv(
            self.daily_stats_csv, DAILY_STATS_HEADERS
        )

        # Загружаем последний visit_id из CSV
        self._load_last_visit_id()

        # Загружаем последний тип корма из food_log
        self._load_last_food_type()

        self.logger.info(
            f"📊 FeederAnalytics initialized: "
            f"dir={self.analytics_dir}, "
            f"food={self.current_food_type}, "
            f"next_visit_id={self._next_visit_id}"
        )

    def _init_csv(self, filepath: str, headers: list):
        """Создать CSV-файл с заголовками если не существует."""
        if not os.path.exists(filepath):
            with open(filepath, "w", newline="",
                       encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
            self.logger.info(
                f"  Created CSV: {filepath}"
            )

    def _load_last_visit_id(self):
        """Загрузить последний visit_id из visits.csv."""
        try:
            if not os.path.exists(self.visits_csv):
                return
            with open(self.visits_csv, "r",
                       encoding="utf-8") as f:
                reader = csv.DictReader(f)
                max_id = 0
                for row in reader:
                    try:
                        vid = int(row.get("visit_id", 0))
                        if vid > max_id:
                            max_id = vid
                    except (ValueError, TypeError):
                        pass
                self._next_visit_id = max_id + 1
        except Exception as e:
            self.logger.warning(
                f"Could not load last visit_id: {e}"
            )

    def _load_last_food_type(self):
        """Загрузить последний тип корма из food_log.csv."""
        try:
            if not os.path.exists(self.food_log_csv):
                return
            last_food = None
            with open(self.food_log_csv, "r",
                       encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ft = row.get("food_type", "").strip()
                    if ft:
                        last_food = ft
            if last_food:
                self.current_food_type = last_food
        except Exception as e:
            self.logger.warning(
                f"Could not load last food type: {e}"
            )

    @staticmethod
    def _moscow_now() -> datetime:
        """Текущее время в московской зоне."""
        return datetime.now(MOSCOW_TZ)

    # === API визитов ===

    def visit_started(self, motion_percent: float):
        """
        Вызывается при начале значимого движения
        (птица прилетела).

        Args:
            motion_percent: Процент площади движения
        """
        with self._lock:
            if self._visit_active:
                return  # Визит уже идёт
            self._visit_active = True
            self._visit_start_time = time.time()
            self._visit_motion_areas = [motion_percent]
            self._visit_max_area = motion_percent
            self._last_video_file = None
            self._current_species = None
            self._current_behavior = None

    def visit_update(self, motion_percent: float):
        """
        Вызывается при каждом кадре с движением
        во время визита.

        Args:
            motion_percent: Процент площади движения
        """
        with self._lock:
            if not self._visit_active:
                return
            self._visit_motion_areas.append(
                motion_percent
            )
            if motion_percent > self._visit_max_area:
                self._visit_max_area = motion_percent

    def set_last_video(self, video_file: str):
        """
        Привязать видеофайл к текущему визиту.

        Args:
            video_file: Имя файла видео
        """
        with self._lock:
            self._last_video_file = video_file

    def visit_ended(self):
        """
        Вызывается при окончании значимого движения
        (птица улетела). Записывает визит в CSV.
        """
        with self._lock:
            if not self._visit_active:
                return

            # Рассчитываем метрики визита
            end_time = time.time()
            duration = end_time - self._visit_start_time
            now = self._moscow_now()

            start_dt = datetime.fromtimestamp(
                self._visit_start_time, tz=MOSCOW_TZ
            )

            areas = self._visit_motion_areas
            avg_area = (
                sum(areas) / len(areas)
                if areas else 0.0
            )
            max_area = self._visit_max_area

            visit_id = self._next_visit_id
            self._next_visit_id += 1

            video_file = self._last_video_file or ""

            # Записываем в visits.csv
            row = [
                visit_id,
                start_dt.strftime("%Y-%m-%d"),
                start_dt.strftime("%H:%M:%S"),
                now.strftime("%H:%M:%S"),
                f"{duration:.1f}",
                f"{max_area:.2f}",
                f"{avg_area:.2f}",
                self.current_food_type,
                video_file,
                start_dt.hour,
                start_dt.weekday(),
                self._current_species or "unknown",
                self._current_behavior or "",
            ]
            self._append_csv(self.visits_csv, row)

            self.logger.info(
                f"📊 Visit #{visit_id} recorded: "
                f"{duration:.1f}s, "
                f"max_area={max_area:.2f}%, "
                f"food={self.current_food_type}"
            )

            # Обновляем дневную статистику
            date_str = start_dt.strftime("%Y-%m-%d")
            self._update_daily_stats(date_str)

            # Сбрасываем состояние визита
            self._visit_active = False
            self._visit_start_time = None
            self._visit_motion_areas = []
            self._visit_max_area = 0.0
            self._last_video_file = None
            self._current_species = None
            self._current_behavior = None

    def set_species(self, species_name: str):
        """
        Задать вид птицы для текущего визита.
        Вызывается ML-модулем после классификации.
        """
        with self._lock:
            if self._visit_active:
                self._current_species = species_name

    def set_behavior(self, behavior_en: str):
        """
        Задать поведение для текущего визита.
        Вызывается ML-модулем после классификации.
        """
        with self._lock:
            if self._visit_active:
                self._current_behavior = behavior_en

    # === API корма ===

    def set_food_type(
        self, food_type: str, source: str = "config"
    ):
        """
        Задать текущий тип корма.

        Args:
            food_type: Тип корма (семечки, сало и т.д.)
            source: Откуда задан (telegram / config)
        """
        food_type = food_type.strip()
        if not food_type:
            return

        with self._lock:
            self.current_food_type = food_type
            now = self._moscow_now()
            row = [
                now.isoformat(),
                food_type,
                source,
            ]
            self._append_csv(self.food_log_csv, row)

        self.logger.info(
            f"🥜 Food type changed to: "
            f"{food_type} (source: {source})"
        )

    def get_food_type(self) -> str:
        """Получить текущий тип корма."""
        return self.current_food_type

    # === CSV-операции ===

    def _append_csv(self, filepath: str, row: list):
        """Добавить строку в CSV-файл."""
        try:
            with open(filepath, "a", newline="",
                       encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
        except Exception as e:
            self.logger.error(
                f"Error writing to {filepath}: {e}"
            )

    def _read_visits_csv(self) -> list:
        """Прочитать все визиты из CSV."""
        rows = []
        try:
            if not os.path.exists(self.visits_csv):
                return rows
            with open(self.visits_csv, "r",
                       encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
        except Exception as e:
            self.logger.error(
                f"Error reading visits CSV: {e}"
            )
        return rows

    # === Дневная статистика ===

    def _update_daily_stats(self, date_str: str):
        """
        Пересчитать и обновить daily_stats.csv
        для указанной даты.
        """
        visits = self._read_visits_csv()
        day_visits = [
            v for v in visits
            if v.get("date") == date_str
        ]

        if not day_visits:
            return

        total = len(day_visits)
        durations = []
        hours_count = defaultdict(int)
        first_time = None
        last_time = None
        food_types = defaultdict(int)

        for v in day_visits:
            try:
                dur = float(v.get("duration_sec", 0))
                durations.append(dur)
            except (ValueError, TypeError):
                durations.append(0.0)

            try:
                hour = int(v.get("hour", 0))
                hours_count[hour] += 1
            except (ValueError, TypeError):
                pass

            t_start = v.get("time_start", "")
            if t_start:
                if first_time is None or t_start < first_time:
                    first_time = t_start
                if last_time is None or t_start > last_time:
                    last_time = t_start

            ft = v.get("food_type", "mixed")
            food_types[ft] += 1

        total_dur = sum(durations)
        avg_dur = total_dur / total if total > 0 else 0
        max_dur = max(durations) if durations else 0
        min_dur = min(durations) if durations else 0

        peak_hour = (
            max(hours_count, key=hours_count.get)
            if hours_count else 0
        )
        main_food = (
            max(food_types, key=food_types.get)
            if food_types else "mixed"
        )

        new_row = [
            date_str,
            total,
            f"{total_dur:.1f}",
            f"{avg_dur:.1f}",
            f"{max_dur:.1f}",
            f"{min_dur:.1f}",
            peak_hour,
            main_food,
            first_time or "",
            last_time or "",
        ]

        # Перезаписываем строку для этой даты
        self._upsert_daily_stats(date_str, new_row)

    def _upsert_daily_stats(
        self, date_str: str, new_row: list
    ):
        """
        Вставить или обновить строку в daily_stats.csv
        для указанной даты.
        """
        try:
            rows = []
            found = False
            if os.path.exists(self.daily_stats_csv):
                with open(
                    self.daily_stats_csv, "r",
                    newline="", encoding="utf-8"
                ) as f:
                    reader = csv.reader(f)
                    headers = next(reader, None)
                    for row in reader:
                        if row and row[0] == date_str:
                            rows.append(new_row)
                            found = True
                        else:
                            rows.append(row)

            if not found:
                rows.append(new_row)

            with open(
                self.daily_stats_csv, "w",
                newline="", encoding="utf-8"
            ) as f:
                writer = csv.writer(f)
                writer.writerow(DAILY_STATS_HEADERS)
                for row in rows:
                    writer.writerow(row)

        except Exception as e:
            self.logger.error(
                f"Error updating daily stats: {e}"
            )

    # === Методы статистики по видам ===

    def _read_visits_csv(self) -> list:
        """Прочитать все визиты из CSV."""
        if not os.path.exists(self.visits_csv):
            return []
        try:
            with open(
                self.visits_csv, "r",
                encoding="utf-8",
            ) as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    def get_species_stats(self) -> dict:
        """
        Статистика по видам птиц.

        Returns:
            {species: {visits, avg_duration,
                       total_duration}}
        """
        visits = self._read_visits_csv()
        species_data = defaultdict(
            lambda: {"visits": 0, "durations": []}
        )

        for v in visits:
            sp = v.get("species", "unknown")
            species_data[sp]["visits"] += 1
            try:
                dur = float(
                    v.get("duration_sec", 0)
                )
                species_data[sp][
                    "durations"
                ].append(dur)
            except (ValueError, TypeError):
                pass

        result = {}
        for s, data in species_data.items():
            durs = data["durations"]
            result[s] = {
                "visits": data["visits"],
                "avg_duration": (
                    sum(durs) / len(durs)
                    if durs else 0
                ),
                "total_duration": sum(durs),
            }
        return result

    def get_species_food_preference(self) -> dict:
        """
        Предпочтения корма по видам птиц.

        Returns:
            {species: {food_type: count}}
        """
        visits = self._read_visits_csv()
        species_food = defaultdict(
            lambda: defaultdict(int)
        )

        for v in visits:
            sp = v.get("species", "unknown")
            food = v.get("food_type", "mixed")
            species_food[sp][food] += 1

        return {
            s: dict(f)
            for s, f in species_food.items()
        }

    def format_species_stats(self) -> str:
        """
        Форматированное сообщение со статистикой
        по видам (для Telegram /species команды).
        """
        stats = self.get_species_stats()

        if not stats:
            return (
                "📊 <b>Статистика по видам</b>\n\n"
                "Данных пока нет."
            )

        sorted_sp = sorted(
            stats.items(),
            key=lambda x: x[1]["visits"],
            reverse=True,
        )

        food_prefs = (
            self.get_species_food_preference()
        )

        lines = []
        for name, s in sorted_sp:
            fp = food_prefs.get(name, {})
            food_info = ""
            if fp:
                top_food = max(fp, key=fp.get)
                food_info = (
                    f" (корм: {top_food})"
                )
            lines.append(
                f"  <b>{name}</b>: "
                f"{s['visits']} визитов, "
                f"средн. {s['avg_duration']:.1f}с"
                f"{food_info}"
            )

        return (
            "📊 <b>Статистика по видам</b>\n\n"
            + "\n".join(lines)
        )

    # === Методы получения статистики ===

    def get_daily_stats(
        self, date_str: str = None
    ) -> dict:
        """
        Получить статистику за конкретный день.

        Args:
            date_str: Дата в формате YYYY-MM-DD
                      (None = сегодня)
        Returns:
            Словарь со статистикой
        """
        if date_str is None:
            date_str = self._moscow_now().strftime(
                "%Y-%m-%d"
            )

        visits = self._read_visits_csv()
        day_visits = [
            v for v in visits
            if v.get("date") == date_str
        ]

        if not day_visits:
            return {
                "date": date_str,
                "total_visits": 0,
                "total_duration_sec": 0,
                "avg_duration_sec": 0,
                "max_duration_sec": 0,
                "min_duration_sec": 0,
                "peak_hour": None,
                "food_type": self.current_food_type,
                "first_visit": None,
                "last_visit": None,
                "hourly": {},
            }

        durations = []
        hours_count = defaultdict(int)
        first_time = None
        last_time = None
        food_types = defaultdict(int)

        for v in day_visits:
            try:
                dur = float(
                    v.get("duration_sec", 0)
                )
                durations.append(dur)
            except (ValueError, TypeError):
                durations.append(0.0)

            try:
                hour = int(v.get("hour", 0))
                hours_count[hour] += 1
            except (ValueError, TypeError):
                pass

            t = v.get("time_start", "")
            if t:
                if first_time is None or t < first_time:
                    first_time = t
                if last_time is None or t > last_time:
                    last_time = t

            ft = v.get("food_type", "")
            if ft:
                food_types[ft] += 1

        total = len(day_visits)
        total_dur = sum(durations)
        peak = (
            max(hours_count, key=hours_count.get)
            if hours_count else None
        )
        main_food = (
            max(food_types, key=food_types.get)
            if food_types else self.current_food_type
        )

        return {
            "date": date_str,
            "total_visits": total,
            "total_duration_sec": total_dur,
            "avg_duration_sec": (
                total_dur / total if total > 0 else 0
            ),
            "max_duration_sec": (
                max(durations) if durations else 0
            ),
            "min_duration_sec": (
                min(durations) if durations else 0
            ),
            "peak_hour": peak,
            "food_type": main_food,
            "first_visit": first_time,
            "last_visit": last_time,
            "hourly": dict(hours_count),
        }

    def get_weekly_stats(self) -> dict:
        """
        Получить статистику за последние 7 дней.

        Returns:
            Словарь с агрегированной статистикой
        """
        now = self._moscow_now()
        visits = self._read_visits_csv()

        week_visits = []
        daily_counts = defaultdict(int)

        for i in range(7):
            d = now - timedelta(days=i)
            date_str = d.strftime("%Y-%m-%d")
            day_v = [
                v for v in visits
                if v.get("date") == date_str
            ]
            daily_counts[date_str] = len(day_v)
            week_visits.extend(day_v)

        if not week_visits:
            return {
                "period": "7 days",
                "total_visits": 0,
                "avg_visits_per_day": 0,
                "total_duration_sec": 0,
                "avg_duration_sec": 0,
                "best_day": None,
                "best_day_visits": 0,
                "daily_counts": dict(daily_counts),
            }

        total = len(week_visits)
        durations = []
        for v in week_visits:
            try:
                dur = float(
                    v.get("duration_sec", 0)
                )
                durations.append(dur)
            except (ValueError, TypeError):
                durations.append(0.0)

        total_dur = sum(durations)

        best_day = (
            max(daily_counts, key=daily_counts.get)
            if daily_counts else None
        )
        best_day_cnt = (
            daily_counts[best_day]
            if best_day else 0
        )

        return {
            "period": "7 days",
            "total_visits": total,
            "avg_visits_per_day": total / 7,
            "total_duration_sec": total_dur,
            "avg_duration_sec": (
                total_dur / total if total > 0 else 0
            ),
            "best_day": best_day,
            "best_day_visits": best_day_cnt,
            "daily_counts": dict(daily_counts),
        }

    def get_hourly_distribution(
        self, days: int = 7
    ) -> dict:
        """
        Распределение визитов по часам за N дней.

        Args:
            days: Количество дней для анализа
        Returns:
            Словарь {hour: count}
        """
        now = self._moscow_now()
        visits = self._read_visits_csv()
        hours = defaultdict(int)

        cutoff = (
            now - timedelta(days=days)
        ).strftime("%Y-%m-%d")

        for v in visits:
            if v.get("date", "") >= cutoff:
                try:
                    h = int(v.get("hour", 0))
                    hours[h] += 1
                except (ValueError, TypeError):
                    pass

        return dict(hours)

    def get_food_comparison(self) -> dict:
        """
        Сравнение типов корма по количеству визитов.

        Returns:
            Словарь {food_type: {visits, avg_duration}}
        """
        visits = self._read_visits_csv()
        food_data = defaultdict(
            lambda: {"visits": 0, "durations": []}
        )

        for v in visits:
            ft = v.get("food_type", "unknown")
            food_data[ft]["visits"] += 1
            try:
                dur = float(
                    v.get("duration_sec", 0)
                )
                food_data[ft]["durations"].append(dur)
            except (ValueError, TypeError):
                pass

        result = {}
        for ft, data in food_data.items():
            durs = data["durations"]
            result[ft] = {
                "visits": data["visits"],
                "avg_duration": (
                    sum(durs) / len(durs)
                    if durs else 0
                ),
                "total_duration": sum(durs),
            }

        return result

    # === Форматированный вывод ===

    def format_daily_stats(
        self, date_str: str = None
    ) -> str:
        """
        Форматированное сообщение со статистикой
        за день (для Telegram).
        """
        stats = self.get_daily_stats(date_str)
        date_display = stats["date"]

        if stats["total_visits"] == 0:
            return (
                f"📊 <b>Статистика за {date_display}"
                f"</b>\n\n"
                f"Визитов пока нет."
            )

        # Форматируем пик активности
        peak = stats.get("peak_hour")
        peak_str = (
            f"{peak}:00-{peak+1}:00"
            if peak is not None else "—"
        )

        # Текстовая гистограмма по часам
        hourly = stats.get("hourly", {})
        histogram = self._format_hourly_histogram(
            hourly
        )

        msg = (
            f"📊 <b>Статистика за {date_display}"
            f"</b>\n\n"
            f"🐦 Визитов: "
            f"{stats['total_visits']}\n"
            f"⏱ Средняя длительность: "
            f"{stats['avg_duration_sec']:.1f} сек\n"
            f"📈 Самый длинный: "
            f"{stats['max_duration_sec']:.1f} сек\n"
            f"📉 Самый короткий: "
            f"{stats['min_duration_sec']:.1f} сек\n"
            f"🕐 Пик активности: {peak_str}\n"
            f"🌅 Первый визит: "
            f"{stats['first_visit'] or '—'}\n"
            f"🌆 Последний визит: "
            f"{stats['last_visit'] or '—'}\n"
            f"🥜 Корм: {stats['food_type']}\n"
        )

        if histogram:
            msg += f"\n📊 По часам:\n{histogram}"

        return msg

    def format_weekly_stats(self) -> str:
        """
        Форматированное сообщение со статистикой
        за неделю (для Telegram).
        """
        stats = self.get_weekly_stats()

        if stats["total_visits"] == 0:
            return (
                "📊 <b>Статистика за неделю</b>"
                "\n\nВизитов пока нет."
            )

        daily = stats.get("daily_counts", {})
        daily_lines = []
        for date_str in sorted(
            daily.keys(), reverse=True
        ):
            cnt = daily[date_str]
            bar = "█" * min(cnt, 30)
            daily_lines.append(
                f"  {date_str}: {bar} {cnt}"
            )
        daily_chart = "\n".join(daily_lines)

        msg = (
            f"📊 <b>Статистика за неделю</b>\n\n"
            f"🐦 Всего визитов: "
            f"{stats['total_visits']}\n"
            f"📅 В среднем в день: "
            f"{stats['avg_visits_per_day']:.1f}\n"
            f"⏱ Средняя длительность: "
            f"{stats['avg_duration_sec']:.1f} сек\n"
            f"🏆 Лучший день: "
            f"{stats['best_day']} "
            f"({stats['best_day_visits']} визитов)\n"
        )

        if daily_chart:
            msg += (
                f"\n<pre>"
                f"{daily_chart}"
                f"</pre>"
            )

        return msg

    def format_food_comparison(self) -> str:
        """
        Форматированное сравнение типов корма
        (для Telegram).
        """
        food = self.get_food_comparison()

        if not food:
            return (
                "🥜 <b>Сравнение корма</b>"
                "\n\nДанных пока нет."
            )

        # Сортируем по количеству визитов
        sorted_food = sorted(
            food.items(),
            key=lambda x: x[1]["visits"],
            reverse=True,
        )

        lines = []
        for ft, data in sorted_food:
            lines.append(
                f"  <b>{ft}</b>: "
                f"{data['visits']} визитов, "
                f"средн. {data['avg_duration']:.1f}с"
            )

        msg = (
            "🥜 <b>Сравнение типов корма</b>\n\n"
            + "\n".join(lines)
        )

        return msg

    def format_hourly_stats(
        self, days: int = 7
    ) -> str:
        """
        Текстовая гистограмма по часам
        (для Telegram).
        """
        hourly = self.get_hourly_distribution(days)

        if not hourly:
            return (
                "🕐 <b>Распределение по часам</b>"
                f"\n\nДанных за {days} дней нет."
            )

        histogram = self._format_hourly_histogram(
            hourly
        )

        return (
            f"🕐 <b>Распределение по часам "
            f"(за {days} дн.)</b>\n\n"
            f"<pre>{histogram}</pre>"
        )

    @staticmethod
    def _format_hourly_histogram(
        hourly: dict
    ) -> str:
        """Текстовая гистограмма по часам."""
        if not hourly:
            return ""

        max_val = max(hourly.values()) if hourly else 1
        lines = []
        for h in range(24):
            cnt = hourly.get(h, 0)
            if cnt > 0 or (6 <= h <= 21):
                bar_len = int(
                    cnt / max_val * 15
                ) if max_val > 0 else 0
                bar = "█" * bar_len
                lines.append(
                    f"  {h:02d}:00 {bar} {cnt}"
                )

        return "\n".join(lines)

    # === Метод для get_status() ===

    def get_summary(self) -> dict:
        """
        Краткая сводка для интеграции
        с MotionDetector.get_status().

        Returns:
            Словарь с основными метриками
        """
        today = self._moscow_now().strftime(
            "%Y-%m-%d"
        )
        stats = self.get_daily_stats(today)

        return {
            "today_visits": stats["total_visits"],
            "today_avg_duration": (
                stats["avg_duration_sec"]
            ),
            "current_food": self.current_food_type,
            "visit_active": self._visit_active,
        }
