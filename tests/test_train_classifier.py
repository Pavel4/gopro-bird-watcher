"""
Тесты для scripts/train_species_classifier.py

Тестируем: load_images_from_directory,
detect_device, train_linear_probe,
save_linear_head, save_training_report.
"""

import json
import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

from train_species_classifier import (
    detect_device,
    load_images_from_directory,
    train_linear_probe,
    save_linear_head,
    save_training_report,
)


class TestDetectDevice:
    def test_cpu_fallback(self):
        """detect_device на CPU."""
        device = detect_device("cpu")
        assert device == "cpu"

    def test_auto_returns_string(self):
        """detect_device('auto') возвращает строку."""
        device = detect_device("auto")
        assert device in ("cpu", "cuda", "mps")

    def test_explicit_device(self):
        """Явно заданное устройство."""
        assert detect_device("cuda") == "cuda"


class TestLoadImages:
    def _create_dataset(self, base_dir):
        """Создать тестовую структуру директорий."""
        for cls_id, name in [
            (0, "great_tit"),
            (1, "blue_tit"),
        ]:
            dir_name = f"{cls_id}_{name}"
            cls_dir = os.path.join(
                base_dir, dir_name
            )
            os.makedirs(cls_dir, exist_ok=True)
            # Создаём фейковые изображения
            for i in range(5):
                # Реальный JPEG-файл (минимальный)
                path = os.path.join(
                    cls_dir, f"{i:04d}.jpg"
                )
                # Пишем фейковые данные
                with open(path, "wb") as f:
                    f.write(b"\xff\xd8\xff" + b"x" * 100)
        return base_dir

    def test_load_from_directory(self, tmp_dir):
        """Загрузка путей и меток из директорий."""
        data_dir = self._create_dataset(
            os.path.join(tmp_dir, "data")
        )
        paths, labels, class_names = (
            load_images_from_directory(data_dir)
        )

        assert len(paths) == 10  # 5 + 5
        assert len(labels) == 10
        assert len(class_names) == 2
        assert 0 in class_names
        assert 1 in class_names
        assert class_names[0] == "great_tit"

    def test_empty_dir_exits(self, tmp_dir):
        """Пустая директория -> sys.exit."""
        empty_dir = os.path.join(
            tmp_dir, "empty"
        )
        os.makedirs(empty_dir, exist_ok=True)
        with pytest.raises(SystemExit):
            load_images_from_directory(empty_dir)

    def test_nonexistent_dir_exits(self, tmp_dir):
        """Несуществующая директория -> sys.exit."""
        with pytest.raises(SystemExit):
            load_images_from_directory(
                os.path.join(tmp_dir, "nope")
            )


class TestTrainLinearProbe:
    def test_train_on_synthetic_data(self):
        """
        Обучение на синтетических эмбеддингах.
        Должен вернуть accuracy > 0.
        """
        np.random.seed(42)
        n_samples = 100
        n_classes = 3
        embed_dim = 512

        # Генерируем кластеры
        embeddings = []
        labels = []
        for c in range(n_classes):
            center = np.random.randn(embed_dim)
            samples = (
                center + np.random.randn(
                    n_samples // n_classes,
                    embed_dim,
                ) * 0.1
            )
            embeddings.append(samples)
            labels.extend(
                [c] * (n_samples // n_classes)
            )

        X = np.vstack(embeddings).astype(
            np.float32
        )
        y = np.array(labels)

        clf, accuracy, report = train_linear_probe(
            X, y, test_size=0.3,
        )
        assert accuracy > 0.5
        assert clf is not None
        assert "accuracy" in report

    def test_train_returns_report_dict(self):
        """train_linear_probe возвращает dict."""
        np.random.seed(0)
        X = np.random.randn(50, 512).astype(
            np.float32
        )
        y = np.array([0] * 25 + [1] * 25)

        _, _, report = train_linear_probe(
            X, y, test_size=0.2,
        )
        assert isinstance(report, dict)


class TestSaveLinearHead:
    def test_save_and_load(self, tmp_dir):
        """Сохранение и загрузка .npz."""
        from sklearn.linear_model import (
            LogisticRegression,
        )
        np.random.seed(42)
        X = np.random.randn(30, 512).astype(
            np.float32
        )
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)

        clf = LogisticRegression(max_iter=100)
        clf.fit(X, y)

        path = os.path.join(
            tmp_dir, "test_head.npz"
        )
        class_names = {
            0: "great_tit",
            1: "blue_tit",
            2: "nuthatch",
        }
        save_linear_head(clf, path, class_names)

        assert os.path.exists(path)

        data = np.load(path, allow_pickle=True)
        assert "weight" in data
        assert "bias" in data
        assert "classes" in data
        assert data["weight"].shape == (3, 512)
        assert data["bias"].shape == (3,)


class TestSaveTrainingReport:
    def test_save_report(self, tmp_dir):
        """Сохранение JSON-отчёта."""
        save_training_report(
            output_dir=tmp_dir,
            accuracy=0.794,
            report={
                "0": {
                    "precision": 0.8,
                    "recall": 0.75,
                    "f1-score": 0.77,
                },
                "accuracy": 0.794,
                "macro avg": {
                    "precision": 0.79,
                },
            },
            class_names={0: "great_tit"},
            num_train=1500,
            num_val=400,
        )

        report_path = os.path.join(
            tmp_dir, "training_report.json"
        )
        assert os.path.exists(report_path)

        with open(report_path, "r") as f:
            data = json.load(f)

        assert data["accuracy"] == 0.794
        assert data["num_train"] == 1500
        assert data["num_val"] == 400
        assert "class_names" in data
