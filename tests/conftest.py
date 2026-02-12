"""
Общие фикстуры для тестов GoPro Bird Watcher.
"""

import json
import os
import sys
import tempfile

import pytest
import numpy as np

# Добавляем detector/ и scripts/ в PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "detector"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


@pytest.fixture
def tmp_dir(tmp_path):
    """Временная директория для тестов."""
    return str(tmp_path)


@pytest.fixture
def fake_frame():
    """
    Фейковый BGR-кадр 480x640x3
    (имитация выхода cv2.VideoCapture).
    """
    return np.random.randint(
        0, 255, (480, 640, 3), dtype=np.uint8
    )


@pytest.fixture
def species_labels_dict():
    """
    Словарь species_labels (10 классов).
    """
    return {
        "0": {
            "ru": "Большая синица",
            "en": "Great Tit",
        },
        "1": {
            "ru": "Лазоревка",
            "en": "Blue Tit",
        },
        "2": {
            "ru": "Домовый воробей",
            "en": "House Sparrow",
        },
        "3": {
            "ru": "Полевой воробей",
            "en": "Tree Sparrow",
        },
        "4": {
            "ru": "Поползень",
            "en": "Nuthatch",
        },
        "5": {
            "ru": "Снегирь",
            "en": "Bullfinch",
        },
        "6": {
            "ru": "Большой пёстрый дятел",
            "en": "Great Spotted Woodpecker",
        },
        "7": {
            "ru": "Зеленушка",
            "en": "Greenfinch",
        },
        "8": {
            "ru": "Чиж",
            "en": "Siskin",
        },
        "9": {
            "ru": "Свиристель",
            "en": "Waxwing",
        },
    }


@pytest.fixture
def species_labels_path(tmp_dir, species_labels_dict):
    """
    Путь к временному species_labels.json.
    """
    path = os.path.join(
        tmp_dir, "species_labels.json"
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            species_labels_dict, f,
            ensure_ascii=False,
        )
    return path


@pytest.fixture
def analytics_instance(tmp_dir):
    """
    Готовый экземпляр FeederAnalytics
    во временной директории.
    """
    from feeder_analytics import FeederAnalytics

    analytics_dir = os.path.join(
        tmp_dir, "analytics"
    )
    return FeederAnalytics(
        analytics_dir=analytics_dir,
        default_food_type="seeds",
    )
