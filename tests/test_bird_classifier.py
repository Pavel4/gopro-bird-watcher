"""
Тесты для detector/bird_classifier.py

Тестируем: Detection, SpeciesResult, ClassificationResult,
BirdDetector, SpeciesClassifier, BirdClassifier.
"""

import json
import os
from unittest.mock import (
    MagicMock, patch, PropertyMock,
)

import numpy as np
import pytest

from bird_classifier import (
    Detection,
    SpeciesResult,
    ClassificationResult,
    BirdDetector,
    SpeciesClassifier,
    BirdClassifier,
    COCO_BIRD_CLASS,
    COCO_CLASSES,
)


# === Dataclass тесты ===


class TestDetection:
    def test_create(self):
        d = Detection(
            class_id=14, class_name="bird",
            confidence=0.9,
            x=10, y=20, width=100, height=80,
        )
        assert d.class_id == 14
        assert d.class_name == "bird"
        assert d.confidence == 0.9
        assert d.x == 10
        assert d.width == 100

    def test_coco_bird_class_index(self):
        assert COCO_BIRD_CLASS == 14
        assert COCO_CLASSES[14] == "bird"


class TestSpeciesResult:
    def test_create(self):
        sr = SpeciesResult(
            species_ru="Большая синица",
            species_en="Great Tit",
            confidence=0.85,
        )
        assert sr.species_ru == "Большая синица"
        assert sr.species_en == "Great Tit"
        assert sr.confidence == 0.85


class TestClassificationResult:
    def test_defaults(self):
        cr = ClassificationResult()
        assert cr.bird_detected is False
        assert cr.detections == []
        assert cr.species is None
        assert cr.best_detection is None

    def test_with_detection(self):
        det = Detection(
            class_id=14, class_name="bird",
            confidence=0.8,
            x=0, y=0, width=50, height=50,
        )
        cr = ClassificationResult(
            bird_detected=True,
            detections=[det],
            best_detection=det,
        )
        assert cr.bird_detected is True
        assert len(cr.detections) == 1

    def test_with_species(self):
        sp = SpeciesResult(
            species_ru="Поползень",
            species_en="Nuthatch",
            confidence=0.7,
        )
        cr = ClassificationResult(
            bird_detected=True,
            species=sp,
        )
        assert cr.species.species_en == "Nuthatch"


# === BirdDetector тесты ===


class TestBirdDetector:
    def test_no_model_file(self, tmp_dir):
        """
        BirdDetector без файла модели
        должен gracefully не загружать session.
        """
        fake_path = os.path.join(
            tmp_dir, "nonexistent.onnx"
        )
        detector = BirdDetector(
            model_path=fake_path
        )
        assert detector.session is None

    def test_detect_returns_empty_no_session(
        self, fake_frame,
    ):
        """detect() без session возвращает []."""
        detector = BirdDetector.__new__(
            BirdDetector
        )
        detector.session = None
        detector.logger = MagicMock()
        result = detector.detect(fake_frame)
        assert result == []

    def test_letterbox_shape(self, fake_frame):
        """_letterbox возвращает изображение 640x640."""
        detector = BirdDetector.__new__(
            BirdDetector
        )
        img, pad = detector._letterbox(
            fake_frame, (640, 640)
        )
        assert img.shape[0] == 640
        assert img.shape[1] == 640
        assert len(pad) == 2

    def test_letterbox_preserves_channels(
        self, fake_frame,
    ):
        """_letterbox сохраняет 3 канала."""
        detector = BirdDetector.__new__(
            BirdDetector
        )
        img, _ = detector._letterbox(
            fake_frame, (640, 640)
        )
        assert img.shape[2] == 3


# === SpeciesClassifier тесты ===


class TestSpeciesClassifier:
    def test_load_labels(
        self, tmp_dir, species_labels_path,
    ):
        """Загрузка species_labels.json."""
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.labels = {}
        classifier.logger = MagicMock()
        classifier.labels_path = species_labels_path
        classifier._load_labels()
        assert len(classifier.labels) == 10
        assert "0" in classifier.labels
        assert (
            classifier.labels["0"]["ru"]
            == "Большая синица"
        )

    def test_load_labels_missing_file(
        self, tmp_dir,
    ):
        """
        Отсутствующий labels file
        не крашит, просто warning.
        """
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.labels = {}
        classifier.logger = MagicMock()
        classifier.labels_path = os.path.join(
            tmp_dir, "missing.json"
        )
        classifier._load_labels()
        assert classifier.labels == {}

    def test_not_ready_without_model(
        self, tmp_dir, species_labels_path,
    ):
        """
        is_ready() = False без ONNX session
        и head weights.
        """
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.session = None
        classifier.head_weight = None
        classifier.labels = {"0": {"ru": "T"}}
        assert classifier.is_ready() is False

    def test_not_ready_without_labels(self):
        """is_ready() = False без labels."""
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.session = MagicMock()
        classifier.head_weight = np.zeros((10, 512))
        classifier.labels = {}
        assert classifier.is_ready() is False

    def test_ready_with_all_components(self):
        """is_ready() = True со всеми компонентами."""
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.session = MagicMock()
        classifier.head_weight = np.zeros((10, 512))
        classifier.labels = {"0": {"ru": "T"}}
        assert classifier.is_ready() is True

    def test_preprocess_clip_shape(
        self, fake_frame,
    ):
        """
        _preprocess_clip возвращает
        tensor [1, 3, 224, 224] float32.
        """
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        # Берём кроп из кадра
        crop = fake_frame[100:300, 200:400]
        result = classifier._preprocess_clip(crop)
        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_classify_with_mock_session(
        self, fake_frame,
    ):
        """
        classify() с мок ONNX session
        возвращает SpeciesResult.
        """
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.logger = MagicMock()
        classifier.confidence_threshold = 0.1
        classifier.labels = {
            "0": {
                "ru": "Большая синица",
                "en": "Great Tit",
            },
            "1": {
                "ru": "Лазоревка",
                "en": "Blue Tit",
            },
        }

        # Мок head: 2 класса, 512-dim
        classifier.head_weight = np.random.randn(
            2, 512
        ).astype(np.float32)
        classifier.head_bias = np.zeros(
            2, dtype=np.float32
        )

        # Мок ONNX session
        mock_session = MagicMock()
        mock_input = MagicMock()
        mock_input.name = "pixel_values"
        mock_session.get_inputs.return_value = [
            mock_input
        ]
        # Возвращаем фейковый 512-dim embedding
        fake_embedding = np.random.randn(
            1, 512
        ).astype(np.float32)
        mock_session.run.return_value = [
            fake_embedding
        ]
        classifier.session = mock_session

        bbox = (200, 100, 200, 200)
        result = classifier.classify(
            fake_frame, bbox
        )
        assert result is not None
        assert isinstance(result, SpeciesResult)
        assert result.confidence > 0
        assert result.species_ru in (
            "Большая синица", "Лазоревка",
        )

    def test_classify_returns_none_not_ready(
        self, fake_frame,
    ):
        """classify() возвращает None если не ready."""
        classifier = SpeciesClassifier.__new__(
            SpeciesClassifier
        )
        classifier.session = None
        classifier.head_weight = None
        classifier.labels = {}
        classifier.confidence_threshold = 0.1
        classifier.logger = MagicMock()

        result = classifier.classify(
            fake_frame, (0, 0, 50, 50)
        )
        assert result is None


# === BirdClassifier тесты ===


class TestBirdClassifier:
    @patch(
        "bird_classifier.BirdDetector",
        return_value=MagicMock(),
    )
    @patch.object(
        BirdClassifier, "_download_models",
    )
    def test_init_creates_dirs(
        self, mock_dl, mock_det, tmp_dir,
    ):
        """
        BirdClassifier создаёт model_dir
        и crops_dir (без реальных HTTP-запросов).
        """
        model_dir = os.path.join(
            tmp_dir, "models"
        )
        crops_dir = os.path.join(
            tmp_dir, "crops"
        )
        bc = BirdClassifier(
            model_dir=model_dir,
            crops_dir=crops_dir,
            save_crops=True,
        )
        assert os.path.isdir(model_dir)
        assert os.path.isdir(crops_dir)
        mock_dl.assert_called_once()

    @patch(
        "bird_classifier.BirdDetector",
        return_value=MagicMock(),
    )
    @patch.object(
        BirdClassifier, "_download_models",
    )
    def test_species_threshold_capped(
        self, mock_dl, mock_det,
        tmp_dir, species_labels_path,
    ):
        """
        Порог species должен быть <= 0.15,
        даже если confidence_threshold=0.5.
        """
        model_dir = os.path.join(
            tmp_dir, "models"
        )
        os.makedirs(model_dir, exist_ok=True)

        # Копируем labels в model_dir
        import shutil
        dst = os.path.join(
            model_dir, "species_labels.json"
        )
        shutil.copy(species_labels_path, dst)

        bc = BirdClassifier(
            model_dir=model_dir,
            confidence_threshold=0.5,
            species_enabled=True,
            save_crops=False,
        )

        assert bc.species_classifier is not None
        assert (
            bc.species_classifier
            .confidence_threshold <= 0.15
        )

    def test_get_species_name_with_species(self):
        """get_species_name с видом."""
        bc = BirdClassifier.__new__(BirdClassifier)
        sp = SpeciesResult(
            species_ru="Снегирь",
            species_en="Bullfinch",
            confidence=0.8,
        )
        cr = ClassificationResult(
            bird_detected=True, species=sp,
        )
        assert bc.get_species_name(cr) == "Снегирь"

    def test_get_species_name_bird_only(self):
        """get_species_name без вида -> 'Птица'."""
        bc = BirdClassifier.__new__(BirdClassifier)
        cr = ClassificationResult(
            bird_detected=True, species=None,
        )
        assert bc.get_species_name(cr) == "Птица"

    def test_get_species_name_no_bird(self):
        """get_species_name без детекции -> ''."""
        bc = BirdClassifier.__new__(BirdClassifier)
        cr = ClassificationResult()
        assert bc.get_species_name(cr) == ""

    def test_get_caption_info_no_bird(self):
        """get_caption_info без детекции."""
        bc = BirdClassifier.__new__(BirdClassifier)
        cr = ClassificationResult()
        info = bc.get_caption_info(cr)
        assert info["name"] == "Движение"
        assert info["species"] is None

    def test_get_caption_info_bird_no_species(
        self,
    ):
        """get_caption_info с bird, без species."""
        bc = BirdClassifier.__new__(BirdClassifier)
        det = Detection(
            class_id=14, class_name="bird",
            confidence=0.9,
            x=0, y=0, width=50, height=50,
        )
        cr = ClassificationResult(
            bird_detected=True,
            best_detection=det,
        )
        info = bc.get_caption_info(cr)
        assert info["name"] == "Птица"
        assert info["confidence"] == 0.9
        assert info["species"] is None

    def test_get_caption_info_with_species(self):
        """get_caption_info с видом."""
        bc = BirdClassifier.__new__(BirdClassifier)
        sp = SpeciesResult(
            species_ru="Чиж",
            species_en="Siskin",
            confidence=0.75,
        )
        cr = ClassificationResult(
            bird_detected=True, species=sp,
        )
        info = bc.get_caption_info(cr)
        assert info["name"] == "Чиж"
        assert info["confidence"] == 0.75
        assert info["species"] == "Siskin"

    def test_save_crop_creates_files(
        self, tmp_dir, fake_frame,
    ):
        """save_crop сохраняет файлы в crops_dir."""
        crops_dir = os.path.join(
            tmp_dir, "crops"
        )
        os.makedirs(crops_dir, exist_ok=True)

        bc = BirdClassifier.__new__(BirdClassifier)
        bc.save_crops = True
        bc.crops_dir = crops_dir
        bc.logger = MagicMock()

        det = Detection(
            class_id=14, class_name="bird",
            confidence=0.9,
            x=100, y=100, width=200, height=150,
        )
        cr = ClassificationResult(
            bird_detected=True,
            detections=[det],
            best_detection=det,
        )
        bc.save_crop(fake_frame, cr, visit_id=1)

        # Проверяем что файлы появились
        date_dirs = os.listdir(crops_dir)
        assert len(date_dirs) == 1
        files = os.listdir(
            os.path.join(crops_dir, date_dirs[0])
        )
        assert len(files) >= 2  # full + bird_0
        filenames = sorted(files)
        assert any(
            "full" in f for f in filenames
        )
        assert any(
            "bird_0" in f for f in filenames
        )

    def test_save_crop_disabled(
        self, fake_frame,
    ):
        """save_crop ничего не делает если off."""
        bc = BirdClassifier.__new__(BirdClassifier)
        bc.save_crops = False
        bc.logger = MagicMock()

        det = Detection(
            class_id=14, class_name="bird",
            confidence=0.9,
            x=0, y=0, width=50, height=50,
        )
        cr = ClassificationResult(
            bird_detected=True,
            detections=[det],
            best_detection=det,
        )
        # Не должно крашиться
        bc.save_crop(fake_frame, cr, visit_id=1)
