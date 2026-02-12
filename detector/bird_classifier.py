#!/usr/bin/env python3
"""
Bird Classifier для GoPro Bird Watcher
Классификация видов (Vogel EfficientNet-B2)
+ распознавание поведения (TSM-MobileNetV3).

Пайплайн:
1. Frame diff → bounding box области движения
2. VogelClassifier → вид птицы (8 классов)
3. BehaviorClassifier → поведение (TSM)

Поддержка форматов моделей:
  - TorchScript (.pt) — приоритетный
  - ONNX (.onnx) — fallback

Зависимости: numpy, opencv, torch или onnxruntime
"""

import os
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone, timedelta

from collections import deque

import cv2
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Московское время (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))

# Vogel EfficientNet-B2 модель
# Приоритет: TorchScript (.pt) > ONNX (.onnx)
VOGEL_PT_FILE = "vogel_bird_classifier.pt"
VOGEL_ONNX_FILE = "vogel_bird_classifier.onnx"
VOGEL_PREPROCESS_FILE = "vogel_preprocess.json"
SPECIES_LABELS_FILE = "species_labels.json"

# Дефолтные ImageNet mean/std для EfficientNet
IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406], dtype=np.float32,
)
IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225], dtype=np.float32,
)


@dataclass
class SpeciesResult:
    """Результат классификации вида."""
    species_ru: str
    species_en: str
    confidence: float


# === Поведение птиц ===

# 7 классов поведения для кормушки
BEHAVIOR_CLASSES = {
    0: {"ru": "Кормление", "en": "feeding"},
    1: {
        "ru": "Схватил и улетел",
        "en": "grab_fly",
    },
    2: {"ru": "Сидение", "en": "perching"},
    3: {"ru": "Озирание", "en": "alert"},
    4: {"ru": "Драка", "en": "fighting"},
    5: {"ru": "Прилёт", "en": "arrival"},
    6: {"ru": "Улёт", "en": "departure"},
}

BEHAVIOR_LABELS_FILE = "behavior_labels.json"
BEHAVIOR_MODEL_FILE = "behavior_tsm.onnx"


@dataclass
class BehaviorResult:
    """Результат классификации поведения."""
    behavior_ru: str
    behavior_en: str
    confidence: float
    class_id: int = -1


@dataclass
class ClassificationResult:
    """Полный результат классификации кадра."""
    bird_detected: bool = False
    species: Optional[SpeciesResult] = None
    behavior: Optional[BehaviorResult] = None
    bird_count: int = 0
    # Метод детекции: "vogel", "voting"
    detection_method: str = "vogel"
    # Кол-во голосов при голосовании
    votes: int = 0
    total_frames: int = 0


class VogelClassifier:
    """
    Классификатор видов птиц на основе
    Vogel german-bird-classifier-v2
    (EfficientNet-B2, ONNX).

    Вход: BGR кроп (numpy array)
    Выход: SpeciesResult или None

    Preprocessing: ImageNet normalization
    (resize 224x224, /255, normalize mean/std).
    """

    INPUT_SIZE = 224

    def __init__(
        self,
        model_dir: str,
        confidence_threshold: float = 0.5,
        logger: logging.Logger = None,
    ):
        self.model_dir = model_dir
        self.confidence_threshold = (
            confidence_threshold
        )
        self.logger = (
            logger or logging.getLogger(__name__)
        )
        # ONNX session (fallback)
        self.session = None
        # TorchScript model (primary)
        self.ts_model = None
        self.backend = None  # "torchscript"|"onnx"
        self.labels = {}
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

        self._load_preprocess_config()
        self._load_labels()
        self._load_model()

    def _load_preprocess_config(self):
        """Загрузить конфиг preprocessing."""
        prep_path = os.path.join(
            self.model_dir, VOGEL_PREPROCESS_FILE,
        )
        if not os.path.exists(prep_path):
            return
        try:
            with open(
                prep_path, "r", encoding="utf-8"
            ) as f:
                cfg = json.load(f)
            if "mean" in cfg:
                self.mean = np.array(
                    cfg["mean"], dtype=np.float32
                )
            if "std" in cfg:
                self.std = np.array(
                    cfg["std"], dtype=np.float32
                )
            if "image_size" in cfg:
                self.INPUT_SIZE = cfg["image_size"]
            self.logger.info(
                f"  Vogel preprocess config loaded"
            )
        except Exception as e:
            self.logger.warning(
                f"  Cannot load preprocess "
                f"config: {e}"
            )

    def _load_labels(self):
        """Загрузить маппинг классов из JSON."""
        labels_path = os.path.join(
            self.model_dir, SPECIES_LABELS_FILE,
        )
        if not os.path.exists(labels_path):
            self.logger.warning(
                f"  Species labels not found: "
                f"{labels_path}"
            )
            return
        try:
            with open(
                labels_path, "r", encoding="utf-8"
            ) as f:
                self.labels = json.load(f)
            self.logger.info(
                f"  Species labels: "
                f"{len(self.labels)} classes"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load species "
                f"labels: {e}"
            )

    def _load_model(self):
        """
        Загрузить модель. Приоритет:
        1. TorchScript (.pt) — надёжный
        2. ONNX (.onnx) — fallback
        """
        pt_path = os.path.join(
            self.model_dir, VOGEL_PT_FILE,
        )
        onnx_path = os.path.join(
            self.model_dir, VOGEL_ONNX_FILE,
        )

        # --- TorchScript (.pt) ---
        if (
            os.path.exists(pt_path)
            and TORCH_AVAILABLE
        ):
            try:
                self.ts_model = torch.jit.load(
                    pt_path,
                    map_location="cpu",
                )
                self.ts_model.eval()
                self.backend = "torchscript"
                size_mb = (
                    os.path.getsize(pt_path)
                    / (1024 ** 2)
                )
                self.logger.info(
                    f"  🐦 Vogel model loaded: "
                    f"{VOGEL_PT_FILE}"
                    f" ({size_mb:.1f} MB), "
                    f"backend=TorchScript"
                )
                return
            except Exception as e:
                self.logger.warning(
                    f"  TorchScript load failed"
                    f": {e}, trying ONNX..."
                )

        # --- ONNX (.onnx) fallback ---
        if (
            os.path.exists(onnx_path)
            and ONNX_AVAILABLE
        ):
            try:
                self._load_onnx_model(
                    onnx_path
                )
                return
            except Exception as e:
                self.logger.error(
                    f"Failed to load ONNX: {e}"
                )

        # --- Ничего не найдено ---
        if os.path.exists(pt_path):
            self.logger.error(
                "torch not installed. "
                "Install: pip install torch"
            )
        elif os.path.exists(onnx_path):
            self.logger.error(
                "onnxruntime not installed. "
                "Install: pip install "
                "onnxruntime"
            )
        else:
            self.logger.warning(
                f"  Vogel model not found. "
                f"Run: python scripts/"
                f"export_vogel_onnx.py"
            )

    def _load_onnx_model(self, model_path):
        """Загрузить ONNX модель (fallback)."""
        providers = (
            ort.get_available_providers()
        )
        preferred = [
            p for p in (
                "CoreMLExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            )
            if p in providers
        ]
        # CoreML может глючить — fallback
        try:
            self.session = (
                ort.InferenceSession(
                    model_path,
                    providers=(
                        preferred or providers
                    ),
                )
            )
        except Exception:
            self.logger.warning(
                "  CoreML failed, "
                "trying CPU..."
            )
            self.session = (
                ort.InferenceSession(
                    model_path,
                    providers=[
                        "CPUExecutionProvider"
                    ],
                )
            )

        self.backend = "onnx"
        size_mb = (
            os.path.getsize(model_path)
            / (1024 ** 2)
        )
        used_provider = (
            self.session.get_providers()[0]
            if self.session
            else "none"
        )
        self.logger.info(
            f"  🐦 Vogel model loaded: "
            f"{VOGEL_ONNX_FILE}"
            f" ({size_mb:.1f} MB), "
            f"provider={used_provider}"
        )

    def is_ready(self) -> bool:
        """Проверить готовность."""
        has_model = (
            self.ts_model is not None
            or self.session is not None
        )
        return (
            has_model
            and len(self.labels) > 0
        )

    def _preprocess(
        self, crop: np.ndarray
    ) -> np.ndarray:
        """
        EfficientNet preprocessing:
        BGR → RGB → resize 224 → /255 →
        normalize(mean, std) → CHW → batch.
        """
        rgb = cv2.cvtColor(
            crop, cv2.COLOR_BGR2RGB
        )
        resized = cv2.resize(
            rgb,
            (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR,
        )
        img = resized.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # → [1, 3, H, W]
        return img[np.newaxis, ...]

    def classify(
        self, crop: np.ndarray,
    ) -> Optional[SpeciesResult]:
        """
        Классифицировать кроп изображения.

        Args:
            crop: BGR изображение (numpy array)
        Returns:
            SpeciesResult или None
        """
        if not self.is_ready():
            return None
        if crop.size == 0:
            return None

        try:
            img_data = self._preprocess(crop)
            logits = self._run_inference(
                img_data
            )
            if logits is None:
                return None

            # Softmax
            shifted = (
                logits
                - np.max(
                    logits, axis=-1,
                    keepdims=True,
                )
            )
            exp_l = np.exp(shifted)
            probs = (
                exp_l
                / exp_l.sum(
                    axis=-1, keepdims=True
                )
            )

            class_idx = int(np.argmax(probs[0]))
            confidence = float(
                probs[0, class_idx]
            )

            label = self.labels.get(
                str(class_idx), {}
            )

            # Логируем все вероятности (debug)
            top3_idx = np.argsort(
                probs[0]
            )[::-1][:3]
            top3_str = ", ".join(
                f"{self.labels.get(str(i), {}).get('ru', '?')}"
                f"={probs[0, i]:.0%}"
                for i in top3_idx
            )
            self.logger.debug(
                f"  Vogel top-3: {top3_str}"
            )

            # Логируем низкую уверенность,
            # но НЕ отбрасываем — решение
            # принимает голосование в
            # _classify_from_video.
            ru_name = label.get('ru', '?')
            if confidence < self.confidence_threshold:
                self.logger.info(
                    f"  🐦 Vogel: "
                    f"{ru_name} "
                    f"{confidence:.0%} "
                    f"(low conf)"
                )
            else:
                self.logger.info(
                    f"  🐦 Vogel: "
                    f"{ru_name} "
                    f"{confidence:.0%}"
                )

            return SpeciesResult(
                species_ru=label.get(
                    "ru", "Неизвестный вид"
                ),
                species_en=label.get(
                    "en", "Unknown species"
                ),
                confidence=confidence,
            )
        except Exception as e:
            self.logger.error(
                f"Vogel classification error: "
                f"{e}"
            )
            return None

    def _run_inference(
        self, img_data: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Запустить inference на preprocessed
        данных. Поддерживает TorchScript и ONNX.

        Args:
            img_data: [1, 3, 224, 224] float32
        Returns:
            logits [1, num_classes] или None
        """
        if self.backend == "torchscript":
            tensor = torch.from_numpy(img_data)
            with torch.no_grad():
                logits = self.ts_model(tensor)
            return logits.numpy()
        elif self.backend == "onnx":
            model_inputs = (
                self.session.get_inputs()
            )
            outputs = self.session.run(
                None,
                {
                    model_inputs[0].name:
                        img_data,
                },
            )
            return outputs[0]
        return None

    def classify_with_probs(
        self, crop: np.ndarray,
    ) -> tuple:
        """
        Classify и вернуть (SpeciesResult, probs).
        Используется для голосования.

        Returns:
            (SpeciesResult or None, probs_array)
        """
        if not self.is_ready():
            return None, None
        if crop.size == 0:
            return None, None

        try:
            img_data = self._preprocess(crop)
            logits = self._run_inference(
                img_data
            )
            if logits is None:
                return None, None

            shifted = (
                logits
                - np.max(
                    logits, axis=-1,
                    keepdims=True,
                )
            )
            exp_l = np.exp(shifted)
            probs = (
                exp_l
                / exp_l.sum(
                    axis=-1, keepdims=True
                )
            )

            class_idx = int(np.argmax(probs[0]))
            confidence = float(
                probs[0, class_idx]
            )

            label = self.labels.get(
                str(class_idx), {}
            )

            result = SpeciesResult(
                species_ru=label.get(
                    "ru", "Неизвестный вид"
                ),
                species_en=label.get(
                    "en", "Unknown species"
                ),
                confidence=confidence,
            )
            return result, probs[0]
        except Exception as e:
            self.logger.error(
                f"Vogel classification error: "
                f"{e}"
            )
            return None, None


class BehaviorClassifier:
    """
    TSM-MobileNetV3 классификатор поведения птиц.

    Принимает буфер из N кадров (кропов птицы),
    классифицирует поведение:
    feeding, perching, alert, fighting,
    arrival, departure.

    Архитектура:
    - Temporal Shift Module (TSM) — сдвиг части
      каналов между кадрами для temporal reasoning
    - MobileNetV3 backbone — лёгкий ~10MB
    - Вход: N кропов 224x224 (N = 8 или 16)
    - Выход: 7 классов поведения

    ONNX модель: behavior_tsm.onnx
    """

    INPUT_SIZE = 224
    # ImageNet normalization (MobileNetV3)
    MEAN = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    STD = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    def __init__(
        self,
        model_dir: str,
        confidence_threshold: float = 0.4,
        num_frames: int = 8,
        logger: logging.Logger = None,
    ):
        self.model_dir = model_dir
        self.confidence_threshold = (
            confidence_threshold
        )
        self.num_frames = num_frames
        self.logger = (
            logger or logging.getLogger(__name__)
        )
        self.session = None
        self.labels = dict(BEHAVIOR_CLASSES)

        self._load_labels()
        self._load_model()

    def _load_labels(self):
        """
        Загрузить маппинг классов из JSON
        (если есть кастомный файл).
        """
        labels_path = os.path.join(
            self.model_dir, BEHAVIOR_LABELS_FILE
        )
        if not os.path.exists(labels_path):
            # Используем дефолтные
            return
        try:
            with open(
                labels_path, "r", encoding="utf-8"
            ) as f:
                custom = json.load(f)
            # Мержим с дефолтными
            for k, v in custom.items():
                self.labels[int(k)] = v
            self.logger.info(
                f"  Behavior labels loaded: "
                f"{len(self.labels)} classes"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load behavior "
                f"labels: {e}"
            )

    def _load_model(self):
        """Загрузить ONNX модель поведения."""
        if not ONNX_AVAILABLE:
            return

        model_path = os.path.join(
            self.model_dir, BEHAVIOR_MODEL_FILE
        )
        if not os.path.exists(model_path):
            self.logger.info(
                f"  Behavior model not found "
                f"at {model_path}. "
                f"Run: python scripts/"
                f"train_behavior_classifier.py"
            )
            return

        try:
            providers = (
                ort.get_available_providers()
            )
            preferred = [
                p for p in (
                    "CoreMLExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
                if p in providers
            ]
            self.session = ort.InferenceSession(
                model_path,
                providers=preferred or providers,
            )
            size_mb = (
                os.path.getsize(model_path)
                / (1024 ** 2)
            )
            self.logger.info(
                f"  🎭 Behavior model loaded: "
                f"{BEHAVIOR_MODEL_FILE}"
                f" ({size_mb:.1f} MB), "
                f"provider="
                f"{preferred[0] if preferred else 'auto'}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load behavior "
                f"model: {e}"
            )
            self.session = None

    def is_ready(self) -> bool:
        """Проверить готовность классификатора."""
        return self.session is not None

    def _preprocess_frames(
        self,
        crops: List[np.ndarray],
    ) -> np.ndarray:
        """
        Preprocessing N кропов для TSM.

        Args:
            crops: список BGR кропов птицы
        Returns:
            float32 tensor [1, N, 3, 224, 224]
        """
        processed = []
        for crop in crops:
            rgb = cv2.cvtColor(
                crop, cv2.COLOR_BGR2RGB
            )
            resized = cv2.resize(
                rgb,
                (self.INPUT_SIZE, self.INPUT_SIZE),
                interpolation=cv2.INTER_LINEAR,
            )
            img = (
                resized.astype(np.float32) / 255.0
            )
            img = (img - self.MEAN) / self.STD
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            processed.append(img)

        # [N, 3, H, W] -> [1, N, 3, H, W]
        batch = np.stack(
            processed, axis=0
        )[np.newaxis, ...]
        return batch

    def _sample_frames(
        self,
        frame_buffer: deque,
        bbox: tuple,
        full_frame_h: int,
        full_frame_w: int,
    ) -> List[np.ndarray]:
        """
        Выбрать N кадров из буфера и кропнуть
        по bbox с padding 20%.
        """
        x, y, w, h = bbox
        pad = int(max(w, h) * 0.2)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(full_frame_w, x + w + pad)
        y2 = min(full_frame_h, y + h + pad)

        frames = list(frame_buffer)
        n = len(frames)

        if n == 0:
            return []

        # Равномерная выборка N кадров
        if n >= self.num_frames:
            indices = np.linspace(
                0, n - 1,
                self.num_frames,
                dtype=int,
            )
        else:
            indices = list(range(n))
            while len(indices) < self.num_frames:
                indices.append(n - 1)

        crops = []
        for idx in indices:
            frame = frames[idx]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crops.append(crop)
            else:
                crops.append(
                    frame[
                        full_frame_h // 4:
                        3 * full_frame_h // 4,
                        full_frame_w // 4:
                        3 * full_frame_w // 4,
                    ]
                )
        return crops

    def classify(
        self,
        frame_buffer: deque,
        bbox: tuple,
        frame_shape: tuple,
    ) -> Optional[BehaviorResult]:
        """
        Классифицировать поведение птицы
        по буферу кадров.
        """
        if not self.is_ready():
            return None

        if len(frame_buffer) < 2:
            return None

        h, w = frame_shape[:2]
        crops = self._sample_frames(
            frame_buffer, bbox, h, w
        )
        if len(crops) < self.num_frames:
            return None

        try:
            input_data = self._preprocess_frames(
                crops
            )
            model_inputs = (
                self.session.get_inputs()
            )
            outputs = self.session.run(
                None,
                {model_inputs[0].name: input_data},
            )
            logits = outputs[0]  # (1, num_classes)

            # Softmax
            shifted = (
                logits
                - np.max(
                    logits, axis=-1, keepdims=True
                )
            )
            exp_l = np.exp(shifted)
            probs = (
                exp_l
                / exp_l.sum(
                    axis=-1, keepdims=True
                )
            )

            class_idx = int(np.argmax(probs[0]))
            confidence = float(
                probs[0, class_idx]
            )

            if (
                confidence
                < self.confidence_threshold
            ):
                return None

            label = self.labels.get(
                class_idx, {}
            )
            return BehaviorResult(
                behavior_ru=label.get(
                    "ru", "Неизвестно"
                ),
                behavior_en=label.get(
                    "en", "unknown"
                ),
                confidence=confidence,
                class_id=class_idx,
            )
        except Exception as e:
            self.logger.error(
                f"Behavior classification "
                f"error: {e}"
            )
            return None


class BirdClassifier:
    """
    Главный класс: классификация видов птиц
    (Vogel EfficientNet-B2) + поведения (TSM).

    Без YOLO — детекция через frame diff
    выполняется в motion_detector.py.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        confidence_threshold: float = 0.5,
        species_enabled: bool = True,
        behavior_enabled: bool = False,
        behavior_num_frames: int = 8,
        behavior_confidence: float = 0.4,
        save_crops: bool = True,
        crops_dir: str = "./crops",
        logger: logging.Logger = None,
    ):
        self.model_dir = model_dir
        self.confidence_threshold = (
            confidence_threshold
        )
        self.species_enabled = species_enabled
        self.behavior_enabled = behavior_enabled
        self.save_crops = save_crops
        self.crops_dir = crops_dir
        self.logger = (
            logger or logging.getLogger(__name__)
        )

        self.vogel_classifier = None
        self.behavior_classifier = None

        if (
            not TORCH_AVAILABLE
            and not ONNX_AVAILABLE
        ):
            self.logger.error(
                "❌ Neither torch nor "
                "onnxruntime installed! "
                "ML disabled. Install: "
                "pip install torch"
            )
            return

        # Создаём директории
        os.makedirs(self.model_dir, exist_ok=True)
        if self.save_crops:
            os.makedirs(
                self.crops_dir, exist_ok=True
            )

        # Инициализация Vogel классификатора
        if self.species_enabled:
            self.vogel_classifier = (
                VogelClassifier(
                    model_dir=self.model_dir,
                    confidence_threshold=(
                        self.confidence_threshold
                    ),
                    logger=self.logger,
                )
            )
            if not self.vogel_classifier.is_ready():
                self.logger.warning(
                    "  ⚠️ Vogel classifier not "
                    "ready — model not found. "
                    "Run: python scripts/"
                    "export_vogel_onnx.py"
                )

        # Инициализация BehaviorClassifier
        if self.behavior_enabled:
            self.behavior_classifier = (
                BehaviorClassifier(
                    model_dir=self.model_dir,
                    confidence_threshold=(
                        behavior_confidence
                    ),
                    num_frames=behavior_num_frames,
                    logger=self.logger,
                )
            )
            if not self.behavior_classifier.is_ready():
                self.logger.info(
                    "  Behavior classifier not "
                    "ready — model not found."
                )

    def is_available(self) -> bool:
        """Проверить доступен ли классификатор."""
        return (
            self.vogel_classifier is not None
            and self.vogel_classifier.is_ready()
        )

    def classify_crop(
        self, crop: np.ndarray,
    ) -> Optional[SpeciesResult]:
        """
        Классифицировать один кроп.

        Args:
            crop: BGR кроп (numpy array)
        Returns:
            SpeciesResult или None
        """
        if not self.is_available():
            return None
        return self.vogel_classifier.classify(crop)

    def classify_crop_with_probs(
        self, crop: np.ndarray,
    ) -> tuple:
        """
        Классифицировать кроп и вернуть probs.

        Returns:
            (SpeciesResult or None, probs_array)
        """
        if not self.is_available():
            return None, None
        return (
            self.vogel_classifier
            .classify_with_probs(crop)
        )

    def get_species_name(
        self, result: ClassificationResult
    ) -> str:
        """Получить название вида."""
        if result.species:
            return result.species.species_ru
        if result.bird_detected:
            return "Птица"
        return ""

    def get_caption_info(
        self, result: ClassificationResult
    ) -> dict:
        """
        Данные для caption в Telegram.

        Returns:
            dict с полями name, confidence,
            species, behavior, bird_count и т.д.
        """
        info = {
            "name": "Движение",
            "confidence": 0.0,
            "species": None,
            "species_en": None,
            "behavior": None,
            "behavior_en": None,
            "behavior_confidence": 0.0,
            "bird_count": 0,
            "detection_method": (
                result.detection_method
            ),
            "votes": result.votes,
            "total_frames": result.total_frames,
        }

        if not result.bird_detected:
            return info

        info["bird_count"] = result.bird_count

        if result.species:
            info["name"] = (
                result.species.species_ru
            )
            info["confidence"] = (
                result.species.confidence
            )
            info["species"] = (
                result.species.species_en
            )
            info["species_en"] = (
                result.species.species_en
            )
        else:
            info["name"] = "Птица"

        if result.behavior:
            info["behavior"] = (
                result.behavior.behavior_ru
            )
            info["behavior_en"] = (
                result.behavior.behavior_en
            )
            info["behavior_confidence"] = (
                result.behavior.confidence
            )

        return info
