#!/usr/bin/env python3
"""
Bird Classifier для GoPro Bird Watcher
Детекция птиц (YOLOv8n ONNX) + классификация видов.

Двухстадийный пайплайн:
1. YOLOv8n — детекция птицы в кадре (COCO class 14)
2. Species Classifier — определение вида (опционально)

Зависимости: onnxruntime, numpy, opencv
"""

import os
import json
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    ort = None

# Московское время (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))

# COCO class index для "bird"
COCO_BIRD_CLASS = 14

# URL для автозагрузки YOLOv8n ONNX
YOLOV8N_URL = (
    "https://huggingface.co/SpotLab/YOLOv8Detection"
    "/resolve/3005c6751fb19cdeb6b10c066185908faf66a097"
    "/yolov8n.onnx?download=true"
)
YOLOV8N_FILENAME = "yolov8n.onnx"

# 80 COCO class names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle",
    "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass
class Detection:
    """Результат детекции объекта."""
    class_id: int
    class_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass
class SpeciesResult:
    """Результат классификации вида."""
    species_ru: str
    species_en: str
    confidence: float


@dataclass
class ClassificationResult:
    """Полный результат классификации кадра."""
    bird_detected: bool = False
    detections: List[Detection] = field(
        default_factory=list
    )
    species: Optional[SpeciesResult] = None
    best_detection: Optional[Detection] = None


class BirdDetector:
    """
    Детекция птиц через YOLOv8n ONNX.
    Фильтрует только COCO class 14 = "bird".
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        logger: logging.Logger = None,
    ):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.logger = (
            logger or logging.getLogger(__name__)
        )
        self.session = None
        self.input_width = 640
        self.input_height = 640

        self._load_model()

    def _load_model(self):
        """Загрузить ONNX модель."""
        if not ONNX_AVAILABLE:
            self.logger.error(
                "onnxruntime not installed. "
                "Install: pip install onnxruntime"
            )
            return

        if not os.path.exists(self.model_path):
            self.logger.error(
                f"Model not found: {self.model_path}"
            )
            return

        try:
            providers = ort.get_available_providers()
            preferred = [
                p for p in (
                    "CoreMLExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
                if p in providers
            ]
            self.session = ort.InferenceSession(
                self.model_path,
                providers=preferred or providers,
            )

            # Определяем размер входа из модели
            model_inputs = self.session.get_inputs()
            shape = model_inputs[0].shape
            self.input_height = shape[2]
            self.input_width = shape[3]

            self.logger.info(
                f"  🧠 Bird detector loaded: "
                f"{os.path.basename(self.model_path)} "
                f"({self.input_width}x"
                f"{self.input_height}), "
                f"provider={preferred[0] if preferred else 'auto'}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load bird detector: {e}"
            )
            self.session = None

    def _letterbox(
        self,
        img: np.ndarray,
        new_shape: tuple = (640, 640),
    ) -> tuple:
        """
        Resize с сохранением пропорций + padding.

        Returns:
            (resized_img, (pad_top, pad_left))
        """
        shape = img.shape[:2]  # h, w
        r = min(
            new_shape[0] / shape[0],
            new_shape[1] / shape[1],
        )
        new_unpad = (
            round(shape[1] * r),
            round(shape[0] * r),
        )
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(
                img, new_unpad,
                interpolation=cv2.INTER_LINEAR,
            )

        top = round(dh - 0.1)
        bottom = round(dh + 0.1)
        left = round(dw - 0.1)
        right = round(dw + 0.1)
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )
        return img, (top, left)

    def detect(
        self, frame: np.ndarray
    ) -> List[Detection]:
        """
        Запустить детекцию на кадре.

        Args:
            frame: BGR изображение (numpy array)
        Returns:
            Список детекций птиц
        """
        if self.session is None:
            return []

        img_h, img_w = frame.shape[:2]

        # Preprocessing
        img_rgb = cv2.cvtColor(
            frame, cv2.COLOR_BGR2RGB
        )
        img_resized, pad = self._letterbox(
            img_rgb,
            (self.input_height, self.input_width),
        )
        img_data = (
            np.array(img_resized, dtype=np.float32)
            / 255.0
        )
        img_data = np.transpose(
            img_data, (2, 0, 1)
        )  # HWC -> CHW
        img_data = img_data[np.newaxis, ...]

        # Inference
        model_inputs = self.session.get_inputs()
        outputs = self.session.run(
            None,
            {model_inputs[0].name: img_data},
        )

        # Postprocessing
        output = np.transpose(
            np.squeeze(outputs[0])
        )
        rows = output.shape[0]

        boxes = []
        scores = []
        class_ids = []

        gain = min(
            self.input_height / img_h,
            self.input_width / img_w,
        )

        for i in range(rows):
            classes_scores = output[i][4:]
            max_score = float(np.amax(classes_scores))

            if max_score < self.confidence_threshold:
                continue

            class_id = int(np.argmax(classes_scores))

            # Фильтруем только птиц
            if class_id != COCO_BIRD_CLASS:
                continue

            x = output[i][0] - pad[1]
            y = output[i][1] - pad[0]
            w = output[i][2]
            h = output[i][3]

            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)

            # Clamp to image bounds
            left = max(0, left)
            top = max(0, top)
            width = min(width, img_w - left)
            height = min(height, img_h - top)

            if width > 0 and height > 0:
                boxes.append(
                    [left, top, width, height]
                )
                scores.append(max_score)
                class_ids.append(class_id)

        # NMS
        if not boxes:
            return []

        indices = cv2.dnn.NMSBoxes(
            boxes, scores,
            self.confidence_threshold,
            self.iou_threshold,
        )

        detections = []
        for i in np.array(indices).flatten():
            box = boxes[int(i)]
            detections.append(Detection(
                class_id=class_ids[int(i)],
                class_name=COCO_CLASSES[
                    class_ids[int(i)]
                ],
                confidence=scores[int(i)],
                x=box[0],
                y=box[1],
                width=box[2],
                height=box[3],
            ))

        return detections


class SpeciesClassifier:
    """
    CLIP-based классификатор видов птиц.

    Двухкомпонентная архитектура:
    1. clip_visual.onnx — CLIP ViT-B/32 visual encoder
       (извлекает 512-мерный эмбеддинг из кропа)
    2. species_head_weights.npz — линейный
       классификатор (sklearn LogisticRegression)

    Preprocessing использует CLIP-нормализацию
    (отличается от ImageNet!).
    """

    # CLIP preprocessing constants
    CLIP_INPUT_SIZE = 224
    CLIP_MEAN = np.array(
        [0.48145466, 0.4578275, 0.40821073],
        dtype=np.float32,
    )
    CLIP_STD = np.array(
        [0.26862954, 0.26130258, 0.27577711],
        dtype=np.float32,
    )

    CLIP_VISUAL_FILENAME = "clip_visual.onnx"
    SPECIES_HEAD_FILENAME = (
        "species_head_weights.npz"
    )

    def __init__(
        self,
        model_dir: str,
        labels_path: str,
        confidence_threshold: float = 0.3,
        logger: logging.Logger = None,
    ):
        self.model_dir = model_dir
        self.labels_path = labels_path
        self.confidence_threshold = (
            confidence_threshold
        )
        self.logger = (
            logger or logging.getLogger(__name__)
        )
        self.session = None
        self.labels = {}
        self.head_weight = None
        self.head_bias = None
        self.head_classes = None

        self._load_labels()
        self._check_clip_visual()
        self._load_model()
        self._load_head()

    def _load_labels(self):
        """Загрузить маппинг классов из JSON."""
        try:
            if not os.path.exists(self.labels_path):
                self.logger.warning(
                    f"Labels not found: "
                    f"{self.labels_path}"
                )
                return
            with open(
                self.labels_path, "r",
                encoding="utf-8",
            ) as f:
                self.labels = json.load(f)
            self.logger.info(
                f"  Species labels loaded: "
                f"{len(self.labels)} classes"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load labels: {e}"
            )

    def _check_clip_visual(self):
        """
        Проверить наличие CLIP visual ONNX.
        Модель экспортируется скриптом обучения:
        python scripts/train_species_classifier.py
        """
        clip_path = os.path.join(
            self.model_dir,
            self.CLIP_VISUAL_FILENAME,
        )
        if not os.path.exists(clip_path):
            self.logger.info(
                f"  CLIP visual model not found "
                f"at {clip_path}. "
                f"Run: python scripts/"
                f"train_species_classifier.py"
            )

    def _load_model(self):
        """Загрузить CLIP visual ONNX модель."""
        if not ONNX_AVAILABLE:
            return

        clip_path = os.path.join(
            self.model_dir,
            self.CLIP_VISUAL_FILENAME,
        )
        if not os.path.exists(clip_path):
            self.logger.info(
                "  CLIP visual model not found "
                "— detection only mode"
            )
            return

        try:
            providers = ort.get_available_providers()
            preferred = [
                p for p in (
                    "CoreMLExecutionProvider",
                    "CUDAExecutionProvider",
                    "CPUExecutionProvider",
                )
                if p in providers
            ]
            self.session = ort.InferenceSession(
                clip_path,
                providers=preferred or providers,
            )

            size_mb = (
                os.path.getsize(clip_path)
                / (1024 ** 2)
            )
            self.logger.info(
                f"  🔬 CLIP visual loaded: "
                f"{self.CLIP_VISUAL_FILENAME}"
                f" ({size_mb:.0f} MB), "
                f"provider="
                f"{preferred[0] if preferred else 'auto'}"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load CLIP visual: {e}"
            )
            self.session = None

    def _load_head(self):
        """
        Загрузить linear head
        (species_head_weights.npz).
        """
        head_path = os.path.join(
            self.model_dir,
            self.SPECIES_HEAD_FILENAME,
        )
        if not os.path.exists(head_path):
            self.logger.info(
                "  Species head weights not found "
                f"at {head_path} — "
                "run train_species_classifier.py "
                "first"
            )
            return

        try:
            data = np.load(head_path)
            self.head_weight = data["weight"]
            self.head_bias = data["bias"]
            self.head_classes = data.get(
                "classes", None
            )
            n_classes = self.head_weight.shape[0]
            embed_dim = self.head_weight.shape[1]
            self.logger.info(
                f"  🧠 Species head loaded: "
                f"{n_classes} classes, "
                f"{embed_dim}-dim embeddings"
            )
        except Exception as e:
            self.logger.error(
                f"Failed to load species head: {e}"
            )
            self.head_weight = None

    def _preprocess_clip(
        self, crop: np.ndarray,
    ) -> np.ndarray:
        """
        CLIP preprocessing для кропа.

        Args:
            crop: BGR изображение (numpy)
        Returns:
            float32 tensor [1, 3, 224, 224]
        """
        crop_rgb = cv2.cvtColor(
            crop, cv2.COLOR_BGR2RGB
        )
        # Resize с bicubic интерполяцией
        crop_resized = cv2.resize(
            crop_rgb,
            (
                self.CLIP_INPUT_SIZE,
                self.CLIP_INPUT_SIZE,
            ),
            interpolation=cv2.INTER_CUBIC,
        )
        img_data = (
            crop_resized.astype(np.float32) / 255.0
        )
        # CLIP normalization
        img_data = (
            (img_data - self.CLIP_MEAN)
            / self.CLIP_STD
        )
        # HWC -> CHW -> NCHW
        img_data = np.transpose(
            img_data, (2, 0, 1)
        )
        img_data = img_data[np.newaxis, ...]
        return img_data

    def is_ready(self) -> bool:
        """
        Проверить готовность классификатора.
        Нужен CLIP visual + linear head + labels.
        """
        return (
            self.session is not None
            and self.head_weight is not None
            and len(self.labels) > 0
        )

    def classify(
        self,
        frame: np.ndarray,
        bbox: tuple,
    ) -> Optional[SpeciesResult]:
        """
        Классифицировать вид птицы по кропу.

        Пайплайн:
        1. Кропнуть bbox с padding 20%
        2. CLIP preprocessing (224x224, CLIP norm)
        3. CLIP visual encoder -> 512-dim embedding
        4. Linear head: matmul + bias -> logits
        5. Softmax -> class probabilities

        Args:
            frame: Полный BGR кадр
            bbox: (x, y, w, h) bounding box птицы
        Returns:
            SpeciesResult или None
        """
        if not self.is_ready():
            return None

        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        # Добавляем padding вокруг bbox (20%)
        pad_size = int(max(w, h) * 0.2)
        x1 = max(0, x - pad_size)
        y1 = max(0, y - pad_size)
        x2 = min(img_w, x + w + pad_size)
        y2 = min(img_h, y + h + pad_size)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            # CLIP preprocessing
            img_data = self._preprocess_clip(crop)

            # CLIP visual encoder -> embedding
            model_inputs = (
                self.session.get_inputs()
            )
            outputs = self.session.run(
                None,
                {model_inputs[0].name: img_data},
            )
            embedding = outputs[0]  # (1, 512)

            # L2 нормализация эмбеддинга
            norm = np.linalg.norm(
                embedding, axis=-1, keepdims=True
            )
            if norm > 0:
                embedding = embedding / norm

            # Linear head: logits = X @ W^T + b
            logits = (
                embedding @ self.head_weight.T
                + self.head_bias
            )

            # Softmax
            logits_shifted = (
                logits - np.max(logits, axis=-1,
                                keepdims=True)
            )
            exp_logits = np.exp(logits_shifted)
            probs = (
                exp_logits
                / exp_logits.sum(
                    axis=-1, keepdims=True
                )
            )

            class_idx = int(
                np.argmax(probs[0])
            )
            confidence = float(probs[0, class_idx])

            if (
                confidence
                < self.confidence_threshold
            ):
                return None

            label = self.labels.get(
                str(class_idx), {}
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
                f"Species classification error: {e}"
            )
            return None


class BirdClassifier:
    """
    Главный класс: двухстадийный пайплайн
    детекции + классификации птиц.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        confidence_threshold: float = 0.5,
        species_enabled: bool = False,
        save_crops: bool = True,
        crops_dir: str = "./crops",
        logger: logging.Logger = None,
    ):
        self.model_dir = model_dir
        self.confidence_threshold = (
            confidence_threshold
        )
        self.species_enabled = species_enabled
        self.save_crops = save_crops
        self.crops_dir = crops_dir
        self.logger = (
            logger or logging.getLogger(__name__)
        )

        self.detector = None
        self.species_classifier = None

        if not ONNX_AVAILABLE:
            self.logger.error(
                "❌ onnxruntime not installed! "
                "ML disabled. Install: "
                "pip install onnxruntime"
            )
            return

        # Создаём директории
        os.makedirs(self.model_dir, exist_ok=True)
        if self.save_crops:
            os.makedirs(
                self.crops_dir, exist_ok=True
            )

        # Автозагрузка моделей
        self._download_models()

        # Инициализация детектора
        detector_path = os.path.join(
            self.model_dir, YOLOV8N_FILENAME
        )
        if os.path.exists(detector_path):
            self.detector = BirdDetector(
                model_path=detector_path,
                confidence_threshold=(
                    self.confidence_threshold
                ),
                logger=self.logger,
            )

        # Инициализация CLIP-based
        # классификатора видов
        # Порог для species ниже чем для детекции:
        # CLIP linear probe на 10 классах даёт
        # confidence ~0.15-0.5 для правильных
        # предсказаний (uniform = 0.1)
        species_threshold = min(
            self.confidence_threshold, 0.15
        )
        if self.species_enabled:
            labels_path = os.path.join(
                self.model_dir,
                "species_labels.json",
            )
            self.species_classifier = (
                SpeciesClassifier(
                    model_dir=self.model_dir,
                    labels_path=labels_path,
                    confidence_threshold=(
                        species_threshold
                    ),
                    logger=self.logger,
                )
            )
            if not self.species_classifier.is_ready():
                self.logger.info(
                    "  Species classifier not ready"
                    " — CLIP visual or head "
                    "weights missing. "
                    "Run train_species_classifier"
                    ".py first."
                )
                # Оставляем classifier для
                # автозагрузки CLIP при след.
                # запуске, но classify() вернёт None

    def _download_models(self):
        """Автозагрузка моделей при первом запуске."""
        yolo_path = os.path.join(
            self.model_dir, YOLOV8N_FILENAME
        )

        if os.path.exists(yolo_path):
            return

        self.logger.info(
            f"📥 Downloading {YOLOV8N_FILENAME} "
            f"(~13 MB)..."
        )
        try:
            urllib.request.urlretrieve(
                YOLOV8N_URL, yolo_path
            )
            size_mb = (
                os.path.getsize(yolo_path)
                / (1024 ** 2)
            )
            self.logger.info(
                f"  ✅ Downloaded: {YOLOV8N_FILENAME}"
                f" ({size_mb:.1f} MB)"
            )
        except Exception as e:
            self.logger.error(
                f"  ❌ Failed to download "
                f"{YOLOV8N_FILENAME}: {e}"
            )
            # Удаляем неполный файл
            if os.path.exists(yolo_path):
                try:
                    os.remove(yolo_path)
                except Exception:
                    pass

    def is_available(self) -> bool:
        """Проверить доступен ли классификатор."""
        return self.detector is not None

    def process_frame(
        self, frame: np.ndarray
    ) -> ClassificationResult:
        """
        Полный пайплайн: детекция + классификация.

        Args:
            frame: BGR кадр (numpy array)
        Returns:
            ClassificationResult
        """
        result = ClassificationResult()

        if not self.is_available():
            return result

        # Stage 1: детекция птиц
        detections = self.detector.detect(frame)

        if not detections:
            return result

        result.bird_detected = True
        result.detections = detections

        # Выбираем лучшую детекцию
        # (макс. confidence)
        best = max(
            detections, key=lambda d: d.confidence
        )
        result.best_detection = best

        # Stage 2: классификация вида
        if (
            self.species_classifier
            and self.species_enabled
        ):
            bbox = (
                best.x, best.y,
                best.width, best.height,
            )
            species = (
                self.species_classifier.classify(
                    frame, bbox
                )
            )
            if species:
                result.species = species

        return result

    def save_crop(
        self,
        frame: np.ndarray,
        result: ClassificationResult,
        visit_id: int = 0,
    ):
        """
        Сохранить кроп птицы и полный кадр
        для будущего обучения.

        Args:
            frame: Полный BGR кадр
            result: Результат классификации
            visit_id: ID визита
        """
        if not self.save_crops:
            return
        if not result.bird_detected:
            return
        if result.best_detection is None:
            return

        now = datetime.now(MOSCOW_TZ)
        date_dir = os.path.join(
            self.crops_dir,
            now.strftime("%Y-%m-%d"),
        )
        os.makedirs(date_dir, exist_ok=True)

        prefix = f"visit_{visit_id:04d}"

        # Сохраняем полный кадр
        full_path = os.path.join(
            date_dir, f"{prefix}_full.jpg"
        )
        cv2.imwrite(full_path, frame)

        # Сохраняем кроп каждой детекции
        for i, det in enumerate(result.detections):
            x1 = max(0, det.x)
            y1 = max(0, det.y)
            x2 = min(
                frame.shape[1], det.x + det.width
            )
            y2 = min(
                frame.shape[0], det.y + det.height
            )
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = os.path.join(
                    date_dir,
                    f"{prefix}_bird_{i}.jpg",
                )
                cv2.imwrite(crop_path, crop)

        # Сохраняем кадр с bbox для визуализации
        annotated = frame.copy()
        for det in result.detections:
            cv2.rectangle(
                annotated,
                (det.x, det.y),
                (det.x + det.width,
                 det.y + det.height),
                (0, 255, 0), 2,
            )
            label = f"bird {det.confidence:.0%}"
            if result.species:
                label = (
                    f"{result.species.species_ru} "
                    f"{result.species.confidence:.0%}"
                )
            cv2.putText(
                annotated, label,
                (det.x, det.y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 255, 0), 2,
            )

        annotated_path = os.path.join(
            date_dir, f"{prefix}_annotated.jpg"
        )
        cv2.imwrite(annotated_path, annotated)

    def get_species_name(
        self, result: ClassificationResult
    ) -> str:
        """
        Получить название вида для отображения.

        Returns:
            Название на русском или "Птица"
        """
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
            {"name": str, "confidence": float,
             "species": str or None}
        """
        if not result.bird_detected:
            return {
                "name": "Движение",
                "confidence": 0.0,
                "species": None,
            }

        best = result.best_detection
        conf = best.confidence if best else 0.0

        if result.species:
            return {
                "name": result.species.species_ru,
                "confidence": (
                    result.species.confidence
                ),
                "species": (
                    result.species.species_en
                ),
            }

        return {
            "name": "Птица",
            "confidence": conf,
            "species": None,
        }
