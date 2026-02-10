#!/usr/bin/env python3
"""
Bird Classifier для GoPro Bird Watcher
Детекция птиц (YOLOv8n ONNX) + классификация видов
+ распознавание поведения.

Трёхстадийный пайплайн:
1. YOLOv8n — детекция птицы в кадре (COCO class 14)
2. Species Classifier — определение вида (CLIP)
3. Behavior Classifier — поведение (TSM-MobileNetV3)

Зависимости: onnxruntime, numpy, opencv
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


# === Поведение птиц ===

# 6 классов поведения для кормушки
BEHAVIOR_CLASSES = {
    0: {"ru": "Кормление", "en": "feeding"},
    1: {"ru": "Сидение", "en": "perching"},
    2: {"ru": "Озирание", "en": "alert"},
    3: {"ru": "Драка", "en": "fighting"},
    4: {"ru": "Прилёт", "en": "arrival"},
    5: {"ru": "Улёт", "en": "departure"},
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
    detections: List[Detection] = field(
        default_factory=list
    )
    species: Optional[SpeciesResult] = None
    behavior: Optional[BehaviorResult] = None
    best_detection: Optional[Detection] = None
    bird_count: int = 0


def save_bird_crops(
    frame: np.ndarray,
    result: 'ClassificationResult',
    crops_dir: str,
    visit_id: int = 0,
    logger: logging.Logger = None,
):
    """
    Сохранить кропы птиц и аннотированный кадр.

    Общая утилита для BirdClassifier и
    RemoteBirdClassifier (без дублирования).

    Сохраняет:
    - Полный кадр (visit_XXXX_full.jpg)
    - Кроп каждой детекции (visit_XXXX_bird_N.jpg)
    - Аннотированный кадр с bbox (annotated.jpg)
    """
    if not result.bird_detected:
        return
    if result.best_detection is None:
        return

    now = datetime.now(MOSCOW_TZ)
    date_dir = os.path.join(
        crops_dir,
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

    def detect_tiled(
        self, frame: np.ndarray,
    ) -> List[Detection]:
        """
        Тайловая детекция: разбить кадр на
        2 горизонтальных тайла с перекрытием
        и запустить YOLO на каждом.

        Объекты в тайле ~2x крупнее → лучше
        детекция мелких птиц через стекло.

        Используется как fallback когда обычный
        detect() ничего не нашёл.
        """
        if self.session is None:
            return []

        img_h, img_w = frame.shape[:2]
        if img_w < 640:
            return self.detect(frame)

        # 2 тайла с 10% перекрытием
        overlap = int(img_w * 0.1)
        tile_w = img_w // 2 + overlap

        tiles = [
            (0, 0, tile_w, img_h),
            (img_w - tile_w, 0, tile_w, img_h),
        ]

        all_dets = []
        for tx, ty, tw, th in tiles:
            crop = frame[ty:ty+th, tx:tx+tw]
            dets = self.detect(crop)
            # Корректируем координаты
            # к оригинальному кадру
            for d in dets:
                d.x += tx
                d.y += ty
            all_dets.extend(dets)

        # Простой NMS для дедупликации
        # (перекрытие может дать дубли)
        if len(all_dets) > 1:
            all_dets = self._nms_dedup(all_dets)

        return all_dets

    @staticmethod
    def _nms_dedup(
        dets: List[Detection],
        iou_thresh: float = 0.3,
    ) -> List[Detection]:
        """Убрать дубликаты детекций из тайлов."""
        if len(dets) <= 1:
            return dets
        # Сортировка по confidence
        dets.sort(
            key=lambda d: d.confidence,
            reverse=True,
        )
        keep = []
        for d in dets:
            is_dup = False
            for k in keep:
                # IoU
                x1 = max(d.x, k.x)
                y1 = max(d.y, k.y)
                x2 = min(
                    d.x + d.width,
                    k.x + k.width,
                )
                y2 = min(
                    d.y + d.height,
                    k.y + k.height,
                )
                inter = max(0, x2 - x1) * max(
                    0, y2 - y1
                )
                area_d = d.width * d.height
                area_k = k.width * k.height
                union = area_d + area_k - inter
                if union > 0:
                    iou = inter / union
                    if iou > iou_thresh:
                        is_dup = True
                        break
            if not is_dup:
                keep.append(d)
        return keep

    def detect_all_debug(
        self, frame: np.ndarray,
        min_score: float = 0.1,
        top_k: int = 5,
    ) -> List[Detection]:
        """
        Детекция ВСЕХ объектов (не только птиц)
        с низким порогом для отладки.

        Возвращает top_k обнаружений с макс.
        confidence (любые COCO классы).
        """
        if self.session is None:
            return []

        img_h, img_w = frame.shape[:2]

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
        )
        img_data = img_data[np.newaxis, ...]

        model_inputs = self.session.get_inputs()
        outputs = self.session.run(
            None,
            {model_inputs[0].name: img_data},
        )

        output = np.transpose(
            np.squeeze(outputs[0])
        )
        rows = output.shape[0]

        gain = min(
            self.input_height / img_h,
            self.input_width / img_w,
        )

        all_dets = []
        for i in range(rows):
            classes_scores = output[i][4:]
            max_score = float(
                np.amax(classes_scores)
            )
            if max_score < min_score:
                continue
            class_id = int(
                np.argmax(classes_scores)
            )

            x = output[i][0] - pad[1]
            y = output[i][1] - pad[0]
            w = output[i][2]
            h = output[i][3]
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)
            left = max(0, left)
            top = max(0, top)
            width = min(width, img_w - left)
            height = min(height, img_h - top)

            if width > 0 and height > 0:
                all_dets.append(Detection(
                    class_id=class_id,
                    class_name=COCO_CLASSES[
                        class_id
                    ],
                    confidence=max_score,
                    x=left,
                    y=top,
                    width=width,
                    height=height,
                ))

        # Сортируем по confidence, берём top_k
        all_dets.sort(
            key=lambda d: d.confidence,
            reverse=True,
        )
        return all_dets[:top_k]


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
            used_provider = (
                preferred[0] if preferred else "auto"
            )
            try:
                self.session = ort.InferenceSession(
                    clip_path,
                    providers=(
                        preferred or providers
                    ),
                )
            except Exception as coreml_err:
                # CoreML может не поддерживать
                # CLIP ViT-B/32 — fallback на CPU
                self.logger.warning(
                    f"  CLIP: {preferred[0] if preferred else 'auto'}"
                    f" failed ({coreml_err}), "
                    f"fallback to CPU"
                )
                self.session = ort.InferenceSession(
                    clip_path,
                    providers=[
                        "CPUExecutionProvider"
                    ],
                )
                used_provider = (
                    "CPUExecutionProvider"
                )

            size_mb = (
                os.path.getsize(clip_path)
                / (1024 ** 2)
            )
            self.logger.info(
                f"  🔬 CLIP visual loaded: "
                f"{self.CLIP_VISUAL_FILENAME}"
                f" ({size_mb:.0f} MB), "
                f"provider={used_provider}"
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

    def detect_bird_clip(
        self,
        frame: np.ndarray,
        threshold: float = 0.35,
    ) -> Optional[SpeciesResult]:
        """
        CLIP-based обнаружение птицы на кадре.
        Сканирует 3 региона: левая половина,
        правая половина, центр.

        Fallback для случая когда YOLO ничего
        не нашёл. Работает потому что CLIP
        на кропах даёт >0.5 для верного вида
        и <0.25 для пустых кадров.

        Returns:
            SpeciesResult если птица найдена,
            None если нет.
        """
        if not self.is_ready():
            return None

        img_h, img_w = frame.shape[:2]
        if img_w < 200 or img_h < 200:
            return None

        cx = img_w // 2
        qw = img_w // 4
        # 3 региона: лево, право, центр
        regions = [
            (0, 0, cx + qw, img_h),
            (cx - qw, 0, img_w, img_h),
            (qw, 0, img_w - qw, img_h),
        ]

        best_result = None
        best_conf = 0.0

        for x1, y1, x2, y2 in regions:
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            try:
                img_data = (
                    self._preprocess_clip(crop)
                )
                inp = self.session.get_inputs()
                out = self.session.run(
                    None,
                    {inp[0].name: img_data},
                )
                emb = out[0]
                norm = np.linalg.norm(
                    emb, axis=-1, keepdims=True
                )
                if norm > 0:
                    emb = emb / norm
                logits = (
                    emb @ self.head_weight.T
                    + self.head_bias
                )
                shifted = logits - np.max(
                    logits, axis=-1,
                    keepdims=True,
                )
                exp_l = np.exp(shifted)
                probs = exp_l / exp_l.sum(
                    axis=-1, keepdims=True,
                )
                ci = int(np.argmax(probs[0]))
                conf = float(probs[0, ci])

                if conf > best_conf:
                    best_conf = conf
                    if conf >= threshold:
                        label = self.labels.get(
                            str(ci), {}
                        )
                        best_result = SpeciesResult(
                            species_ru=label.get(
                                "ru",
                                f"cls_{ci}",
                            ),
                            species_en=label.get(
                                "en",
                                f"cls_{ci}",
                            ),
                            confidence=conf,
                            class_index=ci,
                        )
            except Exception:
                continue

        return best_result

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
    - Выход: 6 классов поведения

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
        Preprocessing N кропов для TSM-MobileNetV3.

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

        При нехватке кадров — дублируем последний.

        Args:
            frame_buffer: deque с кадрами (BGR)
            bbox: (x, y, w, h) детекции птицы
            full_frame_h: высота полного кадра
            full_frame_w: ширина полного кадра
        Returns:
            Список из N кропов
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
            # Мало кадров — берём все + дублируем
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
                # Fallback: центр кадра
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

        Args:
            frame_buffer: deque с BGR кадрами
            bbox: (x, y, w, h) лучшей детекции
            frame_shape: (h, w) полного кадра
        Returns:
            BehaviorResult или None
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
    Главный класс: трёхстадийный пайплайн
    детекции + классификации птиц + поведения.
    """

    def __init__(
        self,
        model_dir: str = "./models",
        confidence_threshold: float = 0.5,
        species_enabled: bool = False,
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

        self.detector = None
        self.species_classifier = None
        self.behavior_classifier = None

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

        # Инициализация BehaviorClassifier
        # (TSM-MobileNetV3)
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
                    "ready — model not found. "
                    "Run train_behavior_"
                    "classifier.py first."
                )

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
        self,
        frame: np.ndarray,
        frame_buffer: deque = None,
    ) -> ClassificationResult:
        """
        Полный пайплайн:
        детекция + классификация + поведение.

        Args:
            frame: BGR кадр (numpy array)
            frame_buffer: deque с кадрами для
                behavior (опционально)
        Returns:
            ClassificationResult
        """
        result = ClassificationResult()

        if not self.is_available():
            return result

        # Stage 1: детекция птиц
        detections = self.detector.detect(frame)

        # Fallback: тайловая детекция если обычная
        # не нашла (птица мелкая → тайлы помогают)
        if not detections:
            detections = (
                self.detector.detect_tiled(frame)
            )

        if not detections:
            return result

        result.bird_detected = True
        result.detections = detections
        result.bird_count = len(detections)

        # Выбираем лучшую детекцию
        # (макс. confidence)
        best = max(
            detections, key=lambda d: d.confidence
        )
        result.best_detection = best

        bbox = (
            best.x, best.y,
            best.width, best.height,
        )

        # Stage 2: классификация вида
        if (
            self.species_classifier
            and self.species_enabled
        ):
            species = (
                self.species_classifier.classify(
                    frame, bbox
                )
            )
            if species:
                result.species = species

        # Stage 3: классификация поведения
        if (
            self.behavior_classifier
            and self.behavior_enabled
            and frame_buffer
            and len(frame_buffer) >= 2
        ):
            behavior = (
                self.behavior_classifier.classify(
                    frame_buffer,
                    bbox,
                    frame.shape,
                )
            )
            if behavior:
                result.behavior = behavior

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

        Делегирует к save_bird_crops().
        """
        if not self.save_crops:
            return
        save_bird_crops(
            frame=frame,
            result=result,
            crops_dir=self.crops_dir,
            visit_id=visit_id,
            logger=self.logger,
        )

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
             "species": str or None,
             "behavior": str or None,
             "behavior_en": str or None,
             "behavior_confidence": float,
             "bird_count": int}
        """
        info = {
            "name": "Движение",
            "confidence": 0.0,
            "species": None,
            "behavior": None,
            "behavior_en": None,
            "behavior_confidence": 0.0,
            "bird_count": 0,
        }

        if not result.bird_detected:
            return info

        best = result.best_detection
        conf = best.confidence if best else 0.0
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
        else:
            info["name"] = "Птица"
            info["confidence"] = conf

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
