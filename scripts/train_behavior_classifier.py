#!/usr/bin/env python3
"""
Обучение классификатора поведения птиц
(TSM-MobileNetV3).

Пайплайн:
1. Загрузка видеоклипов из data/behavior_train/
2. Извлечение N кадров из каждого клипа
3. Fine-tuning TSM-MobileNetV3
4. Экспорт в ONNX для production-инференса

Структура данных:
    data/behavior_train/
    ├── feeding/
    │   ├── clip_001.mp4
    │   ├── clip_002.mp4
    │   └── ...
    ├── perching/
    ├── alert/
    ├── fighting/
    ├── arrival/
    └── departure/

Источники данных для обучения:
- Кропы из Telegram-разметки
  (analytics/behavior_reports.csv)
- Нарезанные клипы из recordings/motion/
- Внешние датасеты (WetlandBirds и др.)

Использование:
    pip install -r scripts/requirements-behavior.txt
    python scripts/train_behavior_classifier.py \
        --data_dir data/behavior_train \
        --output_dir models \
        --num_frames 8

Параметры:
    --data_dir    Директория с клипами по классам
    --output_dir  Директория для моделей
    --num_frames  Кадров на клип (default 8)
    --batch_size  Размер батча (default 8)
    --epochs      Число эпох (default 30)
    --lr          Learning rate (default 1e-3)
    --device      Устройство: cpu/cuda/mps (auto)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] %(message)s"
    ),
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
logging.getLogger("PIL").setLevel(
    logging.WARNING
)


# === Классы поведения ===

BEHAVIOR_CLASSES = {
    0: {"ru": "Кормление", "en": "feeding"},
    1: {"ru": "Сидение", "en": "perching"},
    2: {"ru": "Озирание", "en": "alert"},
    3: {"ru": "Драка", "en": "fighting"},
    4: {"ru": "Прилёт", "en": "arrival"},
    5: {"ru": "Улёт", "en": "departure"},
}

BEHAVIOR_EN_TO_ID = {
    v["en"]: k for k, v in
    BEHAVIOR_CLASSES.items()
}


def detect_device(preferred: str = "auto") -> str:
    """
    Определить лучшее доступное устройство.

    Returns:
        "cuda", "mps", или "cpu"
    """
    import torch

    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return "mps"
    return "cpu"


def load_video_clips(
    data_dir: str,
    num_frames: int = 8,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """
    Загрузить видеоклипы из директории.

    Структура:
        data_dir/
        ├── feeding/   (class 0)
        ├── perching/  (class 1)
        └── ...

    Поддерживает:
    - .mp4, .avi, .mov видео — извлекает N кадров
    - Папки с .jpg/.png — N изображений как кадры

    Args:
        data_dir: путь к директории
        num_frames: кадров на клип
    Returns:
        (clips, labels, paths)
        clips: list of [N, H, W, 3] numpy arrays
        labels: list of int class IDs
        paths: list of source file paths
    """
    import cv2

    clips = []
    labels = []
    paths = []

    for class_name, class_id in sorted(
        BEHAVIOR_EN_TO_ID.items()
    ):
        class_dir = os.path.join(
            data_dir, class_name
        )
        if not os.path.isdir(class_dir):
            logger.warning(
                f"  Directory not found: "
                f"{class_dir}, skipping "
                f"{class_name}"
            )
            continue

        files = sorted(os.listdir(class_dir))
        video_exts = {
            ".mp4", ".avi", ".mov", ".mkv"
        }
        image_exts = {
            ".jpg", ".jpeg", ".png", ".bmp"
        }

        # Собираем видеофайлы
        video_files = [
            f for f in files
            if os.path.splitext(f)[1].lower()
            in video_exts
        ]
        # Собираем изображения
        image_files = [
            f for f in files
            if os.path.splitext(f)[1].lower()
            in image_exts
        ]

        # Обрабатываем видео
        for vf in video_files:
            vpath = os.path.join(class_dir, vf)
            frames = _extract_frames(
                vpath, num_frames
            )
            if frames is not None:
                clips.append(frames)
                labels.append(class_id)
                paths.append(vpath)

        # Изображения: группируем по num_frames
        if image_files and not video_files:
            for i in range(
                0, len(image_files), num_frames
            ):
                batch = image_files[
                    i:i + num_frames
                ]
                if len(batch) < 2:
                    continue
                frames = []
                for imgf in batch:
                    imgpath = os.path.join(
                        class_dir, imgf
                    )
                    img = cv2.imread(imgpath)
                    if img is not None:
                        frames.append(img)

                if len(frames) >= 2:
                    # Pad до num_frames
                    while (
                        len(frames) < num_frames
                    ):
                        frames.append(frames[-1])
                    frames = frames[:num_frames]
                    clips.append(
                        np.stack(frames, axis=0)
                    )
                    labels.append(class_id)
                    paths.append(
                        os.path.join(
                            class_dir, batch[0]
                        )
                    )

        logger.info(
            f"  {class_name}: "
            f"{len([l for l in labels if l == class_id])} "
            f"clips"
        )

    logger.info(
        f"Total: {len(clips)} clips, "
        f"{len(set(labels))} classes"
    )
    return clips, labels, paths


def _extract_frames(
    video_path: str,
    num_frames: int,
) -> np.ndarray:
    """
    Извлечь N равномерно распределённых кадров
    из видео.

    Args:
        video_path: путь к видеофайлу
        num_frames: количество кадров
    Returns:
        numpy array [N, H, W, 3] или None
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(
            f"Cannot open: {video_path}"
        )
        return None

    total = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )
    if total < 2:
        cap.release()
        return None

    indices = np.linspace(
        0, total - 1, num_frames, dtype=int
    )
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        elif frames:
            frames.append(frames[-1])

    cap.release()

    if len(frames) < num_frames:
        while len(frames) < num_frames:
            frames.append(frames[-1])

    return np.stack(frames[:num_frames], axis=0)


def build_tsm_mobilenetv3(
    num_classes: int,
    num_frames: int = 8,
    pretrained: bool = True,
):
    """
    Построить TSM-MobileNetV3 модель.

    Архитектура:
    - MobileNetV3-Small backbone (pretrained)
    - Temporal Shift Module (TSM) на каждом
      InvertedResidual блоке
    - Global Average Pooling
    - Linear head → num_classes

    Args:
        num_classes: количество классов поведения
        num_frames: кадров на клип
        pretrained: использовать ImageNet веса
    Returns:
        torch.nn.Module
    """
    import torch
    import torch.nn as nn
    from torchvision.models import (
        mobilenet_v3_small,
        MobileNet_V3_Small_Weights,
    )

    class TemporalShift(nn.Module):
        """
        Temporal Shift Module (TSM).
        Сдвигает 1/8 каналов вперёд и 1/8 назад
        по временной оси для temporal reasoning.
        """

        def __init__(
            self,
            n_frames: int,
            shift_div: int = 8,
        ):
            super().__init__()
            self.n_frames = n_frames
            self.shift_div = shift_div

        def forward(self, x):
            # x: [B*T, C, H, W]
            bt, c, h, w = x.shape
            t = self.n_frames
            b = bt // t

            x = x.view(b, t, c, h, w)

            fold = c // self.shift_div
            out = x.clone()

            # Сдвиг вперёд (первые fold каналов)
            out[:, 1:, :fold, :, :] = (
                x[:, :-1, :fold, :, :]
            )
            # Сдвиг назад (следующие fold каналов)
            out[:, :-1, fold:2*fold, :, :] = (
                x[:, 1:, fold:2*fold, :, :]
            )
            # Остальные каналы без изменений

            return out.view(bt, c, h, w)

    class TSMMobileNetV3(nn.Module):
        """TSM + MobileNetV3-Small."""

        def __init__(
            self,
            num_classes: int,
            n_frames: int,
            pretrained: bool = True,
        ):
            super().__init__()
            self.n_frames = n_frames

            # Загружаем pretrained backbone
            weights = (
                MobileNet_V3_Small_Weights
                .IMAGENET1K_V1
                if pretrained else None
            )
            base = mobilenet_v3_small(
                weights=weights
            )

            # Features (convolutional layers)
            self.features = base.features

            # Вставляем TSM после каждого блока
            self.tsm_modules = nn.ModuleList()
            for i in range(len(self.features)):
                self.tsm_modules.append(
                    TemporalShift(n_frames)
                )

            # Classifier head
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            in_features = 576  # MobileNetV3-Small
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 256),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(256, num_classes),
            )

        def forward(self, x):
            # x: [B, T, C, H, W]
            b, t, c, h, w = x.shape
            # Merge batch and time
            x = x.view(b * t, c, h, w)

            # Forward через features + TSM
            for i, block in enumerate(
                self.features
            ):
                x = self.tsm_modules[i](x)
                x = block(x)

            # Global pooling
            x = self.avgpool(x)
            x = x.view(b, t, -1)

            # Temporal average pooling
            x = x.mean(dim=1)  # [B, features]

            # Classification
            x = self.classifier(x)
            return x

    model = TSMMobileNetV3(
        num_classes=num_classes,
        n_frames=num_frames,
        pretrained=pretrained,
    )
    return model


def preprocess_clips(
    clips: List[np.ndarray],
    input_size: int = 224,
) -> np.ndarray:
    """
    Preprocessing клипов для TSM-MobileNetV3.

    Args:
        clips: list of [N, H, W, 3] BGR arrays
        input_size: размер входа (224)
    Returns:
        float32 array [len(clips), N, 3, H, W]
    """
    import cv2

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )
    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    processed = []
    for clip in clips:
        frames = []
        for i in range(clip.shape[0]):
            frame = clip[i]
            # BGR -> RGB
            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )
            # Resize
            resized = cv2.resize(
                rgb,
                (input_size, input_size),
                interpolation=cv2.INTER_LINEAR,
            )
            # Normalize
            img = (
                resized.astype(np.float32)
                / 255.0
            )
            img = (img - mean) / std
            # HWC -> CHW
            img = np.transpose(img, (2, 0, 1))
            frames.append(img)
        processed.append(
            np.stack(frames, axis=0)
        )

    return np.array(processed, dtype=np.float32)


def train_model(
    model,
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: np.ndarray,
    val_labels: np.ndarray,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
) -> dict:
    """
    Обучить TSM-MobileNetV3 модель.

    Args:
        model: TSMMobileNetV3
        train_data: [N, T, C, H, W]
        train_labels: [N]
        val_data, val_labels: validation set
        epochs, batch_size, lr: гиперпараметры
        device: устройство
    Returns:
        dict с метриками
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import (
        TensorDataset, DataLoader,
    )

    model = model.to(device)

    train_tensor = torch.FloatTensor(train_data)
    train_labels_t = torch.LongTensor(
        train_labels
    )
    val_tensor = torch.FloatTensor(val_data)
    val_labels_t = torch.LongTensor(val_labels)

    train_ds = TensorDataset(
        train_tensor, train_labels_t
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs
    )

    best_val_acc = 0.0
    best_state = None
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += (
                predicted.eq(batch_y)
                .sum().item()
            )

        scheduler.step()

        train_acc = correct / total
        avg_loss = (
            total_loss / len(train_loader)
        )

        # Validation
        model.eval()
        with torch.no_grad():
            val_x = val_tensor.to(device)
            val_y = val_labels_t.to(device)
            val_out = model(val_x)
            _, val_pred = val_out.max(1)
            val_acc = (
                val_pred.eq(val_y)
                .sum().item() / len(val_y)
            )

        history["train_loss"].append(avg_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.cpu().clone()
                for k, v in
                model.state_dict().items()
            }

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                f"  Epoch {epoch+1:3d}/{epochs}"
                f"  loss={avg_loss:.4f}"
                f"  train_acc={train_acc:.3f}"
                f"  val_acc={val_acc:.3f}"
                f"  best={best_val_acc:.3f}"
            )

    # Загружаем лучшие веса
    if best_state:
        model.load_state_dict(best_state)

    return {
        "best_val_acc": best_val_acc,
        "history": history,
    }


def export_to_onnx(
    model,
    output_path: str,
    num_frames: int = 8,
    input_size: int = 224,
):
    """
    Экспорт модели в ONNX формат.

    Args:
        model: TSMMobileNetV3
        output_path: путь для .onnx файла
        num_frames: кадров на клип
        input_size: размер входа
    """
    import torch

    model.eval()
    model = model.cpu()

    dummy_input = torch.randn(
        1, num_frames, 3, input_size, input_size
    )

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=14,
    )

    size_mb = (
        os.path.getsize(output_path)
        / (1024 ** 2)
    )
    logger.info(
        f"ONNX exported: {output_path} "
        f"({size_mb:.1f} MB)"
    )


def save_training_report(
    output_dir: str,
    metrics: dict,
    num_classes: int,
    num_frames: int,
    train_size: int,
    val_size: int,
):
    """Сохранить отчёт об обучении в JSON."""
    report = {
        "model": "TSM-MobileNetV3-Small",
        "task": "behavior_classification",
        "num_classes": num_classes,
        "num_frames": num_frames,
        "classes": {
            str(k): v
            for k, v in BEHAVIOR_CLASSES.items()
        },
        "train_size": train_size,
        "val_size": val_size,
        "best_val_accuracy": (
            metrics["best_val_acc"]
        ),
        "timestamp": time.strftime(
            "%Y-%m-%dT%H:%M:%S"
        ),
    }
    path = os.path.join(
        output_dir,
        "behavior_training_report.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            report, f,
            indent=2, ensure_ascii=False,
        )
    logger.info(f"Report saved: {path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train TSM-MobileNetV3 "
            "behavior classifier"
        )
    )
    parser.add_argument(
        "--data_dir",
        default="data/behavior_train",
        help="Directory with behavior clips",
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Output directory for models",
    )
    parser.add_argument(
        "--num_frames",
        type=int, default=8,
        help="Frames per clip (default 8)",
    )
    parser.add_argument(
        "--batch_size",
        type=int, default=8,
        help="Batch size (default 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int, default=30,
        help="Training epochs (default 30)",
    )
    parser.add_argument(
        "--lr",
        type=float, default=1e-3,
        help="Learning rate (default 1e-3)",
    )
    parser.add_argument(
        "--test_size",
        type=float, default=0.2,
        help="Validation split (default 0.2)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device: auto/cpu/cuda/mps",
    )
    parser.add_argument(
        "--skip_onnx_export",
        action="store_true",
        help="Skip ONNX export",
    )
    args = parser.parse_args()

    logger.info(
        "=== TSM-MobileNetV3 Behavior "
        "Classifier Training ==="
    )

    # Проверяем данные
    if not os.path.isdir(args.data_dir):
        logger.error(
            f"Data directory not found: "
            f"{args.data_dir}"
        )
        logger.info(
            "Create the directory and add "
            "video clips:"
        )
        logger.info(
            f"  {args.data_dir}/feeding/"
        )
        logger.info(
            f"  {args.data_dir}/perching/"
        )
        logger.info(
            f"  {args.data_dir}/alert/"
        )
        logger.info("  ...")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Устройство
    device = detect_device(args.device)
    logger.info(f"Device: {device}")

    # 1. Загрузка данных
    logger.info(
        f"\n📁 Loading clips from "
        f"{args.data_dir}..."
    )
    clips, labels, paths = load_video_clips(
        args.data_dir, args.num_frames
    )

    if len(clips) < 10:
        logger.error(
            f"Not enough data: {len(clips)} "
            f"clips (need at least 10)"
        )
        sys.exit(1)

    num_classes = len(set(labels))
    logger.info(
        f"Loaded {len(clips)} clips, "
        f"{num_classes} classes"
    )

    # 2. Preprocessing
    logger.info("\n🔄 Preprocessing clips...")
    data = preprocess_clips(clips)
    labels_np = np.array(labels)

    # 3. Train/val split
    from sklearn.model_selection import (
        train_test_split,
    )
    (
        train_data, val_data,
        train_labels, val_labels,
    ) = train_test_split(
        data, labels_np,
        test_size=args.test_size,
        stratify=labels_np,
        random_state=42,
    )
    logger.info(
        f"Train: {len(train_data)}, "
        f"Val: {len(val_data)}"
    )

    # 4. Обучение
    logger.info(
        "\n🏋️ Training TSM-MobileNetV3..."
    )
    model = build_tsm_mobilenetv3(
        num_classes=num_classes,
        num_frames=args.num_frames,
        pretrained=True,
    )
    metrics = train_model(
        model,
        train_data, train_labels,
        val_data, val_labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
    )
    logger.info(
        f"\n✅ Best val accuracy: "
        f"{metrics['best_val_acc']:.3f}"
    )

    # 5. ONNX экспорт
    if not args.skip_onnx_export:
        logger.info("\n📦 Exporting to ONNX...")
        onnx_path = os.path.join(
            args.output_dir,
            "behavior_tsm.onnx",
        )
        export_to_onnx(
            model, onnx_path,
            num_frames=args.num_frames,
        )

    # 6. Сохраняем labels
    labels_path = os.path.join(
        args.output_dir,
        "behavior_labels.json",
    )
    with open(
        labels_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                str(k): v
                for k, v in
                BEHAVIOR_CLASSES.items()
            },
            f, indent=2, ensure_ascii=False,
        )
    logger.info(f"Labels saved: {labels_path}")

    # 7. Отчёт
    save_training_report(
        args.output_dir,
        metrics,
        num_classes=num_classes,
        num_frames=args.num_frames,
        train_size=len(train_data),
        val_size=len(val_data),
    )

    logger.info("\n🎉 Training complete!")
    logger.info(
        f"Model: {args.output_dir}/"
        f"behavior_tsm.onnx"
    )
    logger.info(
        f"Labels: {args.output_dir}/"
        f"behavior_labels.json"
    )
    logger.info(
        "Enable in config: "
        "ML_BEHAVIOR_ENABLED=true"
    )


if __name__ == "__main__":
    main()
