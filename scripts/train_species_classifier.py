#!/usr/bin/env python3
"""
Обучение классификатора видов птиц на CLIP-эмбеддингах.

Пайплайн:
1. Загрузка изображений из data/train/
2. Извлечение 512-мерных CLIP ViT-B/32 эмбеддингов
3. Обучение линейного классификатора (LogisticRegression)
4. Экспорт CLIP visual encoder в ONNX
5. Сохранение весов linear head в .npz

Использование:
    pip install -r scripts/requirements-train.txt
    python scripts/train_species_classifier.py \
        --data_dir data/train \
        --output_dir models

Параметры:
    --data_dir   Директория с фото (из download)
    --output_dir Директория для моделей
    --batch_size Размер батча для CLIP (default 32)
    --device     Устройство: cpu/cuda/mps (auto)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Подавляем лишние логи от torch/clip
logging.getLogger("PIL").setLevel(logging.WARNING)


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


def load_images_from_directory(
    data_dir: str,
) -> tuple:
    """
    Загрузить пути изображений и метки классов.

    Ожидаемая структура:
        data_dir/
            0_great_tit/
                0001.jpg
                0002.jpg
            1_blue_tit/
                0001.jpg
            ...

    Returns:
        (image_paths, labels, class_names)
        - image_paths: list[str]
        - labels: list[int] (class_id)
        - class_names: dict[int, str]
    """
    image_paths = []
    labels = []
    class_names = {}

    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(
            f"Директория не найдена: {data_dir}"
        )
        sys.exit(1)

    # Перебираем поддиректории
    subdirs = sorted([
        d for d in data_path.iterdir()
        if d.is_dir()
    ])

    if not subdirs:
        logger.error(
            f"Нет поддиректорий в {data_dir}. "
            f"Сначала запустите download_bird_images.py"
        )
        sys.exit(1)

    for subdir in subdirs:
        dir_name = subdir.name

        # Парсим class_id из имени директории
        # Формат: "0_great_tit", "1_blue_tit"
        parts = dir_name.split("_", 1)
        try:
            class_id = int(parts[0])
        except ValueError:
            logger.warning(
                f"Пропускаем {dir_name}: "
                f"не удалось извлечь class_id"
            )
            continue

        species_name = (
            parts[1] if len(parts) > 1
            else dir_name
        )
        class_names[class_id] = species_name

        # Собираем изображения
        images = sorted([
            str(f) for f in subdir.iterdir()
            if f.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".webp"
            )
        ])

        if not images:
            logger.warning(
                f"  {species_name}: нет изображений"
            )
            continue

        image_paths.extend(images)
        labels.extend([class_id] * len(images))

        logger.info(
            f"  Класс {class_id} ({species_name}): "
            f"{len(images)} изображений"
        )

    logger.info(
        f"Всего: {len(image_paths)} изображений, "
        f"{len(class_names)} классов"
    )

    return image_paths, labels, class_names


def extract_clip_embeddings(
    image_paths: list,
    device: str = "cpu",
    batch_size: int = 32,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
) -> np.ndarray:
    """
    Извлечь CLIP-эмбеддинги для списка изображений.

    Args:
        image_paths: пути к изображениям
        device: устройство (cpu/cuda/mps)
        batch_size: размер батча
        clip_model_name: имя модели CLIP
        clip_pretrained: набор весов

    Returns:
        numpy array shape (N, 512)
    """
    import torch
    import open_clip
    from PIL import Image

    logger.info(
        f"Загрузка CLIP {clip_model_name} "
        f"({clip_pretrained}) на {device}..."
    )

    model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name,
        pretrained=clip_pretrained,
        device=device,
    )
    model.eval()

    all_embeddings = []
    total = len(image_paths)
    failed = 0

    logger.info(
        f"Извлечение эмбеддингов: "
        f"{total} изображений, "
        f"batch_size={batch_size}..."
    )

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_paths = image_paths[start:end]

        batch_tensors = []
        valid_indices = []

        for i, path in enumerate(batch_paths):
            try:
                img = Image.open(path).convert("RGB")
                tensor = preprocess(img)
                batch_tensors.append(tensor)
                valid_indices.append(
                    start + i
                )
            except Exception as e:
                logger.debug(
                    f"Ошибка загрузки {path}: {e}"
                )
                failed += 1
                # Добавляем нулевой эмбеддинг
                batch_tensors.append(
                    torch.zeros(3, 224, 224)
                )

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            features = model.encode_image(batch)
            # L2 нормализация
            features = features / features.norm(
                dim=-1, keepdim=True
            )

        embeddings = features.cpu().numpy()
        all_embeddings.append(embeddings)

        progress = min(end, total)
        if (
            progress % (batch_size * 5) == 0
            or progress == total
        ):
            logger.info(
                f"  Прогресс: {progress}/{total} "
                f"({100 * progress / total:.0f}%)"
            )

    if failed > 0:
        logger.warning(
            f"Не удалось загрузить {failed} "
            f"изображений"
        )

    result = np.concatenate(all_embeddings, axis=0)
    logger.info(
        f"Эмбеддинги: shape={result.shape}, "
        f"dtype={result.dtype}"
    )
    return result


def train_linear_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Обучить линейный классификатор на эмбеддингах.

    Args:
        embeddings: (N, 512) float32
        labels: (N,) int
        test_size: доля валидации
        random_state: seed

    Returns:
        (model, accuracy, report_dict)
    """
    from sklearn.linear_model import (
        LogisticRegression,
    )
    from sklearn.model_selection import (
        train_test_split,
    )
    from sklearn.metrics import (
        classification_report,
        accuracy_score,
    )

    logger.info(
        f"Разделение train/val: "
        f"{1 - test_size:.0%}/{test_size:.0%}..."
    )

    X_train, X_val, y_train, y_val = (
        train_test_split(
            embeddings,
            labels,
            test_size=test_size,
            random_state=random_state,
            stratify=labels,
        )
    )

    logger.info(
        f"  Train: {len(X_train)}, "
        f"Val: {len(X_val)}"
    )

    logger.info("Обучение LogisticRegression...")
    t0 = time.time()

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    elapsed = time.time() - t0
    logger.info(
        f"  Обучение завершено за {elapsed:.1f}с"
    )

    # Оценка на валидации
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    report = classification_report(
        y_val, y_pred,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_val, y_pred,
        zero_division=0,
    )

    logger.info(
        f"\n=== Результаты валидации ===\n"
        f"Accuracy: {accuracy:.4f} "
        f"({accuracy:.1%})\n\n"
        f"{report_text}"
    )

    return clf, accuracy, report


def save_linear_head(
    clf,
    output_path: str,
    class_names: dict,
):
    """
    Сохранить веса линейного классификатора в .npz.

    Формат:
        weight: (num_classes, 512)
        bias: (num_classes,)
        classes: (num_classes,) — class_ids
        class_names: JSON строка

    Args:
        clf: обученный LogisticRegression
        output_path: путь для .npz файла
        class_names: {class_id: species_name}
    """
    weight = clf.coef_.astype(np.float32)
    bias = clf.intercept_.astype(np.float32)
    classes = clf.classes_.astype(np.int32)

    np.savez(
        output_path,
        weight=weight,
        bias=bias,
        classes=classes,
        class_names_json=json.dumps(
            class_names, ensure_ascii=False
        ),
    )

    logger.info(
        f"Linear head сохранён: {output_path}"
    )
    logger.info(
        f"  weight: {weight.shape}, "
        f"bias: {bias.shape}, "
        f"classes: {classes}"
    )


def export_clip_visual_onnx(
    output_path: str,
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    device: str = "cpu",
):
    """
    Экспортировать CLIP visual encoder в ONNX.

    Args:
        output_path: путь для .onnx файла
        clip_model_name: имя модели
        clip_pretrained: набор весов
        device: устройство для экспорта
    """
    import torch
    import open_clip

    if os.path.exists(output_path):
        size_mb = (
            os.path.getsize(output_path)
            / (1024 ** 2)
        )
        logger.info(
            f"CLIP visual ONNX уже существует: "
            f"{output_path} ({size_mb:.1f} MB)"
        )
        return

    logger.info(
        f"Экспорт CLIP visual encoder в ONNX..."
    )

    model, _, _ = (
        open_clip.create_model_and_transforms(
            clip_model_name,
            pretrained=clip_pretrained,
            device="cpu",  # Экспорт на CPU
        )
    )
    model.eval()

    # Создаём обёртку для visual encoder
    class VisualEncoder(torch.nn.Module):
        def __init__(self, clip_model):
            super().__init__()
            self.visual = clip_model.visual

        def forward(self, x):
            features = self.visual(x)
            # L2 нормализация
            features = features / features.norm(
                dim=-1, keepdim=True
            )
            return features

    visual_encoder = VisualEncoder(model)
    visual_encoder.eval()

    # Dummy input: batch=1, 3x224x224
    dummy_input = torch.randn(
        1, 3, 224, 224, dtype=torch.float32
    )

    logger.info(
        f"  Экспорт в {output_path}..."
    )

    torch.onnx.export(
        visual_encoder,
        dummy_input,
        output_path,
        input_names=["pixel_values"],
        output_names=["image_embeds"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "image_embeds": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    size_mb = (
        os.path.getsize(output_path)
        / (1024 ** 2)
    )
    logger.info(
        f"  CLIP visual ONNX сохранён: "
        f"{output_path} ({size_mb:.1f} MB)"
    )


def verify_onnx_model(
    onnx_path: str,
    head_path: str,
    sample_embedding: np.ndarray = None,
):
    """
    Верифицировать ONNX модель и linear head.

    Args:
        onnx_path: путь к clip_visual.onnx
        head_path: путь к species_head_weights.npz
        sample_embedding: тестовый эмбеддинг
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.warning(
            "onnxruntime не установлен, "
            "пропускаем верификацию"
        )
        return

    logger.info("Верификация ONNX модели...")

    # Проверяем CLIP visual
    session = ort.InferenceSession(
        onnx_path,
        providers=["CPUExecutionProvider"],
    )
    dummy = np.random.randn(
        1, 3, 224, 224
    ).astype(np.float32)
    outputs = session.run(None, {
        session.get_inputs()[0].name: dummy,
    })
    embed = outputs[0]
    logger.info(
        f"  CLIP visual output: "
        f"shape={embed.shape}, "
        f"dtype={embed.dtype}"
    )

    # Проверяем linear head
    head = np.load(head_path)
    weight = head["weight"]
    bias = head["bias"]
    logger.info(
        f"  Linear head: weight={weight.shape}, "
        f"bias={bias.shape}"
    )

    # Тестовый прогон
    logits = embed @ weight.T + bias
    probs = np.exp(logits) / np.exp(logits).sum(
        axis=-1, keepdims=True
    )
    pred_class = int(np.argmax(probs, axis=-1)[0])
    pred_conf = float(probs[0, pred_class])
    logger.info(
        f"  Тестовый прогон: class={pred_class}, "
        f"confidence={pred_conf:.4f}"
    )
    logger.info("  Верификация пройдена!")


def save_training_report(
    output_dir: str,
    accuracy: float,
    report: dict,
    class_names: dict,
    num_train: int,
    num_val: int,
):
    """Сохранить отчёт об обучении в JSON."""
    report_path = os.path.join(
        output_dir, "training_report.json"
    )
    data = {
        "model": "CLIP ViT-B/32 + LinearProbe",
        "accuracy": accuracy,
        "num_train": num_train,
        "num_val": num_val,
        "num_classes": len(class_names),
        "class_names": class_names,
        "per_class_metrics": {
            str(k): v for k, v in report.items()
            if k not in (
                "accuracy", "macro avg",
                "weighted avg",
            )
        },
        "macro_avg": report.get(
            "macro avg", {}
        ),
        "weighted_avg": report.get(
            "weighted avg", {}
        ),
    }

    with open(
        report_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            data, f,
            ensure_ascii=False, indent=2,
        )

    logger.info(
        f"Отчёт сохранён: {report_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Обучение CLIP-based классификатора "
            "видов птиц"
        )
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/train",
        help="Директория с фото птиц",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="models",
        help="Директория для сохранения моделей",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Размер батча для CLIP",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Устройство: auto/cpu/cuda/mps",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-B-32",
        help="Модель CLIP (default: ViT-B-32)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="Доля валидации (default: 0.2)",
    )
    parser.add_argument(
        "--skip_onnx_export",
        action="store_true",
        help="Не экспортировать CLIP в ONNX",
    )

    args = parser.parse_args()

    # Определяем устройство
    device = detect_device(args.device)
    logger.info(f"Устройство: {device}")

    # Создаём выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)

    # Шаг 1: Загрузка данных
    logger.info(
        f"\n=== Шаг 1: Загрузка данных ==="
    )
    image_paths, labels, class_names = (
        load_images_from_directory(args.data_dir)
    )

    if len(image_paths) == 0:
        logger.error("Нет данных для обучения!")
        sys.exit(1)

    labels_np = np.array(labels)

    # Шаг 2: Извлечение CLIP эмбеддингов
    logger.info(
        f"\n=== Шаг 2: Извлечение CLIP "
        f"эмбеддингов ==="
    )
    embeddings = extract_clip_embeddings(
        image_paths,
        device=device,
        batch_size=args.batch_size,
        clip_model_name=args.clip_model,
    )

    # Сохраняем эмбеддинги (для повторного обучения)
    embeddings_path = os.path.join(
        args.output_dir, "clip_embeddings.npz"
    )
    np.savez_compressed(
        embeddings_path,
        embeddings=embeddings,
        labels=labels_np,
    )
    logger.info(
        f"Эмбеддинги сохранены: {embeddings_path}"
    )

    # Шаг 3: Обучение linear probe
    logger.info(
        f"\n=== Шаг 3: Обучение linear probe ==="
    )
    clf, accuracy, report = train_linear_probe(
        embeddings, labels_np,
        test_size=args.test_size,
    )

    # Шаг 4: Сохранение весов
    logger.info(
        f"\n=== Шаг 4: Сохранение модели ==="
    )
    head_path = os.path.join(
        args.output_dir,
        "species_head_weights.npz",
    )
    save_linear_head(clf, head_path, class_names)

    # Шаг 5: Экспорт CLIP visual в ONNX
    if not args.skip_onnx_export:
        logger.info(
            f"\n=== Шаг 5: Экспорт CLIP "
            f"visual ONNX ==="
        )
        onnx_path = os.path.join(
            args.output_dir, "clip_visual.onnx"
        )
        export_clip_visual_onnx(
            onnx_path,
            clip_model_name=args.clip_model,
            device="cpu",
        )

        # Верификация
        verify_onnx_model(
            onnx_path, head_path
        )
    else:
        logger.info(
            "Экспорт CLIP в ONNX пропущен "
            "(--skip_onnx_export)"
        )

    # Шаг 6: Отчёт
    from sklearn.model_selection import (
        train_test_split,
    )
    _, X_val, _, _ = train_test_split(
        embeddings, labels_np,
        test_size=args.test_size,
        random_state=42,
        stratify=labels_np,
    )
    save_training_report(
        args.output_dir,
        accuracy=accuracy,
        report=report,
        class_names=class_names,
        num_train=len(embeddings) - len(X_val),
        num_val=len(X_val),
    )

    logger.info(
        f"\n=== Готово! ===\n"
        f"Accuracy: {accuracy:.1%}\n"
        f"Файлы в {args.output_dir}/:\n"
        f"  - species_head_weights.npz "
        f"(linear head)\n"
        f"  - clip_visual.onnx "
        f"(CLIP visual encoder)\n"
        f"  - clip_embeddings.npz "
        f"(кеш эмбеддингов)\n"
        f"  - training_report.json "
        f"(отчёт)\n"
    )


if __name__ == "__main__":
    main()
