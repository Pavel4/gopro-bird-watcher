#!/usr/bin/env python3
"""
Экспорт модели Vogel german-bird-classifier-v2
из HuggingFace.

Экспортирует в TorchScript (.pt) — надёжный
формат, работающий через PyTorch.

ONNX экспорт для EfficientNet сломан
(padding-операции некорректно конвертируются),
поэтому используется TorchScript.

Использование:
    pip install transformers torch pillow numpy
    python scripts/export_vogel_onnx.py \
        --output_dir models

Результат:
    models/vogel_bird_classifier.pt  (~33 MB)
    models/species_labels.json
    models/vogel_preprocess.json
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s [%(levelname)s] "
        "%(message)s"
    ),
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = (
    "kamera-linux/german-bird-classifier-v2"
)
PT_FILENAME = "vogel_bird_classifier.pt"
# Устаревший ONNX файл (не работает)
ONNX_FILENAME = "vogel_bird_classifier.onnx"


def _make_wrapper(hf_model):
    """
    Создать nn.Module обёртку для HF модели,
    возвращающую только logits tensor.
    Нужна для torch.jit.trace (HF model
    возвращает ModelOutput, а trace
    требует plain tensor).
    """
    import torch.nn as nn

    class Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, pixel_values):
            return self.model(
                pixel_values
            ).logits

    return Wrapper(hf_model)


def _verify_torchscript(
    pt_path, processor, model
):
    """
    Верификация: сравнить PyTorch и
    TorchScript на тестовом изображении.
    """
    import torch
    from PIL import Image

    logger.info("Verifying TorchScript model...")

    # Тестовое изображение (фиксированный seed)
    rng = np.random.RandomState(42)
    test_img = Image.fromarray(
        rng.randint(
            0, 255, (224, 224, 3),
            dtype=np.uint8,
        )
    )

    # PyTorch inference
    inputs = processor(
        images=test_img,
        return_tensors="pt",
    )
    pixel_values = inputs["pixel_values"]
    with torch.no_grad():
        pt_logits = model(
            pixel_values
        ).logits.numpy()[0]

    # TorchScript inference
    ts_model = torch.jit.load(
        pt_path,
        map_location="cpu",
    )
    ts_model.eval()
    with torch.no_grad():
        ts_logits = ts_model(
            pixel_values
        ).numpy()[0]

    # Сравниваем
    diff = np.abs(pt_logits - ts_logits)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    pt_class = int(np.argmax(pt_logits))
    ts_class = int(np.argmax(ts_logits))

    logger.info(
        f"  PyTorch:     class={pt_class}, "
        f"logit={pt_logits[pt_class]:.4f}"
    )
    logger.info(
        f"  TorchScript: class={ts_class}, "
        f"logit={ts_logits[ts_class]:.4f}"
    )
    logger.info(
        f"  Diff: max={max_diff:.6f}, "
        f"mean={mean_diff:.6f}"
    )

    if pt_class != ts_class:
        logger.error(
            "  ❌ Different top classes!"
        )
        return False

    if max_diff > 0.01:
        logger.warning(
            f"  ⚠️ Large diff ({max_diff:.4f})"
        )

    # Если модели идеально совпадают —
    # экспорт корректен. Пропускаем проверку
    # uniform distribution (на случайном шуме
    # модель может законно давать низкий
    # confidence).
    if max_diff < 0.001:
        logger.info(
            "  ✅ Verification passed! "
            "(exact match)"
        )
        return True

    # Проверяем uniform distribution
    # только если есть расхождения.
    # 1/num_classes = базовый uniform порог.
    ts_shifted = ts_logits - np.max(ts_logits)
    ts_probs = (
        np.exp(ts_shifted)
        / np.sum(np.exp(ts_shifted))
    )
    max_prob = float(np.max(ts_probs))
    n_classes = len(ts_logits)
    uniform_threshold = (
        1.0 / n_classes + 0.02
    )
    if max_prob < uniform_threshold:
        logger.error(
            "  ❌ Uniform distribution "
            f"(max_prob={max_prob:.1%})"
        )
        return False

    logger.info("  ✅ Verification passed!")
    return True


def export_to_torchscript(output_dir: str):
    """
    Скачать модель с HuggingFace и
    экспортировать в TorchScript.
    """
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Downloading: {MODEL_NAME}..."
    )
    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
    )
    model = (
        AutoModelForImageClassification
        .from_pretrained(MODEL_NAME)
    )
    model.eval()

    pt_path = os.path.join(
        output_dir, PT_FILENAME
    )

    # --- TorchScript export via trace ---
    logger.info("Exporting TorchScript...")
    wrapper = _make_wrapper(model)
    dummy = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper, dummy
        )

    torch.jit.save(traced, pt_path)

    size_mb = os.path.getsize(pt_path) / 1e6
    logger.info(
        f"TorchScript: {PT_FILENAME} "
        f"({size_mb:.1f} MB)"
    )

    # --- Верификация ---
    ok = _verify_torchscript(
        pt_path, processor, model
    )
    if not ok:
        logger.error(
            "❌ TorchScript verification "
            "failed!"
        )
        sys.exit(1)

    # --- Удалить старый ONNX если есть ---
    old_onnx = os.path.join(
        output_dir, ONNX_FILENAME
    )
    if os.path.exists(old_onnx):
        os.remove(old_onnx)
        logger.info(
            f"Removed old ONNX: "
            f"{ONNX_FILENAME}"
        )

    # --- Сохраняем labels ---
    id2label = model.config.id2label
    logger.info(f"Classes: {id2label}")

    de_to_ru_en = {
        "Blaumeise": {
            "ru": "Лазоревка",
            "en": "Blue Tit",
        },
        "Grünling": {
            "ru": "Зеленушка",
            "en": "Greenfinch",
        },
        "Haussperling": {
            "ru": "Домовый воробей",
            "en": "House Sparrow",
        },
        "Kernbeißer": {
            "ru": "Дубонос",
            "en": "Hawfinch",
        },
        "Kleiber": {
            "ru": "Поползень",
            "en": "Nuthatch",
        },
        "Kohlmeise": {
            "ru": "Большая синица",
            "en": "Great Tit",
        },
        "Rotkehlchen": {
            "ru": "Зарянка",
            "en": "European Robin",
        },
        "Sumpfmeise": {
            "ru": "Болотная гаичка",
            "en": "Marsh Tit",
        },
    }

    # Case-insensitive lookup
    de_lower_map = {
        k.lower(): v
        for k, v in de_to_ru_en.items()
    }

    labels = {}
    for idx_str, de_name in id2label.items():
        idx = str(idx_str)
        mapping = de_lower_map.get(
            de_name.lower()
        )
        if mapping:
            labels[idx] = {
                "ru": mapping["ru"],
                "en": mapping["en"],
                "de": de_name,
            }
        else:
            labels[idx] = {
                "ru": de_name,
                "en": de_name,
                "de": de_name,
            }

    labels_path = os.path.join(
        output_dir, "species_labels.json"
    )
    # Архив старых лейблов
    if os.path.exists(labels_path):
        archive_path = os.path.join(
            output_dir,
            "species_labels_archive.json",
        )
        if os.path.exists(archive_path):
            os.remove(archive_path)
        os.rename(labels_path, archive_path)
        logger.info(
            f"Old labels archived → "
            f"{archive_path}"
        )

    with open(
        labels_path, "w", encoding="utf-8"
    ) as f:
        json.dump(
            labels, f,
            ensure_ascii=False, indent=2,
        )
    logger.info(
        f"✅ Labels saved: {labels_path}"
    )

    # --- Preprocessing config ---
    prep_config = {
        "image_size": 224,
        "mean": list(processor.image_mean),
        "std": list(processor.image_std),
        "do_rescale": processor.do_rescale,
        "rescale_factor": (
            processor.rescale_factor
        ),
    }
    prep_path = os.path.join(
        output_dir, "vogel_preprocess.json",
    )
    with open(
        prep_path, "w", encoding="utf-8"
    ) as f:
        json.dump(prep_config, f, indent=2)
    logger.info(
        f"✅ Preprocessing config: {prep_path}"
    )

    logger.info("")
    logger.info("=== DONE ===")
    logger.info(f"  Model:  {pt_path}")
    logger.info(f"  Size:   {size_mb:.1f} MB")
    logger.info(f"  Labels: {labels_path}")
    logger.info(f"  Prep:   {prep_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export Vogel bird classifier "
            "to TorchScript"
        )
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Output directory",
    )
    args = parser.parse_args()
    export_to_torchscript(args.output_dir)


if __name__ == "__main__":
    main()
