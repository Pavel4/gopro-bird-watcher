#!/usr/bin/env python3
"""
Экспорт модели Vogel german-bird-classifier-v2
из HuggingFace в ONNX формат.

Использование:
    pip install transformers torch onnx onnxscript pillow
    python scripts/export_vogel_onnx.py \
        --output_dir models

Результат:
    models/vogel_bird_classifier.onnx  (~16 MB)
    models/species_labels.json (обновлённый)
"""

import argparse
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODEL_NAME = (
    "kamera-linux/german-bird-classifier-v2"
)
ONNX_FILENAME = "vogel_bird_classifier.onnx"


def export_to_onnx(output_dir: str):
    """
    Скачать модель с HuggingFace и
    экспортировать в ONNX.
    """
    import torch
    from transformers import (
        AutoImageProcessor,
        AutoModelForImageClassification,
    )

    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Downloading model: {MODEL_NAME}..."
    )
    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )
    model = AutoModelForImageClassification \
        .from_pretrained(MODEL_NAME)
    model.eval()

    # Dummy input для экспорта
    # EfficientNet-B2: 224x224
    dummy = torch.randn(1, 3, 224, 224)

    onnx_path = os.path.join(
        output_dir, ONNX_FILENAME
    )

    logger.info(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["pixel_values"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
        },
        opset_version=14,
    )

    size_mb = os.path.getsize(onnx_path) / 1e6
    logger.info(
        f"✅ Exported: {ONNX_FILENAME} "
        f"({size_mb:.1f} MB)"
    )

    # Сохраняем labels
    id2label = model.config.id2label
    logger.info(f"Classes: {id2label}")

    # Маппинг немецких → русские названия
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

    labels = {}
    for idx_str, de_name in id2label.items():
        idx = str(idx_str)
        mapping = de_to_ru_en.get(de_name)
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

    # Сохраняем preprocessing config
    prep_config = {
        "image_size": 224,
        "mean": list(
            processor.image_mean
        ),
        "std": list(
            processor.image_std
        ),
        "do_rescale": processor.do_rescale,
        "rescale_factor": (
            processor.rescale_factor
        ),
    }
    prep_path = os.path.join(
        output_dir,
        "vogel_preprocess.json",
    )
    with open(
        prep_path, "w", encoding="utf-8"
    ) as f:
        json.dump(prep_config, f, indent=2)
    logger.info(
        f"✅ Preprocessing config: {prep_path}"
    )

    logger.info("Done!")
    logger.info(
        f"  Model:  {onnx_path}"
    )
    logger.info(
        f"  Labels: {labels_path}"
    )
    logger.info(
        f"  Prep:   {prep_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Export Vogel bird classifier "
            "to ONNX"
        )
    )
    parser.add_argument(
        "--output_dir",
        default="models",
        help="Output directory for ONNX model",
    )
    args = parser.parse_args()
    export_to_onnx(args.output_dir)


if __name__ == "__main__":
    main()
