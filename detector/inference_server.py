#!/usr/bin/env python3
"""
gRPC Inference Server для GoPro Bird Watcher.

Роль: compute — принимает кадры от edge-устройств
(Raspberry Pi), выполняет ML-инференс (YOLOv8n +
CLIP + TSM) и возвращает результат классификации.

Запуск:
    python detector/inference_server.py

Переменные окружения:
    GRPC_SERVER_PORT    — порт (default: 50051)
    ML_MODEL_DIR        — путь к моделям
    ML_CONFIDENCE_THRESHOLD — порог детекции
    ML_SPECIES_ENABLED  — классификация видов
    ML_BEHAVIOR_ENABLED — классификация поведения
"""

import os
import sys
import time
import signal
import logging
from concurrent import futures
from collections import deque

import cv2
import numpy as np
import grpc

# Добавляем parent в path для импортов
sys.path.insert(
    0, os.path.dirname(os.path.abspath(__file__))
)

from generated import inference_pb2
from generated import inference_pb2_grpc
from bird_classifier import (
    BirdClassifier,
    ClassificationResult,
    Detection,
)

logger = logging.getLogger("inference_server")

# gRPC server options
_MAX_RECV_MB = 50  # Принимаем большие кадры
_MAX_SEND_MB = 10  # Ответы компактные

_SERVER_OPTIONS = [
    (
        "grpc.max_receive_message_length",
        _MAX_RECV_MB * 1024 * 1024,
    ),
    (
        "grpc.max_send_message_length",
        _MAX_SEND_MB * 1024 * 1024,
    ),
]


def _setup_logging():
    """Настройка логирования для сервера."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _decode_jpeg(jpeg_bytes: bytes) -> np.ndarray:
    """Декодировать JPEG bytes -> BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def _detection_to_msg(det: Detection):
    """Detection dataclass -> protobuf DetectionMsg."""
    return inference_pb2.DetectionMsg(
        class_id=det.class_id,
        class_name=det.class_name,
        confidence=det.confidence,
        x=det.x,
        y=det.y,
        width=det.width,
        height=det.height,
    )


def _result_to_response(
    result: ClassificationResult,
) -> inference_pb2.ClassificationResponse:
    """ClassificationResult -> protobuf response."""
    resp = inference_pb2.ClassificationResponse(
        bird_detected=result.bird_detected,
        bird_count=result.bird_count,
    )

    # Detections
    for det in result.detections:
        resp.detections.append(
            _detection_to_msg(det)
        )

    # Best detection
    if result.best_detection:
        resp.best_detection.CopyFrom(
            _detection_to_msg(result.best_detection)
        )

    # Species
    if result.species:
        resp.species.CopyFrom(
            inference_pb2.SpeciesMsg(
                species_ru=(
                    result.species.species_ru
                ),
                species_en=(
                    result.species.species_en
                ),
                confidence=(
                    result.species.confidence
                ),
            )
        )

    # Behavior
    if result.behavior:
        resp.behavior.CopyFrom(
            inference_pb2.BehaviorMsg(
                behavior_ru=(
                    result.behavior.behavior_ru
                ),
                behavior_en=(
                    result.behavior.behavior_en
                ),
                confidence=(
                    result.behavior.confidence
                ),
                class_id=(
                    result.behavior.class_id
                ),
            )
        )

    return resp


class InferenceServicer(
    inference_pb2_grpc.InferenceServiceServicer
):
    """gRPC-сервер ML-инференса."""

    def __init__(self, classifier: BirdClassifier):
        self.classifier = classifier
        self._request_count = 0

    def ClassifyFrame(self, request, context):
        """
        Обработка запроса на классификацию кадра.

        Декодирует JPEG, вызывает BirdClassifier,
        возвращает результат в protobuf.
        """
        start = time.time()
        self._request_count += 1
        req_id = self._request_count

        # Декодируем основной кадр
        if not request.frame_jpeg:
            context.set_code(
                grpc.StatusCode.INVALID_ARGUMENT
            )
            context.set_details(
                "frame_jpeg is empty"
            )
            return (
                inference_pb2.ClassificationResponse()
            )

        frame = _decode_jpeg(request.frame_jpeg)
        if frame is None:
            context.set_code(
                grpc.StatusCode.INVALID_ARGUMENT
            )
            context.set_details(
                "Failed to decode JPEG"
            )
            return (
                inference_pb2.ClassificationResponse()
            )

        # Декодируем буфер кадров для behavior
        frame_buffer = None
        if (
            request.behavior_enabled
            and request.behavior_frames
        ):
            frame_buffer = deque()
            for jpeg_bytes in (
                request.behavior_frames
            ):
                buf_frame = _decode_jpeg(jpeg_bytes)
                if buf_frame is not None:
                    frame_buffer.append(buf_frame)

        # Временно обновляем флаги классификатора
        # чтобы учесть настройки из запроса
        orig_species = (
            self.classifier.species_enabled
        )
        orig_behavior = (
            self.classifier.behavior_enabled
        )
        self.classifier.species_enabled = (
            request.species_enabled
        )
        self.classifier.behavior_enabled = (
            request.behavior_enabled
        )

        try:
            result = self.classifier.process_frame(
                frame,
                frame_buffer=frame_buffer,
            )
        finally:
            # Восстанавливаем оригинальные флаги
            self.classifier.species_enabled = (
                orig_species
            )
            self.classifier.behavior_enabled = (
                orig_behavior
            )

        elapsed_ms = (time.time() - start) * 1000

        # Логирование
        if result.bird_detected:
            species_str = ""
            if result.species:
                sp = result.species
                species_str = (
                    f" | {sp.species_ru}"
                    f" ({sp.confidence:.0%})"
                )
            behavior_str = ""
            if result.behavior:
                bh = result.behavior
                behavior_str = (
                    f" | {bh.behavior_ru}"
                )
            logger.info(
                f"#{req_id} Bird detected"
                f"{species_str}{behavior_str}"
                f" [{elapsed_ms:.0f}ms]"
            )
        else:
            logger.debug(
                f"#{req_id} No bird"
                f" [{elapsed_ms:.0f}ms]"
            )

        return _result_to_response(result)

    def HealthCheck(self, request, context):
        """Проверка здоровья сервера."""
        loaded = []
        if self.classifier.detector:
            loaded.append("YOLOv8n")
        if self.classifier.species_classifier:
            sc = self.classifier.species_classifier
            if sc.is_ready():
                loaded.append("CLIP-Species")
        if self.classifier.behavior_classifier:
            bc = self.classifier.behavior_classifier
            if bc.is_ready():
                loaded.append("TSM-Behavior")

        # Определяем ONNX provider
        provider = "unknown"
        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                provider = "CUDA"
            elif (
                "CoreMLExecutionProvider" in providers
            ):
                provider = "CoreML"
            else:
                provider = "CPU"
        except ImportError:
            provider = "onnxruntime not available"

        return inference_pb2.HealthResponse(
            ok=self.classifier.is_available(),
            loaded_models=loaded,
            onnx_provider=provider,
        )


def serve():
    """Запуск gRPC-сервера."""
    _setup_logging()

    # Конфигурация из переменных окружения
    port = int(os.environ.get(
        "GRPC_SERVER_PORT", "50051"
    ))
    model_dir = os.environ.get(
        "ML_MODEL_DIR", "./models"
    )
    confidence = float(os.environ.get(
        "ML_CONFIDENCE_THRESHOLD", "0.5"
    ))
    species_enabled = os.environ.get(
        "ML_SPECIES_ENABLED", "true"
    ).lower() in ("true", "1", "yes")
    behavior_enabled = os.environ.get(
        "ML_BEHAVIOR_ENABLED", "false"
    ).lower() in ("true", "1", "yes")
    behavior_num_frames = int(os.environ.get(
        "ML_BEHAVIOR_NUM_FRAMES", "8"
    ))
    behavior_confidence = float(os.environ.get(
        "ML_BEHAVIOR_CONFIDENCE", "0.4"
    ))

    logger.info(
        "═══════════════════════════════════════"
    )
    logger.info(
        "  🧠 Bird Watcher Inference Server"
    )
    logger.info(
        "═══════════════════════════════════════"
    )
    logger.info(f"  Port: {port}")
    logger.info(f"  Models: {model_dir}")
    logger.info(
        f"  Species: {species_enabled}"
    )
    logger.info(
        f"  Behavior: {behavior_enabled}"
    )

    # Инициализация BirdClassifier
    logger.info("Loading ML models...")
    classifier = BirdClassifier(
        model_dir=model_dir,
        confidence_threshold=confidence,
        species_enabled=species_enabled,
        behavior_enabled=behavior_enabled,
        behavior_num_frames=behavior_num_frames,
        behavior_confidence=behavior_confidence,
        save_crops=False,  # Кропы сохраняет edge
        logger=logger,
    )

    if not classifier.is_available():
        logger.error(
            "❌ BirdClassifier not available! "
            "Check models directory."
        )
        sys.exit(1)

    logger.info("✅ ML models loaded")

    # Запуск gRPC-сервера
    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=4
        ),
        options=_SERVER_OPTIONS,
    )
    inference_pb2_grpc.add_InferenceServiceServicer_to_server(
        InferenceServicer(classifier), server
    )
    server.add_insecure_port(f"[::]:{port}")
    server.start()

    logger.info(
        f"🚀 gRPC server listening on "
        f"0.0.0.0:{port}"
    )
    logger.info(
        "═══════════════════════════════════════"
    )

    # Graceful shutdown
    def _shutdown(signum, frame):
        logger.info(
            "Shutting down gRPC server..."
        )
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
