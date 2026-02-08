#!/usr/bin/env python3
"""
gRPC Inference Client для GoPro Bird Watcher.

Роль: edge — отправляет кадры на compute-сервер
для ML-инференса и получает результат.

Класс RemoteBirdClassifier имеет тот же интерфейс,
что и BirdClassifier (Strategy pattern), поэтому
motion_detector.py может использовать любой из них
прозрачно.
"""

import os
import sys
import time
import logging
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
    ClassificationResult,
    Detection,
    SpeciesResult,
    BehaviorResult,
    save_bird_crops,
)


# gRPC message size limits
_MAX_SEND_MB = 50
_MAX_RECV_MB = 10

# gRPC channel options для клиента
_CLIENT_CHANNEL_OPTIONS = [
    (
        "grpc.max_receive_message_length",
        _MAX_RECV_MB * 1024 * 1024,
    ),
    (
        "grpc.max_send_message_length",
        _MAX_SEND_MB * 1024 * 1024,
    ),
    ("grpc.keepalive_time_ms", 30000),
    ("grpc.keepalive_timeout_ms", 10000),
]


class RemoteBirdClassifier:
    """
    gRPC-клиент для удалённого ML-инференса.

    Интерфейс совместим с BirdClassifier:
    - process_frame(frame, frame_buffer)
      -> ClassificationResult
    - is_available() -> bool
    - save_crop(frame, result, visit_id)
    - get_species_name(result) -> str
    - get_caption_info(result) -> dict

    При ошибке связи возвращает пустой результат
    и логирует ошибку (видео продолжает писаться).
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        jpeg_quality: int = 90,
        timeout: float = 5.0,
        save_crops: bool = True,
        crops_dir: str = "./crops",
        species_enabled: bool = True,
        behavior_enabled: bool = False,
        logger: logging.Logger = None,
    ):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.save_crops = save_crops
        self.crops_dir = crops_dir
        self.species_enabled = species_enabled
        self.behavior_enabled = behavior_enabled
        self.logger = (
            logger or logging.getLogger(__name__)
        )

        self._channel = None
        self._stub = None
        self._connected = False
        self._last_error_time = 0
        # Не спамить ошибками чаще 30 сек
        self._error_cooldown = 30.0

        if self.save_crops:
            os.makedirs(
                self.crops_dir, exist_ok=True
            )

        self._connect()

    def _connect(self):
        """Установить gRPC-соединение."""
        target = f"{self.host}:{self.port}"
        try:
            self._channel = grpc.insecure_channel(
                target,
                options=_CLIENT_CHANNEL_OPTIONS,
            )
            self._stub = (
                inference_pb2_grpc
                .InferenceServiceStub(
                    self._channel
                )
            )
            self._connected = True
            self.logger.info(
                f"gRPC client connected to "
                f"{target}"
            )
        except Exception as e:
            self._connected = False
            self.logger.error(
                f"gRPC connect failed: {e}"
            )

    def _reconnect_if_needed(self):
        """Переподключение при необходимости."""
        if not self._connected:
            self._connect()

    def _encode_jpeg(
        self, frame: np.ndarray
    ) -> bytes:
        """Закодировать BGR numpy -> JPEG bytes."""
        params = [
            cv2.IMWRITE_JPEG_QUALITY,
            self.jpeg_quality,
        ]
        ok, buf = cv2.imencode(
            ".jpg", frame, params
        )
        if not ok:
            return b""
        return buf.tobytes()

    def is_available(self) -> bool:
        """Проверить доступность сервера."""
        if not self._stub:
            return False
        try:
            resp = self._stub.HealthCheck(
                inference_pb2.HealthRequest(),
                timeout=2.0,
            )
            return resp.ok
        except grpc.RpcError:
            return False

    def process_frame(
        self,
        frame: np.ndarray,
        frame_buffer: deque = None,
    ) -> ClassificationResult:
        """
        Отправить кадр на compute-сервер
        и получить результат классификации.

        Args:
            frame: BGR кадр (numpy array)
            frame_buffer: deque с кадрами для
                behavior (опционально)
        Returns:
            ClassificationResult
        """
        result = ClassificationResult()

        self._reconnect_if_needed()
        if not self._stub:
            return result

        # Кодируем основной кадр
        jpeg_data = self._encode_jpeg(frame)
        if not jpeg_data:
            self.logger.error(
                "Failed to encode frame to JPEG"
            )
            return result

        # Кодируем буфер для behavior
        behavior_frames = []
        if (
            self.behavior_enabled
            and frame_buffer
            and len(frame_buffer) >= 2
        ):
            for buf_frame in frame_buffer:
                encoded = self._encode_jpeg(
                    buf_frame
                )
                if encoded:
                    behavior_frames.append(encoded)

        # Формируем gRPC-запрос
        request = inference_pb2.FrameRequest(
            frame_jpeg=jpeg_data,
            species_enabled=self.species_enabled,
            behavior_enabled=self.behavior_enabled,
            behavior_frames=behavior_frames,
        )

        try:
            resp = self._stub.ClassifyFrame(
                request,
                timeout=self.timeout,
            )
            return self._response_to_result(resp)
        except grpc.RpcError as e:
            now = time.time()
            if (
                now - self._last_error_time
                > self._error_cooldown
            ):
                self.logger.error(
                    f"gRPC inference error: "
                    f"{e.code().name} - "
                    f"{e.details()}"
                )
                self._last_error_time = now
            # При ошибке — пробуем переподключиться
            self._connected = False
            return result

    def _response_to_result(
        self,
        resp: inference_pb2.ClassificationResponse,
    ) -> ClassificationResult:
        """Protobuf response -> ClassificationResult."""
        result = ClassificationResult()
        result.bird_detected = resp.bird_detected
        result.bird_count = resp.bird_count

        # Detections
        for det_msg in resp.detections:
            result.detections.append(
                Detection(
                    class_id=det_msg.class_id,
                    class_name=det_msg.class_name,
                    confidence=det_msg.confidence,
                    x=det_msg.x,
                    y=det_msg.y,
                    width=det_msg.width,
                    height=det_msg.height,
                )
            )

        # Best detection
        if resp.HasField("best_detection"):
            bd = resp.best_detection
            result.best_detection = Detection(
                class_id=bd.class_id,
                class_name=bd.class_name,
                confidence=bd.confidence,
                x=bd.x,
                y=bd.y,
                width=bd.width,
                height=bd.height,
            )

        # Species
        if resp.HasField("species"):
            sp = resp.species
            if sp.species_ru:
                result.species = SpeciesResult(
                    species_ru=sp.species_ru,
                    species_en=sp.species_en,
                    confidence=sp.confidence,
                )

        # Behavior
        if resp.HasField("behavior"):
            bh = resp.behavior
            if bh.behavior_ru:
                result.behavior = BehaviorResult(
                    behavior_ru=bh.behavior_ru,
                    behavior_en=bh.behavior_en,
                    confidence=bh.confidence,
                    class_id=bh.class_id,
                )

        return result

    def save_crop(
        self,
        frame: np.ndarray,
        result: ClassificationResult,
        visit_id: int = 0,
    ):
        """
        Сохранить кроп птицы локально (на edge).

        Делегирует к save_bird_crops() из
        bird_classifier.py (общая логика).
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
        """Название вида для отображения."""
        if result.species:
            return result.species.species_ru
        if result.bird_detected:
            return "Птица"
        return ""

    def get_caption_info(
        self, result: ClassificationResult
    ) -> dict:
        """Данные для caption в Telegram."""
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

    def close(self):
        """Закрыть gRPC-канал."""
        if self._channel:
            self._channel.close()
            self._channel = None
            self._stub = None
            self._connected = False

    def __del__(self):
        self.close()
