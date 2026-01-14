#!/usr/bin/env python3
"""
Motion Detector for GoPro Bird Watcher
Детектор движения с записью через FFmpeg (со звуком).

Два режима записи:
1. Автоматический (motion) — при обнаружении движения → recordings/motion/
2. Ручной (manual) — по команде RECORD_START/STOP → recordings/manual/
"""

import cv2
import numpy as np
import time
import os
import signal
import sys
import subprocess
import shutil
import glob
from datetime import datetime, timezone, timedelta
from collections import deque
from threading import Thread, Event, Lock
from enum import Enum
import logging

# Московское время (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))


class RecordingType(Enum):
    """Тип записи."""
    NONE = "none"
    MOTION = "motion"  # Автоматическая запись при движении
    MANUAL = "manual"  # Принудительная запись по команде


def setup_logging(log_file: str = None):
    """Настройка логирования в файл и консоль."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


class SegmentRecorder:
    """
    Записывает RTMP поток короткими сегментами через FFmpeg.
    Позволяет потом объединять сегменты в итоговое видео.
    """
    
    def __init__(
        self,
        rtmp_url: str,
        segments_dir: str,
        segment_duration: int = 2,
        max_segments: int = 60,
        logger: logging.Logger = None
    ):
        """
        Args:
            rtmp_url: URL RTMP потока
            segments_dir: Папка для временных сегментов
            segment_duration: Длительность одного сегмента в секундах
            max_segments: Максимальное количество хранимых сегментов
        """
        self.rtmp_url = rtmp_url
        self.segments_dir = segments_dir
        self.segment_duration = segment_duration
        self.max_segments = max_segments
        self.logger = logger or logging.getLogger(__name__)
        
        self.ffmpeg_process = None
        self.is_running = False
        self.stop_event = Event()
        self.lock = Lock()
        
        # Очищаем и создаём папку сегментов
        if os.path.exists(segments_dir):
            shutil.rmtree(segments_dir)
        os.makedirs(segments_dir, exist_ok=True)
        
        self.logger.info(f"SegmentRecorder initialized: {segments_dir}")
    
    def start(self):
        """Запустить запись сегментов."""
        if self.is_running:
            return
        
        self.stop_event.clear()
        self.is_running = True
        
        # Запускаем FFmpeg для записи сегментов
        # Формат: segment_%05d.ts (segment_00001.ts, segment_00002.ts, ...)
        segment_pattern = os.path.join(self.segments_dir, "seg_%05d.ts")
        
        cmd = [
            "ffmpeg",
            "-y",  # Перезаписывать файлы
            "-i", self.rtmp_url,
            "-c:v", "copy",  # Копируем видео без перекодирования
            "-c:a", "aac",   # Аудио в AAC
            "-f", "segment",
            "-segment_time", str(self.segment_duration),
            "-segment_format", "mpegts",
            "-reset_timestamps", "1",
            "-strftime", "0",
            segment_pattern
        ]
        
        self.logger.info(f"Starting FFmpeg segment recorder...")
        self.logger.debug(f"Command: {' '.join(cmd)}")
        
        try:
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE
            )
            
            # Запускаем поток для очистки старых сегментов
            self.cleanup_thread = Thread(target=self._cleanup_old_segments, daemon=True)
            self.cleanup_thread.start()
            
            self.logger.info("SegmentRecorder started")
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
            self.is_running = False
    
    def stop(self):
        """Остановить запись сегментов."""
        if not self.is_running:
            return
        
        self.stop_event.set()
        self.is_running = False
        
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(b'q')
                self.ffmpeg_process.stdin.flush()
                self.ffmpeg_process.wait(timeout=5)
            except Exception:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
        
        self.logger.info("SegmentRecorder stopped")
    
    def _cleanup_old_segments(self):
        """Удаляет старые сегменты, оставляя только последние max_segments."""
        while not self.stop_event.is_set():
            try:
                with self.lock:
                    segments = self._get_sorted_segments()
                    if len(segments) > self.max_segments:
                        # Удаляем самые старые
                        to_delete = segments[:-self.max_segments]
                        for seg in to_delete:
                            try:
                                os.remove(seg)
                            except Exception:
                                pass
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
            
            time.sleep(self.segment_duration)
    
    def _get_sorted_segments(self) -> list:
        """Получить список сегментов, отсортированных по времени создания."""
        pattern = os.path.join(self.segments_dir, "seg_*.ts")
        segments = glob.glob(pattern)
        return sorted(segments, key=lambda x: os.path.getmtime(x))
    
    def get_recent_segments(self, seconds: int) -> list:
        """
        Получить сегменты за последние N секунд.
        
        Args:
            seconds: Количество секунд
            
        Returns:
            Список путей к сегментам
        """
        with self.lock:
            segments = self._get_sorted_segments()
            
            # Сколько сегментов нам нужно
            num_segments = max(1, seconds // self.segment_duration + 1)
            
            # Берём последние N сегментов
            return segments[-num_segments:] if segments else []
    
    def get_all_segments_since(self, start_time: float) -> list:
        """
        Получить все сегменты с указанного времени.
        
        Args:
            start_time: Unix timestamp начала
            
        Returns:
            Список путей к сегментам
        """
        with self.lock:
            segments = self._get_sorted_segments()
            result = []
            for seg in segments:
                try:
                    if os.path.getmtime(seg) >= start_time:
                        result.append(seg)
                except Exception:
                    pass
            return result


class VideoMerger:
    """Объединяет сегменты в итоговое видео через FFmpeg."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def merge_segments(
        self,
        segments: list,
        output_path: str,
        copy_codec: bool = True
    ) -> bool:
        """
        Объединить сегменты в один файл.
        
        Args:
            segments: Список путей к сегментам
            output_path: Путь к выходному файлу
            copy_codec: Копировать кодеки без перекодирования
            
        Returns:
            True если успешно
        """
        if not segments:
            self.logger.error("No segments to merge")
            return False
        
        # Создаём временный файл со списком сегментов
        list_file = output_path + ".txt"
        
        try:
            with open(list_file, 'w') as f:
                for seg in segments:
                    # Экранируем путь для ffmpeg concat
                    escaped_path = seg.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            # Формируем команду ffmpeg
            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_file,
            ]
            
            if copy_codec:
                cmd.extend(["-c", "copy"])
            else:
                cmd.extend(["-c:v", "libx264", "-c:a", "aac"])
            
            cmd.append(output_path)
            
            self.logger.debug(f"Merge command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=120
            )
            
            if result.returncode != 0:
                self.logger.error(f"FFmpeg merge failed: {result.stderr.decode()}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Merge error: {e}")
            return False
        finally:
            # Удаляем временный файл списка
            if os.path.exists(list_file):
                os.remove(list_file)


class MotionDetector:
    """Детектор движения с записью через FFmpeg (со звуком)."""
    
    def __init__(
        self,
        rtmp_url: str = "rtmp://nginx-rtmp/live",
        output_dir: str = "/recordings",
        log_file: str = "/logs/motion_detector.log",
        buffer_seconds: int = 5,
        post_motion_seconds: int = 5,
        min_contour_area: int = 500,
        min_motion_frames: int = 3,
        motion_area_percent: float = 0.5,
        segment_duration: int = 2
    ):
        """
        Args:
            rtmp_url: URL RTMP потока
            output_dir: Базовая директория для сохранения видео
            log_file: Путь к файлу логов
            buffer_seconds: Секунд до движения для записи
            post_motion_seconds: Секунд после ОКОНЧАНИЯ движения для записи
            min_contour_area: Минимальная площадь одного контура движения (пиксели)
            min_motion_frames: Мин. кадров подряд с движением для начала записи
            motion_area_percent: Мин. % площади кадра с движением для срабатывания
            segment_duration: Длительность одного сегмента записи (сек)
        """
        self.rtmp_url = rtmp_url
        self.output_dir = output_dir
        self.buffer_seconds = buffer_seconds
        self.post_motion_seconds = post_motion_seconds
        self.min_contour_area = min_contour_area
        self.min_motion_frames = min_motion_frames
        self.motion_area_percent = motion_area_percent
        self.segment_duration = segment_duration
        
        # Логирование
        self.logger = setup_logging(log_file)
        
        # Папки
        self.motion_dir = os.path.join(output_dir, "motion")
        self.manual_dir = os.path.join(output_dir, "manual")
        self.segments_dir = os.path.join(output_dir, ".segments")
        os.makedirs(self.motion_dir, exist_ok=True)
        os.makedirs(self.manual_dir, exist_ok=True)
        
        # Компоненты записи
        self.segment_recorder = SegmentRecorder(
            rtmp_url=rtmp_url,
            segments_dir=self.segments_dir,
            segment_duration=segment_duration,
            max_segments=120,  # ~4 минуты буфера
            logger=self.logger
        )
        self.video_merger = VideoMerger(logger=self.logger)
        
        # OpenCV для анализа
        self.cap = None
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0
        self.frame_area = 0
        
        # Background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
        # Состояние записи
        self.is_recording = False
        self.recording_type = RecordingType.NONE
        self.motion_detection_enabled = False
        self.recording_start_time = None
        self.recording_segments_start = None  # Время начала сбора сегментов
        self.buffer_segments = []  # Сегменты буфера (до движения)
        
        # Состояние движения
        self.last_motion_time = 0
        self.consecutive_motion_frames = 0
        self.significant_motion_started = False
        
        # События для управления
        self.stop_event = Event()
        self.stats = {
            'frames_processed': 0,
            'motion_events': 0,
            'significant_motion_events': 0,
            'motion_videos_saved': 0,
            'manual_videos_saved': 0,
            'last_motion': None
        }
        
        self.logger.info(f"Motion detector initialized (with audio support)")
        self.logger.info(f"  Output dirs: motion={self.motion_dir}, manual={self.manual_dir}")
        self.logger.info(f"  Buffer: {buffer_seconds}s before, {post_motion_seconds}s after")
        self.logger.info(f"  Segment duration: {segment_duration}s")
    
    def get_moscow_time(self) -> datetime:
        """Получить текущее время по Москве."""
        return datetime.now(MOSCOW_TZ)
    
    def format_duration(self, seconds: float) -> str:
        """Форматирование продолжительности в MMmSSs."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
    
    def connect(self) -> bool:
        """Подключение к RTMP потоку для анализа."""
        self.logger.info(f"Connecting to {self.rtmp_url}...")
        
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
        
        self.cap = cv2.VideoCapture(self.rtmp_url)
        
        if not self.cap.isOpened():
            self.logger.error(f"Failed to connect to {self.rtmp_url}")
            return False
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_area = self.frame_width * self.frame_height
        
        self.logger.info(
            f"Connected: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
        )
        return True
    
    def detect_motion(self, frame: np.ndarray) -> tuple:
        """
        Детекция движения в кадре.
        
        Returns:
            (motion_detected, motion_area_percent)
        """
        fg_mask = self.background_subtractor.apply(frame)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        total_motion_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.min_contour_area:
                total_motion_area += area
        
        motion_percent = (total_motion_area / self.frame_area) * 100 if self.frame_area else 0
        motion_detected = motion_percent >= self.motion_area_percent
        
        return motion_detected, motion_percent
    
    def start_recording(self, rec_type: RecordingType):
        """
        Начать запись видео.
        
        Args:
            rec_type: Тип записи (MOTION или MANUAL)
        """
        if self.is_recording:
            return
        
        # Запоминаем сегменты буфера (до начала записи)
        if rec_type == RecordingType.MOTION:
            self.buffer_segments = self.segment_recorder.get_recent_segments(
                self.buffer_seconds
            )
        else:
            self.buffer_segments = []
        
        self.is_recording = True
        self.recording_type = rec_type
        self.recording_start_time = time.time()
        self.recording_segments_start = time.time()
        
        type_str = "🐦 MOTION" if rec_type == RecordingType.MOTION else "🎬 MANUAL"
        buffer_info = f", buffer: {len(self.buffer_segments)} segments" if self.buffer_segments else ""
        self.logger.info(f"▶ {type_str} recording started{buffer_info}")
    
    def stop_recording(self):
        """Остановить запись и сохранить видео."""
        if not self.is_recording:
            return
        
        # Собираем сегменты с момента начала записи
        new_segments = self.segment_recorder.get_all_segments_since(
            self.recording_segments_start
        )
        
        # Объединяем буфер + новые сегменты
        all_segments = self.buffer_segments + new_segments
        
        # Убираем дубликаты, сохраняя порядок
        seen = set()
        unique_segments = []
        for seg in all_segments:
            if seg not in seen:
                seen.add(seg)
                unique_segments.append(seg)
        
        if not unique_segments:
            self.logger.warning("No segments to save")
            self._reset_recording_state()
            return
        
        # Формируем имя файла
        now = self.get_moscow_time()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        duration = time.time() - self.recording_start_time
        duration_str = self.format_duration(duration)
        
        if self.recording_type == RecordingType.MOTION:
            prefix = "bird"
            output_folder = self.motion_dir
            self.stats['motion_videos_saved'] += 1
            type_str = "🐦 MOTION"
        else:
            prefix = "manual"
            output_folder = self.manual_dir
            self.stats['manual_videos_saved'] += 1
            type_str = "🎬 MANUAL"
        
        filename = f"{prefix}_{timestamp}_{duration_str}.mp4"
        filepath = os.path.join(output_folder, filename)
        
        # Объединяем сегменты
        self.logger.info(f"Merging {len(unique_segments)} segments...")
        
        if self.video_merger.merge_segments(unique_segments, filepath):
            self.logger.info(f"■ {type_str} recording saved: {filename} ({duration:.1f}s)")
        else:
            self.logger.error(f"Failed to save recording: {filename}")
        
        self._reset_recording_state()
    
    def _reset_recording_state(self):
        """Сбросить состояние записи."""
        self.is_recording = False
        self.recording_type = RecordingType.NONE
        self.recording_start_time = None
        self.recording_segments_start = None
        self.buffer_segments = []
    
    def process_frame(self, frame: np.ndarray):
        """Обработка одного кадра."""
        current_time = time.time()
        
        # Детекция движения
        motion, motion_percent = self.detect_motion(frame)
        
        if motion:
            self.consecutive_motion_frames += 1
            self.stats['motion_events'] += 1
            
            if self.consecutive_motion_frames >= self.min_motion_frames:
                self.last_motion_time = current_time
                
                if not self.significant_motion_started:
                    self.significant_motion_started = True
                    self.stats['significant_motion_events'] += 1
                    self.stats['last_motion'] = self.get_moscow_time().isoformat()
                    self.logger.info(
                        f"🐦 Significant motion detected! "
                        f"(area: {motion_percent:.2f}%, "
                        f"event #{self.stats['significant_motion_events']})"
                    )
                    
                    # Начинаем MOTION запись если включена авто-детекция
                    if self.motion_detection_enabled and not self.is_recording:
                        self.start_recording(RecordingType.MOTION)
        else:
            self.consecutive_motion_frames = 0
        
        # Проверяем окончание движения (только для MOTION записи)
        if self.significant_motion_started:
            time_since_last_motion = current_time - self.last_motion_time
            
            if time_since_last_motion > self.post_motion_seconds:
                self.significant_motion_started = False
                self.logger.info(
                    f"   Motion ended. {self.post_motion_seconds}s buffer recorded."
                )
                # Останавливаем только MOTION запись
                if self.is_recording and self.recording_type == RecordingType.MOTION:
                    self.stop_recording()
        
        self.stats['frames_processed'] += 1
    
    # === Команды управления ===
    
    def enable_motion_detection(self):
        """Включить автоматическую запись при движении."""
        self.motion_detection_enabled = True
        self.logger.info("✅ MOTION detection ENABLED - auto-save on significant motion")
    
    def disable_motion_detection(self):
        """Выключить автоматическую запись при движении."""
        self.motion_detection_enabled = False
        if self.is_recording and self.recording_type == RecordingType.MOTION:
            self.stop_recording()
        self.logger.info("⏹ MOTION detection DISABLED")
    
    def start_manual_recording(self):
        """Начать принудительную запись (ручной режим)."""
        if self.is_recording:
            if self.recording_type == RecordingType.MANUAL:
                self.logger.warning("Manual recording already in progress")
            else:
                self.logger.warning("Cannot start manual: motion recording active")
            return
        self.start_recording(RecordingType.MANUAL)
    
    def stop_manual_recording(self):
        """Остановить принудительную запись."""
        if not self.is_recording:
            self.logger.warning("No recording in progress")
            return
        if self.recording_type != RecordingType.MANUAL:
            self.logger.warning("Cannot stop: current recording is not manual")
            return
        self.stop_recording()
    
    def get_status(self) -> dict:
        """Получить текущий статус."""
        return {
            'motion_detection_enabled': self.motion_detection_enabled,
            'is_recording': self.is_recording,
            'recording_type': self.recording_type.value,
            'segment_recorder_running': self.segment_recorder.is_running,
            'stats': self.stats
        }
    
    def run(self):
        """Основной цикл обработки."""
        # Запускаем запись сегментов
        self.segment_recorder.start()
        time.sleep(2)  # Даём время FFmpeg стартовать
        
        # Подключаемся к потоку для анализа
        if not self.connect():
            for attempt in range(5):
                self.logger.info(f"Reconnect attempt {attempt + 1}/5...")
                time.sleep(5)
                if self.connect():
                    break
            else:
                self.logger.error("Failed to connect after 5 attempts")
                self.segment_recorder.stop()
                return
        
        self.logger.info("Starting motion detection loop...")
        
        reconnect_attempts = 0
        max_reconnect_attempts = 10
        
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            
            if not ret:
                reconnect_attempts += 1
                self.logger.warning(
                    f"Frame read failed. Reconnect attempt {reconnect_attempts}"
                )
                
                if reconnect_attempts > max_reconnect_attempts:
                    self.logger.error("Max reconnect attempts reached. Exiting.")
                    break
                
                time.sleep(2)
                self.cap.release()
                if not self.connect():
                    continue
                reconnect_attempts = 0
                continue
            
            reconnect_attempts = 0
            self.process_frame(frame)
        
        self.cleanup()
    
    def cleanup(self):
        """Очистка ресурсов."""
        self.logger.info("Cleaning up...")
        if self.is_recording:
            self.stop_recording()
        self.segment_recorder.stop()
        if self.cap:
            self.cap.release()
        self.logger.info(f"Final stats: {self.stats}")
    
    def stop(self):
        """Остановить детектор."""
        self.logger.info("Stop signal received")
        self.stop_event.set()


def monitor_control_file(detector: MotionDetector, control_file: str):
    """
    Мониторит файл управления.
    
    Команды:
    - MOTION_ON   — включить авто-запись при движении
    - MOTION_OFF  — выключить авто-запись при движении
    - RECORD_START — начать принудительную (ручную) запись
    - RECORD_STOP  — остановить принудительную запись
    - STATUS       — показать статус
    """
    logger = detector.logger
    logger.info(f"Control file: {control_file}")
    logger.info("Commands: MOTION_ON, MOTION_OFF, RECORD_START, RECORD_STOP, STATUS")
    
    while not detector.stop_event.is_set():
        try:
            if os.path.exists(control_file):
                with open(control_file, 'r') as f:
                    command = f.read().strip().upper()
                
                os.remove(control_file)
                
                if command == "MOTION_ON":
                    detector.enable_motion_detection()
                elif command == "MOTION_OFF":
                    detector.disable_motion_detection()
                elif command == "RECORD_START":
                    detector.start_manual_recording()
                elif command == "RECORD_STOP":
                    detector.stop_manual_recording()
                elif command == "STATUS":
                    status = detector.get_status()
                    logger.info(f"Status: {status}")
                else:
                    logger.warning(f"Unknown command: {command}")
        except Exception as e:
            logger.error(f"Control file error: {e}")
        
        time.sleep(1)


def load_config(config_path: str = None) -> dict:
    """
    Загрузка конфигурации из файла.
    
    Приоритет:
    1. Файл config.env (высший приоритет, для dev-режима)
    2. Переменные окружения (для Docker без config.env)
    3. Значения по умолчанию
    """
    defaults = {
        "RTMP_URL": "rtmp://nginx-rtmp/live",
        "OUTPUT_DIR": "/recordings",
        "LOG_FILE": "/logs/motion_detector.log",
        "CONTROL_FILE": "/tmp/control/command",
        "BUFFER_SECONDS": "5",
        "POST_MOTION_SECONDS": "5",
        "MIN_CONTOUR_AREA": "500",
        "MIN_MOTION_FRAMES": "3",
        "MOTION_AREA_PERCENT": "0.5",
        "AUTO_START_MOTION": "false",
        "SEGMENT_DURATION": "2",
    }
    
    config = defaults.copy()
    
    # Сначала применяем переменные окружения
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
    # Ищем config.env в разных местах
    if config_path is None:
        possible_paths = [
            "/app/config.env",
            os.path.join(os.path.dirname(__file__), "..", "config.env"),
            "config.env",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    # Файл config.env имеет ВЫСШИЙ приоритет
    if config_path and os.path.exists(config_path):
        print(f"📋 Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    return config


def main():
    """Точка входа."""
    config = load_config()
    
    rtmp_url = config["RTMP_URL"]
    output_dir = config["OUTPUT_DIR"]
    log_file = config["LOG_FILE"]
    buffer_seconds = int(config["BUFFER_SECONDS"])
    post_motion_seconds = int(config["POST_MOTION_SECONDS"])
    min_contour_area = int(config["MIN_CONTOUR_AREA"])
    min_motion_frames = int(config["MIN_MOTION_FRAMES"])
    motion_area_percent = float(config["MOTION_AREA_PERCENT"])
    auto_start_motion = config["AUTO_START_MOTION"].lower() == "true"
    control_file = config["CONTROL_FILE"]
    segment_duration = int(config.get("SEGMENT_DURATION", "2"))
    
    detector = MotionDetector(
        rtmp_url=rtmp_url,
        output_dir=output_dir,
        log_file=log_file,
        buffer_seconds=buffer_seconds,
        post_motion_seconds=post_motion_seconds,
        min_contour_area=min_contour_area,
        min_motion_frames=min_motion_frames,
        motion_area_percent=motion_area_percent,
        segment_duration=segment_duration
    )
    
    def signal_handler(sig, frame):
        detector.logger.info(f"Received signal {sig}")
        detector.stop()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    control_thread = Thread(
        target=monitor_control_file, args=(detector, control_file), daemon=True
    )
    control_thread.start()
    
    if auto_start_motion:
        detector.enable_motion_detection()
    
    detector.run()


if __name__ == "__main__":
    main()
