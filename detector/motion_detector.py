#!/usr/bin/env python3
"""
Motion Detector for GoPro Bird Watcher
Детектор движения с кольцевым буфером для записи моментов с птицами.

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
from datetime import datetime, timezone, timedelta
from collections import deque
from threading import Thread, Event
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


class MotionDetector:
    """Детектор движения с кольцевым буфером и записью."""
    
    def __init__(
        self,
        rtmp_url: str = "rtmp://nginx-rtmp/live",
        output_dir: str = "/recordings",
        log_file: str = "/logs/motion_detector.log",
        buffer_seconds: int = 5,
        post_motion_seconds: int = 5,
        min_contour_area: int = 500,
        min_motion_frames: int = 3,
        motion_area_percent: float = 0.5
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
        """
        self.rtmp_url = rtmp_url
        self.output_dir = output_dir
        self.buffer_seconds = buffer_seconds
        self.post_motion_seconds = post_motion_seconds
        self.min_contour_area = min_contour_area
        self.min_motion_frames = min_motion_frames
        self.motion_area_percent = motion_area_percent
        
        # Логирование
        self.logger = setup_logging(log_file)
        
        self.cap = None
        self.fps = 30
        self.frame_width = 0
        self.frame_height = 0
        self.frame_area = 0
        
        # Кольцевой буфер для хранения последних N секунд
        self.frame_buffer = deque(maxlen=buffer_seconds * self.fps)
        
        # Состояние записи
        self.is_recording = False
        self.recording_type = RecordingType.NONE
        self.motion_detection_enabled = False  # Авто-запись при движении
        self.video_writer = None
        self.recording_start_time = None
        self.current_recording_file = None
        
        # Состояние движения
        self.last_motion_time = 0
        self.consecutive_motion_frames = 0
        self.significant_motion_started = False
        
        # Background subtractor
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )
        
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
        
        # Создаём подпапки для разных типов записей
        self.motion_dir = os.path.join(output_dir, "motion")
        self.manual_dir = os.path.join(output_dir, "manual")
        os.makedirs(self.motion_dir, exist_ok=True)
        os.makedirs(self.manual_dir, exist_ok=True)
        
        self.logger.info(f"Motion detector initialized")
        self.logger.info(f"  Output dirs: motion={self.motion_dir}, manual={self.manual_dir}")
        self.logger.info(f"  Buffer: {buffer_seconds}s before, {post_motion_seconds}s after")
        self.logger.info(f"  Thresholds: min_frames={min_motion_frames}, "
                        f"area_percent={motion_area_percent}%")
    
    def get_moscow_time(self) -> datetime:
        """Получить текущее время по Москве."""
        return datetime.now(MOSCOW_TZ)
    
    def format_duration(self, seconds: float) -> str:
        """Форматирование продолжительности в MM:SS."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
    
    def connect(self) -> bool:
        """Подключение к RTMP потоку."""
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
        
        # Обновляем размер буфера под актуальный FPS
        self.frame_buffer = deque(maxlen=self.buffer_seconds * self.fps)
        
        self.logger.info(
            f"Connected: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
        )
        return True
    
    def detect_motion(self, frame: np.ndarray) -> tuple[bool, float]:
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
    
    def start_recording(self, rec_type: RecordingType, include_buffer: bool = True):
        """
        Начать запись видео.
        
        Args:
            rec_type: Тип записи (MOTION или MANUAL)
            include_buffer: Включить ли буфер (кадры до начала записи)
        """
        if self.is_recording:
            return
        
        now = self.get_moscow_time()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        if rec_type == RecordingType.MOTION:
            prefix = "bird"
            output_folder = self.motion_dir
        else:
            prefix = "manual"
            output_folder = self.manual_dir
        
        filename = f"{prefix}_{timestamp}.mp4"
        filepath = os.path.join(output_folder, filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            filepath, fourcc, self.fps, (self.frame_width, self.frame_height)
        )
        
        buffer_frames_count = 0
        if include_buffer:
            buffer_frames = list(self.frame_buffer)
            for buffered_frame in buffer_frames:
                self.video_writer.write(buffered_frame)
            buffer_frames_count = len(buffer_frames)
        
        self.is_recording = True
        self.recording_type = rec_type
        self.recording_start_time = time.time()
        self.current_recording_file = filepath
        
        buffer_duration = buffer_frames_count / self.fps if self.fps else 0
        type_str = "🐦 MOTION" if rec_type == RecordingType.MOTION else "🎬 MANUAL"
        self.logger.info(
            f"▶ {type_str} recording started: {filename} "
            f"(buffer: {buffer_duration:.1f}s, {buffer_frames_count} frames)"
        )
    
    def stop_recording(self):
        """Остановить запись видео и переименовать файл с продолжительностью."""
        if not self.is_recording:
            return
        
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
        
        duration = time.time() - self.recording_start_time
        duration_str = self.format_duration(duration)
        
        final_filename = None
        if self.current_recording_file and os.path.exists(self.current_recording_file):
            old_path = self.current_recording_file
            dir_name = os.path.dirname(old_path)
            base_name = os.path.basename(old_path)
            name_without_ext = os.path.splitext(base_name)[0]
            new_filename = f"{name_without_ext}_{duration_str}.mp4"
            new_path = os.path.join(dir_name, new_filename)
            
            os.rename(old_path, new_path)
            final_filename = new_filename
        
        # Обновляем статистику
        if self.recording_type == RecordingType.MOTION:
            self.stats['motion_videos_saved'] += 1
            type_str = "🐦 MOTION"
        else:
            self.stats['manual_videos_saved'] += 1
            type_str = "🎬 MANUAL"
        
        self.logger.info(f"■ {type_str} recording saved: {final_filename} ({duration:.1f}s)")
        
        self.is_recording = False
        self.recording_type = RecordingType.NONE
        self.recording_start_time = None
        self.current_recording_file = None
    
    def process_frame(self, frame: np.ndarray):
        """Обработка одного кадра."""
        current_time = time.time()
        
        # Добавляем кадр в буфер (всегда, для записи "до движения")
        self.frame_buffer.append(frame.copy())
        
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
                    # и не идёт ручная запись
                    if (self.motion_detection_enabled and 
                        not self.is_recording):
                        self.start_recording(RecordingType.MOTION, include_buffer=True)
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
                # Останавливаем только MOTION запись, не MANUAL
                if self.is_recording and self.recording_type == RecordingType.MOTION:
                    self.stop_recording()
        
        # Если идёт запись, пишем текущий кадр
        if self.is_recording and self.video_writer:
            self.video_writer.write(frame)
        
        self.stats['frames_processed'] += 1
    
    # === Команды управления ===
    
    def enable_motion_detection(self):
        """Включить автоматическую запись при движении."""
        self.motion_detection_enabled = True
        self.logger.info("✅ MOTION detection ENABLED - auto-save on significant motion")
    
    def disable_motion_detection(self):
        """Выключить автоматическую запись при движении."""
        self.motion_detection_enabled = False
        # Если идёт MOTION запись, останавливаем её
        if self.is_recording and self.recording_type == RecordingType.MOTION:
            self.stop_recording()
        self.logger.info("⏹ MOTION detection DISABLED")
    
    def start_manual_recording(self):
        """Начать принудительную запись (ручной режим)."""
        if self.is_recording:
            if self.recording_type == RecordingType.MANUAL:
                self.logger.warning("Manual recording already in progress")
            else:
                self.logger.warning("Cannot start manual recording: motion recording active")
            return
        self.start_recording(RecordingType.MANUAL, include_buffer=False)
    
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
            'stats': self.stats
        }
    
    def run(self):
        """Основной цикл обработки."""
        if not self.connect():
            for attempt in range(5):
                self.logger.info(f"Reconnect attempt {attempt + 1}/5...")
                time.sleep(5)
                if self.connect():
                    break
            else:
                self.logger.error("Failed to connect after 5 attempts")
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
    # Значения по умолчанию
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
    
    # Файл config.env имеет ВЫСШИЙ приоритет (переопределяет env vars)
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
    # Загружаем конфигурацию
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
    
    detector = MotionDetector(
        rtmp_url=rtmp_url,
        output_dir=output_dir,
        log_file=log_file,
        buffer_seconds=buffer_seconds,
        post_motion_seconds=post_motion_seconds,
        min_contour_area=min_contour_area,
        min_motion_frames=min_motion_frames,
        motion_area_percent=motion_area_percent
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
