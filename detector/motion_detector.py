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
import platform
import asyncio
from datetime import datetime, timezone, timedelta
from threading import Thread, Event, Lock
from enum import Enum
import logging

# Локальные модули
try:
    from storage_manager import StorageManager
except ImportError:
    # Если запускаем из другой директории
    try:
        from detector.storage_manager import StorageManager
    except ImportError:
        StorageManager = None  # Будет работать без управления хранилищем

try:
    from telegram_bot import TelegramNotifier
except ImportError:
    try:
        from detector.telegram_bot import TelegramNotifier
    except ImportError:
        TelegramNotifier = None  # Будет работать без Telegram

# Московское время (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))

# === Константы по умолчанию ===
DEFAULT_RTMP_URL = "rtmp://nginx-rtmp/live"
DEFAULT_OUTPUT_DIR = "/recordings"
DEFAULT_LOG_FILE = "/logs/motion_detector.log"
DEFAULT_CONTROL_FILE = "/tmp/control/command"
DEFAULT_BUFFER_SECONDS = 5
DEFAULT_POST_MOTION_SECONDS = 5
DEFAULT_MIN_CONTOUR_AREA = 500
DEFAULT_MIN_MOTION_FRAMES = 3
DEFAULT_MOTION_AREA_PERCENT = 0.5
DEFAULT_EXTEND_MOTION_PERCENT = 0.2
DEFAULT_SEGMENT_DURATION = 1
DEFAULT_AUTO_START_MOTION = False
DEFAULT_DEBUG_MOTION = False

# ROI (Region of Interest) - область кормушки
DEFAULT_ROI_ENABLED = False
DEFAULT_ROI_X = 0
DEFAULT_ROI_Y = 0
DEFAULT_ROI_WIDTH = 0   # 0 = весь кадр
DEFAULT_ROI_HEIGHT = 0  # 0 = весь кадр

# USB Webcam режим (альтернатива RTMP)
DEFAULT_INPUT_SOURCE = "rtmp"  # "rtmp" или "usb"
DEFAULT_USB_DEVICE = "/dev/video0"
DEFAULT_USB_RESOLUTION = "1080"  # 480, 720, 1080
DEFAULT_USB_FPS = 30

# Управление хранилищем
DEFAULT_MAX_RECORDING_AGE_DAYS = 30
DEFAULT_MIN_FREE_SPACE_GB = 10.0
DEFAULT_AUTO_CLEANUP_ENABLED = True
DEFAULT_CLEANUP_INTERVAL_HOURS = 1

# Telegram бот
DEFAULT_TELEGRAM_ENABLED = False
DEFAULT_TELEGRAM_BOT_TOKEN = ""
DEFAULT_TELEGRAM_CHAT_ID = ""
DEFAULT_TELEGRAM_SEND_ON_MOTION = True
DEFAULT_TELEGRAM_SEND_MANUAL = False
DEFAULT_TELEGRAM_MAX_VIDEO_MB = 45.0


class RecordingType(Enum):
    """Тип записи."""
    NONE = "none"
    MOTION = "motion"
    MANUAL = "manual"


def setup_logging(log_file: str = None):
    """Настройка логирования в файл и консоль."""
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    # INFO уровень для чистых логов (DEBUG засоряет логи FFmpeg сообщениями)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


class FileCapture:
    """
    Читает кадры из растущего .ts файла, который
    FFmpeg пишет в реальном времени.
    На macOS: вместо второго доступа к камере,
    OpenCV читает тот же файл что записывает FFmpeg.
    Периодически переоткрывает файл чтобы видеть
    новые данные.
    """

    def __init__(self, ts_path, width, height, fps,
                 logger=None):
        self._log = logger or logging.getLogger(__name__)
        self.ts_path = ts_path
        self.width = width
        self.height = height
        self.fps = fps
        self._cap = None
        self._eof_count = 0
        self._max_eof = 5  # переоткрыть после 5 EOF
        self._opened = True
        # Подавляем h264 warnings от FFmpeg при чтении
        # растущего .ts файла (corrupted macroblock и т.п.)
        # Наши логи идут через stdout — не затронуты.
        self._suppress_ffmpeg_warnings()
        self._reopen()

    @staticmethod
    def _suppress_ffmpeg_warnings():
        """Перенаправить C-level stderr в /dev/null."""
        try:
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 2)  # fd 2 = stderr
            os.close(devnull_fd)
        except Exception:
            pass

    def _reopen(self):
        """Переоткрыть файл и перемотать к концу."""
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(self.ts_path)
        if self._cap.isOpened():
            total = int(
                self._cap.get(cv2.CAP_PROP_FRAME_COUNT)
            )
            # Перемотать к последним кадрам
            if total > 10:
                self._cap.set(
                    cv2.CAP_PROP_POS_FRAMES, total - 5
                )
            self._eof_count = 0

    def isOpened(self):
        return self._opened and os.path.exists(self.ts_path)

    def read(self):
        if not self._opened or not self._cap:
            return False, None
        ret, frame = self._cap.read()
        if ret:
            self._eof_count = 0
            return True, frame
        # EOF — файл ещё пишется, подождать
        self._eof_count += 1
        if self._eof_count >= self._max_eof:
            self._reopen()
        else:
            time.sleep(0.033)  # ~30fps
        return False, None

    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        return 0.0

    def set(self, prop, value):
        pass

    def release(self):
        if self._cap:
            self._cap.release()
        self._opened = False


class SegmentRecorder:
    """
    Записывает видеопоток короткими сегментами через FFmpeg.
    Поддерживает RTMP и USB (V4L2) источники.
    Сегменты именуются с timestamp для точного выбора по времени.
    
    ВАЖНО: На macOS сегментация не работает с AVFoundation,
    поэтому используется прямая запись в один файл.
    """
    
    def __init__(
        self,
        source_url: str,
        segments_dir: str,
        segment_duration: int = 1,
        max_segments: int = 180,
        logger: logging.Logger = None,
        input_source: str = "rtmp",
        usb_device: str = "/dev/video0",
        usb_resolution: str = "1080",
        usb_fps: int = 30
    ):
        self.source_url = source_url  # RTMP URL или USB device path
        self.segments_dir = segments_dir
        self.segment_duration = segment_duration
        self.max_segments = max_segments
        self.logger = logger or logging.getLogger(__name__)
        
        # Режим работы: rtmp или usb
        self.input_source = input_source.lower()
        self.usb_device = usb_device
        self.usb_resolution = usb_resolution
        self.usb_fps = usb_fps
        
        # Определяем режим записи (сегменты или прямая запись)
        system = platform.system()
        self.use_segments = not (system == "Darwin" and input_source == "usb")
        
        self.ffmpeg_process = None
        self.is_running = False
        self.stop_event = Event()
        self.lock = Lock()
        
        # Флаг для приостановки cleanup во время записи
        self.cleanup_paused = False
        
        # Для режима без сегментов (macOS USB)
        self.direct_output_file = None
        self.recording_start_time = None

        # Очищаем и создаём папку сегментов
        self._clean_segments_dir()
        
        source_info = self.usb_device if self.input_source == "usb" else self.source_url
        mode = "segments" if self.use_segments else "direct"
        self.logger.info(f"SegmentRecorder initialized: {segments_dir}")
        self.logger.info(f"  Input source: {self.input_source.upper()} ({source_info})")
        self.logger.info(f"  Recording mode: {mode}")
    
    def _kill_existing_ffmpeg(self):
        """Убить все существующие FFmpeg процессы записи сегментов."""
        killed = False
        
        # Способ 1: pkill (работает в большинстве контейнеров)
        try:
            result = subprocess.run(
                ["pkill", "-9", "-f", f"ffmpeg.*{self.segments_dir}"],
                capture_output=True,
                timeout=5
            )
            if result.returncode == 0:
                killed = True
                self.logger.info("Killed existing FFmpeg processes via pkill")
        except FileNotFoundError:
            pass
        except Exception:
            pass
        
        # Способ 2: killall ffmpeg (если pkill не сработал)
        if not killed:
            try:
                subprocess.run(["killall", "-9", "ffmpeg"], 
                              capture_output=True, timeout=5)
            except Exception:
                pass
    
    def _clean_segments_dir(self):
        """Безопасно очищаем папку сегментов."""
        # Сначала убиваем старые FFmpeg процессы
        self._kill_existing_ffmpeg()
        time.sleep(0.5)
        
        try:
            if os.path.exists(self.segments_dir):
                # Удаляем только .ts и .txt файлы
                for pattern in ["*.ts", "*.txt"]:
                    for f in glob.glob(os.path.join(self.segments_dir, pattern)):
                        try:
                            os.remove(f)
                        except Exception:
                            pass
            os.makedirs(self.segments_dir, exist_ok=True)
        except Exception as e:
            self.logger.warning(f"Error cleaning segments dir: {e}")
            os.makedirs(self.segments_dir, exist_ok=True)
    
    def _start_ffmpeg(self):
        """Внутренний метод запуска FFmpeg процесса."""
        segment_pattern = os.path.join(self.segments_dir, "seg_%Y%m%d_%H%M%S.ts")
        system = platform.system()
        
        if self.input_source == "usb":
            # USB режим - захват с веб-камеры (зависит от платформы)
            # Определяем разрешение
            resolution_map = {
                "480": "854x480",
                "720": "1280x720",
                "1080": "1920x1080"
            }
            resolution = resolution_map.get(self.usb_resolution, "1280x720")
            
            if system == "Darwin":  # macOS - используем AVFoundation
                # Автоопределение GoPro для FFmpeg
                ffmpeg_device = self.usb_device
                if self.usb_device.lower() == "auto":
                    gopro_index = detect_gopro_macos(self.logger)
                    if gopro_index >= 0:
                        ffmpeg_device = str(gopro_index)
                        self.logger.info(
                            f"🎯 FFmpeg: auto-detected GoPro at index {gopro_index}"
                        )
                    else:
                        ffmpeg_device = "0"
                        self.logger.warning(
                            "FFmpeg: GoPro not found, falling back to device 0"
                        )
                
                # macOS: Прямая запись без сегментации (AVFoundation не поддерживает)
                self.logger.info(
                    f"macOS: Using AVFoundation for direct capture "
                    f"from device {ffmpeg_device}"
                )
                
                # Временный файл для записи (будет переименован при финализации)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                self.direct_output_file = os.path.join(
                    self.segments_dir, 
                    f"temp_recording_{timestamp}.ts"
                )
                self.recording_start_time = time.time()
                
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "warning",
                    "-f", "avfoundation",
                    "-framerate", str(self.usb_fps),
                    "-video_size", resolution,
                    "-i", ffmpeg_device,
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-g", "30",
                    "-crf", "23",
                    "-f", "mpegts",
                    self.direct_output_file
                ]
                self.logger.info(
                    f"  Recording: {self.direct_output_file}"
                )
                self.logger.info(
                    "  Keyframe interval: 1s (-g 30)"
                )
                self.logger.info(
                    "  Analysis: from .ts file (shared)"
                )
                self.logger.warning(
                    "  macOS: NO pre-buffer "
                    "(recording starts from now)"
                )
            else:  # Linux - используем V4L2
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-loglevel", "warning",
                    # Входные параметры для V4L2
                    "-f", "v4l2",
                    "-input_format", "mjpeg",
                    "-video_size", resolution,
                    "-framerate", str(self.usb_fps),
                    "-i", self.usb_device,
                    # Кодирование
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-crf", "23",
                    # Сегментация
                    "-f", "segment",
                    "-segment_time", str(self.segment_duration),
                    "-segment_format", "mpegts",
                    "-segment_atclocktime", "1",
                    "-reset_timestamps", "1",
                    "-strftime", "1",
                    segment_pattern
                ]
        else:
            # RTMP режим - копируем поток
            cmd = [
                "ffmpeg",
                "-y",
                "-loglevel", "warning",
                "-i", self.source_url,
                "-c:v", "copy",
                "-c:a", "aac",
                "-f", "segment",
                "-segment_time", str(self.segment_duration),
                "-segment_format", "mpegts",
                "-segment_atclocktime", "1",
                "-reset_timestamps", "1",
                "-strftime", "1",
                segment_pattern
            ]
        
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE
        )
        
        # Поток для логирования stderr
        def log_stderr():
            try:
                for line in self.ffmpeg_process.stderr:
                    line = line.decode().strip()
                    if line:
                        self.logger.debug(f"FFmpeg: {line}")
            except Exception:
                pass
        
        Thread(target=log_stderr, daemon=True).start()
        
        self.last_segment_time = time.time()
        return self.ffmpeg_process
    
    def _monitor_ffmpeg(self):
        """Мониторинг и автоперезапуск FFmpeg."""
        restart_count = 0
        max_restarts = 20
        
        # Даём FFmpeg время на старт
        time.sleep(5)
        
        while not self.stop_event.is_set() and restart_count < max_restarts:
            time.sleep(5)
            
            if self.stop_event.is_set():
                break
            
            # Проверяем живой ли процесс
            if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None:
                exit_code = self.ffmpeg_process.poll() if self.ffmpeg_process else "N/A"
                restart_count += 1
                self.logger.warning(
                    f"⚠️ FFmpeg died (exit={exit_code})! "
                    f"Restarting ({restart_count}/{max_restarts})..."
                )
                time.sleep(2)
                try:
                    self._start_ffmpeg()
                    self.logger.info("✅ FFmpeg restarted successfully")
                    time.sleep(3)  # Даём время на старт
                except Exception as e:
                    self.logger.error(f"Failed to restart FFmpeg: {e}")
                    time.sleep(5)
                continue
            
            # Проверяем создаются ли новые сегменты (только если уже есть сегменты)
            try:
                segments = glob.glob(os.path.join(self.segments_dir, "seg_*.ts"))
                if len(segments) < 3:
                    continue
                
                # Безопасно получаем время самого нового сегмента
                newest_time = 0
                for seg in segments:
                    try:
                        mtime = os.path.getmtime(seg)
                        if mtime > newest_time:
                            newest_time = mtime
                    except FileNotFoundError:
                        continue  # Файл удалён, пропускаем
                
                if newest_time == 0:
                    continue
                
                stale_seconds = time.time() - newest_time
                
                # Если сегменты не создавались более 15 секунд — проблема
                if stale_seconds > 15:
                    restart_count += 1
                    self.logger.warning(
                        f"⚠️ No new segments for {stale_seconds:.0f}s! "
                        f"Restarting FFmpeg ({restart_count}/{max_restarts})..."
                    )
                    try:
                        if self.ffmpeg_process:
                            self.ffmpeg_process.kill()
                            self.ffmpeg_process.wait(timeout=3)
                    except Exception:
                        pass
                    
                    time.sleep(2)
                    try:
                        self._start_ffmpeg()
                        self.logger.info("✅ FFmpeg restarted (stale segments)")
                        time.sleep(3)
                    except Exception as e:
                        self.logger.error(f"Failed to restart FFmpeg: {e}")
                        time.sleep(5)
            except Exception:
                pass  # Игнорируем ошибки в мониторинге
        
        if restart_count >= max_restarts:
            self.logger.error(f"❌ FFmpeg failed {max_restarts} times. Giving up.")
    
    def start(self):
        """Запустить запись сегментов."""
        if self.is_running:
            return
        
        self.stop_event.clear()
        self.is_running = True
        
        self.logger.info("Starting FFmpeg segment recorder...")
        
        try:
            self._start_ffmpeg()
            
            # Поток мониторинга FFmpeg (перезапуск при падении)
            self.monitor_thread = Thread(target=self._monitor_ffmpeg, daemon=True)
            self.monitor_thread.start()
            
            # Поток очистки старых сегментов
            self.cleanup_thread = Thread(target=self._cleanup_old_segments, daemon=True)
            self.cleanup_thread.start()
            
            self.logger.info("SegmentRecorder started (with auto-restart)")
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg: {e}")
            self.is_running = False
    
    def stop(self):
        """Остановить запись сегментов."""
        if not self.is_running:
            return
        
        self.stop_event.set()
        self.is_running = False
        
        # Пробуем мягко остановить
        if self.ffmpeg_process:
            try:
                self.ffmpeg_process.stdin.write(b'q')
                self.ffmpeg_process.stdin.flush()
                self.ffmpeg_process.wait(timeout=3)
            except Exception:
                pass
            
            # Если ещё жив - убиваем
            if self.ffmpeg_process.poll() is None:
                try:
                    self.ffmpeg_process.kill()
                    self.ffmpeg_process.wait(timeout=2)
                except Exception:
                    pass
            self.ffmpeg_process = None
        
        # Убиваем ВСЕ процессы FFmpeg которые пишут в нашу папку
        self._kill_existing_ffmpeg()
        
        self.logger.info("SegmentRecorder stopped")
    
    def pause_cleanup(self):
        """Приостановить очистку сегментов."""
        self.cleanup_paused = True
    
    def resume_cleanup(self):
        """Возобновить очистку сегментов."""
        self.cleanup_paused = False
    
    def _cleanup_old_segments(self):
        """Удаляет старые сегменты, оставляя последние max_segments."""
        # Для режима прямой записи cleanup не нужен
        if not self.use_segments:
            return
        
        while not self.stop_event.is_set():
            if not self.cleanup_paused:
                try:
                    with self.lock:
                        segments = self._get_sorted_segments()
                        if len(segments) > self.max_segments:
                            to_delete = segments[:-self.max_segments]
                            for seg in to_delete:
                                try:
                                    os.remove(seg)
                                except FileNotFoundError:
                                    pass  # Уже удалён, это нормально
                                except Exception:
                                    pass
                except Exception:
                    pass
            
            time.sleep(self.segment_duration * 5)
    
    def _get_sorted_segments(self) -> list:
        """Получить список сегментов, отсортированных по имени (timestamp)."""
        pattern = os.path.join(self.segments_dir, "seg_*.ts")
        segments = glob.glob(pattern)
        return sorted(segments)
    
    def _get_segment_time(self, segment_path: str) -> float:
        """Получить время создания сегмента (mtime файла)."""
        try:
            return os.path.getmtime(segment_path)
        except Exception:
            return 0
    
    def get_segments_in_time_range(
        self, 
        start_time: float, 
        end_time: float = None
    ) -> list:
        """
        Получить сегменты в указанном временном диапазоне.
        
        Args:
            start_time: Unix timestamp начала
            end_time: Unix timestamp конца, None = текущее время
        
        Returns:
            Список путей к сегментам или прямой файл записи (для macOS)
        """
        if end_time is None:
            end_time = time.time()
        
        with self.lock:
            # Режим прямой записи (macOS): возвращаем один файл
            if not self.use_segments:
                if (self.direct_output_file and 
                    os.path.exists(self.direct_output_file) and
                    os.path.getsize(self.direct_output_file) > 1000):
                    return [self.direct_output_file]
                return []
            
            # Режим сегментов: собираем в диапазоне
            segments = self._get_sorted_segments()
            result = []
            
            # Расширяем диапазон на 2 сегмента с каждой стороны
            margin = self.segment_duration * 2
            
            for seg in segments:
                try:
                    if not os.path.exists(seg):
                        continue
                    
                    seg_time = os.path.getmtime(seg)
                    seg_size = os.path.getsize(seg)
                    
                    # Проверяем что сегмент в нужном диапазоне и не пустой
                    if (seg_time >= start_time - margin and 
                        seg_time <= end_time + margin and
                        seg_size > 1000):  # Минимум 1KB
                        result.append(seg)
                except Exception:
                    continue
            
            # Сортируем по времени создания
            result.sort(key=lambda x: os.path.getmtime(x))
            
            return result
    
    def get_current_time(self) -> float:
        """Получить текущее время."""
        return time.time()


class VideoMerger:
    """Объединяет сегменты в итоговое видео через FFmpeg."""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def merge_segments(
        self, 
        segments: list, 
        output_path: str,
        crop_params: tuple = None,
        time_range: tuple = None
    ) -> bool:
        """
        Объединить сегменты в один файл.
        
        Args:
            segments: Список путей к сегментам (или один файл для прямой записи)
            output_path: Путь для выходного файла
            crop_params: Кортеж (x, y, width, height) для обрезки или None
            time_range: Кортеж (start_seconds, duration_seconds) для вырезки
                       из непрерывной записи (macOS режим)
        
        Returns:
            True если успешно
        """
        if not segments:
            self.logger.error("No segments to merge")
            return False
        
        # Фильтруем только существующие и непустые файлы
        valid_segments = []
        for s in segments:
            if os.path.exists(s):
                size = os.path.getsize(s)
                if size > 1000:  # Минимум 1KB
                    valid_segments.append(s)
        
        if not valid_segments:
            self.logger.error("No valid segments to merge")
            return False
        
        if len(valid_segments) != len(segments):
            self.logger.warning(
                f"Filtered segments: {len(valid_segments)}/{len(segments)} valid"
            )
        
        # Режим прямой записи (macOS): один файл, вырезаем нужный кусок
        if len(valid_segments) == 1 and time_range:
            start_sec, duration_sec = time_range
            self.logger.info(
                f"Direct recording mode: extracting {duration_sec:.1f}s "
                f"starting at {start_sec:.1f}s"
            )
            
            input_file = valid_segments[0]
            # -ss ПОСЛЕ -i = output seeking (точный,
            # но медленнее). Перекодируем для чистого
            # начала видео (без чёрных/битых кадров).
            cmd = [
                "ffmpeg",
                "-y",
                "-i", input_file,
                "-ss", str(start_sec),
                "-t", str(duration_sec),
            ]
            
            if crop_params:
                x, y, w, h = crop_params
                self.logger.info(
                    f"Applying crop: {w}x{h} at ({x}, {y})"
                )
                cmd.extend([
                    "-vf", f"crop={w}:{h}:{x}:{y}",
                ])

            cmd.extend([
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-an",  # Без аудио (USB-камера)
            ])
            
            cmd.append(output_path)
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300
            )
            
            success = result.returncode == 0 and os.path.exists(output_path)
            
            if not success:
                stderr = result.stderr.decode()
                error_lines = [l for l in stderr.split('\n') 
                              if 'error' in l.lower()]
                if error_lines:
                    self.logger.error(f"FFmpeg error: {error_lines[-1][:200]}")
            
            return success
        
        # Создаём временный файл со списком сегментов
        list_file = output_path + ".concat.txt"
        
        try:
            with open(list_file, 'w') as f:
                for seg in valid_segments:
                    # Абсолютный путь для надёжности
                    abs_path = os.path.abspath(seg)
                    escaped_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
            # Базовая команда
            if crop_params:
                # С обрезкой - нужно перекодировать видео
                x, y, w, h = crop_params
                self.logger.info(f"Applying crop: {w}x{h} at ({x}, {y})")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-vf", f"crop={w}:{h}:{x}:{y}",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
                    "-c:a", "aac",
                    "-b:a", "128k",
                    output_path
                ]
            else:
                # Без обрезки - просто копируем потоки
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-f", "concat",
                    "-safe", "0",
                    "-i", list_file,
                    "-c", "copy",
                    output_path
                ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                timeout=300  # Увеличен таймаут для перекодирования
            )
            
            success = result.returncode == 0 and os.path.exists(output_path)
            
            if not success:
                stderr_output = result.stderr.decode()
                # Ищем реальную ошибку (пропускаем header)
                error_lines = [l for l in stderr_output.split('\n') 
                              if 'error' in l.lower() or 'invalid' in l.lower()]
                if error_lines:
                    self.logger.error(f"FFmpeg error: {error_lines[-1][:200]}")
                else:
                    # Показываем последние строки
                    self.logger.error(f"FFmpeg failed: {stderr_output[-500:]}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Merge error: {e}")
            return False
        finally:
            # Всегда удаляем временный файл списка
            try:
                if os.path.exists(list_file):
                    os.remove(list_file)
            except Exception:
                pass


def detect_gopro_macos(logger=None) -> int:
    """
    Автоопределение индекса GoPro на macOS через FFmpeg.
    
    Returns:
        Индекс GoPro устройства или -1 если не найдено
    """
    import re

    def _print(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-f", "avfoundation",
                "-list_devices", "true", "-i", ""
            ],
            capture_output=True, text=True, timeout=5
        )
        # Выводим полный список устройств для отладки
        devices = []
        for line in result.stderr.split('\n'):
            if 'AVFoundation' in line and ']' in line:
                match = re.search(
                    r'\[(\d+)\]\s*(.*)', line
                )
                if match:
                    devices.append(
                        (int(match.group(1)), match.group(2))
                    )
        if devices:
            _print("AVFoundation устройства:")
            for idx, name in devices:
                marker = " <-- GoPro" if (
                    'gopro' in name.lower()
                ) else ""
                _print(f"  [{idx}] {name}{marker}")

        for line in result.stderr.split('\n'):
            if 'gopro' in line.lower():
                match = re.search(r'\[(\d+)\]', line)
                if match:
                    idx = int(match.group(1))
                    _print(
                        f"GoPro найдена: AVFoundation"
                        f" index {idx}"
                    )
                    return idx
        _print(
            "GoPro не найдена в списке "
            "AVFoundation устройств"
        )
        return -1
    except Exception as e:
        _print(f"Ошибка автоопределения GoPro: {e}")
        return -1


def get_video_duration(filepath: str) -> float:
    """Получить длительность видео через ffprobe."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True,
            timeout=10
        )
        return float(result.stdout.decode().strip())
    except Exception:
        return 0.0


class MotionDetector:
    """Детектор движения с записью через FFmpeg (со звуком)."""
    
    def __init__(
        self,
        rtmp_url: str = DEFAULT_RTMP_URL,
        output_dir: str = DEFAULT_OUTPUT_DIR,
        log_file: str = DEFAULT_LOG_FILE,
        buffer_seconds: int = DEFAULT_BUFFER_SECONDS,
        post_motion_seconds: int = DEFAULT_POST_MOTION_SECONDS,
        min_contour_area: int = DEFAULT_MIN_CONTOUR_AREA,
        min_motion_frames: int = DEFAULT_MIN_MOTION_FRAMES,
        motion_area_percent: float = DEFAULT_MOTION_AREA_PERCENT,
        extend_motion_percent: float = DEFAULT_EXTEND_MOTION_PERCENT,
        debug_motion: bool = DEFAULT_DEBUG_MOTION,
        segment_duration: int = DEFAULT_SEGMENT_DURATION,
        roi_enabled: bool = DEFAULT_ROI_ENABLED,
        roi_x: int = DEFAULT_ROI_X,
        roi_y: int = DEFAULT_ROI_Y,
        roi_width: int = DEFAULT_ROI_WIDTH,
        roi_height: int = DEFAULT_ROI_HEIGHT,
        input_source: str = DEFAULT_INPUT_SOURCE,
        usb_device: str = DEFAULT_USB_DEVICE,
        usb_resolution: str = DEFAULT_USB_RESOLUTION,
        usb_fps: int = DEFAULT_USB_FPS,
        max_recording_age_days: int = DEFAULT_MAX_RECORDING_AGE_DAYS,
        min_free_space_gb: float = DEFAULT_MIN_FREE_SPACE_GB,
        auto_cleanup_enabled: bool = DEFAULT_AUTO_CLEANUP_ENABLED,
        cleanup_interval_hours: int = DEFAULT_CLEANUP_INTERVAL_HOURS,
        telegram_enabled: bool = DEFAULT_TELEGRAM_ENABLED,
        telegram_bot_token: str = DEFAULT_TELEGRAM_BOT_TOKEN,
        telegram_chat_id: str = DEFAULT_TELEGRAM_CHAT_ID,
        telegram_send_on_motion: bool = DEFAULT_TELEGRAM_SEND_ON_MOTION,
        telegram_send_manual: bool = DEFAULT_TELEGRAM_SEND_MANUAL,
        telegram_max_video_mb: float = DEFAULT_TELEGRAM_MAX_VIDEO_MB
    ):
        self.rtmp_url = rtmp_url
        self.output_dir = output_dir
        self.buffer_seconds = buffer_seconds
        self.post_motion_seconds = post_motion_seconds
        self.min_contour_area = min_contour_area
        self.min_motion_frames = min_motion_frames
        self.motion_area_percent = motion_area_percent
        self.extend_motion_percent = extend_motion_percent
        self.debug_motion = debug_motion
        self.segment_duration = segment_duration
        
        # ROI (Region of Interest) - область кормушки
        self.roi_enabled = roi_enabled
        self.roi_x = roi_x
        self.roi_y = roi_y
        self.roi_width = roi_width
        self.roi_height = roi_height
        
        # USB режим
        self.input_source = input_source.lower()
        self.usb_device = usb_device
        self.usb_resolution = usb_resolution
        self.usb_fps = usb_fps
        
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
            source_url=rtmp_url,
            segments_dir=self.segments_dir,
            segment_duration=segment_duration,
            max_segments=300,  # ~5 минут буфера
            logger=self.logger,
            input_source=self.input_source,
            usb_device=self.usb_device,
            usb_resolution=self.usb_resolution,
            usb_fps=self.usb_fps
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
        
        # Состояние записи (с блокировкой для thread-safety)
        self.recording_lock = Lock()
        self.is_recording = False
        self.recording_type = RecordingType.NONE
        self.motion_detection_enabled = False
        
        # Временные метки для выбора сегментов
        self.recording_buffer_start_time = None  # Начало буфера (ДО движения)
        self.recording_start_time = None          # Начало записи (момент движения)
        
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
        
        self.logger.info(f"Motion detector initialized")
        self.logger.info(f"  Output dirs: motion={self.motion_dir}, manual={self.manual_dir}")
        self.logger.info(f"  Buffer: {buffer_seconds}s before, {post_motion_seconds}s after")
        self.logger.info(f"  Segment duration: {segment_duration}s")
        self.logger.info(
            f"  Motion thresholds: start={motion_area_percent}%, "
            f"extend={extend_motion_percent}%"
        )
        
        # Логируем режим ввода
        if self.input_source == "usb":
            self.logger.info(f"  📹 INPUT: USB Webcam ({self.usb_device})")
            self.logger.info(f"     Resolution: {self.usb_resolution}p @ {self.usb_fps}fps")
        else:
            self.logger.info(f"  📡 INPUT: RTMP ({self.rtmp_url})")
        
        if debug_motion:
            self.logger.info(f"  DEBUG MODE: motion % will be logged")
        
        # Логируем настройки ROI
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            self.logger.info(
                f"  🎯 ROI ENABLED: {self.roi_width}x{self.roi_height} "
                f"at ({self.roi_x}, {self.roi_y})"
            )
            self.logger.info(
                f"     Motion detection and video crop will use ROI area only"
            )
        
        # Storage Manager для автоматической очистки
        self.storage_manager = None
        if StorageManager and auto_cleanup_enabled:
            try:
                self.storage_manager = StorageManager(
                    recordings_dir=self.output_dir,
                    max_age_days=max_recording_age_days,
                    min_free_gb=min_free_space_gb,
                    cleanup_interval_hours=cleanup_interval_hours,
                    logger=self.logger
                )
                self.logger.info(
                    f"  🗑️ STORAGE MANAGER: cleanup every {cleanup_interval_hours}h, "
                    f"max age {max_recording_age_days} days"
                )
            except Exception as e:
                self.logger.warning(f"Failed to init StorageManager: {e}")
        elif not auto_cleanup_enabled:
            self.logger.info("  Storage cleanup: disabled")
        elif not StorageManager:
            self.logger.warning(
                "  ⚠️ StorageManager not available (storage_manager.py not found)"
            )
        else:
            self.logger.info(f"  ROI disabled - using full frame")
        
        # Telegram Bot для отправки уведомлений
        self.telegram_notifier = None
        self.telegram_enabled = telegram_enabled
        self.telegram_send_on_motion = telegram_send_on_motion
        self.telegram_send_manual = telegram_send_manual
        
        if telegram_enabled and TelegramNotifier:
            if telegram_bot_token and telegram_chat_id:
                try:
                    self.telegram_notifier = TelegramNotifier(
                        bot_token=telegram_bot_token,
                        chat_id=telegram_chat_id,
                        send_on_motion=telegram_send_on_motion,
                        send_manual=telegram_send_manual,
                        max_video_mb=telegram_max_video_mb,
                        logger=self.logger
                    )
                    self.logger.info(
                        f"  📱 TELEGRAM BOT: enabled for chat {telegram_chat_id}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to init Telegram bot: {e}")
            else:
                self.logger.warning("Telegram enabled but token/chat_id not set")
        elif telegram_enabled and not TelegramNotifier:
            self.logger.warning(
                "  ⚠️ Telegram enabled but aiogram not installed "
                "(pip install aiogram==3.24.0)"
            )
    
    def get_moscow_time(self) -> datetime:
        """Получить текущее время по Москве."""
        return datetime.now(MOSCOW_TZ)
    
    def format_duration(self, seconds: float) -> str:
        """Форматирование продолжительности в MMmSSs."""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}m{secs:02d}s"
    
    def connect(self) -> bool:
        """Подключение к видеоисточнику (RTMP или USB) для анализа."""
        if self.input_source == "usb":
            return self._connect_usb()
        else:
            return self._connect_rtmp()
    
    def _connect_usb(self) -> bool:
        """Подключение к USB веб-камере (поддерживает Linux и macOS)."""
        system = platform.system()
        self.logger.info(
            f"Connecting to USB device {self.usb_device} on {system}..."
        )
        
        # Определяем индекс камеры в зависимости от платформы
        if system == "Darwin":  # macOS
            # На macOS OpenCV и FFmpeg нумеруют камеры
            # по-разному. Вместо открытия камеры через
            # cv2.VideoCapture (активирует вебку MacBook),
            # читаем кадры из .ts файла, который FFmpeg
            # уже пишет с GoPro. Один процесс — один файл.
            ts_file = self.segment_recorder.direct_output_file
            if not ts_file or not os.path.exists(ts_file):
                self.logger.error(
                    f"macOS: recording file not found: "
                    f"{ts_file}"
                )
                return False

            resolution_map = {
                "480": (854, 480),
                "720": (1280, 720),
                "1080": (1920, 1080)
            }
            w, h = resolution_map.get(
                self.usb_resolution, (1280, 720)
            )

            self.cap = FileCapture(
                ts_file, w, h,
                self.usb_fps, self.logger
            )

            if not self.cap.isOpened():
                self.logger.error(
                    f"Cannot open recording file: "
                    f"{ts_file}"
                )
                return False

            self.fps = self.usb_fps
            self.frame_width = w
            self.frame_height = h
            self.frame_area = w * h

            self.logger.info(
                f"macOS: analysis from .ts file, "
                f"{w}x{h} @ {self.fps}fps"
            )
            return True

        elif system == "Linux":
            # На Linux используем путь устройства /dev/videoX
            if self.usb_device.startswith("/dev/video"):
                device_index = int(self.usb_device.replace("/dev/video", ""))
                self.cap = cv2.VideoCapture(device_index)
            else:
                # Попытка использовать как путь или индекс
                try:
                    device_index = int(self.usb_device)
                    self.cap = cv2.VideoCapture(device_index)
                except ValueError:
                    self.cap = cv2.VideoCapture(self.usb_device)
        
        else:
            # Другие платформы - попытка использовать как есть
            self.logger.warning(f"Unknown platform: {system}, trying as-is")
            try:
                device_index = int(self.usb_device)
                self.cap = cv2.VideoCapture(device_index)
            except ValueError:
                self.cap = cv2.VideoCapture(self.usb_device)
        
        if not self.cap.isOpened():
            self.logger.error(f"Failed to open USB device {self.usb_device}")
            return False
        
        # Устанавливаем разрешение
        resolution_map = {
            "480": (854, 480),
            "720": (1280, 720),
            "1080": (1920, 1080)
        }
        width, height = resolution_map.get(self.usb_resolution, (1280, 720))
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, self.usb_fps)
        
        # Устанавливаем формат MJPEG для GoPro (на macOS может не работать)
        if system != "Darwin":
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Буфер минимальный для снижения задержки
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Получаем реальные параметры
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or self.usb_fps
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_area = self.frame_width * self.frame_height
        
        self.logger.info(
            f"USB connected: {self.frame_width}x{self.frame_height} @ "
            f"{self.fps}fps on {system}"
        )
        return True
    
    def _connect_rtmp(self) -> bool:
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
        
        self.logger.info(
            f"RTMP connected: {self.frame_width}x{self.frame_height} @ {self.fps}fps"
        )
        return True
    
    def detect_motion(self, frame: np.ndarray) -> tuple:
        """
        Детекция движения в кадре.
        
        Если ROI включен - анализируется только область интереса.
        """
        # Определяем область для анализа
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            # Обрезаем кадр до ROI области
            roi_frame = frame[
                self.roi_y:self.roi_y + self.roi_height,
                self.roi_x:self.roi_x + self.roi_width
            ]
            analysis_frame = roi_frame
            analysis_area = self.roi_width * self.roi_height
        else:
            analysis_frame = frame
            analysis_area = self.frame_area
        
        fg_mask = self.background_subtractor.apply(analysis_frame)
        
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
        
        motion_percent = (total_motion_area / analysis_area) * 100 if analysis_area else 0
        motion_detected = motion_percent >= self.motion_area_percent
        
        return motion_detected, motion_percent
    
    def _check_segments_fresh(self, max_age: float = 5.0) -> bool:
        """Проверить что сегменты свежие (FFmpeg работает)."""
        # Режим прямой записи (macOS)
        if not self.segment_recorder.use_segments:
            if self.segment_recorder.direct_output_file:
                if os.path.exists(self.segment_recorder.direct_output_file):
                    age = time.time() - os.path.getmtime(
                        self.segment_recorder.direct_output_file
                    )
                    return age < max_age
            return False
        
        # Режим сегментов
        segments = glob.glob(
            os.path.join(self.segment_recorder.segments_dir, "seg_*.ts")
        )
        if not segments:
            return False
        
        newest = max(segments, key=os.path.getmtime)
        age = time.time() - os.path.getmtime(newest)
        return age < max_age
    
    def start_recording(self, rec_type: RecordingType):
        """Начать запись видео."""
        with self.recording_lock:
            if self.is_recording:
                return
            
            # Проверяем что FFmpeg пишет свежие сегменты
            if not self._check_segments_fresh(max_age=10.0):
                self.logger.warning("⚠️ Segments are stale! Skipping recording.")
                return
            
            current_time = time.time()
            
            # Помечаем что запись началась СРАЗУ (до логирования)
            self.is_recording = True
            self.recording_type = rec_type
            
            # Приостанавливаем cleanup
            self.segment_recorder.pause_cleanup()
            
            # Запоминаем временные метки
            if rec_type == RecordingType.MOTION:
                self.recording_buffer_start_time = current_time - self.buffer_seconds
            else:
                self.recording_buffer_start_time = current_time
            
            self.recording_start_time = current_time
            
            type_str = "🐦 MOTION" if rec_type == RecordingType.MOTION else "🎬 MANUAL"
            self.logger.info(
                f"▶ {type_str} recording started "
                f"(buffer from {self.buffer_seconds}s ago)"
            )
    
    def stop_recording(self):
        """Остановить запись и сохранить видео."""
        with self.recording_lock:
            if not self.is_recording:
                return
            # Помечаем сразу что не записываем (чтобы не вызвали повторно)
            was_recording_type = self.recording_type
            self.is_recording = False
        
        # Остальная работа вне блокировки (занимает время)
        
        # Ждём пока FFmpeg допишет последние сегменты
        # (post_motion_seconds уже прошли, нужно только дождаться финализации)
        wait_time = self.segment_duration + 1
        self.logger.info(f"Finalizing recording ({wait_time}s)...")
        time.sleep(wait_time)
        
        # Время окончания = СЕЙЧАС (после ожидания), чтобы включить все сегменты
        recording_end_time = time.time()
        
        # Получаем сегменты в нужном временном диапазоне
        segments = self.segment_recorder.get_segments_in_time_range(
            start_time=self.recording_buffer_start_time,
            end_time=recording_end_time
        )
        
        expected_duration = recording_end_time - self.recording_buffer_start_time
        actual_duration = len(segments) * self.segment_duration
        
        self.logger.info(
            f"Segments: {len(segments)} (~{actual_duration}s), "
            f"expected: {expected_duration:.1f}s"
        )
        
        if not segments:
            self.logger.warning("No segments found for recording")
            self._reset_recording_state()
            return
        
        # Проверяем что сегменты действительно свежие
        newest_segment_time = max(os.path.getmtime(s) for s in segments)
        if newest_segment_time < self.recording_start_time - 5:
            self.logger.warning(
                f"⚠️ Segments are stale! Newest: {newest_segment_time:.0f}, "
                f"recording started: {self.recording_start_time:.0f}"
            )
            self._reset_recording_state()
            return
        
        # Формируем имя файла
        now = self.get_moscow_time()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        
        if was_recording_type == RecordingType.MOTION:
            prefix = "bird"
            output_folder = self.motion_dir
            self.stats['motion_videos_saved'] += 1
            type_str = "🐦 MOTION"
        else:
            prefix = "manual"
            output_folder = self.manual_dir
            self.stats['manual_videos_saved'] += 1
            type_str = "🎬 MANUAL"
        
        # Временный файл
        temp_filepath = os.path.join(output_folder, f"{prefix}_{timestamp}_temp.mp4")
        
        # Определяем параметры crop (если ROI включен)
        crop_params = None
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            crop_params = (self.roi_x, self.roi_y, self.roi_width, self.roi_height)
        
        # Объединяем сегменты
        self.logger.info(f"Merging {len(segments)} segments...")
        
        # Для режима прямой записи (macOS) вычисляем time_range
        time_range = None
        if not self.segment_recorder.use_segments and len(segments) == 1:
            # Вычисляем смещение от начала файла записи
            ffmpeg_start = self.segment_recorder.recording_start_time
            if ffmpeg_start:
                start_offset = max(0, self.recording_buffer_start_time - ffmpeg_start)
                duration = recording_end_time - self.recording_buffer_start_time
                time_range = (start_offset, duration)
                self.logger.info(
                    f"Direct mode: cutting from {start_offset:.1f}s, "
                    f"duration {duration:.1f}s"
                )
        
        if self.video_merger.merge_segments(
            segments, temp_filepath, crop_params, time_range
        ):
            # Проверяем что файл реально создался
            if not os.path.exists(temp_filepath):
                self.logger.error(f"Merge reported success but file not found: {temp_filepath}")
                self._reset_recording_state()
                return
            
            # Получаем точную длительность
            real_duration = get_video_duration(temp_filepath)
            
            if real_duration > 0:
                duration_str = self.format_duration(real_duration)
                final_filename = f"{prefix}_{timestamp}_{duration_str}.mp4"
                final_filepath = os.path.join(output_folder, final_filename)
                
                try:
                    os.rename(temp_filepath, final_filepath)
                    self.logger.info(
                        f"■ {type_str} saved: {final_filename} "
                        f"(duration: {real_duration:.1f}s)"
                    )
                    
                    # Отправить видео в Telegram (если включено)
                    self._send_to_telegram_async(
                        final_filepath,
                        was_recording_type,
                        real_duration
                    )
                except Exception as e:
                    self.logger.error(f"Failed to rename: {e}")
                    # Файл существует - просто используем temp имя
                    if os.path.exists(temp_filepath):
                        self.logger.info(f"■ {type_str} saved: {prefix}_{timestamp}_temp.mp4")
                        # Отправить видео в Telegram
                        self._send_to_telegram_async(
                            temp_filepath,
                            was_recording_type,
                            real_duration
                        )
            else:
                self.logger.warning("Could not get duration")
                final_filepath = temp_filepath
                self.logger.info(f"■ {type_str} saved: {prefix}_{timestamp}_temp.mp4")
                # Отправить видео в Telegram
                if os.path.exists(temp_filepath):
                    self._send_to_telegram_async(
                        temp_filepath,
                        was_recording_type,
                        0.0
                    )
        else:
            self.logger.error("Failed to merge segments")
            # Удаляем пустой temp файл если есть
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception:
                    pass
        
        self._reset_recording_state()
    
    def _reset_recording_state(self):
        """Сбросить состояние записи."""
        self.is_recording = False
        self.recording_type = RecordingType.NONE
        self.recording_start_time = None
        self.recording_buffer_start_time = None
        
        # Возобновляем cleanup
        self.segment_recorder.resume_cleanup()
    
    def _send_to_telegram_async(
        self,
        video_path: str,
        recording_type: RecordingType,
        duration: float
    ):
        """
        Отправить видео в Telegram асинхронно (в отдельном потоке).
        
        Args:
            video_path: Путь к видео файлу
            recording_type: Тип записи (MOTION или MANUAL)
            duration: Длительность видео в секундах
        """
        # Проверяем что Telegram включен
        if not self.telegram_notifier:
            return
        
        # Проверяем настройки отправки
        should_send = False
        if recording_type == RecordingType.MOTION and self.telegram_send_on_motion:
            should_send = True
        elif recording_type == RecordingType.MANUAL and self.telegram_send_manual:
            should_send = True
        
        if not should_send:
            return
        
        # Формируем caption
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        duration_str = self.format_duration(duration)
        
        if recording_type == RecordingType.MOTION:
            emoji = "🐦"
            type_name = "Птица обнаружена"
        else:
            emoji = "🎬"
            type_name = "Ручная запись"
        
        caption = (
            f"{emoji} <b>{type_name}!</b>\n"
            f"📅 {timestamp}\n"
            f"⏱ Длительность: {duration_str}"
        )
        
        # Запускаем отправку в отдельном потоке с НОВЫМ Bot
        notifier = self.telegram_notifier
        logger = self.logger
        
        def send_video_thread():
            try:
                from aiogram import Bot
                from aiogram.types import FSInputFile
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def _send():
                    # Создаём НОВЫЙ Bot с тем же токеном
                    bot = Bot(token=notifier.bot_token)
                    try:
                        # Проверяем размер, сжимаем если нужно
                        final_path = video_path
                        size_mb = os.path.getsize(video_path) / (1024**2)
                        
                        if size_mb > notifier.max_video_mb:
                            logger.info(
                                f"  📱 Compressing {size_mb:.1f}MB..."
                            )
                            compressed = await notifier._compress_video(video_path)
                            if compressed and os.path.exists(compressed):
                                final_path = compressed
                        
                        video_file = FSInputFile(final_path)
                        await bot.send_video(
                            chat_id=notifier.chat_id,
                            video=video_file,
                            caption=caption[:1024] if caption else None,
                            parse_mode="HTML",
                            supports_streaming=True
                        )
                        
                        logger.info(
                            f"  📱 Sent to Telegram: {os.path.basename(video_path)}"
                        )
                        
                        # Удаляем сжатую версию
                        if final_path != video_path and os.path.exists(final_path):
                            os.remove(final_path)
                        
                    finally:
                        await bot.session.close()
                
                loop.run_until_complete(_send())
                loop.close()
            except Exception as e:
                logger.error(f"Error sending to Telegram: {e}", exc_info=True)
        
        thread = Thread(target=send_video_thread, daemon=True)
        thread.start()
    
    def process_frame(self, frame: np.ndarray):
        """Обработка одного кадра."""
        current_time = time.time()
        
        # Детекция движения
        significant_motion, motion_percent = self.detect_motion(frame)
        
        # Любое движение выше порога продлевает запись
        any_motion = motion_percent >= self.extend_motion_percent
        
        # DEBUG: логируем движение во время записи
        if self.debug_motion and self.is_recording:
            # Логируем каждую секунду чтобы не спамить
            if not hasattr(self, '_last_debug_log') or \
               current_time - self._last_debug_log >= 1.0:
                self._last_debug_log = current_time
                time_since = current_time - self.last_motion_time
                status = "📍" if any_motion else "⚪"
                self.logger.debug(
                    f"{status} Motion: {motion_percent:.2f}% "
                    f"(extend threshold: {self.extend_motion_percent}%), "
                    f"time since last: {time_since:.1f}s"
                )
        
        # Обновляем время последнего движения при ЛЮБОМ движении выше порога
        if any_motion and self.significant_motion_started:
            self.last_motion_time = current_time
        
        if significant_motion:
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
                    
                    # Начинаем MOTION запись
                    if self.motion_detection_enabled and not self.is_recording:
                        self.start_recording(RecordingType.MOTION)
        else:
            self.consecutive_motion_frames = 0
        
        # Проверяем окончание движения
        if self.significant_motion_started:
            time_since_last_motion = current_time - self.last_motion_time
            
            # DEBUG: показываем обратный отсчёт перед остановкой
            if self.debug_motion and time_since_last_motion > 1.0:
                remaining = self.post_motion_seconds - time_since_last_motion
                if remaining > 0 and int(remaining) != getattr(self, '_last_countdown', -1):
                    self._last_countdown = int(remaining)
                    self.logger.info(
                        f"   ⏳ No motion for {time_since_last_motion:.1f}s, "
                        f"stopping in {remaining:.0f}s..."
                    )
            
            # Движение прекратилось, ждём post_motion_seconds
            if time_since_last_motion > self.post_motion_seconds:
                self.significant_motion_started = False
                self._last_countdown = -1  # Reset countdown
                
                total_recording_time = current_time - self.recording_start_time \
                    if self.recording_start_time else 0
                
                self.logger.info(
                    f"   ⏹ Motion stopped. Recorded {total_recording_time:.1f}s total "
                    f"(incl. {self.post_motion_seconds}s post-buffer)"
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
        """Начать принудительную запись."""
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
        roi_info = None
        if self.roi_enabled and self.roi_width > 0 and self.roi_height > 0:
            roi_info = {
                'enabled': True,
                'x': self.roi_x,
                'y': self.roi_y,
                'width': self.roi_width,
                'height': self.roi_height
            }
        
        input_info = {
            'source': self.input_source,
            'device': self.usb_device if self.input_source == 'usb' else self.rtmp_url,
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'fps': self.fps
        }
        
        return {
            'motion_detection_enabled': self.motion_detection_enabled,
            'is_recording': self.is_recording,
            'recording_type': self.recording_type.value,
            'segment_recorder_running': self.segment_recorder.is_running,
            'input': input_info,
            'roi': roi_info,
            'stats': self.stats
        }
    
    def run(self):
        """Основной цикл обработки."""
        # Запускаем запись сегментов
        self.segment_recorder.start()
        
        # Запускаем Storage Manager
        if self.storage_manager:
            self.storage_manager.start()
        
        # Ждём пока накопятся сегменты для буфера
        wait_for_buffer = self.buffer_seconds + 2
        self.logger.info(f"Waiting {wait_for_buffer}s for buffer to fill...")
        time.sleep(wait_for_buffer)
        
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
        
        is_file_capture = isinstance(self.cap, FileCapture)

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            
            if not ret:
                if is_file_capture:
                    # FileCapture: EOF — файл ещё пишется,
                    # read() сам обрабатывает переоткрытие
                    continue

                reconnect_attempts += 1
                self.logger.warning(
                    f"Frame read failed. "
                    f"Reconnect attempt {reconnect_attempts}"
                )
                
                if reconnect_attempts > max_reconnect_attempts:
                    self.logger.error(
                        "Max reconnect attempts reached."
                    )
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
        if self.storage_manager:
            self.storage_manager.stop()
        if self.cap:
            self.cap.release()
        self.logger.info(f"Final stats: {self.stats}")
    
    def stop(self):
        """Остановить детектор."""
        self.logger.info("Stop signal received")
        self.stop_event.set()


def monitor_control_file(detector: MotionDetector, control_file: str):
    """Мониторит файл управления."""
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
    """Загрузка конфигурации из файла."""
    defaults = {
        "RTMP_URL": DEFAULT_RTMP_URL,
        "OUTPUT_DIR": DEFAULT_OUTPUT_DIR,
        "LOG_FILE": DEFAULT_LOG_FILE,
        "CONTROL_FILE": DEFAULT_CONTROL_FILE,
        "BUFFER_SECONDS": str(DEFAULT_BUFFER_SECONDS),
        "POST_MOTION_SECONDS": str(DEFAULT_POST_MOTION_SECONDS),
        "MIN_CONTOUR_AREA": str(DEFAULT_MIN_CONTOUR_AREA),
        "MIN_MOTION_FRAMES": str(DEFAULT_MIN_MOTION_FRAMES),
        "MOTION_AREA_PERCENT": str(DEFAULT_MOTION_AREA_PERCENT),
        "AUTO_START_MOTION": str(DEFAULT_AUTO_START_MOTION).lower(),
        "SEGMENT_DURATION": str(DEFAULT_SEGMENT_DURATION),
        "EXTEND_MOTION_PERCENT": str(DEFAULT_EXTEND_MOTION_PERCENT),
        "DEBUG_MOTION": str(DEFAULT_DEBUG_MOTION).lower(),
        # ROI параметры
        "ROI_ENABLED": str(DEFAULT_ROI_ENABLED).lower(),
        "ROI_X": str(DEFAULT_ROI_X),
        "ROI_Y": str(DEFAULT_ROI_Y),
        "ROI_WIDTH": str(DEFAULT_ROI_WIDTH),
        "ROI_HEIGHT": str(DEFAULT_ROI_HEIGHT),
        # USB параметры
        "INPUT_SOURCE": DEFAULT_INPUT_SOURCE,
        "USB_DEVICE": DEFAULT_USB_DEVICE,
        "USB_RESOLUTION": DEFAULT_USB_RESOLUTION,
        "USB_FPS": str(DEFAULT_USB_FPS),
    }
    
    config = defaults.copy()
    
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
    if config_path is None:
        # Приоритет: config.macos.env > config.pi.env > config.env
        possible_paths = [
            "config.macos.env",  # macOS конфиг (приоритет)
            "config.pi.env",     # Raspberry Pi конфиг
            "config.env",        # Общий конфиг
            os.path.join(os.path.dirname(__file__), "..", "config.macos.env"),
            os.path.join(os.path.dirname(__file__), "..", "config.pi.env"),
            os.path.join(os.path.dirname(__file__), "..", "config.env"),
            "/app/config.env",
        ]
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        print(f"📋 Loading config from: {config_path}")
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    
    # Загружаем credentials.env (секреты отдельно от конфига)
    creds_paths = [
        "credentials.env",
        os.path.join(os.path.dirname(__file__), "..", "credentials.env"),
    ]
    for creds_path in creds_paths:
        if os.path.exists(creds_path):
            print(f"🔐 Loading credentials from: {creds_path}")
            with open(creds_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # Credentials перезаписывают пустые значения
                        if value and (not config.get(key) or config.get(key) == ""):
                            config[key] = value
            break
    
    # Переменные окружения имеют наивысший приоритет
    for key in list(config.keys()):
        env_value = os.environ.get(key)
        if env_value is not None and env_value != "":
            config[key] = env_value
    
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
    extend_motion_percent = float(config["EXTEND_MOTION_PERCENT"])
    debug_motion = config["DEBUG_MOTION"].lower() == "true"
    auto_start_motion = config["AUTO_START_MOTION"].lower() == "true"
    control_file = config["CONTROL_FILE"]
    segment_duration = int(config.get("SEGMENT_DURATION", "1"))
    
    # ROI параметры
    roi_enabled = config.get("ROI_ENABLED", "false").lower() == "true"
    roi_x = int(config.get("ROI_X", "0"))
    roi_y = int(config.get("ROI_Y", "0"))
    roi_width = int(config.get("ROI_WIDTH", "0"))
    roi_height = int(config.get("ROI_HEIGHT", "0"))
    
    # USB параметры
    input_source = config.get("INPUT_SOURCE", "rtmp").lower()
    usb_device = config.get("USB_DEVICE", "/dev/video0")
    usb_resolution = config.get("USB_RESOLUTION", "1080")
    usb_fps = int(config.get("USB_FPS", "30"))

    # Автоопределение GoPro ОДИН РАЗ (для обоих: FFmpeg и OpenCV)
    # ВАЖНО: используем print(), а не logging.*,
    # чтобы не сломать logging.basicConfig() в setup_logging()
    if (
        input_source == "usb"
        and isinstance(usb_device, str)
        and usb_device.lower() == "auto"
        and platform.system() == "Darwin"
    ):
        detected_index = detect_gopro_macos()
        if detected_index >= 0:
            usb_device = str(detected_index)
            print(
                f"GoPro auto-detected at index "
                f"{detected_index}, using for all"
            )
        else:
            usb_device = "0"
            print(
                "GoPro not found, fallback to index 0"
            )
    
    # Storage Manager параметры
    max_recording_age_days = int(config.get("MAX_RECORDING_AGE_DAYS", "30"))
    min_free_space_gb = float(config.get("MIN_FREE_SPACE_GB", "10.0"))
    auto_cleanup_enabled = config.get("AUTO_CLEANUP_ENABLED", "true").lower() == "true"
    cleanup_interval_hours = int(config.get("CLEANUP_INTERVAL_HOURS", "1"))
    
    # Telegram параметры
    telegram_enabled = config.get("TELEGRAM_ENABLED", "false").lower() == "true"
    telegram_bot_token = config.get("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id = config.get("TELEGRAM_CHAT_ID", "")
    telegram_send_on_motion = config.get(
        "TELEGRAM_SEND_ON_MOTION", "true"
    ).lower() == "true"
    telegram_send_manual = config.get(
        "TELEGRAM_SEND_MANUAL", "false"
    ).lower() == "true"
    telegram_max_video_mb = float(config.get("TELEGRAM_MAX_VIDEO_MB", "45.0"))
    
    detector = MotionDetector(
        rtmp_url=rtmp_url,
        output_dir=output_dir,
        log_file=log_file,
        buffer_seconds=buffer_seconds,
        post_motion_seconds=post_motion_seconds,
        min_contour_area=min_contour_area,
        min_motion_frames=min_motion_frames,
        motion_area_percent=motion_area_percent,
        extend_motion_percent=extend_motion_percent,
        debug_motion=debug_motion,
        segment_duration=segment_duration,
        roi_enabled=roi_enabled,
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        input_source=input_source,
        usb_device=usb_device,
        usb_resolution=usb_resolution,
        usb_fps=usb_fps,
        max_recording_age_days=max_recording_age_days,
        min_free_space_gb=min_free_space_gb,
        auto_cleanup_enabled=auto_cleanup_enabled,
        cleanup_interval_hours=cleanup_interval_hours,
        telegram_enabled=telegram_enabled,
        telegram_bot_token=telegram_bot_token,
        telegram_chat_id=telegram_chat_id,
        telegram_send_on_motion=telegram_send_on_motion,
        telegram_send_manual=telegram_send_manual,
        telegram_max_video_mb=telegram_max_video_mb
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
    
    # Запускаем Telegram bot polling в отдельном потоке (если включен)
    if detector.telegram_notifier:
        def run_telegram_polling():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(detector.telegram_notifier.start_polling())
            except Exception as e:
                detector.logger.error(f"Telegram polling error: {e}")
        
        telegram_thread = Thread(target=run_telegram_polling, daemon=True)
        telegram_thread.start()
        detector.logger.info("Telegram bot polling started in background")
    
    if auto_start_motion:
        detector.enable_motion_detection()
    
    detector.run()


if __name__ == "__main__":
    main()
