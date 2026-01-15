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
from threading import Thread, Event, Lock
from enum import Enum
import logging

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
    
    # DEBUG уровень для отладки движения
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


class SegmentRecorder:
    """
    Записывает RTMP поток короткими сегментами через FFmpeg.
    Сегменты именуются с timestamp для точного выбора по времени.
    """
    
    def __init__(
        self,
        rtmp_url: str,
        segments_dir: str,
        segment_duration: int = 1,
        max_segments: int = 180,
        logger: logging.Logger = None
    ):
        self.rtmp_url = rtmp_url
        self.segments_dir = segments_dir
        self.segment_duration = segment_duration
        self.max_segments = max_segments
        self.logger = logger or logging.getLogger(__name__)
        
        self.ffmpeg_process = None
        self.is_running = False
        self.stop_event = Event()
        self.lock = Lock()
        
        # Флаг для приостановки cleanup во время записи
        self.cleanup_paused = False
        
        # Очищаем и создаём папку сегментов
        self._clean_segments_dir()
        
        self.logger.info(f"SegmentRecorder initialized: {segments_dir}")
    
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
        
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "warning",
            "-i", self.rtmp_url,
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
        """
        if end_time is None:
            end_time = time.time()
        
        with self.lock:
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
    
    def merge_segments(self, segments: list, output_path: str) -> bool:
        """Объединить сегменты в один файл."""
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
        
        # Создаём временный файл со списком сегментов
        list_file = output_path + ".concat.txt"
        
        try:
            with open(list_file, 'w') as f:
                for seg in valid_segments:
                    # Абсолютный путь для надёжности
                    abs_path = os.path.abspath(seg)
                    escaped_path = abs_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")
            
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
                timeout=120
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
        segment_duration: int = DEFAULT_SEGMENT_DURATION
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
            max_segments=300,  # ~5 минут буфера
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
        
        self.logger.info(f"Motion detector initialized (with audio support)")
        self.logger.info(f"  Output dirs: motion={self.motion_dir}, manual={self.manual_dir}")
        self.logger.info(f"  Buffer: {buffer_seconds}s before, {post_motion_seconds}s after")
        self.logger.info(f"  Segment duration: {segment_duration}s")
        self.logger.info(
            f"  Motion thresholds: start={motion_area_percent}%, "
            f"extend={extend_motion_percent}%"
        )
        if debug_motion:
            self.logger.info(f"  DEBUG MODE: motion % will be logged")
    
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
        """Детекция движения в кадре."""
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
    
    def _check_segments_fresh(self, max_age: float = 5.0) -> bool:
        """Проверить что сегменты свежие (FFmpeg работает)."""
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
        
        # Объединяем сегменты
        self.logger.info(f"Merging {len(segments)} segments...")
        
        if self.video_merger.merge_segments(segments, temp_filepath):
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
                except Exception as e:
                    self.logger.error(f"Failed to rename: {e}")
                    # Файл существует - просто используем temp имя
                    if os.path.exists(temp_filepath):
                        self.logger.info(f"■ {type_str} saved: {prefix}_{timestamp}_temp.mp4")
            else:
                self.logger.warning("Could not get duration")
                self.logger.info(f"■ {type_str} saved: {prefix}_{timestamp}_temp.mp4")
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
    }
    
    config = defaults.copy()
    
    for key in config:
        env_value = os.environ.get(key)
        if env_value is not None:
            config[key] = env_value
    
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
    extend_motion_percent = float(config["EXTEND_MOTION_PERCENT"])
    debug_motion = config["DEBUG_MOTION"].lower() == "true"
    auto_start_motion = config["AUTO_START_MOTION"].lower() == "true"
    control_file = config["CONTROL_FILE"]
    segment_duration = int(config.get("SEGMENT_DURATION", "1"))
    
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
