#!/usr/bin/env python3
"""
Storage Manager для GoPro Bird Watcher
Управление дисковым пространством: автоматическая очистка старых записей,
мониторинг свободного места.
"""

import os
import time
import glob
import shutil
import logging
from typing import Optional, List, Tuple
from threading import Thread, Event


class StorageManager:
    """
    Управление хранилищем:
    - Автоматическое удаление записей старше N дней
    - Мониторинг свободного места на диске
    - Предупреждения при заполнении хранилища
    """
    
    def __init__(
        self,
        recordings_dir: str,
        max_age_days: int = 30,
        min_free_gb: float = 10.0,
        cleanup_interval_hours: int = 1,
        logger: logging.Logger = None
    ):
        """
        Args:
            recordings_dir: Путь к директории с записями
            max_age_days: Максимальный возраст записей (дни), 0 = отключено
            min_free_gb: Минимум свободного места (GB) для предупреждения
            cleanup_interval_hours: Интервал проверки (часы)
            logger: Логгер
        """
        self.recordings_dir = recordings_dir
        self.max_age_days = max_age_days
        self.min_free_gb = min_free_gb
        self.cleanup_interval_hours = cleanup_interval_hours
        self.logger = logger or logging.getLogger(__name__)
        
        self.stop_event = Event()
        self.cleanup_thread: Optional[Thread] = None
        
        # Создаем директорию если не существует
        os.makedirs(self.recordings_dir, exist_ok=True)
        
        self.logger.info(
            f"StorageManager initialized: max_age={max_age_days} days, "
            f"min_free={min_free_gb}GB, interval={cleanup_interval_hours}h"
        )
    
    def start(self):
        """Запустить фоновую очистку."""
        if self.cleanup_thread is not None:
            self.logger.warning("StorageManager already running")
            return
        
        self.stop_event.clear()
        self.cleanup_thread = Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        self.logger.info("StorageManager started")
    
    def stop(self):
        """Остановить фоновую очистку."""
        if self.cleanup_thread is None:
            return
        
        self.stop_event.set()
        self.cleanup_thread.join(timeout=5)
        self.cleanup_thread = None
        self.logger.info("StorageManager stopped")
    
    def _cleanup_loop(self):
        """Основной цикл фоновой очистки."""
        # Первая очистка сразу при старте
        self._cleanup_cycle()
        
        # Затем по расписанию
        interval_seconds = self.cleanup_interval_hours * 3600
        
        while not self.stop_event.is_set():
            # Ждем с возможностью прерывания
            if self.stop_event.wait(timeout=interval_seconds):
                break  # Получили сигнал остановки
            
            self._cleanup_cycle()
    
    def _cleanup_cycle(self):
        """Один цикл проверки и очистки."""
        try:
            # 1. Проверяем свободное место
            disk_warning = self.check_free_space()
            if disk_warning:
                self.logger.warning(disk_warning)
            
            # 2. Удаляем старые записи (если включено)
            if self.max_age_days > 0:
                deleted_count, freed_mb = self.cleanup_old_recordings()
                if deleted_count > 0:
                    self.logger.info(
                        f"Cleaned up {deleted_count} old recordings, "
                        f"freed {freed_mb:.1f}MB"
                    )
        
        except Exception as e:
            self.logger.error(f"Error in cleanup cycle: {e}", exc_info=True)
    
    def get_disk_usage(self) -> Tuple[float, float, float, float]:
        """
        Получить использование диска.
        
        Returns:
            (total_gb, used_gb, free_gb, percent_used)
        """
        usage = shutil.disk_usage(self.recordings_dir)
        
        total_gb = usage.total / (1024**3)
        used_gb = usage.used / (1024**3)
        free_gb = usage.free / (1024**3)
        percent_used = (usage.used / usage.total * 100) if usage.total > 0 else 0
        
        return total_gb, used_gb, free_gb, percent_used
    
    def check_free_space(self) -> Optional[str]:
        """
        Проверить свободное место, вернуть предупреждение если мало.
        
        Returns:
            Строка с предупреждением или None
        """
        total_gb, used_gb, free_gb, percent_used = self.get_disk_usage()
        
        # Проверка по абсолютному значению
        if free_gb < self.min_free_gb:
            return (
                f"⚠️ Low disk space: {free_gb:.1f}GB free "
                f"(< {self.min_free_gb}GB threshold)"
            )
        
        # Проверка по процентам (> 90% заполнено)
        if percent_used > 90:
            return (
                f"⚠️ Disk almost full: {percent_used:.1f}% used "
                f"({free_gb:.1f}GB free)"
            )
        
        return None
    
    def get_all_recordings(self) -> List[str]:
        """
        Получить список всех записей (MP4 файлов).
        
        Returns:
            Список путей к файлам
        """
        recordings = []
        
        # Ищем в motion/ и manual/ подпапках
        for subdir in ["motion", "manual"]:
            pattern = os.path.join(self.recordings_dir, subdir, "*.mp4")
            recordings.extend(glob.glob(pattern))
        
        # Также в корне (на случай если структура другая)
        pattern = os.path.join(self.recordings_dir, "*.mp4")
        recordings.extend(glob.glob(pattern))
        
        return recordings
    
    def cleanup_old_recordings(self) -> Tuple[int, float]:
        """
        Удалить записи старше max_age_days.
        
        Returns:
            (количество_удаленных, освобождено_MB)
        """
        if self.max_age_days <= 0:
            return 0, 0.0
        
        cutoff_time = time.time() - (self.max_age_days * 24 * 3600)
        recordings = self.get_all_recordings()
        
        deleted_count = 0
        freed_bytes = 0
        
        for filepath in recordings:
            try:
                # Проверяем время создания файла
                file_time = os.path.getmtime(filepath)
                
                if file_time < cutoff_time:
                    # Файл старше порога - удаляем
                    file_size = os.path.getsize(filepath)
                    os.remove(filepath)
                    
                    deleted_count += 1
                    freed_bytes += file_size
                    
                    # Логируем только если DEBUG уровень
                    age_days = (time.time() - file_time) / (24 * 3600)
                    self.logger.debug(
                        f"Deleted old recording: {os.path.basename(filepath)} "
                        f"(age: {age_days:.1f} days)"
                    )
            
            except FileNotFoundError:
                # Файл уже удален, пропускаем
                pass
            except Exception as e:
                self.logger.error(
                    f"Error deleting {filepath}: {e}",
                    exc_info=True
                )
        
        freed_mb = freed_bytes / (1024**2)
        return deleted_count, freed_mb
    
    def get_recordings_stats(self) -> dict:
        """
        Получить статистику по записям.
        
        Returns:
            Словарь со статистикой
        """
        recordings = self.get_all_recordings()
        
        if not recordings:
            return {
                "count": 0,
                "total_size_mb": 0.0,
                "oldest_age_days": 0.0,
                "newest_age_days": 0.0
            }
        
        total_size = 0
        oldest_time = float('inf')
        newest_time = 0
        
        for filepath in recordings:
            try:
                total_size += os.path.getsize(filepath)
                file_time = os.path.getmtime(filepath)
                oldest_time = min(oldest_time, file_time)
                newest_time = max(newest_time, file_time)
            except Exception:
                pass
        
        current_time = time.time()
        oldest_age_days = (current_time - oldest_time) / (24 * 3600)
        newest_age_days = (current_time - newest_time) / (24 * 3600)
        
        return {
            "count": len(recordings),
            "total_size_mb": total_size / (1024**2),
            "oldest_age_days": oldest_age_days,
            "newest_age_days": newest_age_days
        }
    
    def cleanup_specific_recordings(
        self,
        count: int = None,
        size_mb: float = None
    ) -> Tuple[int, float]:
        """
        Удалить N самых старых записей или записи общим размером size_mb.
        
        Args:
            count: Количество записей для удаления (None = не ограничено)
            size_mb: Целевой размер для освобождения в MB (None = не ограничено)
        
        Returns:
            (количество_удаленных, освобождено_MB)
        """
        recordings = self.get_all_recordings()
        
        if not recordings:
            return 0, 0.0
        
        # Сортируем по времени (старые первыми)
        recordings_with_time = []
        for filepath in recordings:
            try:
                mtime = os.path.getmtime(filepath)
                size = os.path.getsize(filepath)
                recordings_with_time.append((filepath, mtime, size))
            except Exception:
                pass
        
        recordings_with_time.sort(key=lambda x: x[1])  # Сортировка по времени
        
        deleted_count = 0
        freed_bytes = 0
        
        for filepath, mtime, size in recordings_with_time:
            # Проверяем условия остановки
            if count is not None and deleted_count >= count:
                break
            if size_mb is not None and freed_bytes >= (size_mb * 1024**2):
                break
            
            try:
                os.remove(filepath)
                deleted_count += 1
                freed_bytes += size
                
                self.logger.debug(f"Deleted: {os.path.basename(filepath)}")
            except Exception as e:
                self.logger.error(f"Error deleting {filepath}: {e}")
        
        freed_mb = freed_bytes / (1024**2)
        return deleted_count, freed_mb


def main():
    """Тестовый запуск StorageManager."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Пример использования
    manager = StorageManager(
        recordings_dir="./recordings",
        max_age_days=30,
        min_free_gb=10.0,
        cleanup_interval_hours=1
    )
    
    # Проверка диска
    total, used, free, percent = manager.get_disk_usage()
    print(f"Disk: {free:.1f}GB free / {total:.1f}GB total ({percent:.1f}% used)")
    
    # Статистика записей
    stats = manager.get_recordings_stats()
    print(f"Recordings: {stats['count']} files, {stats['total_size_mb']:.1f}MB")
    if stats['count'] > 0:
        print(
            f"  Oldest: {stats['oldest_age_days']:.1f} days, "
            f"Newest: {stats['newest_age_days']:.1f} days"
        )
    
    # Запуск фоновой очистки
    manager.start()
    
    try:
        print("StorageManager running (Ctrl+C to stop)...")
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("\nStopping...")
        manager.stop()


if __name__ == "__main__":
    main()
