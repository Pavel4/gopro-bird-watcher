#!/usr/bin/env python3
"""
Telegram Bot для GoPro Bird Watcher
Автоматическая отправка видео при обнаружении птиц, команды управления.
"""

import os
import asyncio
import logging
import subprocess
from typing import Optional
from datetime import datetime

try:
    from aiogram import Bot, Dispatcher, F
    from aiogram import types
    from aiogram.filters import Command
    from aiogram.types import FSInputFile
    AIOGRAM_AVAILABLE = True
except ImportError:
    AIOGRAM_AVAILABLE = False
    Bot = None
    Dispatcher = None
    types = None


class TelegramNotifier:
    """
    Telegram бот для уведомлений и отправки видео.
    Работает в асинхронном режиме через aiogram 3.x.
    """
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        send_on_motion: bool = True,
        send_manual: bool = False,
        max_video_mb: float = 45.0,
        recordings_dir: str = None,
        logger: logging.Logger = None
    ):
        """
        Args:
            bot_token: Токен бота от @BotFather
            chat_id: ID чата куда отправлять уведомления
            send_on_motion: Отправлять видео при движении
            send_manual: Отправлять видео при ручной записи
            max_video_mb: Макс. размер видео (MB)
            recordings_dir: Путь к папке записей
            logger: Логгер
        """
        if not AIOGRAM_AVAILABLE:
            raise ImportError(
                "aiogram not installed. Install: pip install aiogram==3.24.0"
            )
        
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.send_on_motion = send_on_motion
        self.send_manual = send_manual
        self.max_video_mb = max_video_mb
        self.recordings_dir = (
            recordings_dir or "/app/recordings"
        )
        self.logger = logger or logging.getLogger(__name__)
        
        # Создаем бота и диспетчер
        self.bot = Bot(token=bot_token)
        self.dp = Dispatcher()
        
        # Регистрируем обработчики команд
        self._register_handlers()
        
        # Для передачи статистики от детектора
        self.detector_stats = {}
        
        # Ссылка на аналитику (устанавливается из MotionDetector)
        self.analytics = None
        
        self.logger.info(
            f"TelegramNotifier initialized "
            f"for chat {chat_id}"
        )
    
    def _register_handlers(self):
        """Регистрация обработчиков команд бота."""
        # Команды
        self.dp.message.register(
            self.cmd_start, Command("start")
        )
        self.dp.message.register(
            self.cmd_help, Command("help")
        )
        self.dp.message.register(
            self.cmd_status, Command("status")
        )
        self.dp.message.register(
            self.cmd_latest, Command("latest")
        )
        self.dp.message.register(
            self.cmd_food, Command("food")
        )
        self.dp.message.register(
            self.cmd_stats, Command("stats")
        )
        self.dp.message.register(
            self.cmd_species, Command("species")
        )
    
    async def cmd_start(self, message: types.Message):
        """Команда /start."""
        welcome_text = (
            "🐦 <b>GoPro Bird Watcher Bot</b>\n\n"
            "Я буду присылать видео при обнаружении "
            "птиц на кормушке.\n\n"
            "Доступные команды:\n"
            "/status - Статус системы\n"
            "/latest - Последние записи\n"
            "/stats - Статистика визитов\n"
            "/food - Управление типом корма\n"
            "/species - Статистика по видам птиц\n"
            "/help - Справка"
        )
        await message.answer(
            welcome_text, parse_mode="HTML"
        )
    
    async def cmd_help(self, message: types.Message):
        """Команда /help."""
        help_text = (
            "<b>Команды бота:</b>\n\n"
            "/start - Приветствие\n"
            "/status - Статус системы\n"
            "/latest - Последние 5 записей\n"
            "/stats - Статистика за сегодня\n"
            "/stats week - Статистика за неделю\n"
            "/stats food - Сравнение корма\n"
            "/stats hours - По часам (7 дней)\n"
            "/food - Текущий корм\n"
            "/food семечки - Задать тип корма\n"
            "/species - Статистика по видам\n"
            "/help - Эта справка\n\n"
            "<b>Автоматические уведомления:</b>\n"
            "• Видео при обнаружении птицы\n"
            "• Сжатие если > 50MB"
        )
        await message.answer(
            help_text, parse_mode="HTML"
        )
    
    async def cmd_status(self, message: types.Message):
        """Команда /status - статус системы."""
        try:
            # Получаем статистику (должна передаваться из детектора)
            stats = self.detector_stats
            
            status_text = "<b>📊 Статус системы</b>\n\n"
            
            if stats:
                status_text += f"🎬 Записей (motion): {stats.get('motion_videos_saved', 0)}\n"
                status_text += f"🎥 Записей (manual): {stats.get('manual_videos_saved', 0)}\n"
                status_text += f"📹 Кадров обработано: {stats.get('frames_processed', 0)}\n"
                status_text += f"🔍 Событий движения: {stats.get('motion_events', 0)}\n"
                
                last_motion = stats.get('last_motion')
                if last_motion:
                    status_text += f"⏱ Последнее движение: {last_motion}\n"
            else:
                status_text += "ℹ️ Статистика недоступна\n"
            
            # Проверяем свободное место
            try:
                import shutil
                usage = shutil.disk_usage(
                    self.recordings_dir
                )
                free_gb = usage.free / (1024**3)
                total_gb = usage.total / (1024**3)
                percent_used = (usage.used / usage.total * 100)
                
                status_text += f"\n💾 Диск: {free_gb:.1f}GB / {total_gb:.1f}GB "
                status_text += f"({percent_used:.1f}% использовано)"
                
                if free_gb < 10:
                    status_text += "\n⚠️ Мало места на диске!"
            except Exception:
                pass
            
            await message.answer(status_text, parse_mode="HTML")
        
        except Exception as e:
            self.logger.error(f"Error in cmd_status: {e}", exc_info=True)
            await message.answer("❌ Ошибка при получении статуса")
    
    async def cmd_latest(self, message: types.Message):
        """Команда /latest - последние записи."""
        try:
            import glob
            
            # Ищем последние 5 записей
            recordings = []
            rec_dir = self.recordings_dir
            for subdir in ["motion", "manual"]:
                pattern = os.path.join(
                    rec_dir, subdir, "*.mp4"
                )
                recordings.extend(glob.glob(pattern))
            
            if not recordings:
                await message.answer("📭 Пока нет записей")
                return
            
            # Сортируем по времени (новые первые)
            recordings.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            recordings = recordings[:5]
            
            response_text = f"<b>📹 Последние {len(recordings)} записей:</b>\n\n"
            
            for i, filepath in enumerate(recordings, 1):
                filename = os.path.basename(filepath)
                size_mb = os.path.getsize(filepath) / (1024**2)
                mtime = os.path.getmtime(filepath)
                timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                
                response_text += f"{i}. {filename}\n"
                response_text += f"   Размер: {size_mb:.1f}MB, {timestamp}\n\n"
            
            await message.answer(response_text, parse_mode="HTML")
        
        except Exception as e:
            self.logger.error(f"Error in cmd_latest: {e}", exc_info=True)
            await message.answer("❌ Ошибка при получении списка")
    
    async def cmd_food(self, message: types.Message):
        """
        Команда /food — управление типом корма.
        /food — показать текущий корм
        /food <тип> — задать новый тип корма
        """
        try:
            if not self.analytics:
                await message.answer(
                    "ℹ️ Аналитика отключена.\n"
                    "Включите ANALYTICS_ENABLED=true"
                )
                return

            # Парсим аргумент после /food
            text = message.text or ""
            parts = text.strip().split(maxsplit=1)

            if len(parts) < 2:
                # Просто /food — показать текущий
                current = self.analytics.get_food_type()
                await message.answer(
                    f"🥜 Текущий корм: "
                    f"<b>{current}</b>\n\n"
                    f"Чтобы изменить:\n"
                    f"/food семечки\n"
                    f"/food сало\n"
                    f"/food орехи\n"
                    f"/food смешанный",
                    parse_mode="HTML",
                )
                return

            # /food <тип> — задать новый
            new_food = parts[1].strip()
            self.analytics.set_food_type(
                new_food, source="telegram"
            )
            await message.answer(
                f"✅ Корм изменён на: "
                f"<b>{new_food}</b>",
                parse_mode="HTML",
            )

        except Exception as e:
            self.logger.error(
                f"Error in cmd_food: {e}",
                exc_info=True,
            )
            await message.answer(
                "❌ Ошибка при обработке команды"
            )

    async def cmd_stats(self, message: types.Message):
        """
        Команда /stats — статистика визитов.
        /stats — за сегодня
        /stats week — за неделю
        /stats food — сравнение корма
        /stats hours — распределение по часам
        """
        try:
            if not self.analytics:
                await message.answer(
                    "ℹ️ Аналитика отключена.\n"
                    "Включите ANALYTICS_ENABLED=true"
                )
                return

            # Парсим субкоманду
            text = message.text or ""
            parts = text.strip().split()
            subcmd = (
                parts[1].lower()
                if len(parts) > 1 else "day"
            )

            if subcmd == "week":
                msg = (
                    self.analytics.format_weekly_stats()
                )
            elif subcmd == "food":
                msg = (
                    self.analytics
                    .format_food_comparison()
                )
            elif subcmd == "hours":
                msg = (
                    self.analytics
                    .format_hourly_stats(days=7)
                )
            else:
                # /stats или /stats day
                msg = (
                    self.analytics
                    .format_daily_stats()
                )

            await message.answer(
                msg, parse_mode="HTML"
            )

        except Exception as e:
            self.logger.error(
                f"Error in cmd_stats: {e}",
                exc_info=True,
            )
            await message.answer(
                "❌ Ошибка при получении статистики"
            )

    async def cmd_species(
        self, message: types.Message,
    ):
        """Команда /species — статистика по видам."""
        try:
            if not self.analytics:
                await message.answer(
                    "ℹ️ Аналитика отключена.\n"
                    "Включите ANALYTICS_ENABLED=true"
                )
                return

            msg = (
                self.analytics
                .format_species_stats()
            )
            await message.answer(
                msg, parse_mode="HTML"
            )
        except Exception as e:
            self.logger.error(
                f"Error in cmd_species: {e}",
                exc_info=True,
            )
            await message.answer(
                "❌ Ошибка при получении "
                "статистики по видам"
            )

    async def send_video(
        self,
        video_path: str,
        caption: str = None,
        compress_if_needed: bool = True
    ) -> bool:
        """
        Отправить видео в Telegram.
        
        Args:
            video_path: Путь к видео файлу
            caption: Подпись к видео
            compress_if_needed: Сжимать если файл > max_video_mb
        
        Returns:
            True если отправлено успешно
        """
        try:
            if not os.path.exists(video_path):
                self.logger.error(f"Video file not found: {video_path}")
                return False
            
            # Проверяем размер файла
            size_mb = os.path.getsize(video_path) / (1024**2)
            final_path = video_path
            compressed = False
            
            if size_mb > self.max_video_mb and compress_if_needed:
                self.logger.info(
                    f"Video {size_mb:.1f}MB > {self.max_video_mb}MB, compressing..."
                )
                compressed_path = await self._compress_video(video_path)
                
                if compressed_path and os.path.exists(compressed_path):
                    final_path = compressed_path
                    compressed = True
                    new_size_mb = os.path.getsize(final_path) / (1024**2)
                    self.logger.info(
                        f"Compressed: {size_mb:.1f}MB → {new_size_mb:.1f}MB"
                    )
                else:
                    self.logger.warning("Compression failed, sending original")
            
            # Подготавливаем caption
            final_caption = caption or ""
            if compressed:
                final_caption += "\n\n🗜 Сжато для Telegram"
            
            # Отправляем видео
            video_file = FSInputFile(final_path)
            await self.bot.send_video(
                chat_id=self.chat_id,
                video=video_file,
                caption=final_caption[:1024] if final_caption else None,
                supports_streaming=True
            )
            
            self.logger.info(f"Video sent to Telegram: {os.path.basename(video_path)}")
            
            # Удаляем сжатую версию если создавали
            if compressed and final_path != video_path:
                try:
                    os.remove(final_path)
                except Exception:
                    pass
            
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to send video: {e}", exc_info=True)
            return False
    
    async def _compress_video(self, input_path: str) -> Optional[str]:
        """
        Сжать видео для Telegram (макс 50MB).
        
        Args:
            input_path: Путь к исходному видео
        
        Returns:
            Путь к сжатому видео или None если ошибка
        """
        output_path = input_path.replace(".mp4", "_compressed.mp4")
        
        try:
            # FFmpeg команда для сжатия
            # CRF 28 = более высокое сжатие, но все еще хорошее качество
            cmd = [
                "ffmpeg",
                "-y",  # Перезаписать если существует
                "-i", input_path,
                "-c:v", "libx264",
                "-crf", "28",  # Константа качества (выше = меньше размер)
                "-preset", "fast",
                "-c:a", "aac",
                "-b:a", "96k",  # Битрейт аудио
                "-movflags", "+faststart",  # Оптимизация для потокового
                output_path
            ]
            
            # Запускаем FFmpeg
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await result.communicate()
            
            if result.returncode == 0 and os.path.exists(output_path):
                return output_path
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                self.logger.error(f"FFmpeg compression failed: {error_msg[-500:]}")
                return None
        
        except Exception as e:
            self.logger.error(f"Error compressing video: {e}", exc_info=True)
            return None
    
    async def send_message(self, text: str, parse_mode: str = None) -> bool:
        """
        Отправить текстовое сообщение.
        
        Args:
            text: Текст сообщения
            parse_mode: "HTML" или "Markdown"
        
        Returns:
            True если отправлено успешно
        """
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=parse_mode
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send message: {e}", exc_info=True)
            return False
    
    async def start_polling(self):
        """Запустить polling для получения команд от пользователя."""
        try:
            self.logger.info("Starting Telegram bot polling...")
            # handle_signals=False - чтобы работать в фоновом потоке
            await self.dp.start_polling(self.bot, handle_signals=False)
        except Exception as e:
            self.logger.error(f"Error in bot polling: {e}", exc_info=True)
    
    async def close(self):
        """Закрыть соединение с Telegram."""
        try:
            await self.bot.session.close()
        except Exception:
            pass


def main():
    """Тестовый запуск бота."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Проверяем переменные окружения
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    
    if not bot_token or not chat_id:
        print("❌ Error: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set")
        print("\nExample:")
        print("  export TELEGRAM_BOT_TOKEN='123456:ABC-DEF...'")
        print("  export TELEGRAM_CHAT_ID='123456789'")
        return
    
    # Создаем бота
    notifier = TelegramNotifier(
        bot_token=bot_token,
        chat_id=chat_id,
        send_on_motion=True,
        max_video_mb=45.0
    )
    
    # Запускаем polling
    print(f"🤖 Telegram bot started for chat {chat_id}")
    print("Send /start to the bot to test it")
    print("Press Ctrl+C to stop")
    
    try:
        asyncio.run(notifier.start_polling())
    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
