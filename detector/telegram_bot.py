#!/usr/bin/env python3
"""
Telegram Bot для GoPro Bird Watcher
Автоматическая отправка видео при обнаружении птиц, команды управления.
"""

import os
import csv
import json
import shutil
import asyncio
import logging
import subprocess
from typing import Optional, Dict
from datetime import datetime, timezone, timedelta

# Московское время (UTC+3)
MOSCOW_TZ = timezone(timedelta(hours=3))

# Заголовки CSV для репортов
REPORTS_HEADERS = [
    "timestamp",
    "visit_id",
    "original_species",
    "corrected_species",
    "confirmed",
    "video_file",
]

try:
    from aiogram import Bot, Dispatcher, F
    from aiogram import types
    from aiogram.filters import Command
    from aiogram.types import (
        FSInputFile,
        InlineKeyboardMarkup,
        InlineKeyboardButton,
        CallbackQuery,
    )
    AIOGRAM_AVAILABLE = True
except ImportError:
    AIOGRAM_AVAILABLE = False
    Bot = None
    Dispatcher = None
    types = None
    InlineKeyboardMarkup = None
    InlineKeyboardButton = None
    CallbackQuery = None


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
        crops_dir: str = "./crops",
        analytics_dir: str = "./analytics",
        logger: logging.Logger = None
    ):
        """
        Args:
            bot_token: Токен бота от @BotFather
            chat_id: ID чата куда отправлять
            send_on_motion: Отправлять при движении
            send_manual: Отправлять при ручной записи
            max_video_mb: Макс. размер видео (MB)
            recordings_dir: Путь к папке записей
            crops_dir: Путь к папке кропов птиц
            analytics_dir: Путь к папке аналитики
            logger: Логгер
        """
        if not AIOGRAM_AVAILABLE:
            raise ImportError(
                "aiogram not installed. "
                "Install: pip install aiogram==3.24.0"
            )

        self.bot_token = bot_token
        self.chat_id = chat_id
        self.send_on_motion = send_on_motion
        self.send_manual = send_manual
        self.max_video_mb = max_video_mb
        self.recordings_dir = (
            recordings_dir or "/app/recordings"
        )
        self.crops_dir = crops_dir
        self.analytics_dir = analytics_dir
        self.logger = (
            logger or logging.getLogger(__name__)
        )

        # Создаем бота и диспетчер
        self.bot = Bot(token=bot_token)
        self.dp = Dispatcher()

        # Регистрируем обработчики команд
        self._register_handlers()

        # Для передачи статистики от детектора
        self.detector_stats = {}

        # Ссылка на аналитику (из MotionDetector)
        self.analytics = None

        # Репорты: ожидающие обратной связи
        # {visit_id: {species_en, species_ru, ...}}
        self._pending_reports: Dict[int, dict] = {}

        # Загружаем метки видов для inline-кнопок
        self._species_labels = (
            self._load_species_labels()
        )

        # CSV для репортов
        self._reports_csv = os.path.join(
            self.analytics_dir, "species_reports.csv"
        )
        self._init_reports_csv()

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

        # Callback-хэндлеры для inline-кнопок
        self.dp.callback_query.register(
            self.on_correct_callback,
            F.data.startswith("correct:"),
        )
        self.dp.callback_query.register(
            self.on_wrong_callback,
            F.data.startswith("wrong:"),
        )
        self.dp.callback_query.register(
            self.on_fix_callback,
            F.data.startswith("fix:"),
        )
        self.dp.callback_query.register(
            self.on_behavior_callback,
            F.data.startswith("bhv:"),
        )

    # === Вспомогательные методы ===

    def _load_species_labels(self) -> dict:
        """Загрузить метки видов из JSON."""
        labels_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "species_labels.json",
        )
        try:
            with open(labels_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(
                f"Cannot load species labels: {e}"
            )
            return {}

    def _init_reports_csv(self):
        """Инициализировать CSV для репортов."""
        os.makedirs(self.analytics_dir, exist_ok=True)
        if not os.path.exists(self._reports_csv):
            try:
                with open(
                    self._reports_csv, "w",
                    newline="", encoding="utf-8",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(REPORTS_HEADERS)
            except Exception as e:
                self.logger.error(
                    f"Cannot create reports CSV: {e}"
                )

    # === Inline-клавиатуры ===

    def build_species_keyboard(
        self, visit_id: int
    ) -> InlineKeyboardMarkup:
        """
        Inline-клавиатура: Верно / Неверно.

        Args:
            visit_id: ID визита для callback_data
        Returns:
            InlineKeyboardMarkup
        """
        return InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(
                        text="✅ Верно",
                        callback_data=(
                            f"correct:{visit_id}"
                        ),
                    ),
                    InlineKeyboardButton(
                        text="❌ Неверно",
                        callback_data=(
                            f"wrong:{visit_id}"
                        ),
                    ),
                ]
            ]
        )

    def _build_correction_keyboard(
        self, visit_id: int
    ) -> InlineKeyboardMarkup:
        """
        Inline-клавиатура со списком видов
        для исправления.

        Args:
            visit_id: ID визита
        Returns:
            InlineKeyboardMarkup с 10 видами
        """
        rows = []
        row = []
        for sid, info in sorted(
            self._species_labels.items(),
            key=lambda x: int(x[0]),
        ):
            btn = InlineKeyboardButton(
                text=info["ru"],
                callback_data=(
                    f"fix:{visit_id}:{sid}"
                ),
            )
            row.append(btn)
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)

        # Кнопка "Не птица"
        rows.append([
            InlineKeyboardButton(
                text="🚫 Не птица",
                callback_data=(
                    f"fix:{visit_id}:none"
                ),
            ),
        ])
        return InlineKeyboardMarkup(
            inline_keyboard=rows
        )

    def build_behavior_keyboard(
        self, visit_id: int
    ) -> InlineKeyboardMarkup:
        """
        Inline-клавиатура для разметки поведения.
        6 кнопок поведения для сбора данных.

        Args:
            visit_id: ID визита для callback_data
        Returns:
            InlineKeyboardMarkup
        """
        behaviors = [
            ("🍽 Кормление", "feeding"),
            ("🪹 Сидение", "perching"),
            ("👀 Озирание", "alert"),
            ("⚔️ Драка", "fighting"),
            ("➡️ Прилёт", "arrival"),
            ("⬅️ Улёт", "departure"),
        ]
        rows = []
        row = []
        for text, bhv_id in behaviors:
            row.append(
                InlineKeyboardButton(
                    text=text,
                    callback_data=(
                        f"bhv:{visit_id}:{bhv_id}"
                    ),
                )
            )
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        return InlineKeyboardMarkup(
            inline_keyboard=rows
        )

    def register_pending_report(
        self,
        visit_id: int,
        species_en: str,
        species_ru: str,
        confidence: float,
        video_file: str = "",
        behavior_en: str = "",
    ):
        """
        Зарегистрировать ожидающий репорт.
        Вызывается из MotionDetector при отправке
        видео с определённым видом.
        """
        self._pending_reports[visit_id] = {
            "species_en": species_en,
            "species_ru": species_ru,
            "confidence": confidence,
            "video_file": video_file,
            "behavior_en": behavior_en,
            "date": datetime.now(
                MOSCOW_TZ
            ).strftime("%Y-%m-%d"),
        }
        # Лимит: храним не более 100 записей
        if len(self._pending_reports) > 100:
            oldest = min(
                self._pending_reports.keys()
            )
            del self._pending_reports[oldest]

    # === Callback-хэндлеры ===

    async def on_correct_callback(
        self, callback: CallbackQuery
    ):
        """Пользователь подтвердил вид: Верно."""
        try:
            visit_id = int(
                callback.data.split(":")[1]
            )
            info = self._pending_reports.pop(
                visit_id, None
            )
            if not info:
                await callback.answer(
                    "⏳ Репорт устарел"
                )
                return

            self._save_species_report(
                visit_id=visit_id,
                original_species=info["species_en"],
                corrected_species="",
                confirmed=True,
                video_file=info.get(
                    "video_file", ""
                ),
            )

            await callback.answer(
                "✅ Спасибо! Отмечено как верное."
            )
            # Убираем кнопки, обновляем caption
            if callback.message:
                old_caption = (
                    callback.message.caption or ""
                )
                new_caption = (
                    old_caption
                    + "\n\n✅ Вид подтверждён"
                )
                try:
                    await callback.message.edit_caption(
                        caption=new_caption[:1024],
                        parse_mode="HTML",
                        reply_markup=None,
                    )
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(
                f"on_correct_callback error: {e}",
                exc_info=True,
            )
            await callback.answer("❌ Ошибка")

    async def on_wrong_callback(
        self, callback: CallbackQuery
    ):
        """Пользователь нажал Неверно — показать виды."""
        try:
            visit_id = int(
                callback.data.split(":")[1]
            )
            info = self._pending_reports.get(
                visit_id
            )
            if not info:
                await callback.answer(
                    "⏳ Репорт устарел"
                )
                return

            keyboard = (
                self._build_correction_keyboard(
                    visit_id
                )
            )
            await callback.answer()

            if callback.message:
                old_caption = (
                    callback.message.caption or ""
                )
                new_caption = (
                    old_caption
                    + "\n\n❓ Выберите правильный вид:"
                )
                try:
                    await callback.message.edit_caption(
                        caption=new_caption[:1024],
                        parse_mode="HTML",
                        reply_markup=keyboard,
                    )
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(
                f"on_wrong_callback error: {e}",
                exc_info=True,
            )
            await callback.answer("❌ Ошибка")

    async def on_fix_callback(
        self, callback: CallbackQuery
    ):
        """
        Пользователь выбрал правильный вид
        из списка.
        """
        try:
            parts = callback.data.split(":")
            visit_id = int(parts[1])
            species_id = parts[2]

            info = self._pending_reports.pop(
                visit_id, None
            )
            if not info:
                await callback.answer(
                    "⏳ Репорт устарел"
                )
                return

            # Определяем правильный вид
            if species_id == "none":
                corrected_en = "not_a_bird"
                corrected_ru = "Не птица"
            else:
                label = self._species_labels.get(
                    species_id, {}
                )
                corrected_en = label.get(
                    "en", f"unknown_{species_id}"
                )
                corrected_ru = label.get(
                    "ru", corrected_en
                )

            self._save_species_report(
                visit_id=visit_id,
                original_species=info["species_en"],
                corrected_species=corrected_en,
                confirmed=False,
                video_file=info.get(
                    "video_file", ""
                ),
            )

            # Копируем кроп для переобучения
            if corrected_en != "not_a_bird":
                self._copy_crop_for_retraining(
                    visit_id=visit_id,
                    correct_species_en=corrected_en,
                    date_str=info.get("date", ""),
                )

            await callback.answer(
                f"✅ Исправлено на: {corrected_ru}"
            )

            if callback.message:
                old_caption = (
                    callback.message.caption or ""
                )
                # Убираем "Выберите правильный вид"
                old_caption = old_caption.replace(
                    "\n\n❓ Выберите правильный вид:",
                    "",
                )
                new_caption = (
                    old_caption
                    + f"\n\n🔄 Исправлено: "
                    f"{corrected_ru}"
                )
                try:
                    await callback.message.edit_caption(
                        caption=new_caption[:1024],
                        parse_mode="HTML",
                        reply_markup=None,
                    )
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(
                f"on_fix_callback error: {e}",
                exc_info=True,
            )
            await callback.answer("❌ Ошибка")

    async def on_behavior_callback(
        self, callback: CallbackQuery
    ):
        """
        Пользователь выбрал поведение птицы.
        Сохраняет данные для обучения модели.
        """
        try:
            parts = callback.data.split(":")
            if len(parts) < 3:
                await callback.answer(
                    "❌ Ошибка формата"
                )
                return

            visit_id = int(parts[1])
            behavior_en = parts[2]

            behavior_names = {
                "feeding": "Кормление",
                "perching": "Сидение",
                "alert": "Озирание",
                "fighting": "Драка",
                "arrival": "Прилёт",
                "departure": "Улёт",
            }
            behavior_ru = behavior_names.get(
                behavior_en, behavior_en
            )

            # Сохраняем в CSV для обучения
            self._save_behavior_report(
                visit_id=visit_id,
                behavior_en=behavior_en,
            )

            await callback.answer(
                f"🎭 Поведение: {behavior_ru}"
            )

            if callback.message:
                old_caption = (
                    callback.message.caption or ""
                )
                new_caption = (
                    old_caption
                    + f"\n\n🎭 Размечено: "
                    f"{behavior_ru}"
                )
                try:
                    await (
                        callback.message
                        .edit_caption(
                            caption=(
                                new_caption[:1024]
                            ),
                            parse_mode="HTML",
                            reply_markup=None,
                        )
                    )
                except Exception:
                    pass

        except Exception as e:
            self.logger.error(
                f"on_behavior_callback "
                f"error: {e}",
                exc_info=True,
            )
            await callback.answer("❌ Ошибка")

    def _save_behavior_report(
        self,
        visit_id: int,
        behavior_en: str,
    ):
        """
        Сохранить разметку поведения в CSV.
        Файл: analytics/behavior_reports.csv
        """
        behavior_csv = os.path.join(
            self.analytics_dir,
            "behavior_reports.csv",
        )
        headers = [
            "timestamp",
            "visit_id",
            "behavior",
        ]
        # Создаём файл если нет
        if not os.path.exists(behavior_csv):
            try:
                with open(
                    behavior_csv, "w",
                    newline="", encoding="utf-8",
                ) as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
            except Exception as e:
                self.logger.error(
                    f"Cannot create behavior "
                    f"CSV: {e}"
                )
                return
        # Записываем
        try:
            timestamp = datetime.now(
                MOSCOW_TZ
            ).isoformat()
            row = [
                timestamp,
                visit_id,
                behavior_en,
            ]
            with open(
                behavior_csv, "a",
                newline="", encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(row)
            self.logger.info(
                f"Behavior report saved: "
                f"visit={visit_id}, "
                f"behavior={behavior_en}"
            )
        except Exception as e:
            self.logger.error(
                f"Error saving behavior "
                f"report: {e}"
            )

    # === Сохранение репортов ===

    def _save_species_report(
        self,
        visit_id: int,
        original_species: str,
        corrected_species: str,
        confirmed: bool,
        video_file: str = "",
    ):
        """Сохранить репорт в CSV."""
        try:
            timestamp = datetime.now(
                MOSCOW_TZ
            ).strftime("%Y-%m-%d %H:%M:%S")
            row = [
                timestamp,
                visit_id,
                original_species,
                corrected_species,
                str(confirmed).lower(),
                video_file,
            ]
            with open(
                self._reports_csv, "a",
                newline="", encoding="utf-8",
            ) as f:
                writer = csv.writer(f)
                writer.writerow(row)
            self.logger.info(
                f"Species report saved: "
                f"visit={visit_id}, "
                f"confirmed={confirmed}, "
                f"corrected={corrected_species}"
            )
        except Exception as e:
            self.logger.error(
                f"Error saving report: {e}",
                exc_info=True,
            )

    def _copy_crop_for_retraining(
        self,
        visit_id: int,
        correct_species_en: str,
        date_str: str = "",
    ):
        """
        Скопировать кроп птицы в директорию
        data/corrections/{species_en}/ для
        переобучения.
        """
        try:
            if not date_str:
                date_str = datetime.now(
                    MOSCOW_TZ
                ).strftime("%Y-%m-%d")

            prefix = f"visit_{visit_id:04d}"
            crop_date_dir = os.path.join(
                self.crops_dir, date_str,
            )

            if not os.path.isdir(crop_date_dir):
                self.logger.warning(
                    f"Crop dir not found: "
                    f"{crop_date_dir}"
                )
                return

            # Ищем кропы данного визита
            crop_files = [
                f for f in os.listdir(crop_date_dir)
                if f.startswith(prefix)
                and "bird" in f
            ]

            if not crop_files:
                self.logger.warning(
                    f"No crops for {prefix} in "
                    f"{crop_date_dir}"
                )
                return

            # Папка назначения
            corrections_dir = os.path.join(
                os.path.dirname(
                    os.path.dirname(__file__)
                ),
                "data",
                "corrections",
                correct_species_en.lower().replace(
                    " ", "_"
                ),
            )
            os.makedirs(corrections_dir, exist_ok=True)

            for fname in crop_files:
                src = os.path.join(
                    crop_date_dir, fname
                )
                dst = os.path.join(
                    corrections_dir, fname
                )
                shutil.copy2(src, dst)
                self.logger.info(
                    f"Crop copied: {fname} -> "
                    f"{corrections_dir}"
                )

        except Exception as e:
            self.logger.error(
                f"Error copying crop: {e}",
                exc_info=True,
            )

    # === Команды бота ===

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
            "/stats week - За неделю\n"
            "/stats food - Сравнение корма\n"
            "/stats hours - По часам (7 дней)\n"
            "/food - Текущий корм\n"
            "/food семечки - Задать тип корма\n"
            "/species - Статистика по видам\n"
            "/help - Эта справка\n\n"
            "<b>Автоматические уведомления:</b>\n"
            "• Видео при обнаружении птицы\n"
            "• Сжатие если > 50MB\n\n"
            "<b>Обратная связь по ML:</b>\n"
            "Под каждым видео с определением "
            "вида есть кнопки:\n"
            "✅ Верно — подтвердить вид\n"
            "❌ Неверно — выбрать правильный вид\n"
            "Ваши исправления помогают улучшить "
            "модель!"
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
