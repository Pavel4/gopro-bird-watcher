"""
Тесты для detector/telegram_bot.py

Тестируем: TelegramNotifier — наличие команд,
обработчики, интеграция с analytics.
"""

import os
import asyncio
from unittest.mock import (
    MagicMock, AsyncMock, patch,
)

import pytest

from telegram_bot import (
    TelegramNotifier,
    AIOGRAM_AVAILABLE,
)


# Пропускаем если aiogram не установлен
pytestmark = pytest.mark.skipif(
    not AIOGRAM_AVAILABLE,
    reason="aiogram not installed",
)


@pytest.fixture
def notifier():
    """
    Экземпляр TelegramNotifier с фейковым
    токеном.
    """
    bot = TelegramNotifier(
        bot_token="123456:ABC-DEF",
        chat_id="12345",
    )
    return bot


class TestTelegramCommands:
    def test_cmd_species_method_exists(
        self, notifier,
    ):
        """Метод cmd_species существует."""
        assert hasattr(notifier, "cmd_species")
        assert callable(notifier.cmd_species)

    def test_all_handlers_registered(
        self, notifier,
    ):
        """Все ожидаемые команды зарегистрированы."""
        expected = [
            "cmd_start",
            "cmd_help",
            "cmd_status",
            "cmd_latest",
            "cmd_food",
            "cmd_stats",
            "cmd_species",
        ]
        for cmd in expected:
            assert hasattr(notifier, cmd), (
                f"Handler {cmd} не найден"
            )

    @pytest.mark.asyncio
    async def test_cmd_species_no_analytics(
        self, notifier,
    ):
        """
        /species без analytics отвечает
        'аналитика отключена'.
        """
        notifier.analytics = None

        message = AsyncMock()
        message.text = "/species"

        await notifier.cmd_species(message)

        message.answer.assert_called_once()
        call_text = (
            message.answer.call_args[0][0]
        )
        assert "отключена" in call_text

    @pytest.mark.asyncio
    async def test_cmd_species_with_analytics(
        self, notifier,
    ):
        """
        /species с analytics вызывает
        format_species_stats.
        """
        mock_analytics = MagicMock()
        mock_analytics.format_species_stats \
            .return_value = (
                "<b>Статистика</b>\n"
                "Синица: 5 визитов"
            )
        notifier.analytics = mock_analytics

        message = AsyncMock()
        message.text = "/species"

        await notifier.cmd_species(message)

        mock_analytics \
            .format_species_stats \
            .assert_called_once()
        message.answer.assert_called_once()

    @pytest.mark.asyncio
    async def test_cmd_help_includes_species(
        self, notifier,
    ):
        """
        /help содержит упоминание /species.
        """
        message = AsyncMock()
        message.text = "/help"

        await notifier.cmd_help(message)

        message.answer.assert_called_once()
        call_text = (
            message.answer.call_args[0][0]
        )
        assert "/species" in call_text
