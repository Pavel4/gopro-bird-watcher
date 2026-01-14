#!/bin/bash
# Запуск dev-контейнера для разработки
# Автоматически поднимает RTMP сервер

cd "$(dirname "$0")/.."

echo "🛠  Подготовка dev-окружения..."

# Создаём директории
mkdir -p recordings logs control

# Останавливаем старые контейнеры если есть
docker-compose --profile dev down 2>/dev/null

# Собираем образ
echo "📦 Сборка образа..."
docker-compose build dev

# Запускаем RTMP сервер и dev-контейнер
echo ""
echo "🚀 Запуск контейнеров..."
docker-compose up -d nginx-rtmp

IP=$(ipconfig getifaddr en0 2>/dev/null || echo "не определён")

echo ""
echo "============================================"
echo "  Dev-окружение готово!"
echo "============================================"
echo ""
echo "📡 RTMP сервер: rtmp://$IP/live"
echo ""
echo "📁 Всё в /app:"
echo "   /app/detector/     - исходники детектора"
echo "   /app/recordings/   - записи"
echo "   /app/logs/         - логи"
echo "   /app/scripts/      - скрипты"
echo ""
echo "🔧 Команды внутри контейнера:"
echo "   python detector/motion_detector.py"
echo "   echo START > control/command"
echo "   echo STOP > control/command"
echo ""
echo "============================================"
echo ""

# Запускаем интерактивный контейнер
exec docker-compose --profile dev run --rm dev
