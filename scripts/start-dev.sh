#!/bin/bash
# Запуск контейнеров для разработки (daemon mode)
# Поднимает RTMP сервер и контейнер детектора

cd "$(dirname "$0")/.."

echo "🛠  Подготовка окружения..."

# Создаём директории
mkdir -p recordings/motion recordings/manual logs control

# Останавливаем старые контейнеры если есть
docker-compose down 2>/dev/null

# Собираем образ
echo "📦 Сборка образа..."
docker-compose build detector

# Запускаем контейнеры в фоне
echo ""
echo "🚀 Запуск контейнеров..."
docker-compose up -d

# Даём секунду на запуск
sleep 1

# Проверяем что контейнеры запустились
if docker ps | grep -q gopro-detector; then
    DETECTOR_STATUS="✅ Запущен"
else
    DETECTOR_STATUS="❌ Ошибка запуска"
fi

if docker ps | grep -q gopro-rtmp-server; then
    RTMP_STATUS="✅ Запущен"
else
    RTMP_STATUS="❌ Ошибка запуска"
fi

IP=$(ipconfig getifaddr en0 2>/dev/null || echo "не определён")

echo ""
echo "============================================"
echo "  Окружение готово!"
echo "============================================"
echo ""
echo "📦 Контейнеры:"
echo "   gopro-detector      - $DETECTOR_STATUS"
echo "   gopro-rtmp-server   - $RTMP_STATUS"
echo ""
echo "📡 RTMP сервер: rtmp://$IP/live"
echo "   (внутри контейнера: rtmp://nginx-rtmp/live)"
echo ""
echo "📁 Всё в /app:"
echo "   /app/detector/     - исходники детектора"
echo "   /app/recordings/   - записи"
echo "   /app/logs/         - логи"
echo "   /app/scripts/      - скрипты"
echo ""
echo "🔧 Подключение к контейнеру:"
echo "   docker exec -it gopro-detector bash"
echo ""
echo "🔧 Команды внутри контейнера:"
echo "   python detector/motion_detector.py"
echo "   echo START > control/command"
echo "   echo STOP > control/command"
echo ""
echo "🔧 Остановка:"
echo "   docker-compose down"
echo ""
echo "============================================"
echo ""
echo "💡 Подключаюсь к контейнеру..."
echo ""

# Автоматически подключаемся к контейнеру
exec docker exec -it gopro-detector bash
