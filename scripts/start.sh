#!/bin/bash
# Запуск всей системы GoPro Bird Watcher

cd "$(dirname "$0")/.."

echo "🚀 Запуск GoPro Bird Watcher..."
echo ""

# Создаём необходимые директории
mkdir -p recordings control

# Пересобираем и запускаем контейнеры
docker-compose up -d --build

echo ""
echo "✅ Система запущена!"
echo ""
echo "📡 RTMP URL для GoPro: rtmp://$(ipconfig getifaddr en0 2>/dev/null)/live"
echo ""
echo "📺 Просмотр потока:"
echo "   VLC: vlc rtmp://localhost/live"
echo "   Или: open -a VLC rtmp://localhost/live"
echo ""
echo "🎬 Управление записью:"
echo "   Включить:  ./scripts/start-recording.sh"
echo "   Выключить: ./scripts/stop-recording.sh"
echo "   Статус:    ./scripts/status.sh"
echo ""
echo "📊 Статистика RTMP: http://localhost:8080/stat"
echo "📁 Записи: ./recordings/"
