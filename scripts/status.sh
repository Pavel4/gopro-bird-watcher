#!/bin/bash
# Статус системы GoPro Bird Watcher

cd "$(dirname "$0")/.."

echo "📊 Статус GoPro Bird Watcher"
echo "=============================="
echo ""

# Запрос статуса от детектора
echo "STATUS" > control/command
sleep 1

# Проверка Docker контейнеров (если запущено на хосте)
if command -v docker &> /dev/null; then
echo "🐳 Docker контейнеры:"
    docker ps --format "   {{.Names}}: {{.Status}}" 2>/dev/null | grep gopro || \
        echo "   Контейнеры не запущены или Docker недоступен"
echo ""
fi

# Проверка записей
echo "📁 Записи движения (motion/):"
MOTION_COUNT=$(ls -1 recordings/motion/*.mp4 2>/dev/null | wc -l | tr -d ' ')
if [ "$MOTION_COUNT" -gt 0 ]; then
    echo "   Всего: $MOTION_COUNT видео"
    echo "   Последние:"
    ls -lt recordings/motion/*.mp4 2>/dev/null | head -3 | \
        awk '{print "   - " $NF}' | sed 's|recordings/motion/||'
else
    echo "   Пока нет записей"
fi
echo ""

echo "📁 Ручные записи (manual/):"
MANUAL_COUNT=$(ls -1 recordings/manual/*.mp4 2>/dev/null | wc -l | tr -d ' ')
if [ "$MANUAL_COUNT" -gt 0 ]; then
    echo "   Всего: $MANUAL_COUNT видео"
    echo "   Последние:"
    ls -lt recordings/manual/*.mp4 2>/dev/null | head -3 | \
        awk '{print "   - " $NF}' | sed 's|recordings/manual/||'
else
    echo "   Пока нет записей"
fi
echo ""

# Логи
echo "📋 Последние записи в логе:"
if [ -f "logs/motion_detector.log" ]; then
    tail -5 logs/motion_detector.log | sed 's/^/   /'
else
    echo "   Лог-файл не найден"
fi
