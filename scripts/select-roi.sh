#!/bin/bash
# Интерактивный выбор области кормушки (ROI)
#
# Использование:
#   ./scripts/select-roi.sh              # Захват с RTMP потока
#   ./scripts/select-roi.sh --image FILE # Использовать существующее изображение
#
# После выбора координаты сохраняются в config.env
# Перезапустите детектор для применения: docker-compose restart detector

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🎯 Инструмент выбора области кормушки (ROI)${NC}"
echo ""

# Проверяем что контейнер запущен
if ! docker ps --format '{{.Names}}' | grep -q "gopro-detector"; then
    echo -e "${YELLOW}⚠️  Контейнер gopro-detector не запущен${NC}"
    echo "   Запустите: docker-compose up -d"
    exit 1
fi

# Проверяем наличие X11 дисплея
if [ -z "$DISPLAY" ]; then
    echo -e "${YELLOW}⚠️  Переменная DISPLAY не установлена${NC}"
    echo "   Для интерактивного выбора нужен X11 дисплей"
    echo ""
    echo "   Альтернативы:"
    echo "   1. Сохраните кадр: docker exec gopro-detector python detector/select_roi.py --save-frame /app/frame.jpg --no-save"
    echo "   2. Откройте изображение в редакторе и определите координаты вручную"
    echo "   3. Запишите координаты в config.env:"
    echo "      ROI_ENABLED=true"
    echo "      ROI_X=100"
    echo "      ROI_Y=50"
    echo "      ROI_WIDTH=640"
    echo "      ROI_HEIGHT=480"
    exit 1
fi

# Передаём аргументы в Python скрипт
echo "Запуск интерактивного выбора..."
echo ""

# Для X11 forwarding в контейнере
docker exec -it \
    -e DISPLAY="$DISPLAY" \
    gopro-detector \
    python detector/select_roi.py "$@"

echo ""
echo -e "${GREEN}✅ Готово!${NC}"
echo ""
echo "Если ROI был сохранён, перезапустите детектор:"
echo "   docker-compose restart detector"
echo ""
echo "Или запустите вручную внутри контейнера:"
echo "   docker exec -it gopro-detector python detector/motion_detector.py"
