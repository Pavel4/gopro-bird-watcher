#!/bin/bash
# Интерактивный выбор области кормушки (ROI)
#
# Использование:
#   ./scripts/select-roi.sh                # USB auto (GoPro)
#   ./scripts/select-roi.sh --usb 0        # USB по индексу
#   ./scripts/select-roi.sh --image FILE   # Из изображения
#   ./scripts/select-roi.sh --rtmp URL     # С RTMP потока
#
# После выбора координаты автоматически сохраняются
# в config.env (платформо-зависимый).
# Перезапустите детектор для применения.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Инструмент выбора области кормушки (ROI)${NC}"
echo ""

# Определяем режим: native или docker
USE_DOCKER=false

# Если передан --docker — используем Docker
for arg in "$@"; do
    if [ "$arg" = "--docker" ]; then
        USE_DOCKER=true
        break
    fi
done

# ========== NATIVE MODE ==========
if [ "$USE_DOCKER" = false ]; then
    echo "Режим: нативный (без Docker)"
    echo ""

    # Проверяем Python
    PYTHON_CMD=""
    if [ -d "$PROJECT_DIR/venv" ]; then
        PYTHON_CMD="$PROJECT_DIR/venv/bin/python"
        echo -e "${GREEN}Виртуальное окружение: venv${NC}"
    elif command -v python3 &>/dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &>/dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}Python не найден!${NC}"
        echo "Установите Python 3.8+ или создайте venv:"
        echo "  python3 -m venv venv"
        echo "  source venv/bin/activate"
        echo "  pip install -r detector/requirements.txt"
        exit 1
    fi

    # Проверяем opencv
    if ! "$PYTHON_CMD" -c "import cv2" 2>/dev/null; then
        echo -e "${YELLOW}OpenCV не установлен${NC}"
        echo "Установите: pip install opencv-python"
        exit 1
    fi

    # Фильтруем --docker из аргументов
    ARGS=()
    for arg in "$@"; do
        if [ "$arg" != "--docker" ]; then
            ARGS+=("$arg")
        fi
    done

    # Если нет аргументов — по умолчанию USB auto
    if [ ${#ARGS[@]} -eq 0 ]; then
        echo "Источник: USB auto (автоопределение GoPro)"
        ARGS=("--usb" "auto")
    fi

    echo "Запуск интерактивного выбора..."
    echo ""

    "$PYTHON_CMD" "$PROJECT_DIR/detector/select_roi.py" \
        "${ARGS[@]}"

    echo ""
    echo -e "${GREEN}Готово!${NC}"
    echo ""
    echo "Перезапустите детектор для применения:"
    echo "   ./run-native.sh"
    exit 0
fi

# ========== DOCKER MODE ==========
echo "Режим: Docker"
echo ""

# Проверяем что контейнер запущен
if ! docker ps --format '{{.Names}}' \
    | grep -q "gopro-detector"; then
    echo -e "${YELLOW}Контейнер gopro-detector не запущен${NC}"
    echo "   Запустите: docker-compose up -d"
    exit 1
fi

# Проверяем наличие X11 дисплея
if [ -z "$DISPLAY" ]; then
    echo -e "${YELLOW}Переменная DISPLAY не установлена${NC}"
    echo "   Для интерактивного выбора нужен X11 дисплей"
    echo ""
    echo "   Альтернативы:"
    echo "   1. Сохраните кадр:"
    echo "      docker exec gopro-detector python \\"
    echo "        detector/select_roi.py \\"
    echo "        --save-frame /app/frame.jpg --no-save"
    echo "   2. Откройте frame.jpg и определите координаты"
    echo "   3. Запишите в config.env:"
    echo "      ROI_ENABLED=true"
    echo "      ROI_X=100"
    echo "      ROI_Y=50"
    echo "      ROI_WIDTH=640"
    echo "      ROI_HEIGHT=480"
    exit 1
fi

# Фильтруем --docker из аргументов
ARGS=()
for arg in "$@"; do
    if [ "$arg" != "--docker" ]; then
        ARGS+=("$arg")
    fi
done

echo "Запуск интерактивного выбора..."
echo ""

# Для X11 forwarding в контейнере
docker exec -it \
    -e DISPLAY="$DISPLAY" \
    gopro-detector \
    python detector/select_roi.py "${ARGS[@]}"

echo ""
echo -e "${GREEN}Готово!${NC}"
echo ""
echo "Перезапустите детектор для применения:"
echo "   docker-compose restart detector"
