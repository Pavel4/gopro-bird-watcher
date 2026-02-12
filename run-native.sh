#!/bin/bash
# Запуск детектора движения в нативной среде (без Docker)
# Автоматически определяет платформу и использует нужный конфиг
#
# Флаги:
#   --role ROLE    Роль устройства: edge, compute, standalone (default)
#   --full-frame   Детектировать на всём кадре (игнорировать ROI)
#   --no-crop      Не обрезать видео (игнорировать CROP)

set -e

cd "$(dirname "$0")"

# Парсинг аргументов
OVERRIDE_ROI=""
OVERRIDE_CROP=""
CROP_ARGS=""       # --crop X,Y,W,H
CROP_PAD_ARG=""    # --crop-pad N
CROP_SCALE_ARG=""  # --crop-scale WxH
SHOW_HELP=""
DEVICE_ROLE=""     # edge, compute, standalone

while [[ $# -gt 0 ]]; do
    case "$1" in
        --role)
            shift
            DEVICE_ROLE="${1:-}"
            ;;
        --full-frame|--no-roi)
            OVERRIDE_ROI="true"
            ;;
        --no-crop)
            OVERRIDE_CROP="true"
            ;;
        --crop)
            shift
            CROP_ARGS="${1:-}"
            ;;
        --crop-pad)
            shift
            CROP_PAD_ARG="${1:-}"
            ;;
        --crop-scale)
            shift
            CROP_SCALE_ARG="${1:-}"
            ;;
        --help|-h)
            SHOW_HELP="true"
            ;;
    esac
    shift
done

if [ -n "$SHOW_HELP" ]; then
    echo "Использование: ./run-native.sh [флаги]"
    echo ""
    echo "Флаги:"
    echo "  --role ROLE        Роль устройства:"
    echo "      standalone     Всё локально (default)"
    echo "      edge           Камера + детекция, ML через gRPC"
    echo "      compute        gRPC-сервер ML-инференса"
    echo "  --full-frame       Детекция на всём кадре (игнорировать ROI)"
    echo "  --no-crop          Не обрезать видео"
    echo "  --crop X,Y,W,H    Обрезка: абсолютные координаты"
    echo "                     Пример: --crop 200,50,800,600"
    echo "  --crop-pad N       Обрезка: отступ N пикселей от ROI"
    echo "                     Центрируется на ROI, расширяется на N px"
    echo "                     Пример: --crop-pad 150"
    echo "  --crop-scale WxH   Масштаб после обрезки"
    echo "                     Пример: --crop-scale 1280x720"
    echo ""
    echo "Примеры:"
    echo "  ./run-native.sh                            # standalone"
    echo "  ./run-native.sh --role edge                # edge (Pi)"
    echo "  ./run-native.sh --role compute             # ML server"
    echo "  ./run-native.sh --crop-pad 100             # ROI + 100px"
    echo "  ./run-native.sh --crop 200,50,800,600      # точные координаты"
    echo "  ./run-native.sh --crop-pad 150 --crop-scale 1280x720"
    echo "  ./run-native.sh --full-frame               # весь кадр"
    exit 0
fi

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════${NC}"
if [[ "$DEVICE_ROLE" == "compute" ]]; then
    echo -e "${BLUE}   🧠 Bird Watcher Inference Server${NC}"
else
    echo -e "${BLUE}   🎥 Запуск детектора движения GoPro (Native)${NC}"
fi
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""

# Определяем платформу
if [[ "$OSTYPE" == "darwin"* ]]; then
    CONFIG_FILE="config.macos.env"
    PLATFORM="macOS"
    PLATFORM_EMOJI="🍎"
elif [[ -f "/etc/rpi-issue" ]] || grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null; then
    CONFIG_FILE="config.pi.env"
    PLATFORM="Raspberry Pi"
    PLATFORM_EMOJI="🥧"
else
    CONFIG_FILE="config.env"
    PLATFORM="Linux"
    PLATFORM_EMOJI="🐧"
fi

# Для роли compute — переопределяем конфиг
if [[ "$DEVICE_ROLE" == "compute" ]]; then
    CONFIG_FILE="config.compute.env"
fi

echo -e "${GREEN}$PLATFORM_EMOJI Платформа: $PLATFORM${NC}"
echo -e "${GREEN}📝 Конфигурация: $CONFIG_FILE${NC}"
echo ""

# Проверка конфигурационного файла
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Конфигурационный файл не найден: $CONFIG_FILE${NC}"
    echo ""
    echo "Создайте конфигурацию на основе примера:"
    if [[ "$PLATFORM" == "macOS" ]]; then
        echo "  cp config.env config.macos.env"
        echo "  nano config.macos.env"
        echo ""
        echo "ВАЖНО для macOS:"
        echo "  USB_DEVICE=1  # Используйте ИНДЕКС камеры, а не путь!"
    else
        echo "  cp config.env.example config.env"
        echo "  nano config.env"
    fi
    exit 1
fi

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Виртуальное окружение не найдено${NC}"
    echo ""
    echo "Создание виртуального окружения..."
    python3 -m venv venv
    
    echo "Установка зависимостей..."
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r detector/requirements.txt
    
    echo -e "${GREEN}✅ Виртуальное окружение создано${NC}"
    echo ""
else
    echo -e "${GREEN}✅ Виртуальное окружение найдено${NC}"
fi

# Активация виртуального окружения
echo -e "${BLUE}🔧 Активация виртуального окружения...${NC}"
source venv/bin/activate

# Проверка Python зависимостей
echo -e "${BLUE}🔍 Проверка зависимостей...${NC}"
if ! python -c "import cv2" 2>/dev/null; then
    echo -e "${RED}❌ OpenCV не установлен${NC}"
    echo "Установка зависимостей..."
    pip install -r detector/requirements.txt
fi

# Загрузка credentials (если есть)
if [ -f "credentials.env" ]; then
    echo -e "${BLUE}🔐 Загрузка credentials.env...${NC}"
    export $(cat "credentials.env" | grep -v '^#' | grep -v '^$' | xargs)
else
    echo -e "${YELLOW}⚠️  credentials.env не найден${NC}"
    echo "   Создайте: cp credentials.env.example credentials.env"
fi

# Загрузка конфигурации (не перезаписывает уже заданные переменные)
echo -e "${BLUE}⚙️  Загрузка конфигурации...${NC}"
while IFS='=' read -r key value; do
    # Пропускаем комментарии и пустые строки
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    key=$(echo "$key" | xargs)
    # Не перезаписываем уже заданные переменные (из credentials.env)
    if [ -z "${!key}" ]; then
        export "$key=$value"
    fi
done < "$CONFIG_FILE"

# Создание необходимых директорий
echo -e "${BLUE}📁 Создание директорий...${NC}"
if [[ "$DEVICE_ROLE" == "compute" ]]; then
    mkdir -p models logs
else
    mkdir -p recordings/motion recordings/manual logs control
fi

# Проверки для edge/standalone (не compute)
if [[ "$DEVICE_ROLE" != "compute" ]]; then

# Проверка GoPro подключения (только для информации)
echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${YELLOW}📹 Проверка GoPro подключения${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"

if [[ "$PLATFORM" == "macOS" ]]; then
    echo "Платформа: macOS"
    echo "USB устройство: индекс $USB_DEVICE"
    echo ""
    echo "Для проверки доступных камер:"
    echo "  ffmpeg -f avfoundation -list_devices true -i \"\""
    echo ""
    echo "Убедитесь что:"
    echo "  1. GoPro подключена через USB-C"
    echo "  2. На GoPro: Настройки → USB → GoPro Connect"
    echo "  3. Камера включена"
else
    echo "Платформа: Linux"
    echo "USB устройство: $USB_DEVICE"
    
    if [ -e "$USB_DEVICE" ]; then
        echo -e "${GREEN}✅ Устройство найдено: $USB_DEVICE${NC}"
        
        # Дополнительная информация если доступен v4l2-ctl
        if command -v v4l2-ctl &> /dev/null; then
            echo ""
            echo "Информация о устройстве:"
            v4l2-ctl --device="$USB_DEVICE" --info 2>/dev/null | head -5 || true
        fi
    else
        echo -e "${YELLOW}⚠️  Устройство не найдено: $USB_DEVICE${NC}"
        echo ""
        echo "Доступные видеоустройства:"
        ls -la /dev/video* 2>/dev/null || echo "  Нет видеоустройств"
        echo ""
        echo "Убедитесь что:"
        echo "  1. GoPro подключена через USB-C"
        echo "  2. На GoPro: Настройки → USB → GoPro Connect"
        echo "  3. Устройство определилось в системе"
    fi
fi

echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${GREEN}🚀 Запуск детектора движения...${NC}"
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""

# Показываем основные настройки
echo "Настройки:"
echo "  Источник: $INPUT_SOURCE"
if [[ "$INPUT_SOURCE" == "usb" ]]; then
    echo "  USB устройство: $USB_DEVICE"
    echo "  Разрешение: ${USB_RESOLUTION}p @ ${USB_FPS}fps"
else
    echo "  RTMP URL: $RTMP_URL"
fi
echo "  Записи: $OUTPUT_DIR"
echo "  Логи: $LOG_FILE"
echo "  Автостарт: ${AUTO_START_MOTION:-false}"
echo ""

# Проверка FFmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}❌ FFmpeg не установлен!${NC}"
    echo ""
    if [[ "$PLATFORM" == "macOS" ]]; then
        echo "Установите FFmpeg:"
        echo "  brew install ffmpeg"
    else
        echo "Установите FFmpeg:"
        echo "  sudo apt install ffmpeg"
    fi
    exit 1
fi

fi  # конец проверок для edge/standalone

echo -e "${GREEN}✅ Все проверки пройдены${NC}"

# Применяем флаги командной строки
if [ -n "$OVERRIDE_ROI" ]; then
    export ROI_ENABLED=false
    echo -e "${YELLOW}🔲 --full-frame: ROI отключён, детекция на всём кадре${NC}"
fi
if [ -n "$OVERRIDE_CROP" ]; then
    export CROP_VIDEO_ENABLED=false
    echo -e "${YELLOW}🔲 --no-crop: обрезка видео отключена${NC}"
fi
if [ -n "$CROP_ARGS" ]; then
    # --crop X,Y,W,H — абсолютные координаты
    IFS=',' read -r CX CY CW CH <<< "$CROP_ARGS"
    export CROP_VIDEO_ENABLED=true
    export CROP_X="$CX"
    export CROP_Y="$CY"
    export CROP_WIDTH="$CW"
    export CROP_HEIGHT="$CH"
    echo -e "${GREEN}🔲 --crop: ${CW}x${CH} at (${CX},${CY})${NC}"
fi
if [ -n "$CROP_PAD_ARG" ]; then
    # --crop-pad N — отступ от ROI (центрирование)
    export CROP_VIDEO_ENABLED=true
    export CROP_PAD="$CROP_PAD_ARG"
    echo -e "${GREEN}🔲 --crop-pad: ${CROP_PAD_ARG}px вокруг ROI${NC}"
fi
if [ -n "$CROP_SCALE_ARG" ]; then
    export CROP_VIDEO_ENABLED=true
    export CROP_SCALE="$CROP_SCALE_ARG"
    echo -e "${GREEN}🔲 --crop-scale: ${CROP_SCALE_ARG}${NC}"
fi

# Применяем роль устройства
if [[ "$DEVICE_ROLE" == "edge" ]]; then
    export INFERENCE_MODE=remote
    echo -e "${GREEN}🔗 --role edge: ML инференс через gRPC (remote)${NC}"
    echo -e "${GREEN}   Сервер: ${INFERENCE_SERVER_HOST:-localhost}:${INFERENCE_SERVER_PORT:-50051}${NC}"
fi

echo ""
echo "Для остановки нажмите Ctrl+C"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""

# Запуск в зависимости от роли
if [[ "$DEVICE_ROLE" == "compute" ]]; then
    echo -e "${GREEN}🧠 Запуск Inference Server (compute)...${NC}"
    exec python detector/inference_server.py
else
    # edge или standalone — запускаем детектор
    exec python detector/motion_detector.py
fi
