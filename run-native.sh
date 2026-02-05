#!/bin/bash
# Запуск детектора движения в нативной среде (без Docker)
# Автоматически определяет платформу и использует нужный конфиг

set -e

cd "$(dirname "$0")"

# Цвета для вывода
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   🎥 Запуск детектора движения GoPro (Native)${NC}"
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
mkdir -p recordings/motion recordings/manual logs control

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

echo -e "${GREEN}✅ Все проверки пройдены${NC}"
echo ""
echo "Для остановки нажмите Ctrl+C"
echo ""
echo -e "${BLUE}════════════════════════════════════════════════${NC}"
echo ""

# Запуск детектора
exec python detector/motion_detector.py
