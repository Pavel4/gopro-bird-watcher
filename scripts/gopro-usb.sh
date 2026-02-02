#!/bin/bash
# Скрипт для работы с GoPro в USB режиме
#
# Использование:
#   ./scripts/gopro-usb.sh check    - проверить подключение GoPro
#   ./scripts/gopro-usb.sh list     - список видеоустройств
#   ./scripts/gopro-usb.sh test     - тестовый захват кадра
#   ./scripts/gopro-usb.sh info     - информация о настройках

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Загружаем конфиг
USB_DEVICE="${USB_DEVICE:-/dev/video0}"
USB_RESOLUTION="${USB_RESOLUTION:-1080}"

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  GoPro USB Webcam Mode${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

cmd_list() {
    echo -e "${GREEN}📹 Список видеоустройств:${NC}"
    echo ""
    
    if [ -d /dev ]; then
        for device in /dev/video*; do
            if [ -e "$device" ]; then
                echo -n "  $device"
                # Пробуем получить имя устройства
                if command -v v4l2-ctl &> /dev/null; then
                    name=$(v4l2-ctl --device="$device" --info 2>/dev/null | grep "Card type" | cut -d: -f2 | xargs)
                    if [ -n "$name" ]; then
                        echo " - $name"
                    else
                        echo ""
                    fi
                else
                    echo ""
                fi
            fi
        done
    else
        echo -e "${RED}  /dev не доступен${NC}"
    fi
    echo ""
}

cmd_check() {
    echo -e "${GREEN}🔍 Проверка подключения GoPro...${NC}"
    echo ""
    
    # Проверяем наличие устройства
    if [ ! -e "$USB_DEVICE" ]; then
        echo -e "${RED}❌ Устройство $USB_DEVICE не найдено${NC}"
        echo ""
        echo "Убедитесь что:"
        echo "  1. GoPro подключена по USB-C кабелю"
        echo "  2. На GoPro: Настройки → Подключения → USB → GoPro Connect"
        echo "  3. GoPro включена"
        echo ""
        cmd_list
        return 1
    fi
    
    echo -e "${GREEN}✅ Устройство $USB_DEVICE найдено${NC}"
    
    # Проверяем доступность через v4l2
    if command -v v4l2-ctl &> /dev/null; then
        echo ""
        echo -e "${BLUE}📋 Информация об устройстве:${NC}"
        v4l2-ctl --device="$USB_DEVICE" --info 2>/dev/null | head -20 || true
        
        echo ""
        echo -e "${BLUE}📐 Поддерживаемые форматы:${NC}"
        v4l2-ctl --device="$USB_DEVICE" --list-formats-ext 2>/dev/null | head -30 || true
    fi
    
    echo ""
    echo -e "${GREEN}✅ GoPro готова к работе в USB режиме${NC}"
}

cmd_test() {
    echo -e "${GREEN}🎬 Тестовый захват кадра...${NC}"
    echo ""
    
    if [ ! -e "$USB_DEVICE" ]; then
        echo -e "${RED}❌ Устройство $USB_DEVICE не найдено${NC}"
        return 1
    fi
    
    OUTPUT_FILE="$PROJECT_DIR/test_frame.jpg"
    
    # Захватываем один кадр через ffmpeg
    echo "Захват с $USB_DEVICE..."
    
    if command -v ffmpeg &> /dev/null; then
        ffmpeg -y -f v4l2 -input_format mjpeg -video_size 1280x720 \
            -i "$USB_DEVICE" -frames:v 1 "$OUTPUT_FILE" 2>/dev/null
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo -e "${GREEN}✅ Кадр сохранён: $OUTPUT_FILE${NC}"
            ls -la "$OUTPUT_FILE"
        else
            echo -e "${RED}❌ Не удалось захватить кадр${NC}"
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️ ffmpeg не установлен, пропускаем тест${NC}"
    fi
}

cmd_info() {
    echo -e "${GREEN}📋 Текущие настройки USB режима:${NC}"
    echo ""
    echo "  INPUT_SOURCE:    ${INPUT_SOURCE:-usb}"
    echo "  USB_DEVICE:      ${USB_DEVICE:-/dev/video0}"
    echo "  USB_RESOLUTION:  ${USB_RESOLUTION:-1080}p"
    echo "  USB_FPS:         ${USB_FPS:-30}"
    echo ""
    echo -e "${BLUE}Инструкция по настройке GoPro:${NC}"
    echo ""
    echo "  1. Подключите GoPro к компьютеру через USB-C кабель"
    echo "  2. На GoPro откройте: Настройки → Подключения → USB"
    echo "  3. Выберите режим: GoPro Connect (не MTP!)"
    echo "  4. GoPro должна определиться как /dev/videoX"
    echo ""
    echo "  Проверка: ./scripts/gopro-usb.sh check"
}

print_usage() {
    echo "Использование: $0 <команда>"
    echo ""
    echo "Команды:"
    echo "  check   - проверить подключение GoPro"
    echo "  list    - список видеоустройств"
    echo "  test    - тестовый захват кадра"
    echo "  info    - информация о настройках"
    echo ""
}

# Main
print_header

case "${1:-info}" in
    check)
        cmd_check
        ;;
    list)
        cmd_list
        ;;
    test)
        cmd_test
        ;;
    info)
        cmd_info
        ;;
    -h|--help|help)
        print_usage
        ;;
    *)
        echo -e "${RED}Неизвестная команда: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
