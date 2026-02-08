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

# Определяем платформу
if [[ "$(uname -s)" == "Darwin" ]]; then
    PLATFORM="macOS"
else
    PLATFORM="Linux"
fi

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
    
    if [[ "$PLATFORM" == "macOS" ]]; then
        # macOS: используем ffmpeg для списка устройств
        echo "  Используйте ffmpeg для просмотра доступных камер:"
        echo ""
        if command -v ffmpeg &> /dev/null; then
            ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | \
                grep -E "AVFoundation (video|audio) devices:" -A 20 | \
                grep -E "\[AVFoundation" || true
        else
            echo -e "${RED}  ffmpeg не установлен${NC}"
            echo "  Установите: brew install ffmpeg"
        fi
        
        echo ""
        echo "  Камеры обычно имеют индексы:"
        echo "    0 - встроенная камера Mac"
        echo "    1 - первая внешняя камера (GoPro)"
        echo "    2 - вторая внешняя камера"
    else
        # Linux: используем v4l2-ctl
        if [ -d /dev ]; then
            for device in /dev/video*; do
                if [ -e "$device" ]; then
                    echo -n "  $device"
                    # Пробуем получить имя устройства
                    if command -v v4l2-ctl &> /dev/null; then
                        name=$(
                            v4l2-ctl --device="$device" --info 2>/dev/null | \
                            grep "Card type" | cut -d: -f2 | xargs
                        )
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
    fi
    echo ""
}

cmd_check() {
    echo -e "${GREEN}🔍 Проверка подключения GoPro...${NC}"
    echo ""
    
    if [[ "$PLATFORM" == "macOS" ]]; then
        # macOS: проверяем через system_profiler
        echo "Поиск GoPro среди USB устройств..."
        echo ""
        
        if command -v system_profiler &> /dev/null; then
            gopro_found=$(
                system_profiler SPUSBDataType 2>/dev/null | \
                grep -i "gopro" || echo ""
            )
            
            if [ -n "$gopro_found" ]; then
                echo -e "${GREEN}✅ GoPro найдена среди USB устройств${NC}"
                echo ""
                echo "$gopro_found"
            else
                echo -e "${YELLOW}⚠️ GoPro не найдена в списке USB${NC}"
                echo ""
                echo "Проверьте:"
                echo "  1. GoPro подключена через USB-C кабель"
                echo "  2. На GoPro: Настройки → Подключения → USB"
                echo "  3. Выбран режим: GoPro Connect (не MTP!)"
                echo "  4. GoPro включена"
            fi
        fi
        
        echo ""
        echo "Проверка доступности камеры для ffmpeg..."
        if command -v ffmpeg &> /dev/null; then
            echo ""
            ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | \
                grep -E "\[AVFoundation" | head -10 || true
            echo ""
            echo -e "${BLUE}💡 Используйте индекс камеры в USB_DEVICE${NC}"
            echo "   Обычно GoPro это индекс 1 (если 0 - встроенная камера)"
        else
            echo -e "${RED}❌ ffmpeg не установлен${NC}"
            echo "Установите: brew install ffmpeg"
        fi
    else
        # Linux: проверяем /dev/videoX
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
            v4l2-ctl --device="$USB_DEVICE" --info 2>/dev/null | \
                head -20 || true
            
            echo ""
            echo -e "${BLUE}📐 Поддерживаемые форматы:${NC}"
            v4l2-ctl --device="$USB_DEVICE" --list-formats-ext 2>/dev/null | \
                head -30 || true
        fi
        
        echo ""
        echo -e "${GREEN}✅ GoPro готова к работе в USB режиме${NC}"
    fi
}

cmd_test() {
    echo -e "${GREEN}🎬 Тестовый захват кадра...${NC}"
    echo ""
    
    OUTPUT_FILE="$PROJECT_DIR/test_frame.jpg"
    
    # Захватываем один кадр через ffmpeg
    echo "Захват с $USB_DEVICE..."
    
    if command -v ffmpeg &> /dev/null; then
        if [[ "$PLATFORM" == "macOS" ]]; then
            # macOS: используем AVFoundation
            ffmpeg -y -f avfoundation -framerate 30 -video_size 1280x720 \
                -i "$USB_DEVICE" -frames:v 1 "$OUTPUT_FILE" 2>/dev/null
        else
            # Linux: используем V4L2
            if [ ! -e "$USB_DEVICE" ]; then
                echo -e "${RED}❌ Устройство $USB_DEVICE не найдено${NC}"
                return 1
            fi
            
            ffmpeg -y -f v4l2 -input_format mjpeg -video_size 1280x720 \
                -i "$USB_DEVICE" -frames:v 1 "$OUTPUT_FILE" 2>/dev/null
        fi
        
        if [ -f "$OUTPUT_FILE" ]; then
            echo -e "${GREEN}✅ Кадр сохранён: $OUTPUT_FILE${NC}"
            ls -lh "$OUTPUT_FILE"
        else
            echo -e "${RED}❌ Не удалось захватить кадр${NC}"
            echo ""
            echo "Попробуйте:"
            if [[ "$PLATFORM" == "macOS" ]]; then
                echo "  - Изменить USB_DEVICE на другой индекс (0, 1, 2)"
                echo "  - Проверить список камер: ./scripts/gopro-usb.sh list"
            else
                echo "  - Проверить что GoPro подключена"
                echo "  - Попробовать другое устройство: /dev/video1, /dev/video2"
            fi
            return 1
        fi
    else
        echo -e "${YELLOW}⚠️ ffmpeg не установлен${NC}"
        if [[ "$PLATFORM" == "macOS" ]]; then
            echo "Установите: brew install ffmpeg"
        fi
    fi
}

cmd_info() {
    echo -e "${GREEN}📋 Текущие настройки USB режима:${NC}"
    echo ""
    echo "  PLATFORM:        $PLATFORM"
    echo "  INPUT_SOURCE:    ${INPUT_SOURCE:-usb}"
    echo "  USB_DEVICE:      $USB_DEVICE"
    echo "  USB_RESOLUTION:  ${USB_RESOLUTION:-1080}p"
    echo "  USB_FPS:         ${USB_FPS:-30}"
    echo ""
    
    if [[ "$PLATFORM" == "macOS" ]]; then
        echo -e "${BLUE}Инструкция для macOS:${NC}"
        echo ""
        echo "  1. Подключите GoPro через USB-C кабель"
        echo "  2. На GoPro: Настройки → Подключения → USB"
        echo "  3. Выберите: GoPro Connect (не MTP!)"
        echo "  4. В config.macos.env установите USB_DEVICE=1"
        echo "     (0 = встроенная камера, 1 = первая внешняя)"
        echo ""
        echo "  Запуск детектора:"
        echo "    python detector/motion_detector.py"
        echo ""
        echo "  Проверка: ./scripts/gopro-usb.sh check"
        echo "  Список камер: ./scripts/gopro-usb.sh list"
    else
        echo -e "${BLUE}Инструкция для Linux:${NC}"
        echo ""
        echo "  1. Подключите GoPro к компьютеру через USB-C кабель"
        echo "  2. На GoPro откройте: Настройки → Подключения → USB"
        echo "  3. Выберите режим: GoPro Connect (не MTP!)"
        echo "  4. GoPro должна определиться как /dev/videoX"
        echo ""
        echo "  Проверка: ./scripts/gopro-usb.sh check"
    fi
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
