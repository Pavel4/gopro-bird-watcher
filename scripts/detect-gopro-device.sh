#!/bin/bash
# Автоматическое определение индекса GoPro Webcam на macOS

detect_gopro_macos() {
    # Получаем список всех видеоустройств
    devices=$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1)
    
    # Ищем GoPro Webcam и извлекаем индекс
    gopro_index=$(echo "$devices" | grep -i "gopro" | grep -oE '\[[0-9]+\]' | head -1 | tr -d '[]')
    
    if [ -n "$gopro_index" ]; then
        echo "$gopro_index"
        return 0
    else
        return 1
    fi
}

# Основная логика
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    gopro_index=$(detect_gopro_macos)
    
    if [ $? -eq 0 ]; then
        echo "✅ GoPro найдена: индекс $gopro_index"
        echo "$gopro_index"
        exit 0
    else
        echo "❌ GoPro не найдена" >&2
        echo "Доступные устройства:" >&2
        ffmpeg -f avfoundation -list_devices true -i "" 2>&1 | grep -A 10 "video devices" >&2
        exit 1
    fi
else
    # Linux - ищем /dev/video*
    for device in /dev/video*; do
        if [ -e "$device" ]; then
            # Проверяем название устройства
            device_name=$(v4l2-ctl --device="$device" --info 2>/dev/null | grep "Card type" | cut -d: -f2 | xargs)
            
            if echo "$device_name" | grep -qi "gopro"; then
                echo "✅ GoPro найдена: $device"
                echo "$device"
                exit 0
            fi
        fi
    done
    
    echo "❌ GoPro не найдена" >&2
    echo "Доступные устройства:" >&2
    ls -la /dev/video* 2>&1 >&2
    exit 1
fi
