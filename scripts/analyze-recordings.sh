#!/bin/bash
# Анализ сохранённых записей с движением
# Показывает статистику и детали видеофайлов

set -e

cd "$(dirname "$0")/.."

# Цвета
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  Анализ записей детектора движения${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Статистика по директориям
echo -e "${CYAN}📊 Общая статистика:${NC}"
echo ""

MOTION_COUNT=$(find recordings/motion -name "*.mp4" -type f 2>/dev/null | wc -l)
MANUAL_COUNT=$(find recordings/manual -name "*.mp4" -type f 2>/dev/null | wc -l)
TOTAL_SIZE=$(du -sh recordings/motion 2>/dev/null | awk '{print $1}')

echo "  Автоматические записи (движение): $MOTION_COUNT видео"
echo "  Ручные записи: $MANUAL_COUNT видео"
echo "  Общий размер (motion): $TOTAL_SIZE"
echo ""

# Анализ последних записей
echo -e "${CYAN}🎥 Последние 10 записей с движением:${NC}"
echo ""
printf "%-45s %10s %15s %10s\n" "Файл" "Размер" "Разрешение" "Длина"
echo "--------------------------------------------------------------------------------"

find recordings/motion -name "*.mp4" -type f | sort -r | head -10 | while read file; do
    filename=$(basename "$file")
    size=$(ls -lh "$file" | awk '{print $5}')
    
    # Получаем параметры видео через ffprobe
    if command -v ffprobe &> /dev/null; then
        resolution=$(
            ffprobe -v error -select_streams v:0 \
            -show_entries stream=width,height \
            -of csv=s=x:p=0 "$file" 2>/dev/null
        )
        duration=$(
            ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null | \
            awk '{printf "%.0fs", $1}'
        )
    else
        resolution="N/A"
        duration="N/A"
    fi
    
    printf "%-45s %10s %15s %10s\n" \
        "${filename:0:44}" "$size" "$resolution" "$duration"
done

echo ""
echo -e "${CYAN}📅 Распределение по датам:${NC}"
echo ""

# Группируем по датам (из имени файла bird_2026-01-16_...)
find recordings/motion -name "bird_*.mp4" -type f | \
    grep -oP '\d{4}-\d{2}-\d{2}' | sort | uniq -c | \
    awk '{printf "  %s: %2d записей\n", $2, $1}'

echo ""
echo -e "${CYAN}⏱️ Анализ длительности:${NC}"
echo ""

if command -v ffprobe &> /dev/null; then
    total_duration=0
    count=0
    
    find recordings/motion -name "*.mp4" -type f | head -20 | while read file; do
        duration=$(
            ffprobe -v error -show_entries format=duration \
            -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null
        )
        if [ -n "$duration" ]; then
            echo "$duration"
        fi
    done > /tmp/durations.txt
    
    if [ -s /tmp/durations.txt ]; then
        avg_duration=$(awk '{sum+=$1; count++} END {print sum/count}' /tmp/durations.txt)
        total_duration=$(awk '{sum+=$1} END {print sum}' /tmp/durations.txt)
        count=$(wc -l < /tmp/durations.txt)
        
        printf "  Проанализировано: %d видео\n" "$count"
        printf "  Средняя длительность: %.1f секунд\n" "$avg_duration"
        printf "  Общая длительность: %.1f секунд (%.1f минут)\n" \
            "$total_duration" "$(echo "$total_duration/60" | bc -l)"
        
        rm -f /tmp/durations.txt
    fi
else
    echo "  ffprobe не установлен - пропускаем анализ"
fi

echo ""
echo -e "${CYAN}🎬 Примеры использования:${NC}"
echo ""
echo "  # Просмотр последнего видео"
echo "  ffplay \$(ls -t recordings/motion/*.mp4 | head -1)"
echo ""
echo "  # Создание превью (первый кадр)"
echo "  ffmpeg -i recordings/motion/bird_2026-01-16_16-29-47_00m17s.mp4 \\"
echo "         -vframes 1 preview.jpg"
echo ""
echo "  # Объединение нескольких записей"
echo "  ffmpeg -f concat -safe 0 -i filelist.txt -c copy output.mp4"
echo ""

echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}✅ Анализ завершён!${NC}"
echo ""
echo "Система успешно записывает видео при обнаружении движения"
echo "Записи сохраняются в: recordings/motion/"
echo ""
