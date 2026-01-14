#!/bin/bash
# Начать принудительную (ручную) запись
# Записи сохраняются в recordings/manual/

cd "$(dirname "$0")/.."

echo "RECORD_START" > control/command

echo "🎬 Ручная запись НАЧАТА"
echo "   📁 Записи: recordings/manual/"
echo "   📝 Формат: manual_ГГГГ-ММ-ДД_ЧЧ-ММ-СС_XXmYYs.mp4"
echo ""
echo "   Для остановки: ./scripts/record-stop.sh"
