#!/bin/bash
# Просмотр RTMP потока

echo "📺 Открываю поток в VLC..."

# Попытка открыть в VLC
if [ -d "/Applications/VLC.app" ]; then
    /Applications/VLC.app/Contents/MacOS/VLC rtmp://localhost/live/gopro 2>/dev/null
elif command -v ffplay &> /dev/null; then
    echo "VLC не найден, использую ffplay..."
    ffplay -fflags nobuffer rtmp://localhost/live/gopro
else
    echo "❌ Не найден VLC или ffplay"
    echo ""
    echo "Установите VLC: https://www.videolan.org/vlc/"
    echo "Или используйте OBS Studio с URL: rtmp://localhost/live/gopro"
fi
