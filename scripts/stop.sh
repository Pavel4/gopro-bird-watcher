#!/bin/bash
# Остановка системы GoPro Bird Watcher

cd "$(dirname "$0")/.."

echo "⏹️  Остановка GoPro Bird Watcher..."

docker-compose down

echo ""
echo "✅ Система остановлена"
echo "📁 Записи сохранены в: $(pwd)/recordings/"
