#!/bin/bash
# Скрипт автоматического развертывания на Raspberry Pi
#
# Использование:
#   ./scripts/deploy-to-pi.sh [user@hostname]
#
# Пример:
#   ./scripts/deploy-to-pi.sh pi@raspberrypi.local
#

set -e

# Цвета
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}================================================${NC}"
    echo -e "${BLUE}  Deploy to Raspberry Pi${NC}"
    echo -e "${BLUE}================================================${NC}"
    echo ""
}

# Проверка параметров
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: Raspberry Pi host required${NC}"
    echo ""
    echo "Usage: $0 [user@hostname]"
    echo ""
    echo "Examples:"
    echo "  $0 pi@raspberrypi.local"
    echo "  $0 pi@192.168.1.100"
    exit 1
fi

PI_HOST=$1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
REMOTE_DIR="/home/pi/gopro-bird-watcher"

print_header

echo -e "${GREEN}🎯 Target: $PI_HOST${NC}"
echo -e "${GREEN}📁 Project: $PROJECT_DIR${NC}"
echo ""

# Проверка SSH подключения
echo -e "${BLUE}1. Проверка SSH подключения...${NC}"
if ssh -o ConnectTimeout=5 "$PI_HOST" "echo 'OK'" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ SSH подключение работает${NC}"
else
    echo -e "${RED}❌ Не удалось подключиться к $PI_HOST${NC}"
    echo ""
    echo "Убедитесь что:"
    echo "  - Raspberry Pi включен и в сети"
    echo "  - SSH включен (sudo raspi-config)"
    echo "  - Правильный hostname/IP"
    echo "  - SSH ключ настроен или будет запрошен пароль"
    exit 1
fi

echo ""

# Создание директории на Pi
echo -e "${BLUE}2. Создание директории на Pi...${NC}"
ssh "$PI_HOST" "mkdir -p $REMOTE_DIR"
echo -e "${GREEN}✅ Директория создана: $REMOTE_DIR${NC}"
echo ""

# Синхронизация файлов
echo -e "${BLUE}3. Синхронизация проекта...${NC}"
rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'recordings/' \
    --exclude 'logs/' \
    --exclude '.segments/' \
    --exclude 'node_modules/' \
    --exclude 'test_frame.jpg' \
    "$PROJECT_DIR/" "$PI_HOST:$REMOTE_DIR/"

echo -e "${GREEN}✅ Файлы синхронизированы${NC}"
echo ""

# Проверка Docker
echo -e "${BLUE}4. Проверка Docker на Pi...${NC}"
if ssh "$PI_HOST" "command -v docker > /dev/null 2>&1"; then
    echo -e "${GREEN}✅ Docker установлен${NC}"
else
    echo -e "${YELLOW}⚠️ Docker не установлен${NC}"
    echo ""
    echo "Установите Docker на Raspberry Pi:"
    echo "  curl -fsSL https://get.docker.com -o get-docker.sh"
    echo "  sudo sh get-docker.sh"
    echo "  sudo usermod -aG docker pi"
    echo "  sudo reboot"
    echo ""
    echo "После установки запустите скрипт снова."
    exit 1
fi

# Проверка docker-compose
if ssh "$PI_HOST" "command -v docker-compose > /dev/null 2>&1"; then
    echo -e "${GREEN}✅ Docker Compose установлен${NC}"
else
    echo -e "${YELLOW}⚠️ Docker Compose не установлен${NC}"
    echo ""
    echo "Установите Docker Compose:"
    echo "  sudo apt install docker-compose -y"
    exit 1
fi

echo ""

# Сборка образа
echo -e "${BLUE}5. Сборка Docker образа на Pi...${NC}"
ssh "$PI_HOST" "cd $REMOTE_DIR && docker-compose -f docker-compose.pi.yml build"
echo -e "${GREEN}✅ Образ собран${NC}"
echo ""

# Проверка конфигурации
echo -e "${BLUE}6. Проверка конфигурации...${NC}"
if ssh "$PI_HOST" "[ -f $REMOTE_DIR/config.env ]"; then
    echo -e "${GREEN}✅ config.env существует${NC}"
else
    echo -e "${YELLOW}⚠️ config.env не найден${NC}"
    echo ""
    echo "Скопируйте config.pi.env в config.env и настройте:"
    echo "  ssh $PI_HOST"
    echo "  cd $REMOTE_DIR"
    echo "  cp config.pi.env config.env"
    echo "  nano config.env"
fi

echo ""

# Проверка GoPro
echo -e "${BLUE}7. Проверка GoPro подключения...${NC}"
if ssh "$PI_HOST" "ls /dev/video0 > /dev/null 2>&1"; then
    echo -e "${GREEN}✅ /dev/video0 найден${NC}"
else
    echo -e "${YELLOW}⚠️ /dev/video0 не найден${NC}"
    echo ""
    echo "Подключите GoPro через USB-C и выберите 'GoPro Connect'"
fi

echo ""

# Финальные инструкции
echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}  Развертывание завершено!${NC}"
echo -e "${GREEN}================================================${NC}"
echo ""
echo -e "${BLUE}Следующие шаги:${NC}"
echo ""
echo "1. Подключитесь к Pi:"
echo "   ${YELLOW}ssh $PI_HOST${NC}"
echo ""
echo "2. Перейдите в директорию:"
echo "   ${YELLOW}cd $REMOTE_DIR${NC}"
echo ""
echo "3. Настройте config.env (если еще не сделали):"
echo "   ${YELLOW}cp config.pi.env config.env${NC}"
echo "   ${YELLOW}nano config.env${NC}"
echo ""
echo "4. Запустите контейнеры:"
echo "   ${YELLOW}docker-compose -f docker-compose.pi.yml up -d${NC}"
echo ""
echo "5. Просмотр логов:"
echo "   ${YELLOW}docker-compose -f docker-compose.pi.yml logs -f detector${NC}"
echo ""
echo "6. Подключение к контейнеру:"
echo "   ${YELLOW}docker exec -it gopro-detector-pi bash${NC}"
echo ""
echo "7. Запуск детектора (внутри контейнера):"
echo "   ${YELLOW}python detector/motion_detector.py${NC}"
echo ""
echo -e "${BLUE}Документация:${NC}"
echo "  docs/RASPBERRY_PI.md - полное руководство"
echo "  docs/TELEGRAM_BOT.md - настройка Telegram бота"
echo ""
