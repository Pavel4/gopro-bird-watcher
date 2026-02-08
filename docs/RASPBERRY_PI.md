# Развертывание на Raspberry Pi 5

Руководство по развертыванию GoPro Bird Watcher на Raspberry Pi 5 для автономной работы.

## Требования

- **Raspberry Pi 5** (4GB/8GB RAM)
- **SD карта**: 128GB Class 10 (или SSD через USB 3.0 - рекомендуется)
- **Охлаждение**: Активное (вентилятор) или пассивное (радиатор)
- **Питание**: 5V 5A (официальный БП для Pi 5)
- **GoPro Hero 13** + USB-C кабель
- **Опционально**: USB hub с внешним питанием (для стабильности GoPro)

## Установка Raspberry Pi OS

1. Скачайте [Raspberry Pi Imager](https://www.raspberrypi.com/software/)
2. Установите **Raspberry Pi OS Lite (64-bit)** на SD карту
3. При настройке:
   - Включите SSH
   - Установите hostname, пароль
   - Настройте WiFi (если нужно)

## Подготовка системы

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Установка Docker Compose
sudo apt install docker-compose -y

# Перезагрузка для применения изменений
sudo reboot
```

## Клонирование проекта

```bash
cd ~
git clone <your-repo-url> gopro-bird-watcher
cd gopro-bird-watcher
```

## Настройка

### 1. Конфигурация

```bash
cp config.pi.env config.env
nano config.env
```

Отредактируйте:
```env
# USB устройство
USB_DEVICE=/dev/video0

# Управление хранилищем (важно для 128GB SD)
MAX_RECORDING_AGE_DAYS=30  # Автоудаление записей старше 30 дней
AUTO_CLEANUP_ENABLED=true

# Telegram бот (получите токен через @BotFather)
TELEGRAM_ENABLED=true
TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
TELEGRAM_CHAT_ID=123456789
```

### 2. GoPro подключение

1. Подключите GoPro через USB-C
2. На GoPro: **Настройки** → **Подключения** → **USB** → **GoPro Connect**
3. Проверьте:

```bash
ls /dev/video*
# Должно показать /dev/video0
```

## Запуск

### Сборка и запуск контейнеров

```bash
# Сборка ARM64 образа
docker-compose -f docker-compose.pi.yml build

# Запуск в фоне
docker-compose -f docker-compose.pi.yml up -d

# Просмотр логов
docker-compose -f docker-compose.pi.yml logs -f detector
```

### Подключение к контейнеру

```bash
docker exec -it gopro-detector-pi bash

# Внутри контейнера
python detector/motion_detector.py
```

## Автозапуск при загрузке

Создайте systemd service:

```bash
sudo nano /etc/systemd/system/gopro-watcher.service
```

```ini
[Unit]
Description=GoPro Bird Watcher
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/home/pi/gopro-bird-watcher
ExecStart=/usr/bin/docker-compose -f docker-compose.pi.yml up -d
ExecStop=/usr/bin/docker-compose -f docker-compose.pi.yml down
User=pi

[Install]
WantedBy=multi-user.target
```

Включите сервис:

```bash
sudo systemctl enable gopro-watcher
sudo systemctl start gopro-watcher
sudo systemctl status gopro-watcher
```

## Мониторинг

### Температура CPU

```bash
# Прямая проверка
cat /sys/class/thermal/thermal_zone0/temp
# Выведет температуру в милиградусах (65000 = 65°C)

# Удобная команда
vcgencmd measure_temp
```

**Рекомендации:**
- < 75°C - нормально
- 75-85°C - установите охлаждение
- > 85°C - критично, система может троттлить

### Использование диска

```bash
df -h /app/recordings
```

### Логи

```bash
# Docker логи
docker-compose -f docker-compose.pi.yml logs -f

# Логи детектора
tail -f logs/motion_detector.log
```

## Оптимизация производительности

### Если 1080p тормозит

В `config.env`:
```env
USB_RESOLUTION=720  # Снизить до 720p
USB_FPS=20          # Снизить FPS до 20
```

### SSD вместо SD карты (рекомендуется)

1. Подключите SSD через USB 3.0
2. Примонтируйте к `/app/recordings`:

```bash
sudo mkfs.ext4 /dev/sda1
sudo mount /dev/sda1 /home/pi/gopro-bird-watcher/recordings

# Автомонтирование в /etc/fstab
/dev/sda1 /home/pi/gopro-bird-watcher/recordings ext4 defaults 0 2
```

## Troubleshooting

### USB устройство не определяется

```bash
# Проверка USB устройств
lsusb | grep GoPro

# Права доступа
sudo chmod 666 /dev/video0

# Перезагрузка
sudo reboot
```

### Перегрев Pi

- Установите активное охлаждение
- Снизьте разрешение/FPS
- Проверьте вентиляцию корпуса

### Заполнение диска

Проверьте автоочистку:
```bash
# В config.env
AUTO_CLEANUP_ENABLED=true
MAX_RECORDING_AGE_DAYS=14  # Уменьшите срок хранения
```

## Полезные команды

```bash
# Перезапуск контейнеров
docker-compose -f docker-compose.pi.yml restart

# Остановка
docker-compose -f docker-compose.pi.yml down

# Обновление кода
git pull
docker-compose -f docker-compose.pi.yml build
docker-compose -f docker-compose.pi.yml up -d

# Очистка старых Docker образов
docker system prune -a
```

## Безопасность

```bash
# Обновляйте систему регулярно
sudo apt update && sudo apt upgrade -y

# Настройте firewall
sudo apt install ufw
sudo ufw allow ssh
sudo ufw enable

# Смените пароль по умолчанию
passwd
```

## Дальше

- Настройте Telegram бота (см. [TELEGRAM_BOT.md](TELEGRAM_BOT.md))
- Настройте ROI для точной детекции
- Мониторьте температуру и производительность
