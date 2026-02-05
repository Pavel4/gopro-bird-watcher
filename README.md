# GoPro Bird Watcher

Система автоматической записи видео с GoPro Hero 13 при обнаружении движения.
Идеально подходит для наблюдения за птицами на кормушке.

## Возможности

- **Три платформы:** macOS (Apple Silicon), Raspberry Pi 5, Linux
- **USB Webcam режим** — проводное подключение GoPro, стабильно работает
- **RTMP (WiFi)** — беспроводное подключение, требует настройки сети
- **Автоопределение GoPro** — не нужно вручную указывать индекс камеры (`USB_DEVICE=auto`)
- Умная детекция движения с настраиваемыми порогами
- Два режима записи: **автоматический** (при движении) и **ручной** (по команде)
- Буфер: записывает N секунд **до** и **после** движения (Linux/Pi)
- **ROI (Region of Interest)** — запись только области кормушки
- **Управление хранилищем** — автоматическая очистка старых записей
- **Telegram бот** — автоматическая отправка видео и команды управления
- Время по Москве в именах файлов

## Архитектура

```
GoPro Hero 13 ──USB-C──> Хост (macOS / Raspberry Pi / Linux)
                                │
                          Motion Detector (Python + OpenCV + FFmpeg)
                                │
                                ├──> recordings/motion/   (автозаписи)
                                ├──> recordings/manual/   (ручные записи)
                                ├──> Storage Manager      (автоочистка)
                                └──> Telegram Bot         (уведомления)
```

## Быстрый старт

### 1. Клонирование и установка

```bash
git clone https://github.com/your-username/gopro-bird-watcher.git
cd gopro-bird-watcher

python3 -m venv venv
source venv/bin/activate
pip install -r detector/requirements.txt
```

### 2. Настройка credentials (секреты)

```bash
cp credentials.env.example credentials.env
```

Отредактируйте `credentials.env` — укажите токен Telegram бота и Chat ID:

```env
TELEGRAM_BOT_TOKEN=your_token_from_BotFather
TELEGRAM_CHAT_ID=your_chat_id
```

Подробнее: [Настройка Telegram бота](docs/TELEGRAM_BOT.md)

### 3. Подключение GoPro

На самой камере: **Настройки -> Подключения -> USB -> "GoPro Connect"** (не MTP).
Подключите GoPro по USB-C кабелю и включите камеру.

### 4. Запуск

```bash
./run-native.sh
```

Скрипт автоматически:
- определит платформу (macOS / Raspberry Pi / Linux)
- загрузит нужный конфиг
- загрузит credentials
- запустит детектор

## Конфигурация

Проект использует **два уровня** конфигурации:

| Файл | Назначение | В git? |
|------|-----------|--------|
| `credentials.env` | Секреты (токен бота, chat ID) | **Нет** (.gitignore) |
| `credentials.env.example` | Шаблон для credentials | Да |
| `config.env` | Общие настройки | Да |
| `config.macos.env` | Настройки для macOS | Да |
| `config.pi.env` | Настройки для Raspberry Pi | Да |

**Приоритет загрузки:**
1. Переменные окружения (наивысший приоритет)
2. `credentials.env` (секреты)
3. Платформенный конфиг (`config.macos.env` / `config.pi.env` / `config.env`)

### Основные параметры

```env
# Источник видео
INPUT_SOURCE=usb           # "usb" (рекомендуется) или "rtmp"
USB_DEVICE=auto            # "auto" для автоопределения GoPro, или индекс (0, 1, 2...)
USB_RESOLUTION=1080        # 480, 720, или 1080
USB_FPS=30

# Детекция движения
BUFFER_SECONDS=5           # Секунд ДО движения (только Linux/Pi)
POST_MOTION_SECONDS=5      # Секунд ПОСЛЕ движения
MOTION_AREA_PERCENT=3.0    # Мин. % площади кадра для старта записи
EXTEND_MOTION_PERCENT=0.2  # Мин. % движения чтобы продлить запись

# Telegram
TELEGRAM_ENABLED=true      # Включить отправку видео в Telegram
```

## Команды

| Команда | Описание |
|---------|----------|
| `./run-native.sh` | Запуск детектора (нативно) |
| `./scripts/start.sh` | Запуск в Docker |
| `./scripts/stop.sh` | Остановка |
| `./scripts/status.sh` | Статус |
| `./scripts/gopro-usb.sh check` | Проверить USB подключение GoPro |
| `./scripts/motion-on.sh` | Включить детекцию движения |
| `./scripts/motion-off.sh` | Выключить детекцию движения |
| `./scripts/record-start.sh` | Начать ручную запись |
| `./scripts/record-stop.sh` | Остановить ручную запись |
| `./scripts/select-roi.sh` | Выбрать область кормушки (ROI) |

### Telegram бот

Отправьте боту в Telegram:
- `/start` — приветствие
- `/status` — статус системы
- `/latest` — последние 5 записей

## Структура проекта

```
gopro-bird-watcher/
├── credentials.env.example    # Шаблон для секретов
├── config.env                 # Общие настройки
├── config.macos.env           # Настройки для macOS
├── config.pi.env              # Настройки для Raspberry Pi
├── run-native.sh              # Запуск без Docker
├── docker-compose.yml         # Docker конфигурация
├── docker-compose.pi.yml      # Docker для Raspberry Pi
├── nginx.conf                 # RTMP сервер (WiFi режим)
├── detector/
│   ├── motion_detector.py     # Детектор движения
│   ├── telegram_bot.py        # Telegram бот
│   ├── storage_manager.py     # Управление хранилищем
│   ├── select_roi.py          # Выбор области кормушки
│   ├── requirements.txt       # Python зависимости
│   ├── Dockerfile
│   └── Dockerfile.arm64
├── scripts/                   # Утилиты и скрипты управления
├── recordings/
│   ├── motion/                # Автозаписи при движении
│   └── manual/                # Ручные записи
├── logs/
├── docs/
│   ├── BACKLOG.md             # Известные проблемы и планы
│   ├── NATIVE_SETUP.md        # Руководство по нативному запуску
│   ├── MACOS.md               # Специфика macOS
│   ├── RASPBERRY_PI.md        # Развертывание на Pi
│   └── TELEGRAM_BOT.md        # Настройка Telegram
└── control/
```

## Документация

- [Нативный запуск (macOS / Linux)](docs/NATIVE_SETUP.md)
- [macOS: специфика](docs/MACOS.md)
- [Raspberry Pi: развертывание](docs/RASPBERRY_PI.md)
- [Telegram бот: настройка](docs/TELEGRAM_BOT.md)
- [Backlog: проблемы и планы](docs/BACKLOG.md)

## Настройка ROI (Region of Interest)

ROI позволяет записывать только область кормушки:

```bash
./scripts/select-roi.sh
```

Или вручную в конфиге:

```env
ROI_ENABLED=true
ROI_X=200
ROI_Y=100
ROI_WIDTH=640
ROI_HEIGHT=400
```

## Troubleshooting

### Много ложных срабатываний

Увеличьте пороги:

```env
MIN_MOTION_FRAMES=5
MOTION_AREA_PERCENT=3.0
```

### Пропускает птиц

Уменьшите пороги:

```env
MIN_MOTION_FRAMES=2
MOTION_AREA_PERCENT=1.0
```

### Видео записывается с вебки, а не с GoPro

Установите `USB_DEVICE=auto` — система автоматически найдет GoPro.
Или укажите индекс вручную. На macOS проверьте доступные камеры:

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

## Лицензия

MIT
