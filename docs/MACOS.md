# Запуск на macOS (MacBook M5)

Это руководство для разработки и тестирования системы на MacBook M5 (Apple Silicon) перед развертыванием на Raspberry Pi.

## Требования

- macOS (Apple Silicon - M1/M2/M3/M5)
- Python 3.11+
- FFmpeg
- GoPro Hero 13 с USB-C кабелем

## Установка зависимостей

### 1. Homebrew

```bash
# Если еще не установлен
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### 2. Python 3.11

```bash
brew install python@3.11
```

### 3. FFmpeg

```bash
brew install ffmpeg
```

### 4. Python зависимости

```bash
cd /path/to/gopro-bird-watcher
pip3 install -r detector/requirements.txt
```

## Настройка GoPro

1. **Подключите GoPro Hero 13** через USB-C кабель
2. **На самой камере GoPro:**
   - Откройте: **Настройки** → **Подключения** → **USB**
   - Выберите: **GoPro Connect** (не MTP!)
3. **GoPro должна быть включена**

## Проверка подключения

### Определить индекс камеры

```bash
ffmpeg -f avfoundation -list_devices true -i ""
```

Вывод покажет:
```
[AVFoundation indev] AVFoundation video devices:
[AVFoundation indev] [0] FaceTime HD Camera
[AVFoundation indev] [1] GoPro Camera
```

→ GoPro обычно индекс **1** (0 - встроенная камера Mac)

### Проверка через скрипт

```bash
./scripts/gopro-usb.sh check
./scripts/gopro-usb.sh list
```

## Конфигурация

Скопируйте конфиг для macOS:

```bash
cp config.macos.env config.env
```

Отредактируйте `config.env`:

```env
INPUT_SOURCE=usb
USB_DEVICE=1  # Индекс камеры (0, 1, 2, ...)
USB_RESOLUTION=1080
USB_FPS=30

# Включите Telegram если нужно
TELEGRAM_ENABLED=false
# TELEGRAM_BOT_TOKEN=ваш_токен
# TELEGRAM_CHAT_ID=ваш_chat_id
```

## Запуск

**ВАЖНО:** На macOS запускайте детектор **нативно** (не в Docker):

```bash
# Из корневой директории проекта
python3 detector/motion_detector.py
```

Docker на macOS не имеет доступа к USB устройствам!

## Управление

Откройте новый терминал:

```bash
# Включить автозапись при движении
./scripts/motion-on.sh

# Выключить автозапись
./scripts/motion-off.sh

# Ручная запись
./scripts/record-start.sh
./scripts/record-stop.sh

# Статус
./scripts/status.sh
```

## Troubleshooting

### GoPro не определяется

1. Проверьте USB-C кабель (должен поддерживать передачу данных)
2. Перезагрузите GoPro
3. Убедитесь что выбран режим "GoPro Connect"
4. Попробуйте другой USB порт
5. Закройте приложение GoPro Webcam если оно открыто

### Ошибка "Failed to open USB device"

Попробуйте другой индекс камеры:

```bash
# В config.env
USB_DEVICE=0  # встроенная камера
USB_DEVICE=1  # первая внешняя (GoPro)
USB_DEVICE=2  # вторая внешняя
```

### FFmpeg не находит камеру

Проверьте что FFmpeg установлен с AVFoundation:

```bash
ffmpeg -devices 2>&1 | grep avfoundation
```

Должно быть:
```
avfoundation    AVFoundation input device
```

### Низкий FPS

1. Снизьте разрешение: `USB_RESOLUTION=720`
2. Снизьте FPS: `USB_FPS=20`
3. Закройте другие приложения

## Следующие шаги

После успешного тестирования на macOS:

1. Проверьте что записи сохраняются в `recordings/motion/`
2. Протестируйте автоочистку старых файлов
3. Настройте Telegram бота (опционально)
4. Перенесите на Raspberry Pi (см. [RASPBERRY_PI.md](RASPBERRY_PI.md))

## Полезные команды

```bash
# Просмотр логов
tail -f logs/motion_detector.log

# Очистка всех записей
rm -rf recordings/motion/* recordings/manual/*

# Тест захвата кадра
./scripts/gopro-usb.sh test

# Просмотр записей
ls -lh recordings/motion/
```
