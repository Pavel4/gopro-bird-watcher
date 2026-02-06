#!/usr/bin/env python3
"""
Интерактивный инструмент для выбора области интереса (ROI).
Позволяет захватить кадр с камеры и выбрать область кормушки мышкой.

Использование:
    python select_roi.py --usb auto               # USB (автоопределение)
    python select_roi.py --usb 0                   # USB по индексу
    python select_roi.py --image frame.jpg         # Из изображения
    python select_roi.py --rtmp rtmp://host/live   # С RTMP потока

Примеры:
    python select_roi.py --usb auto                # GoPro по USB
    python select_roi.py --usb auto --save-frame frame.jpg
    python select_roi.py --image frame.jpg         # Готовое изображение
    python select_roi.py --rtmp rtmp://host/live   # RTMP поток
"""

import cv2
import numpy as np
import argparse
import os
import platform
import re
import subprocess
import sys
import time


# Глобальные переменные для callback мыши
roi_start = None
roi_end = None
drawing = False
roi_selected = False


def mouse_callback(event, x, y, flags, param):
    """Обработчик событий мыши для выбора ROI."""
    global roi_start, roi_end, drawing, roi_selected
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        roi_end = (x, y)
        roi_selected = False
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            roi_end = (x, y)
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)
        roi_selected = True


def capture_frame_from_rtmp(rtmp_url: str, timeout: int = 30) -> np.ndarray:
    """
    Захватить один кадр с RTMP потока.
    
    Args:
        rtmp_url: URL RTMP потока
        timeout: Таймаут ожидания в секундах
    
    Returns:
        Кадр как numpy array или None при ошибке
    """
    print(f"📡 Подключение к {rtmp_url}...")
    
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
    cap = cv2.VideoCapture(rtmp_url)
    
    if not cap.isOpened():
        print("❌ Не удалось подключиться к RTMP потоку")
        return None
    
    # Пробуем захватить несколько кадров (первые могут быть битые)
    start_time = time.time()
    frame = None
    
    for attempt in range(100):
        ret, frame = cap.read()
        if ret and frame is not None:
            # Проверяем что кадр не чёрный
            if np.mean(frame) > 10:
                break
        
        if time.time() - start_time > timeout:
            print(f"❌ Таймаут {timeout}с - не удалось получить кадр")
            cap.release()
            return None
        
        time.sleep(0.1)
    
    cap.release()
    
    if frame is None:
        print("❌ Не удалось захватить кадр")
        return None
    
    h, w = frame.shape[:2]
    print(f"✅ Кадр захвачен: {w}x{h}")
    return frame


def load_frame_from_file(image_path: str) -> np.ndarray:
    """
    Загрузить кадр из файла изображения.
    
    Args:
        image_path: Путь к файлу изображения
    
    Returns:
        Кадр как numpy array или None при ошибке
    """
    if not os.path.exists(image_path):
        print(f"❌ Файл не найден: {image_path}")
        return None
    
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ Не удалось загрузить изображение: {image_path}")
        return None
    
    h, w = frame.shape[:2]
    print(f"✅ Изображение загружено: {w}x{h}")
    return frame


def detect_gopro_index() -> int:
    """
    Автоопределение индекса GoPro на macOS через FFmpeg.

    Returns:
        Индекс GoPro устройства или -1 если не найдено
    """
    if platform.system() != "Darwin":
        print("⚠️  Автоопределение GoPro поддерживается только на macOS")
        return -1
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-f", "avfoundation",
                "-list_devices", "true", "-i", ""
            ],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stderr.split('\n'):
            if 'gopro' in line.lower():
                match = re.search(r'\[(\d+)\]', line)
                if match:
                    idx = int(match.group(1))
                    print(f"✅ GoPro найдена: индекс {idx}")
                    return idx
        print("⚠️  GoPro не найдена в списке устройств")
        return -1
    except Exception as e:
        print(f"❌ Ошибка автоопределения GoPro: {e}")
        return -1


def capture_frame_from_usb(
    device_index, timeout: int = 10
) -> np.ndarray:
    """
    Захватить один кадр с USB-камеры.

    На macOS: через FFmpeg (AVFoundation), т.к. OpenCV
    и FFmpeg используют разные индексы устройств.
    На Linux: через OpenCV VideoCapture.

    Args:
        device_index: Индекс камеры (int) или 'auto'
        timeout: Таймаут ожидания в секундах

    Returns:
        Кадр как numpy array или None при ошибке
    """
    # Определяем индекс устройства
    if (isinstance(device_index, str)
            and device_index.lower() == "auto"):
        idx = detect_gopro_index()
        if idx < 0:
            print("⚠️  GoPro не найдена, пробуем камеру 0...")
            idx = 0
    else:
        try:
            idx = int(device_index)
        except ValueError:
            print(
                f"❌ Неверный индекс камеры: "
                f"{device_index}"
            )
            return None

    print(
        f"📹 Подключение к USB камере "
        f"(индекс {idx})..."
    )

    # macOS: захват через FFmpeg (AVFoundation)
    # Индексы FFmpeg и OpenCV не совпадают на macOS!
    if platform.system() == "Darwin":
        return _capture_frame_ffmpeg_macos(idx, timeout)

    # Linux: стандартный OpenCV
    return _capture_frame_opencv(idx, timeout)


def _capture_frame_ffmpeg_macos(
    idx: int, timeout: int = 10
) -> np.ndarray:
    """
    Захватить один кадр через FFmpeg AVFoundation.
    Решает проблему несовпадения индексов камер
    между FFmpeg и OpenCV на macOS.
    """
    import tempfile
    tmp_path = os.path.join(
        tempfile.gettempdir(),
        "select_roi_frame.jpg"
    )
    # Удаляем старый файл если есть
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-f", "avfoundation",
        "-pixel_format", "uyvy422",
        "-framerate", "30",
        "-video_size", "1920x1080",
        "-i", str(idx),
        "-frames:v", "5",
        "-update", "1",
        "-q:v", "2",
        tmp_path,
    ]

    print(
        f"   macOS: захват через FFmpeg "
        f"(AVFoundation, device {idx})..."
    )

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0:
            print(
                f"❌ FFmpeg ошибка: "
                f"{result.stderr[:300]}"
            )
            return None
    except subprocess.TimeoutExpired:
        print(f"❌ Таймаут {timeout}с — FFmpeg не ответил")
        return None
    except Exception as e:
        print(f"❌ Ошибка запуска FFmpeg: {e}")
        return None

    if not os.path.exists(tmp_path):
        print("❌ FFmpeg не создал файл кадра")
        return None

    frame = cv2.imread(tmp_path)
    try:
        os.remove(tmp_path)
    except Exception:
        pass

    if frame is None:
        print("❌ Не удалось прочитать захваченный кадр")
        return None

    h, w = frame.shape[:2]
    print(f"✅ Кадр захвачен с USB камеры: {w}x{h}")
    return frame


def _capture_frame_opencv(
    idx: int, timeout: int = 10
) -> np.ndarray:
    """Захватить кадр через OpenCV (Linux)."""
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"❌ Не удалось открыть камеру {idx}")
        return None

    start_time = time.time()
    frame = None

    for attempt in range(100):
        ret, frame = cap.read()
        if ret and frame is not None:
            if np.mean(frame) > 10:
                break

        if time.time() - start_time > timeout:
            print(
                f"❌ Таймаут {timeout}с — "
                f"не удалось получить кадр"
            )
            cap.release()
            return None

        time.sleep(0.1)

    cap.release()

    if frame is None:
        print("❌ Не удалось захватить кадр с USB камеры")
        return None

    h, w = frame.shape[:2]
    print(f"✅ Кадр захвачен с USB камеры: {w}x{h}")
    return frame


def select_roi_interactive(frame: np.ndarray) -> tuple:
    """
    Интерактивный выбор области ROI с помощью мыши.
    
    Args:
        frame: Исходный кадр
    
    Returns:
        Кортеж (x, y, width, height) или None если отменено
    """
    global roi_start, roi_end, drawing, roi_selected
    
    # Сброс состояния
    roi_start = None
    roi_end = None
    drawing = False
    roi_selected = False
    
    window_name = "Select ROI - Draw rectangle with mouse, ENTER to confirm, R to reset, Q to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Масштабируем для удобного отображения
    h, w = frame.shape[:2]
    max_display_width = 1200
    max_display_height = 800
    
    scale = min(max_display_width / w, max_display_height / h, 1.0)
    display_w = int(w * scale)
    display_h = int(h * scale)
    
    cv2.resizeWindow(window_name, display_w, display_h)
    
    print("\n🖱️  Инструкция:")
    print("   - Нарисуйте прямоугольник мышкой")
    print("   - ENTER - подтвердить выбор")
    print("   - R - сбросить и выбрать заново")
    print("   - Q или ESC - отмена\n")
    
    confirmed_roi = None
    
    while True:
        # Копия кадра для отрисовки
        display = frame.copy()
        
        # Рисуем текущий выбор
        if roi_start and roi_end:
            x1, y1 = roi_start
            x2, y2 = roi_end
            
            # Нормализуем координаты (чтобы работало при рисовании в любом направлении)
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            # Рисуем полупрозрачный затемнённый фон вне ROI
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 0, 0), -1)
            alpha = 0.3
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            
            # Восстанавливаем ROI область
            display[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
            
            # Рисуем рамку
            color = (0, 255, 0) if roi_selected else (0, 255, 255)
            cv2.rectangle(display, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Показываем размеры
            roi_w = x_max - x_min
            roi_h = y_max - y_min
            text = f"ROI: {roi_w}x{roi_h} @ ({x_min}, {y_min})"
            cv2.putText(display, text, (x_min, y_min - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Добавляем подсказку
        help_text = "ENTER=confirm | R=reset | Q=quit"
        cv2.putText(display, help_text, (10, h - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow(window_name, display)
        
        key = cv2.waitKey(30) & 0xFF
        
        if key == ord('q') or key == 27:  # Q или ESC
            print("❌ Отменено пользователем")
            break
        
        elif key == ord('r'):  # R - сброс
            roi_start = None
            roi_end = None
            roi_selected = False
            print("🔄 Сброс выбора")
        
        elif key == 13 and roi_selected:  # ENTER
            x1, y1 = roi_start
            x2, y2 = roi_end
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            roi_w = x_max - x_min
            roi_h = y_max - y_min
            
            if roi_w > 50 and roi_h > 50:
                confirmed_roi = (x_min, y_min, roi_w, roi_h)
                print(f"✅ ROI выбран: x={x_min}, y={y_min}, w={roi_w}, h={roi_h}")
                break
            else:
                print("⚠️  ROI слишком маленький (минимум 50x50)")
    
    cv2.destroyAllWindows()
    return confirmed_roi


def show_roi_preview(frame: np.ndarray, roi: tuple):
    """
    Показать превью обрезанной области.
    
    Args:
        frame: Исходный кадр
        roi: Кортеж (x, y, width, height)
    """
    x, y, w, h = roi
    cropped = frame[y:y+h, x:x+w]
    
    window_name = "ROI Preview - Press any key to continue"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Масштабируем для отображения
    max_display = 800
    scale = min(max_display / w, max_display / h, 1.0)
    display_w = int(w * scale)
    display_h = int(h * scale)
    cv2.resizeWindow(window_name, display_w, display_h)
    
    cv2.imshow(window_name, cropped)
    print(f"\n📺 Превью ROI ({w}x{h}). Нажмите любую клавишу...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def update_config_file(
    config_path: str,
    roi: tuple,
    enable_crop: bool = False
) -> bool:
    """
    Обновить файл конфигурации с ROI и CROP.
    
    Args:
        config_path: Путь к config.env
        roi: Кортеж (x, y, width, height)
        enable_crop: Также включить CROP_VIDEO
    
    Returns:
        True если успешно
    """
    x, y, w, h = roi
    
    # Читаем текущий конфиг
    lines = []
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            lines = f.readlines()
    
    # Параметры для обновления
    params = {
        'ROI_ENABLED': 'true',
        'ROI_X': str(x),
        'ROI_Y': str(y),
        'ROI_WIDTH': str(w),
        'ROI_HEIGHT': str(h),
    }
    
    # Если включаем CROP — ставим те же координаты
    # (fallback: CROP=0 → используются ROI)
    if enable_crop:
        params['CROP_VIDEO_ENABLED'] = 'true'
    
    # Обновляем существующие параметры
    updated_keys = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if (stripped
                and not stripped.startswith('#')
                and '=' in stripped):
            key = stripped.split('=', 1)[0].strip()
            if key in params:
                new_lines.append(
                    f"{key}={params[key]}\n"
                )
                updated_keys.add(key)
                continue
        new_lines.append(line)
    
    # Добавляем недостающие параметры
    missing = set(params.keys()) - updated_keys
    if missing:
        has_roi = any(
            'ROI' in l and '===' in l
            for l in new_lines
        )
        if not has_roi:
            new_lines.append(
                "\n# === ROI — выбрано через "
                "select_roi.py ===\n"
            )
        
        ordered = [
            'ROI_ENABLED', 'ROI_X', 'ROI_Y',
            'ROI_WIDTH', 'ROI_HEIGHT',
            'CROP_VIDEO_ENABLED',
        ]
        for key in ordered:
            if key in missing:
                new_lines.append(
                    f"{key}={params[key]}\n"
                )
    
    # Записываем
    try:
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        print(
            f"✅ Конфигурация сохранена "
            f"в {config_path}"
        )
        return True
    except Exception as e:
        print(f"❌ Ошибка записи: {e}")
        return False


def save_frame(frame: np.ndarray, output_path: str):
    """Сохранить кадр в файл."""
    cv2.imwrite(output_path, frame)
    print(f"💾 Кадр сохранён: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            'Интерактивный выбор области интереса '
            '(ROI) для кормушки'
        )
    )
    parser.add_argument(
        '--usb', '-u',
        nargs='?', const='auto', default=None,
        help=(
            'USB камера: индекс (0,1,...) или "auto" '
            'для автоопределения GoPro (по умолчанию: auto)'
        )
    )
    parser.add_argument(
        '--rtmp', '-r',
        default=None,
        help='URL RTMP потока'
    )
    parser.add_argument(
        '--image', '-i',
        help='Путь к изображению (вместо камеры)'
    )
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Путь к config.env (автоопределение по платформе)'
    )
    parser.add_argument(
        '--save-frame', '-s',
        help='Сохранить захваченный кадр в файл'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help=(
            'Не сохранять в config.env '
            '(только показать координаты)'
        )
    )
    parser.add_argument(
        '--crop',
        action='store_true',
        help=(
            'Также включить обрезку видео по ROI '
            '(CROP_VIDEO_ENABLED=true)'
        )
    )

    args = parser.parse_args()

    # Автоопределение пути к конфигу
    if args.config is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        system = platform.system()
        if system == "Darwin":
            cfg = os.path.join(project_dir, "config.macos.env")
        elif os.path.exists("/proc/device-tree/model"):
            cfg = os.path.join(project_dir, "config.pi.env")
        else:
            cfg = os.path.join(project_dir, "config.env")
        args.config = cfg

    print("=" * 60)
    print("  Инструмент выбора ROI (Region of Interest)")
    print("=" * 60)

    # Получаем кадр (приоритет: image > usb > rtmp)
    frame = None
    if args.image:
        frame = load_frame_from_file(args.image)
    elif args.usb is not None:
        frame = capture_frame_from_usb(args.usb)
    elif args.rtmp:
        frame = capture_frame_from_rtmp(args.rtmp)
    else:
        # По умолчанию — USB auto
        print("Источник не указан, пробуем USB (auto)...")
        frame = capture_frame_from_usb("auto")

    if frame is None:
        print("\n  Совет:")
        print("   --usb auto    : GoPro по USB")
        print("   --usb 0       : камера по индексу")
        print("   --image FILE  : готовое изображение")
        print("   --rtmp URL    : RTMP поток")
        sys.exit(1)
    
    # Сохраняем кадр если нужно
    if args.save_frame:
        save_frame(frame, args.save_frame)
    
    # Интерактивный выбор
    roi = select_roi_interactive(frame)
    
    if roi is None:
        print("\n⚠️  ROI не выбран")
        sys.exit(1)
    
    # Показываем превью
    show_roi_preview(frame, roi)
    
    # Выводим координаты
    x, y, w, h = roi
    print("\n" + "=" * 60)
    print("📋 Координаты ROI (детекция движения):")
    print(f"   ROI_X={x}")
    print(f"   ROI_Y={y}")
    print(f"   ROI_WIDTH={w}")
    print(f"   ROI_HEIGHT={h}")
    print("=" * 60)
    
    # Спрашиваем про обрезку видео
    enable_crop = args.crop
    if not args.no_save and not args.crop:
        print(
            "\n🔲 Включить обрезку видео по этой "
            "области?"
        )
        print(
            "   (видео будет обрезано до ROI и "
            "отправлено в Telegram крупным планом)"
        )
        try:
            answer = input(
                "   [y/N]: "
            ).strip().lower()
            enable_crop = answer in ('y', 'yes', 'д', 'да')
        except (EOFError, KeyboardInterrupt):
            enable_crop = False
    
    if enable_crop:
        print("   ✅ CROP_VIDEO_ENABLED=true")
    
    # Сохраняем в конфиг
    if not args.no_save:
        if update_config_file(
            args.config, roi, enable_crop
        ):
            print(
                f"\n✅ ROI настроен в "
                f"{args.config}!"
            )
            if enable_crop:
                print(
                    "   Видео будет обрезано до "
                    f"{w}x{h}"
                )
                print(
                    "   Для масштабирования: "
                    "CROP_SCALE=1280x720"
                )
            print("   Перезапустите детектор:")
            print("   Native:  ./run-native.sh")
            print(
                "   Docker:  "
                "docker-compose restart detector"
            )
        else:
            print(
                "\n⚠️  Не удалось сохранить. "
                "Добавьте параметры вручную."
            )
    else:
        print(
            "\n  Добавьте эти параметры "
            "в config.env вручную"
        )
    
    print()


if __name__ == "__main__":
    main()
