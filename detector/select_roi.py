#!/usr/bin/env python3
"""
Интерактивный инструмент для выбора области интереса (ROI).
Позволяет захватить кадр с RTMP потока и выбрать область кормушки мышкой.

Использование:
    python select_roi.py [--rtmp URL] [--image PATH] [--config PATH]

Примеры:
    python select_roi.py                           # Захват с RTMP по умолчанию
    python select_roi.py --image frame.jpg         # Использовать изображение
    python select_roi.py --rtmp rtmp://host/live   # Указать RTMP URL
"""

import cv2
import numpy as np
import argparse
import os
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


def update_config_file(config_path: str, roi: tuple) -> bool:
    """
    Обновить файл конфигурации с новыми ROI параметрами.
    
    Args:
        config_path: Путь к config.env
        roi: Кортеж (x, y, width, height)
    
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
    roi_params = {
        'ROI_ENABLED': 'true',
        'ROI_X': str(x),
        'ROI_Y': str(y),
        'ROI_WIDTH': str(w),
        'ROI_HEIGHT': str(h)
    }
    
    # Обновляем существующие или помечаем для добавления
    updated_keys = set()
    new_lines = []
    
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and '=' in stripped:
            key = stripped.split('=', 1)[0].strip()
            if key in roi_params:
                new_lines.append(f"{key}={roi_params[key]}\n")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    
    # Добавляем недостающие параметры
    missing_keys = set(roi_params.keys()) - updated_keys
    if missing_keys:
        # Проверяем есть ли уже секция ROI
        has_roi_section = any('ROI' in line and '===' in line for line in new_lines)
        
        if not has_roi_section:
            new_lines.append("\n# === ROI (Region of Interest) - область кормушки ===\n")
            new_lines.append("# Координаты выбраны через select_roi.py\n")
        
        for key in ['ROI_ENABLED', 'ROI_X', 'ROI_Y', 'ROI_WIDTH', 'ROI_HEIGHT']:
            if key in missing_keys:
                new_lines.append(f"{key}={roi_params[key]}\n")
    
    # Записываем
    try:
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        print(f"✅ Конфигурация сохранена в {config_path}")
        return True
    except Exception as e:
        print(f"❌ Ошибка записи конфигурации: {e}")
        return False


def save_frame(frame: np.ndarray, output_path: str):
    """Сохранить кадр в файл."""
    cv2.imwrite(output_path, frame)
    print(f"💾 Кадр сохранён: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Интерактивный выбор области интереса (ROI) для кормушки'
    )
    parser.add_argument(
        '--rtmp', '-r',
        default='rtmp://nginx-rtmp/live',
        help='URL RTMP потока (по умолчанию: rtmp://nginx-rtmp/live)'
    )
    parser.add_argument(
        '--image', '-i',
        help='Путь к изображению (вместо захвата с RTMP)'
    )
    parser.add_argument(
        '--config', '-c',
        default='/app/config.env',
        help='Путь к config.env (по умолчанию: /app/config.env)'
    )
    parser.add_argument(
        '--save-frame', '-s',
        help='Сохранить захваченный кадр в файл'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Не сохранять в config.env (только показать координаты)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 Инструмент выбора ROI (Region of Interest)")
    print("=" * 60)
    
    # Получаем кадр
    if args.image:
        frame = load_frame_from_file(args.image)
    else:
        frame = capture_frame_from_rtmp(args.rtmp)
    
    if frame is None:
        print("\n💡 Совет: Убедитесь что GoPro стримит на RTMP сервер")
        print("   Или используйте --image для загрузки существующего изображения")
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
    print("📋 Координаты ROI:")
    print(f"   ROI_X={x}")
    print(f"   ROI_Y={y}")
    print(f"   ROI_WIDTH={w}")
    print(f"   ROI_HEIGHT={h}")
    print("=" * 60)
    
    # Сохраняем в конфиг
    if not args.no_save:
        if update_config_file(args.config, roi):
            print("\n✅ ROI настроен! Перезапустите детектор для применения.")
            print("   Команда: docker-compose restart detector")
        else:
            print("\n⚠️  Не удалось сохранить в конфиг. Добавьте параметры вручную.")
    else:
        print("\n📝 Добавьте эти параметры в config.env вручную")
    
    print()


if __name__ == "__main__":
    main()
