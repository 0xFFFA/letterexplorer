#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для распознавания текста с изображений
Поддерживает EasyOCR, PaddleOCR и Tesseract с модулем для рукописного текста
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF для работы с PDF

# Импорты OCR библиотек
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Предупреждение: EasyOCR не установлен")

# PaddleOCR удален из-за нестабильной работы

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Предупреждение: Tesseract не установлен")


class OCRProcessor:
    """Класс для обработки изображений с помощью различных OCR библиотек"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.easyocr_reader = None
        
    def initialize_easyocr(self):
        """Инициализация EasyOCR"""
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR не установлен")
        
        if self.easyocr_reader is None:
            # EasyOCR поддерживает русский язык
            self.easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=self.use_gpu)
        return self.easyocr_reader
    
    # PaddleOCR удален из-за нестабильной работы
    
    def process_with_easyocr(self, image_path):
        """Обработка изображения с помощью EasyOCR"""
        reader = self.initialize_easyocr()
        
        # Читаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Распознавание текста
        results = reader.readtext(image)
        
        # Извлекаем текст
        text_lines = []
        for (bbox, text, confidence) in results:
            if confidence > 0.5:  # Фильтруем по уверенности
                text_lines.append(text)
        
        return '\n'.join(text_lines)
    
    # PaddleOCR удален из-за нестабильной работы
    
    def process_with_tesseract(self, image_path):
        """Обработка изображения с помощью Tesseract"""
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract не установлен")
        
        # Читаем изображение
        image = Image.open(image_path)
        
        # Конфигурация для лучшего распознавания рукописного текста
        # Используем русский язык и специальные настройки
        custom_config = r'--oem 3 --psm 6 -l rus+eng'
        
        # Распознавание текста
        text = pytesseract.image_to_string(image, config=custom_config)
        
        return text
    
    def convert_pdf_to_images(self, pdf_path):
        """Конвертация PDF в изображения"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Увеличиваем разрешение для лучшего качества OCR
            mat = fitz.Matrix(2.0, 2.0)  # 2x увеличение
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            # Конвертируем в numpy array для OpenCV
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(img)
        
        doc.close()
        return images
    
    def save_image_temp(self, image, temp_path):
        """Сохранение изображения во временный файл"""
        cv2.imwrite(temp_path, image)
        return temp_path


def process_image_file(processor, image_path, ocr_method):
    """Обработка одного изображения"""
    print(f"Обрабатываем {image_path} с помощью {ocr_method}")
    
    try:
        if ocr_method.lower() == 'easyocr':
            text = processor.process_with_easyocr(image_path)
        elif ocr_method.lower() == 'tesseract':
            text = processor.process_with_tesseract(image_path)
        else:
            raise ValueError(f"Неподдерживаемый метод OCR: {ocr_method}")
        
        return text
    except Exception as e:
        print(f"Ошибка при обработке {image_path}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description='OCR скрипт для распознавания текста с изображений')
    parser.add_argument('--method', '-m', 
                       choices=['easyocr', 'tesseract'],
                       required=True,
                       help='Выбор OCR библиотеки')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU (если поддерживается)')
    parser.add_argument('--input', '-i', 
                       default='input',
                       help='Каталог с входными файлами (по умолчанию: input)')
    parser.add_argument('--output', '-o',
                       default='output', 
                       help='Каталог для выходных файлов (по умолчанию: output)')
    
    args = parser.parse_args()
    
    # Проверяем доступность выбранной библиотеки
    if args.method == 'easyocr' and not EASYOCR_AVAILABLE:
        print("Ошибка: EasyOCR не установлен")
        sys.exit(1)
    elif args.method == 'tesseract' and not TESSERACT_AVAILABLE:
        print("Ошибка: Tesseract не установлен")
        sys.exit(1)
    
    # Создаем каталоги
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"Ошибка: Каталог {input_dir} не существует")
        sys.exit(1)
    
    # Инициализируем процессор
    processor = OCRProcessor(use_gpu=args.gpu)
    
    # Поддерживаемые форматы
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    pdf_extensions = {'.pdf'}
    
    # Обрабатываем файлы
    processed_files = 0
    
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            if file_ext in image_extensions:
                # Обрабатываем изображение напрямую
                text = process_image_file(processor, str(file_path), args.method)
                
                # Сохраняем результат
                output_file = output_dir / f"{file_path.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"Результат сохранен: {output_file}")
                processed_files += 1
                
            elif file_ext in pdf_extensions:
                # Конвертируем PDF в изображения и обрабатываем каждую страницу
                print(f"Конвертируем PDF: {file_path}")
                images = processor.convert_pdf_to_images(str(file_path))
                
                all_text = []
                for i, image in enumerate(images):
                    # Сохраняем изображение во временный файл
                    temp_path = f"/tmp/temp_page_{i}.png"
                    processor.save_image_temp(image, temp_path)
                    
                    # Обрабатываем изображение
                    text = process_image_file(processor, temp_path, args.method)
                    if text.strip():
                        all_text.append(f"--- Страница {i+1} ---\n{text}\n")
                    
                    # Удаляем временный файл
                    os.remove(temp_path)
                
                # Сохраняем результат
                output_file = output_dir / f"{file_path.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(all_text))
                
                print(f"Результат сохранен: {output_file}")
                processed_files += 1
    
    print(f"\nОбработано файлов: {processed_files}")
    print(f"Использованный метод: {args.method}")
    print(f"GPU: {'Да' if args.gpu else 'Нет'}")


if __name__ == "__main__":
    main()
