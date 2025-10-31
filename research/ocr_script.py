#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для распознавания текста с изображений
Использует EasyOCR для распознавания печатного и рукописного текста
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
    print("❌ Ошибка: EasyOCR не установлен")
    print("   Установите: pip install easyocr")
    sys.exit(1)


class OCRProcessor:
    """Класс для обработки изображений с помощью различных OCR библиотек"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        self.easyocr_reader = None
        
    def initialize_easyocr(self):
        """Инициализация EasyOCR"""
        if self.easyocr_reader is None:
            print("📥 Инициализация EasyOCR...")
            # EasyOCR поддерживает русский и английский языки
            self.easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=self.use_gpu)
            print("✅ EasyOCR готов к работе")
        return self.easyocr_reader
    
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
    
    def convert_pdf_to_images(self, pdf_path, resolution=5.0):
        """Конвертация PDF в изображения"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Увеличиваем разрешение для лучшего качества OCR
            mat = fitz.Matrix(resolution, resolution)
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


def process_image_file(processor, image_path):
    """Обработка одного изображения"""
    try:
        text = processor.process_with_easyocr(image_path)
        return text
    except Exception as e:
        print(f"❌ Ошибка при обработке {image_path}: {e}")
        return ""


def main():
    parser = argparse.ArgumentParser(description='OCR скрипт для распознавания текста с изображений (EasyOCR)')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU (если поддерживается)')
    parser.add_argument('--input', '-i', 
                       default='input',
                       help='Каталог с входными файлами (по умолчанию: input)')
    parser.add_argument('--output', '-o',
                       default='output', 
                       help='Каталог для выходных файлов (по умолчанию: output)')
    parser.add_argument('--resolution', type=float, default=5.0,
                       help='Разрешение для PDF (по умолчанию: 5.0)')
    
    args = parser.parse_args()
    
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
                print(f"📄 Обрабатываем изображение: {file_path.name}")
                text = process_image_file(processor, str(file_path))
                
                # Сохраняем результат
                output_file = output_dir / f"{file_path.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"✅ Результат сохранен: {output_file}")
                processed_files += 1
                
            elif file_ext in pdf_extensions:
                # Конвертируем PDF в изображения и обрабатываем каждую страницу
                print(f"\n📑 Конвертируем PDF: {file_path.name}")
                images = processor.convert_pdf_to_images(str(file_path), resolution=args.resolution)
                
                all_text = []
                for i, image in enumerate(images):
                    print(f"  📄 Обрабатываем страницу {i+1}/{len(images)}")
                    
                    # Сохраняем изображение во временный файл
                    temp_path = f"/tmp/temp_page_{i}.png"
                    processor.save_image_temp(image, temp_path)
                    
                    # Обрабатываем изображение
                    text = process_image_file(processor, temp_path)
                    if text.strip():
                        all_text.append(f"--- Страница {i+1} ---\n{text}\n")
                    
                    # Удаляем временный файл
                    os.remove(temp_path)
                
                # Сохраняем результат
                output_file = output_dir / f"{file_path.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(all_text))
                
                print(f"✅ Результат сохранен: {output_file}")
                processed_files += 1
    
    print(f"\n{'='*60}")
    print(f"✅ Обработано файлов: {processed_files}")
    print(f"🎮 GPU: {'Да' if args.gpu else 'Нет'}")
    print(f"📐 Разрешение: {args.resolution}x")


if __name__ == "__main__":
    main()
