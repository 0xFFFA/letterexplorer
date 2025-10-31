#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR скрипт на базе TrOCR с поддержкой печатного и рукописного текста
Автоматически определяет тип текста и применяет соответствующую модель
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF для работы с PDF
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

# Проверка наличия библиотек
try:
    from transformers import TrOCRProcessor as HFTrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    print("❌ Ошибка: Установите transformers и torch")
    print("   pip install transformers torch pillow")
    sys.exit(1)


class TextTypeDetector:
    """Детектор типа текста: печатный vs рукописный"""
    
    @staticmethod
    def analyze_text_region(image_region):
        """Анализирует область и определяет тип текста"""
        # Конвертируем в grayscale
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region.copy()
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Признаки для определения типа текста
        
        # 1. Вариация толщины штрихов (рукописный текст более неравномерный)
        edges = cv2.Canny(binary, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 2. Количество контуров (рукописный текст имеет больше мелких деталей)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contour_count = len(contours)
        
        # 3. Соотношение черных и белых пикселей
        black_ratio = np.sum(binary > 0) / binary.size
        
        # 4. Стандартное отклонение интенсивности (рукописный более вариативен)
        std_dev = np.std(gray)
        
        # Эвристика для классификации
        handwritten_score = 0
        
        if edge_density > 0.1:  # Много краев
            handwritten_score += 1
        if contour_count > 50:  # Много мелких контуров
            handwritten_score += 1
        if black_ratio < 0.15:  # Тонкие штрихи
            handwritten_score += 1
        if std_dev > 40:  # Высокая вариативность
            handwritten_score += 1
        
        is_handwritten = handwritten_score >= 2
        
        return {
            'is_handwritten': is_handwritten,
            'confidence': handwritten_score / 4.0,
            'edge_density': edge_density,
            'contour_count': contour_count,
            'std_dev': std_dev
        }


class TableDetector:
    """Детектор таблиц на изображении"""
    
    @staticmethod
    def detect_tables(image):
        """Определяет области таблиц"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Обнаружение горизонтальных и вертикальных линий
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Объединяем линии
        table_mask = cv2.add(horizontal_lines, vertical_lines)
        
        # Находим контуры (потенциальные таблицы)
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        table_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Фильтруем слишком маленькие области
            if w > 200 and h > 100:
                table_regions.append((x, y, w, h))
        
        return table_regions
    
    @staticmethod
    def remove_table_lines(image_region):
        """Удаляет линии таблицы для лучшего OCR"""
        gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Удаляем горизонтальные линии
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        remove_horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Удаляем вертикальные линии
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        remove_vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Убираем линии из оригинала
        result = binary.copy()
        result = cv2.subtract(result, remove_horizontal)
        result = cv2.subtract(result, remove_vertical)
        
        # Инвертируем обратно
        result = cv2.bitwise_not(result)
        
        return result


class TrOCRProcessor:
    """Обработчик на базе TrOCR"""
    
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        print(f"🔧 Устройство: {self.device.upper()}")
        
        # Модели
        self.printed_processor = None
        self.printed_model = None
        self.handwritten_processor = None
        self.handwritten_model = None
        
        self.text_detector = TextTypeDetector()
        self.table_detector = TableDetector()
    
    def initialize_printed_model(self):
        """Инициализация модели для печатного текста"""
        if self.printed_processor is None:
            print("📥 Загрузка модели для печатного текста...")
            # Используем базовую модель TrOCR для печатного текста
            self.printed_processor = HFTrOCRProcessor.from_pretrained('microsoft/trocr-base-printed')
            self.printed_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-printed')
            self.printed_model.to(self.device)
            self.printed_model.eval()
            print("✅ Модель для печатного текста загружена")
        return self.printed_processor, self.printed_model
    
    def initialize_handwritten_model(self):
        """Инициализация модели для рукописного текста"""
        if self.handwritten_processor is None:
            print("📥 Загрузка модели для рукописного текста...")
            # Используем модель для рукописного текста
            self.handwritten_processor = HFTrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
            self.handwritten_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
            self.handwritten_model.to(self.device)
            self.handwritten_model.eval()
            print("✅ Модель для рукописного текста загружена")
        return self.handwritten_processor, self.handwritten_model
    
    def preprocess_image(self, image, enhance=True):
        """Предобработка изображения"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if enhance:
            # Увеличение контраста
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Убираем шум
            denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
            
            return denoised
        
        return gray
    
    def segment_text_lines(self, image):
        """Сегментирует изображение на строки текста"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Бинаризация
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Проекция по горизонтали для поиска строк
        horizontal_projection = np.sum(binary, axis=1)
        
        # Находим границы строк
        threshold = np.max(horizontal_projection) * 0.1
        in_text = False
        lines = []
        start_y = 0
        
        for i, value in enumerate(horizontal_projection):
            if not in_text and value > threshold:
                in_text = True
                start_y = i
            elif in_text and value < threshold:
                in_text = False
                if i - start_y > 10:  # Минимальная высота строки
                    lines.append((start_y, i))
        
        return lines
    
    def recognize_line(self, image, line_coords, is_handwritten=False):
        """Распознает одну строку текста"""
        y_start, y_end = line_coords
        # Добавляем отступы
        y_start = max(0, y_start - 5)
        y_end = min(image.shape[0], y_end + 5)
        
        line_img = image[y_start:y_end, :]
        
        # Конвертируем в PIL Image
        if len(line_img.shape) == 2:
            pil_img = Image.fromarray(line_img).convert('RGB')
        else:
            pil_img = Image.fromarray(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
        
        # Выбираем модель
        if is_handwritten:
            processor, model = self.initialize_handwritten_model()
        else:
            processor, model = self.initialize_printed_model()
        
        # Обработка
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)
        
        # Генерация текста
        with torch.no_grad():
            generated_ids = model.generate(pixel_values)
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    
    def process_image(self, image_path, detect_tables=True, auto_detect_handwritten=True):
        """Обработка изображения с автоматическим определением типа текста"""
        print(f"\n📄 Обрабатываем: {image_path}")
        
        # Читаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        all_text = []
        
        # Определяем таблицы
        table_regions = []
        if detect_tables:
            table_regions = self.table_detector.detect_tables(image)
            if table_regions:
                print(f"  📊 Найдено таблиц: {len(table_regions)}")
        
        # Создаем маску для областей вне таблиц
        mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
        for x, y, w, h in table_regions:
            mask[y:y+h, x:x+w] = 0
        
        # Обрабатываем обычный текст (вне таблиц)
        text_image = cv2.bitwise_and(image, image, mask=mask)
        
        # Предобработка
        preprocessed = self.preprocess_image(text_image, enhance=True)
        
        # Сегментация на строки
        lines = self.segment_text_lines(preprocessed)
        print(f"  📝 Найдено строк текста: {len(lines)}")
        
        # Обрабатываем каждую строку
        for idx, line_coords in enumerate(lines):
            y_start, y_end = line_coords
            line_region = preprocessed[y_start:y_end, :]
            
            # Определяем тип текста
            if auto_detect_handwritten and line_region.shape[0] > 10 and line_region.shape[1] > 10:
                text_info = self.text_detector.analyze_text_region(line_region)
                is_handwritten = text_info['is_handwritten']
                
                if is_handwritten:
                    print(f"    ✍️  Строка {idx+1}: рукописный текст (уверенность: {text_info['confidence']:.2f})")
            else:
                is_handwritten = False
            
            # Распознаем строку
            try:
                text = self.recognize_line(preprocessed, line_coords, is_handwritten=is_handwritten)
                if text.strip():
                    all_text.append(text.strip())
            except Exception as e:
                print(f"    ⚠️  Ошибка при распознавании строки {idx+1}: {e}")
        
        # Обрабатываем таблицы
        for idx, (x, y, w, h) in enumerate(table_regions):
            print(f"  📊 Обработка таблицы {idx+1}...")
            table_img = image[y:y+h, x:x+w]
            
            # Удаляем линии таблицы
            table_no_lines = self.table_detector.remove_table_lines(table_img)
            
            # Сегментируем таблицу на строки
            table_lines = self.segment_text_lines(table_no_lines)
            
            table_text = ["\n=== ТАБЛИЦА ==="]
            for line_coords in table_lines:
                try:
                    text = self.recognize_line(table_no_lines, line_coords, is_handwritten=False)
                    if text.strip():
                        table_text.append(text.strip())
                except Exception as e:
                    pass
            
            table_text.append("=== КОНЕЦ ТАБЛИЦЫ ===\n")
            all_text.extend(table_text)
        
        return '\n'.join(all_text)
    
    def convert_pdf_to_images(self, pdf_path, resolution=5.0):
        """Конвертация PDF в изображения"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            mat = fitz.Matrix(resolution, resolution)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            images.append(img)
        
        doc.close()
        return images


def main():
    parser = argparse.ArgumentParser(description='OCR с TrOCR для печатного и рукописного текста')
    parser.add_argument('--input', '-i', 
                       default='input',
                       help='Каталог с входными файлами')
    parser.add_argument('--output', '-o',
                       default='output_trocr', 
                       help='Каталог для выходных файлов')
    parser.add_argument('--no-gpu', action='store_true',
                       help='Не использовать GPU')
    parser.add_argument('--no-tables', action='store_true',
                       help='Отключить определение таблиц')
    parser.add_argument('--no-handwritten', action='store_true',
                       help='Отключить автоматическое определение рукописного текста')
    parser.add_argument('--resolution', type=float, default=5.0,
                       help='Разрешение для PDF (по умолчанию: 5.0)')
    
    args = parser.parse_args()
    
    # Создаем каталоги
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"❌ Ошибка: Каталог {input_dir} не существует")
        sys.exit(1)
    
    # Инициализируем процессор
    processor = TrOCRProcessor(use_gpu=not args.no_gpu)
    
    # Поддерживаемые форматы
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    pdf_extensions = {'.pdf'}
    
    # Обрабатываем файлы
    processed_files = 0
    
    for file_path in sorted(input_dir.iterdir()):
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            if file_ext in image_extensions:
                try:
                    text = processor.process_image(
                        str(file_path),
                        detect_tables=not args.no_tables,
                        auto_detect_handwritten=not args.no_handwritten
                    )
                    
                    output_file = output_dir / f"{file_path.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(text)
                    
                    print(f"✅ Результат сохранен: {output_file}")
                    processed_files += 1
                except Exception as e:
                    print(f"❌ Ошибка при обработке {file_path}: {e}")
                
            elif file_ext in pdf_extensions:
                try:
                    print(f"\n📑 Конвертируем PDF: {file_path}")
                    images = processor.convert_pdf_to_images(str(file_path), resolution=args.resolution)
                    
                    all_text = []
                    for i, image in enumerate(images):
                        print(f"\n📄 Страница {i+1}/{len(images)}")
                        
                        # Сохраняем во временный файл
                        temp_path = f"/tmp/temp_trocr_page_{i}.png"
                        cv2.imwrite(temp_path, image)
                        
                        # Обрабатываем
                        text = processor.process_image(
                            temp_path,
                            detect_tables=not args.no_tables,
                            auto_detect_handwritten=not args.no_handwritten
                        )
                        
                        if text.strip():
                            all_text.append(f"--- Страница {i+1} ---\n{text}\n")
                        
                        # Удаляем временный файл
                        os.remove(temp_path)
                    
                    # Сохраняем результат
                    output_file = output_dir / f"{file_path.stem}.txt"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(all_text))
                    
                    print(f"\n✅ Результат сохранен: {output_file}")
                    processed_files += 1
                except Exception as e:
                    print(f"\n❌ Ошибка при обработке {file_path}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Обработано файлов: {processed_files}")
    print(f"📊 Определение таблиц: {'Включено' if not args.no_tables else 'Отключено'}")
    print(f"✍️  Автоопределение рукописного: {'Включено' if not args.no_handwritten else 'Отключено'}")
    print(f"🎮 GPU: {'Да' if not args.no_gpu and torch.cuda.is_available() else 'Нет'}")


if __name__ == "__main__":
    main()

