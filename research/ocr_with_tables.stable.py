#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Улучшенный OCR скрипт с обработкой таблиц
Использует EasyOCR с предобработкой изображений для таблиц
"""

import argparse
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import fitz  # PyMuPDF для работы с PDF

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("Ошибка: EasyOCR не установлен")
    sys.exit(1)


class TableDetector:
    """Детектор и обработчик таблиц"""
    
    @staticmethod
    def preprocess_for_table(image):
        """Предобработка изображения для улучшения распознавания таблиц"""
        # Конвертируем в grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Увеличиваем контраст
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Бинаризация с адаптивным порогом
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Убираем шум
        denoised = cv2.fastNlMeansDenoising(binary, None, 10, 7, 21)
        
        return denoised
    
    @staticmethod
    def detect_table_regions(image):
        """Определяет области таблиц на изображении"""
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
            if w > 100 and h > 100:
                table_regions.append((x, y, w, h))
        
        return table_regions
    
    @staticmethod
    def extract_table_cells(image, table_region):
        """Извлекает ячейки из таблицы"""
        x, y, w, h = table_region
        table_img = image[y:y+h, x:x+w]
        
        gray = cv2.cvtColor(table_img, cv2.COLOR_BGR2GRAY) if len(table_img.shape) == 3 else table_img
        _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
        
        # Детектируем горизонтальные и вертикальные линии
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
        
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        
        # Объединяем линии
        grid = cv2.add(horizontal, vertical)
        
        # Находим пересечения (углы ячеек)
        contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        cells = []
        for contour in contours:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            if w_c > 20 and h_c > 20:  # Минимальный размер ячейки
                cells.append((x + x_c, y + y_c, w_c, h_c))
        
        return sorted(cells, key=lambda c: (c[1], c[0]))  # Сортируем по строкам, затем столбцам


class OCRProcessor:
    """Класс для обработки изображений с помощью EasyOCR"""
    
    def __init__(self, use_gpu=False, detect_tables=True):
        self.use_gpu = use_gpu
        self.detect_tables = detect_tables
        self.reader = None
        self.table_detector = TableDetector()
        
    def initialize_easyocr(self):
        """Инициализация EasyOCR"""
        if self.reader is None:
            print("Инициализация EasyOCR...")
            self.reader = easyocr.Reader(['ru', 'en'], gpu=self.use_gpu)
        return self.reader
    
    def process_table_region(self, image, table_region):
        """Обработка области таблицы с улучшенной предобработкой"""
        x, y, w, h = table_region
        table_img = image[y:y+h, x:x+w]
        
        # Предобработка для таблиц
        preprocessed = self.table_detector.preprocess_for_table(table_img)
        
        # OCR на предобработанном изображении
        reader = self.initialize_easyocr()
        results = reader.readtext(preprocessed, paragraph=False)
        
        # Сортируем результаты по позиции (сверху вниз, слева направо)
        results_sorted = sorted(results, key=lambda r: (r[0][0][1], r[0][0][0]))
        
        # Форматируем как таблицу
        table_text = ["\n=== ТАБЛИЦА ==="]
        current_row = []
        prev_y = None
        
        for bbox, text, confidence in results_sorted:
            if confidence > 0.3:  # Более низкий порог для таблиц
                y_pos = bbox[0][1]
                
                # Новая строка таблицы
                if prev_y is not None and abs(y_pos - prev_y) > 20:
                    if current_row:
                        table_text.append(" | ".join(current_row))
                        current_row = []
                
                current_row.append(text)
                prev_y = y_pos
        
        if current_row:
            table_text.append(" | ".join(current_row))
        
        table_text.append("=== КОНЕЦ ТАБЛИЦЫ ===\n")
        
        return "\n".join(table_text)
    
    def process_with_easyocr(self, image_path):
        """Обработка изображения с автоматическим определением таблиц"""
        reader = self.initialize_easyocr()
        
        # Читаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        all_text = []
        
        if self.detect_tables:
            # Определяем области таблиц
            table_regions = self.table_detector.detect_table_regions(image)
            
            if table_regions:
                print(f"  Найдено таблиц: {len(table_regions)}")
                
                # Создаем маску для нетабличных областей
                mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
                for x, y, w, h in table_regions:
                    mask[y:y+h, x:x+w] = 0
                
                # OCR для обычного текста (вне таблиц)
                masked_image = cv2.bitwise_and(image, image, mask=mask)
                results = reader.readtext(masked_image)
                
                text_results = []
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:
                        text_results.append((bbox[0][1], text))  # Сохраняем Y-координату
                
                # OCR для таблиц
                table_results = []
                for idx, region in enumerate(table_regions):
                    table_text = self.process_table_region(image, region)
                    table_results.append((region[1], table_text))  # Y-координата таблицы
                
                # Объединяем результаты в правильном порядке
                all_results = text_results + table_results
                all_results.sort(key=lambda r: r[0])
                
                all_text = [text for _, text in all_results]
            else:
                # Если таблиц не найдено, обычный OCR
                results = reader.readtext(image)
                for (bbox, text, confidence) in results:
                    if confidence > 0.5:
                        all_text.append(text)
        else:
            # Обычный OCR без обработки таблиц
            results = reader.readtext(image)
            for (bbox, text, confidence) in results:
                if confidence > 0.5:
                    all_text.append(text)
        
        return '\n'.join(all_text)
    
    def convert_pdf_to_images(self, pdf_path):
        """Конвертация PDF в изображения"""
        doc = fitz.open(pdf_path)
        images = []
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            # Увеличиваем разрешение для лучшего качества OCR
            mat = fitz.Matrix(3.0, 3.0)  # 3x увеличение для таблиц
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


def main():
    parser = argparse.ArgumentParser(description='OCR скрипт с улучшенной обработкой таблиц')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU')
    parser.add_argument('--no-tables', action='store_true',
                       help='Отключить автоматическое определение таблиц')
    parser.add_argument('--input', '-i', 
                       default='input',
                       help='Каталог с входными файлами (по умолчанию: input)')
    parser.add_argument('--output', '-o',
                       default='output_tables', 
                       help='Каталог для выходных файлов (по умолчанию: output_tables)')
    
    args = parser.parse_args()
    
    # Создаем каталоги
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    if not input_dir.exists():
        print(f"Ошибка: Каталог {input_dir} не существует")
        sys.exit(1)
    
    # Инициализируем процессор
    processor = OCRProcessor(use_gpu=args.gpu, detect_tables=not args.no_tables)
    
    # Поддерживаемые форматы
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    pdf_extensions = {'.pdf'}
    
    # Обрабатываем файлы
    processed_files = 0
    
    for file_path in input_dir.iterdir():
        if file_path.is_file():
            file_ext = file_path.suffix.lower()
            
            if file_ext in image_extensions:
                print(f"\nОбрабатываем изображение: {file_path}")
                text = processor.process_with_easyocr(str(file_path))
                
                # Сохраняем результат
                output_file = output_dir / f"{file_path.stem}.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(text)
                
                print(f"Результат сохранен: {output_file}")
                processed_files += 1
                
            elif file_ext in pdf_extensions:
                print(f"\nКонвертируем PDF: {file_path}")
                images = processor.convert_pdf_to_images(str(file_path))
                
                all_text = []
                for i, image in enumerate(images):
                    print(f"  Обрабатываем страницу {i+1}/{len(images)}")
                    
                    # Сохраняем изображение во временный файл
                    temp_path = f"/tmp/temp_page_{i}.png"
                    processor.save_image_temp(image, temp_path)
                    
                    # Обрабатываем изображение
                    text = processor.process_with_easyocr(temp_path)
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
    
    print(f"\n{'='*50}")
    print(f"Обработано файлов: {processed_files}")
    print(f"Определение таблиц: {'Включено' if not args.no_tables else 'Отключено'}")
    print(f"GPU: {'Да' if args.gpu else 'Нет'}")


if __name__ == "__main__":
    main()

