#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR скрипт с использованием EasyOCR для лучшего распознавания таблиц.
Таблицы представляются в Markdown формате.
"""

import argparse
import sys
from pathlib import Path
from pdf2image import convert_from_path
from img2table.document import Image as Img2TableImage
from img2table.ocr import EasyOCR
import pytesseract
from PIL import Image
import pandas as pd
from typing import List, Dict, Any

class EasyOCRProcessor:
    """Обработчик PDF с использованием EasyOCR"""
    
    def __init__(self, pdf_path: Path, output_dir: Path, dpi: int = 400, use_gpu: bool = False, min_confidence: int = 30):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.dpi = dpi
        self.use_gpu = use_gpu
        self.min_confidence = min_confidence
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Инициализируем EasyOCR один раз
        gpu_status = "GPU" if use_gpu else "CPU"
        print(f"🔧 Инициализируем EasyOCR ({gpu_status})...")
        self.ocr = EasyOCR(lang=["ru", "en"], kw={"gpu": use_gpu})
    
    def convert_pdf_to_images(self) -> List[Image.Image]:
        """Конвертирует PDF в изображения"""
        print(f"📄 Конвертируем PDF в изображения (DPI={self.dpi})...")
        try:
            images = convert_from_path(
                str(self.pdf_path),
                dpi=self.dpi,
                fmt='png'
            )
            print(f"✅ Получено {len(images)} страниц")
            return images
        except Exception as e:
            print(f"❌ Ошибка при конвертации PDF: {e}")
            sys.exit(1)
    
    def extract_tables_from_image(self, image_path: Path) -> List[Dict[str, Any]]:
        """Извлекает таблицы из изображения"""
        try:
            # detect_rotation=True для автоматического исправления наклона
            img_doc = Img2TableImage(src=str(image_path), detect_rotation=True)
            tables = img_doc.extract_tables(
                ocr=self.ocr,
                implicit_rows=True,
                borderless_tables=True,
                min_confidence=self.min_confidence
            )
            
            result = []
            for table in tables:
                result.append({
                    'df': table.df,
                    'bbox': table.bbox
                })
            
            return result
        except Exception as e:
            print(f"⚠️  Ошибка при извлечении таблиц: {e}")
            return []
    
    def extract_text_from_image(self, image_path: Path) -> str:
        """Извлекает весь текст со страницы используя Tesseract"""
        try:
            # Используем pytesseract для обычного текста (быстрее чем EasyOCR)
            text = pytesseract.image_to_string(
                str(image_path),
                lang='rus+eng',
                config='--psm 6'
            )
            return text.strip()
        except Exception as e:
            print(f"⚠️  Ошибка при извлечении текста: {e}")
            return ""
    
    def dataframe_to_markdown(self, df: pd.DataFrame) -> str:
        """Конвертирует DataFrame в Markdown таблицу"""
        lines = []
        
        # Заголовок
        headers = []
        for col in df.columns:
            # Обрабатываем пустые заголовки
            header = str(col) if col is not None and str(col) != 'None' else ''
            headers.append(header)
        
        lines.append('| ' + ' | '.join(headers) + ' |')
        lines.append('|' + '|'.join(['---' for _ in headers]) + '|')
        
        # Данные
        for _, row in df.iterrows():
            cells = []
            for val in row:
                # Обрабатываем пустые ячейки
                if pd.isna(val) or val is None or str(val) == 'None':
                    cell = ''
                else:
                    cell = str(val).replace('\n', ' ').strip()
                cells.append(cell)
            lines.append('| ' + ' | '.join(cells) + ' |')
        
        return '\n'.join(lines)
    
    def process_page(self, page_num: int, image: Image.Image, table_counter: int) -> tuple[str, int]:
        """Обрабатывает одну страницу. Возвращает (текст_страницы, обновленный_счетчик_таблиц)"""
        print(f"\n{'='*70}")
        print(f"📄 Обработка страницы {page_num}")
        print(f"{'='*70}")
        
        # Сохраняем изображение страницы
        image_path = self.output_dir / f"page_{page_num}.png"
        image.save(image_path, "PNG")
        
        result_parts = []
        result_parts.append(f"\n{'='*70}")
        result_parts.append(f"СТРАНИЦА {page_num}")
        result_parts.append(f"{'='*70}\n")
        
        # Извлекаем таблицы
        print("🔍 Поиск таблиц...")
        tables = self.extract_tables_from_image(image_path)
        
        if tables:
            print(f"✅ Найдено таблиц: {len(tables)}")
        else:
            print("ℹ️  Таблицы не найдены")
        
        # Извлекаем текст
        print("📝 Извлечение текста...")
        text = self.extract_text_from_image(image_path)
        
        if text:
            result_parts.append("## Текст\n")
            result_parts.append(text)
            result_parts.append("\n")
        
        # Добавляем таблицы со сквозной нумерацией
        if tables:
            result_parts.append(f"\n## Таблицы (найдено: {len(tables)})\n")
            
            for table_info in tables:
                df = table_info['df']
                bbox = table_info['bbox']
                
                # Инкрементируем глобальный счетчик
                table_counter += 1
                
                print(f"📊 Таблица {table_counter}: {df.shape[0]}×{df.shape[1]} (позиция: {bbox})")
                
                result_parts.append(f"\n### Таблица {table_counter}")
                result_parts.append(f"Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")
                result_parts.append(f"Позиция: {bbox}\n")
                
                # Конвертируем в Markdown
                markdown_table = self.dataframe_to_markdown(df)
                result_parts.append(markdown_table)
                result_parts.append("")
                
                # Сохраняем также в CSV со сквозной нумерацией
                csv_path = self.output_dir / f"table{table_counter}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"   💾 CSV: {csv_path}")
        
        return '\n'.join(result_parts), table_counter
    
    def process(self) -> str:
        """Обрабатывает весь PDF"""
        print(f"\n{'='*70}")
        print(f"🚀 ОБРАБОТКА PDF: {self.pdf_path.name}")
        print(f"{'='*70}\n")
        
        # Конвертируем PDF в изображения
        images = self.convert_pdf_to_images()
        
        # Обрабатываем каждую страницу со сквозной нумерацией таблиц
        all_pages = []
        table_counter = 0  # Глобальный счетчик таблиц
        
        for page_num, image in enumerate(images, start=1):
            page_text, table_counter = self.process_page(page_num, image, table_counter)
            all_pages.append(page_text)
        
        # Объединяем всё
        final_text = '\n\n'.join(all_pages)
        
        return final_text
    
    def save_result(self, text: str) -> Path:
        """Сохраняет результат в текстовый файл"""
        output_file = self.output_dir / f"{self.pdf_path.stem}.txt"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"\n{'='*70}")
        print(f"✅ РЕЗУЛЬТАТ СОХРАНЁН")
        print(f"{'='*70}")
        print(f"📝 Текстовый файл: {output_file}")
        print(f"📂 Дополнительные файлы: {self.output_dir}/")
        
        return output_file


def main():
    parser = argparse.ArgumentParser(
        description='OCR с использованием EasyOCR для лучшего распознавания таблиц',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s input/document.pdf
  %(prog)s input/document.pdf --output output/easyocr/
  %(prog)s input/document.pdf --dpi 400
  %(prog)s input/document.pdf --gpu  # использовать GPU для ускорения
        """
    )
    
    parser.add_argument(
        'pdf_file',
        type=str,
        help='Путь к PDF файлу'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Папка для сохранения результатов (по умолчанию: output/[имя_файла]_easyocr/)'
    )
    
    parser.add_argument(
        '--dpi',
        type=int,
        default=400,
        help='DPI для конвертации PDF в изображения (по умолчанию: 400)'
    )
    
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='Использовать GPU для EasyOCR (по умолчанию: CPU)'
    )
    
    parser.add_argument(
        '--min-confidence',
        type=int,
        default=30,
        help='Минимальная уверенность OCR для ячейки таблицы (по умолчанию: 30)'
    )
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    pdf_path = Path(args.pdf_file)
    if not pdf_path.exists():
        print(f"❌ Файл не найден: {pdf_path}")
        sys.exit(1)
    
    if pdf_path.suffix.lower() != '.pdf':
        print(f"❌ Файл должен быть в формате PDF: {pdf_path}")
        sys.exit(1)
    
    # Определяем папку для результатов
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path('output') / f"{pdf_path.stem}_easyocr"
    
    # Обрабатываем PDF
    processor = EasyOCRProcessor(
        pdf_path, 
        output_dir, 
        dpi=args.dpi, 
        use_gpu=args.gpu,
        min_confidence=args.min_confidence
    )
    text = processor.process()
    output_file = processor.save_result(text)
    
    print(f"\n✨ Готово!")


if __name__ == '__main__':
    main()

