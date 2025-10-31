#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор .markup файлов для разметки структуры документа
Автоматически находит потенциальные метаданные, параметры, таблицы и разделы
"""

import re
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional


class MarkupGenerator:
    """Генерирует черновик .markup файла из текста"""
    
    def __init__(self, text: str):
        self.text = text
        self.lines = text.split('\n')
        self.markup = []
    
    def generate(self) -> str:
        """Генерирует полный .markup файл"""
        self.markup = []
        
        self._add_header()
        self._detect_metadata()
        self._detect_parameters()
        self._detect_tables()
        self._detect_sections()
        self._add_footer()
        
        return '\n'.join(self.markup)
    
    def _add_header(self):
        """Добавляет заголовок"""
        self.markup.extend([
            "# ============================================",
            "# ФАЙЛ РАЗМЕТКИ (автосгенерирован)",
            "# ============================================",
            "#",
            "# ИНСТРУКЦИЯ:",
            "# 1. Проверьте строки с меткой [ПРОВЕРЬ]",
            "# 2. Удалите ненужные строки",
            "# 3. Добавьте недостающие элементы",
            "# 4. Запустите: python3 markup_parser.py --markup файл.markup --input файл.txt",
            "#",
            "# СИНТАКСИС:",
            "# текст { тип }                      - точное совпадение",
            "# текст { ключ: значение }           - пара ключ-значение (точное)",
            "# @REGEX паттерн { тип }             - regex-паттерн (для переменных значений)",
            "# @REGEX паттерн { ключ: группа }    - regex с захватом группы",
            "# [TABLE name] ... [END]             - таблица",
            "# [SECTION name] ... [END]           - раздел",
            "#",
            "# ПРИМЕРЫ REGEX:",
            r"# @REGEX \d{2}-\s*\d{4} { document_number }     - номер документа",
            r"# @REGEX \d{2}\.\d{4} { date }                  - дата (ММ.ГГГГ)",
            r"# @REGEX Начальнику\s+\S+ { addressee }         - адресат",
            "#",
            ""
        ])
    
    def _detect_metadata(self):
        """Находит метаданные в начале документа"""
        self.markup.extend([
            "# ============================================",
            "# МЕТАДАННЫЕ ДОКУМЕНТА",
            "# ============================================",
            "[META]",
            ""
        ])
        
        # Ищем номер документа (формат: ЦЦ- ЦЦЦЦ) - используем regex для переиспользования
        doc_num_found = False
        for i, line in enumerate(self.lines[:20]):
            line = line.strip()
            if re.match(r'^\d{2}-\s*\d{3,4}$', line):
                self.markup.append(f"# Номер документа (regex для переиспользования на других документах):")
                self.markup.append(r"@REGEX \d{2}-\s*\d{3,4} { document_number }")
                self.markup.append(f"# Пример из текущего документа: {line}")
                doc_num_found = True
                break
        
        if not doc_num_found:
            self.markup.append("# [ПРОВЕРЬ] Номер документа не найден автоматически")
        
        self.markup.append("")
        
        # Ищем дату (формат: ММ.ГГГГ) - используем regex
        date_found = False
        for i, line in enumerate(self.lines[:30]):
            line = line.strip()
            if re.match(r'^\d{2}\.\d{4}$', line):
                self.markup.append(f"# Дата (regex для переиспользования):")
                self.markup.append(r"@REGEX \d{2}\.\d{4} { date }")
                self.markup.append(f"# Пример из текущего документа: {line}")
                date_found = True
                break
        
        if not date_found:
            self.markup.append("# [ПРОВЕРЬ] Дата не найдена автоматически")
        
        self.markup.append("")
        
        # Ищем адресатов (строки "Начальнику ...") - используем regex для всех сразу
        addressees = []
        for i, line in enumerate(self.lines[:50]):
            line = line.strip()
            if line.startswith('Начальнику'):
                addressees.append(line)
        
        if addressees:
            self.markup.append("# Адресаты (regex для захвата всех вариантов):")
            self.markup.append(r"@REGEX ^Начальнику\s+.+$ { addressee }")
            self.markup.append(f"# Найдено адресатов: {len(addressees)}")
            for i, addr in enumerate(addressees[:3], 1):
                self.markup.append(f"#   {i}. {addr}")
            if len(addressees) > 3:
                self.markup.append(f"#   ... и ещё {len(addressees) - 3}")
        
        self.markup.extend(["", "[END]", ""])
    
    def _detect_parameters(self):
        """Находит технические параметры"""
        self.markup.extend([
            "# ============================================",
            "# ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ",
            "# ============================================",
            "[PARAMS]",
            ""
        ])
        
        # Паттерны для поиска
        patterns = [
            (r'стали марки\s+(\d+)', 'steel_grade', 'Марка стали'),
            (r'код ОЭМК', 'code_oemk', 'Код ОЭМК (значение на следующей строке)'),
            (r'код вида продукции\s+(\d+)', 'production_code', 'Код вида продукции'),
            (r'экспортное наименование\s+([^\n]+)', 'export_name', 'Экспортное наименование'),
            (r'условный аналог\s+([^\n]+)', 'analog', 'Условный аналог'),
            (r'стандарт выплавки\s+([^\n]+)', 'standard', 'Стандарт выплавки'),
        ]
        
        found_params = []
        for pattern, key, description in patterns:
            matches = re.finditer(pattern, self.text, re.IGNORECASE)
            for match in matches:
                context = self._get_context(match.start(), 80)
                if match.groups():
                    found_params.append((context, key, match.group(1), description))
                else:
                    found_params.append((context, key, None, description))
                break  # Берём только первое совпадение
        
        if found_params:
            for context, key, value, description in found_params:
                self.markup.append(f"# {description}")
                if value:
                    # Создаём regex с захватом группы для числовых значений
                    if key == 'steel_grade':
                        self.markup.append(r"@REGEX стали\s+марки\s+(\d+) { steel_grade: 1 }")
                        self.markup.append(f"# Пример: {context} → {value}")
                    elif key == 'production_code':
                        self.markup.append(r"@REGEX код\s+вида\s+продукции\s+(\d+) { production_code: 1 }")
                        self.markup.append(f"# Пример: {context} → {value}")
                    else:
                        # Для остальных - точное совпадение
                        self.markup.append(f"{context} {{ {key}: {value} }}")
                else:
                    self.markup.append(f"{context} {{ {key} }}")
                    self.markup.append("# [ПРОВЕРЬ] Значение на следующей строке - добавьте вручную")
                self.markup.append("")
        else:
            self.markup.append("# [ПРОВЕРЬ] Параметры не найдены автоматически")
            self.markup.append("# Добавьте вручную в формате:")
            self.markup.append("# параметр значение { ключ: значение }")
            self.markup.append("")
        
        self.markup.extend(["[END]", ""])
    
    def _detect_tables(self):
        """Находит таблицы в документе"""
        self.markup.extend([
            "# ============================================",
            "# ТАБЛИЦЫ",
            "# ============================================",
            ""
        ])
        
        # Ищем все упоминания "Таблица N"
        table_pattern = r'Таблица\s*(\d+)'
        tables = []
        
        for match in re.finditer(table_pattern, self.text):
            table_num = match.group(1)
            start_pos = match.start()
            
            # Находим контекст (следующие 20 строк)
            line_num = self.text[:start_pos].count('\n')
            context_lines = self.lines[line_num:line_num + 20]
            
            tables.append({
                'number': table_num,
                'line': line_num,
                'context': context_lines
            })
        
        if tables:
            for table in tables:
                table_name = f"table_{table['number']}"
                self.markup.append(f"[TABLE {table_name}]")
                self.markup.append(f"# Таблица {table['number']}")
                self.markup.append("")
                self.markup.append("# Начало таблицы (regex для поиска номера):")
                self.markup.append(f"@START: Таблица")
                self.markup.append(r"@START: \d+")
                self.markup.append(f"# Пример: Таблица {table['number']}")
                self.markup.append("")
                self.markup.append("# [ПРОВЕРЬ] Определите столбцы таблицы:")
                self.markup.append("# @COLUMNS:")
                
                # Показываем первые строки контекста
                for i, line in enumerate(table['context'][:10]):
                    line = line.strip()
                    if line and not line.startswith('Таблица'):
                        self.markup.append(f"# {line}")
                
                self.markup.append("")
                self.markup.append("# [ПРОВЕРЬ] Определите строки данных:")
                self.markup.append("# @ROW имя_строки: Начало строки")
                self.markup.append("# {{ values }}")
                self.markup.append("")
                self.markup.append("# Конец таблицы (укажите маркер окончания):")
                self.markup.append("# @END: Примечания:")
                self.markup.append("")
                self.markup.append("[END]")
                self.markup.append("")
        else:
            self.markup.append("# [ПРОВЕРЬ] Таблицы не найдены автоматически")
            self.markup.append("# Добавьте вручную если есть таблицы")
            self.markup.append("")
    
    def _detect_sections(self):
        """Находит разделы документа"""
        self.markup.extend([
            "# ============================================",
            "# РАЗДЕЛЫ ДОКУМЕНТА",
            "# ============================================",
            ""
        ])
        
        # Ищем нумерованные разделы (1., 2., 3., ...)
        section_pattern = r'^(\d+)\.\s+(.+)$'
        sections = []
        
        for i, line in enumerate(self.lines):
            match = re.match(section_pattern, line.strip())
            if match:
                section_num = match.group(1)
                section_title = match.group(2)
                sections.append({
                    'number': section_num,
                    'title': section_title,
                    'line': i,
                    'full_line': line.strip()
                })
        
        if sections:
            for i, section in enumerate(sections):
                section_name = f"section_{section['number']}"
                self.markup.append(f"[SECTION {section_name}]")
                self.markup.append(f"# Раздел {section['number']}: {section['title']}")
                self.markup.append("")
                self.markup.append(f"@START: {section['full_line']}")
                
                # Определяем конец раздела (начало следующего или конец документа)
                if i + 1 < len(sections):
                    next_section = sections[i + 1]
                    self.markup.append(f"@END: {next_section['full_line']}")
                else:
                    self.markup.append("@END: Технический директор")
                    self.markup.append("# [ПРОВЕРЬ] Уточните маркер конца раздела")
                
                self.markup.append("")
                self.markup.append("# Подразделы (автоматически по номерам вида 1.1., 1.2., ...):")
                self.markup.append(f"@SUBSECTIONS: ^({section['number']}\\.\\d+\\.?)")
                self.markup.append("")
                self.markup.append("[END]")
                self.markup.append("")
        else:
            self.markup.append("# [ПРОВЕРЬ] Разделы не найдены автоматически")
            self.markup.append("# Попробуйте найти разделы с другой структурой")
            self.markup.append("# Например: 'ПРОИЗВОДСТВО НЛЗ:', 'КОНТРОЛЬ И АТТЕСТАЦИЯ'")
            self.markup.append("")
    
    def _add_footer(self):
        """Добавляет подвал"""
        self.markup.extend([
            "# ============================================",
            "# ДОПОЛНИТЕЛЬНЫЕ ПОЛЯ",
            "# ============================================",
            "[EXTRA]",
            "",
            "# Подписи:",
            "# Технический директор",
            "# {{ name }} { director }",
            "",
            "# План рассылки:",
            "# План рассылки: {{ text }} { distribution }",
            "",
            "[END]",
            "",
            "# ============================================",
            "# КОНЕЦ ФАЙЛА",
            "# ============================================",
            "# Проверьте все строки с меткой [ПРОВЕРЬ]",
            "# Удалите закомментированные строки после проверки",
            "# ============================================",
        ])
    
    def _get_context(self, pos: int, length: int = 60) -> str:
        """Получает контекст вокруг позиции"""
        start = max(0, pos - length // 2)
        end = min(len(self.text), pos + length // 2)
        context = self.text[start:end]
        # Убираем переносы строк для читаемости
        context = context.replace('\n', ' ')
        return context.strip()


def main():
    parser = argparse.ArgumentParser(
        description='Генератор .markup файлов для разметки структуры документа'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Входной текстовый файл')
    parser.add_argument('--output', '-o',
                       help='Выходной .markup файл (по умолчанию: имя_файла.markup)')
    
    args = parser.parse_args()
    
    # Читаем входной файл
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        return
    
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"📄 Загружен файл: {input_path.name} ({len(text)} символов)")
    
    # Генерируем разметку
    print("🔍 Анализируем структуру документа...")
    generator = MarkupGenerator(text)
    markup = generator.generate()
    
    # Определяем выходной файл
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}.markup"
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(markup)
    
    print(f"💾 Разметка сохранена: {output_path}")
    print()
    print("📋 Что дальше:")
    print("   1. Откройте файл .markup")
    print("   2. Найдите строки с [ПРОВЕРЬ]")
    print("   3. Проверьте и исправьте автосгенерированную разметку")
    print("   4. Удалите лишние комментарии")
    print("   5. Запустите: python3 markup_parser.py --markup файл.markup --input файл.txt")
    print()
    
    # Статистика
    check_count = markup.count('[ПРОВЕРЬ]')
    if check_count > 0:
        print(f"⚠️  Найдено мест для проверки: {check_count}")


if __name__ == "__main__":
    main()

