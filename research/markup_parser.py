#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Парсер .markup файлов - извлекает данные из текста по разметке
"""

import re
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple


class MarkupParser:
    """Парсер разметки документов"""
    
    def __init__(self, markup_text: str, document_text: str):
        self.markup_text = markup_text
        self.document_text = document_text
        self.document_lines = document_text.split('\n')
        self.result = {
            "file_info": {},
            "extracted_data": {
                "metadata": {},
                "technical_params": {},
                "sections": {},
                "tables": []
            }
        }
    
    def parse(self) -> Dict[str, Any]:
        """Парсит markup и извлекает данные"""
        # Разбираем markup на секции
        sections = self._split_into_sections()
        
        for section_type, section_content in sections:
            if section_type == 'META':
                self._parse_meta(section_content)
            elif section_type == 'PARAMS':
                self._parse_params(section_content)
            elif section_type.startswith('TABLE'):
                table_name = section_type.split()[1] if len(section_type.split()) > 1 else 'unknown'
                self._parse_table(section_content, table_name)
            elif section_type.startswith('SECTION'):
                section_name = section_type.split()[1] if len(section_type.split()) > 1 else 'unknown'
                self._parse_section(section_content, section_name)
        
        return self.result
    
    def _split_into_sections(self) -> List[Tuple[str, List[str]]]:
        """Разбивает markup на секции"""
        sections = []
        current_section = None
        current_content = []
        
        for line in self.markup_text.split('\n'):
            line_stripped = line.strip()
            
            # Пропускаем комментарии и пустые строки
            if not line_stripped or line_stripped.startswith('#'):
                continue
            
            # Начало секции
            if line_stripped.startswith('[') and not line_stripped.startswith('[END]'):
                # Сохраняем предыдущую секцию
                if current_section:
                    sections.append((current_section, current_content))
                
                # Начинаем новую
                current_section = line_stripped[1:-1]  # Убираем [ ]
                current_content = []
            
            # Конец секции
            elif line_stripped == '[END]':
                if current_section:
                    sections.append((current_section, current_content))
                current_section = None
                current_content = []
            
            # Содержимое секции
            else:
                if current_section:
                    current_content.append(line_stripped)
        
        return sections
    
    def _parse_meta(self, content: List[str]):
        """Парсит метаданные"""
        for line in content:
            if '{' not in line or '}' not in line:
                continue
            
            # Проверяем, regex это или точное совпадение
            is_regex = line.strip().startswith('@REGEX')
            
            if is_regex:
                # Формат: @REGEX паттерн { тип }
                line = line.strip()[6:].strip()  # Убираем "@REGEX"
            
            # Извлекаем текст/паттерн и метку
            text_part = line[:line.index('{')].strip()
            label_part = line[line.index('{')+1:line.index('}')].strip()
            
            # Ищем в документе
            if is_regex:
                values = self._find_by_regex(text_part, label_part, find_all=(label_part.strip() == 'addressee'))
            else:
                values = [self._find_in_document(text_part)]
            
            # Обрабатываем найденные значения
            for value in values:
                if value:
                    # Если метка содержит двоеточие - это пара ключ:значение с захватом группы
                    if ':' in label_part:
                        key, group_num = label_part.split(':', 1)
                        key = key.strip()
                        # Для regex - value уже извлечённое значение группы
                        self.result['extracted_data']['metadata'][key] = value
                    else:
                        # Простая метка
                        key = label_part
                        if key in self.result['extracted_data']['metadata']:
                            # Если ключ уже есть - делаем список
                            existing = self.result['extracted_data']['metadata'][key]
                            if isinstance(existing, list):
                                existing.append(value)
                            else:
                                self.result['extracted_data']['metadata'][key] = [existing, value]
                        else:
                            self.result['extracted_data']['metadata'][key] = value
    
    def _parse_params(self, content: List[str]):
        """Парсит технические параметры"""
        for line in content:
            if '{' not in line or '}' not in line:
                continue
            
            # Проверяем, regex это или точное совпадение
            is_regex = line.strip().startswith('@REGEX')
            
            if is_regex:
                line = line.strip()[6:].strip()  # Убираем "@REGEX"
            
            text_part = line[:line.index('{')].strip()
            label_part = line[line.index('{')+1:line.index('}')].strip()
            
            # Ищем в документе
            if is_regex:
                values = self._find_by_regex(text_part, label_part, find_all=False)
            else:
                values = [self._find_in_document(text_part)]
            
            # Обрабатываем найденные значения (берём только первое для параметров)
            for value in values[:1]:  # Только первое значение
                if value:
                    if ':' in label_part:
                        key, group_or_value = label_part.split(':', 1)
                        key = key.strip()
                        # Для regex - value уже извлечённое значение группы
                        self.result['extracted_data']['technical_params'][key] = value
                    else:
                        key = label_part
                        self.result['extracted_data']['technical_params'][key] = value
    
    def _parse_table(self, content: List[str], table_name: str):
        """Парсит таблицу"""
        table_data = {
            "table_name": table_name,
            "table_number": "",
            "description": "",
            "columns": [],
            "rows": {},
            "data": {}
        }
        
        # Состояние парсера
        mode = None
        
        for line in content:
            line = line.strip()
            
            # @START: маркер начала таблицы (может быть regex)
            if line.startswith('@START:'):
                marker = line[7:].strip()
                # Извлекаем номер таблицы (точное совпадение или regex)
                if marker.isdigit():
                    table_data['table_number'] = marker
                else:
                    # Пробуем как regex для поиска номера
                    try:
                        regex = re.compile(marker)
                        # Ищем в тексте документа после слова "Таблица"
                        for line_text in self.document_lines:
                            if 'Таблица' in line_text:
                                match = regex.search(line_text)
                                if match:
                                    table_data['table_number'] = match.group(0)
                                    break
                    except re.error:
                        # Не regex - просто сохраняем как есть
                        table_data['table_number'] = marker
            
            # @END: маркер конца таблицы
            elif line.startswith('@END:'):
                pass  # Пока не используем
            
            # @COLUMNS: список столбцов
            elif line == '@COLUMNS:':
                mode = 'columns'
            
            # @ROW: строка данных
            elif line.startswith('@ROW'):
                # Формат: @ROW имя_строки: маркер начала
                match = re.match(r'@ROW\s+(\w+):\s*(.+)', line)
                if match:
                    row_name = match.group(1)
                    row_marker = match.group(2)
                    mode = ('row', row_name, row_marker)
            
            # Обычная строка
            else:
                if mode == 'columns':
                    # Добавляем столбец
                    table_data['columns'].append(line)
                
                elif isinstance(mode, tuple) and mode[0] == 'row':
                    row_name, row_marker = mode[1], mode[2]
                    
                    # Если это {{ values }} - извлекаем значения
                    if '{{' in line and 'values' in line:
                        # Ищем строку с маркером в документе
                        values = self._find_table_row_values(row_marker)
                        if values:
                            table_data['rows'][row_name] = values
                        mode = None
        
        self.result['extracted_data']['tables'].append(table_data)
    
    def _parse_section(self, content: List[str], section_name: str):
        """Парсит раздел"""
        section_data = {
            "full_text": "",
            "subsections": {}
        }
        
        start_marker = None
        end_marker = None
        subsection_pattern = None
        
        for line in content:
            if line.startswith('@START:'):
                start_marker = line[7:].strip()
            elif line.startswith('@END:'):
                end_marker = line[5:].strip()
            elif line.startswith('@SUBSECTIONS:'):
                subsection_pattern = line[13:].strip()
        
        if start_marker and end_marker:
            # Извлекаем текст раздела
            section_text = self._extract_text_between(start_marker, end_marker)
            
            if section_text:
                section_data['full_text'] = section_text
                
                # Разбиваем на подразделы
                if subsection_pattern:
                    subsections = self._extract_subsections(section_text, subsection_pattern)
                    section_data['subsections'] = subsections
                
                self.result['extracted_data']['sections'][section_name] = section_data
    
    def _find_in_document(self, text: str) -> Optional[str]:
        """Находит точное совпадение текста в документе"""
        # Нормализуем пробелы
        text_normalized = ' '.join(text.split())
        
        # Ищем точное совпадение
        if text_normalized in self.document_text:
            return text_normalized
        
        # Ищем по строкам
        for line in self.document_lines:
            line_normalized = ' '.join(line.split())
            if text_normalized == line_normalized:
                return line_normalized
            # Частичное совпадение в начале строки
            if line_normalized.startswith(text_normalized):
                return text_normalized
        
        return None
    
    def _find_by_regex(self, pattern: str, label_part: str, find_all: bool = False) -> List[str]:
        """Находит совпадения по regex-паттерну
        
        Args:
            pattern: regex паттерн
            label_part: метка (ключ или ключ:группа)
            find_all: если True, находит все совпадения, иначе только первое
        """
        results = []
        
        try:
            # Компилируем regex с поддержкой многострочных паттернов
            # DOTALL: точка (.) совпадает с переносами строк
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            
            # Определяем, нужно ли захватывать группу
            if ':' in label_part:
                # Формат: { ключ: номер_группы }
                _, group_num_str = label_part.split(':', 1)
                group_num = int(group_num_str.strip()) if group_num_str.strip().isdigit() else 0
                
                # Ищем все совпадения с группами
                for match in regex.finditer(self.document_text):
                    if match.groups() and group_num > 0 and group_num <= len(match.groups()):
                        results.append(match.group(group_num))
                    elif match.groups():
                        # Берём первую группу по умолчанию
                        results.append(match.group(1))
                    else:
                        # Нет групп - берём всё совпадение
                        results.append(match.group(0))
                
                # Если построчно не нашли, ищем в полном тексте
                if not results:
                    for line in self.document_lines:
                        match = regex.search(line)
                        if match:
                            if match.groups() and group_num > 0 and group_num <= len(match.groups()):
                                results.append(match.group(group_num))
                            elif match.groups():
                                results.append(match.group(1))
                            else:
                                results.append(match.group(0))
            else:
                # Без захвата группы - берём всё совпадение
                if find_all:
                    # Находим ВСЕ совпадения (для addressee и т.д.)
                    for match in regex.finditer(self.document_text):
                        results.append(match.group(0))
                    
                    # Если не нашли в полном тексте, ищем построчно
                    if not results:
                        for line in self.document_lines:
                            match = regex.search(line)
                            if match:
                                results.append(match.group(0))
                else:
                    # Находим ПЕРВОЕ совпадение (для document_number, date и т.д.)
                    match = regex.search(self.document_text)
                    if match:
                        results.append(match.group(0))
                    else:
                        # Пробуем построчно
                        for line in self.document_lines:
                            match = regex.search(line)
                            if match:
                                results.append(match.group(0))
                                break  # Берём только первое совпадение
        
        except re.error as e:
            print(f"⚠️  Ошибка в regex паттерне: {pattern}")
            print(f"   {e}")
        
        return results
    
    def _extract_value_from_text(self, pattern: str, found_text: str) -> Optional[str]:
        """Извлекает значение из найденного текста"""
        # Пытаемся найти число в конце
        numbers = re.findall(r'\d+', found_text)
        if numbers:
            return numbers[-1]
        
        # Возвращаем последнее слово
        words = found_text.split()
        if words:
            return words[-1]
        
        return None
    
    def _find_table_row_values(self, marker: str) -> List[str]:
        """Находит значения строки таблицы по маркеру"""
        values = []
        
        # Ищем маркер в документе
        in_row = False
        for line in self.document_lines:
            line_stripped = line.strip()
            
            # Проверяем точное совпадение или вхождение в начале строки
            # Это помогает избежать ложных срабатываний (например, "Начальнику ЭСПЦ" vs "ЭСПЦ")
            marker_found = False
            if line_stripped == marker:
                # Точное совпадение (например, "ЭСПЦ")
                marker_found = True
            elif line_stripped.startswith(marker + ' ') or line_stripped.startswith(marker + ':'):
                # Маркер в начале с пробелом или двоеточием (например, "Рекоменд: значения")
                marker_found = True
            elif marker.endswith(':') and marker in line_stripped:
                # Маркер с двоеточием может быть в любом месте (например, "Рекоменд:")
                marker_found = True
            
            if marker_found:
                in_row = True
                continue
            
            if in_row:
                # Если строка содержит числа или н.б. - это значения
                if re.search(r'\d|н\.б', line_stripped):
                    # Извлекаем все числа и специальные значения
                    tokens = line_stripped.split()
                    for token in tokens:
                        if re.match(r'[\d,\.]+$|н\.б\.?$', token):
                            values.append(token.replace('.', ',') if token != 'н.б.' and token != 'н.б' else token)
                else:
                    # Закончились данные
                    break
        
        return values
    
    def _extract_text_between(self, start_marker: str, end_marker: str) -> str:
        """Извлекает текст между маркерами"""
        text_parts = []
        capturing = False
        
        for line in self.document_lines:
            line_stripped = line.strip()
            
            # Начало захвата
            if start_marker in line_stripped:
                capturing = True
                continue
            
            # Конец захвата
            if end_marker in line_stripped and capturing:
                break
            
            # Захватываем
            if capturing:
                text_parts.append(line)
        
        return '\n'.join(text_parts)
    
    def _extract_subsections(self, section_text: str, pattern: str) -> Dict[str, str]:
        """Извлекает подразделы по паттерну"""
        subsections = {}
        
        try:
            # Компилируем regex
            regex = re.compile(pattern, re.MULTILINE)
            
            # Находим все совпадения
            matches = list(regex.finditer(section_text))
            
            for i, match in enumerate(matches):
                subsection_num = match.group(1)
                start_pos = match.end()
                
                # Конец подраздела - начало следующего или конец текста
                if i + 1 < len(matches):
                    end_pos = matches[i + 1].start()
                else:
                    end_pos = len(section_text)
                
                subsection_text = section_text[start_pos:end_pos].strip()
                subsections[subsection_num] = subsection_text
        
        except re.error as e:
            print(f"⚠️  Ошибка в regex паттерне: {pattern}")
            print(f"   {e}")
        
        return subsections


def main():
    parser = argparse.ArgumentParser(
        description='Парсер .markup файлов - извлекает данные из текста'
    )
    parser.add_argument('--markup', '-m', required=True,
                       help='Файл разметки (.markup)')
    parser.add_argument('--input', '-i', required=True,
                       help='Исходный текстовый файл')
    parser.add_argument('--output', '-o',
                       help='Выходной JSON файл (по умолчанию: имя_файла_parsed.json)')
    
    args = parser.parse_args()
    
    # Читаем файлы
    markup_path = Path(args.markup)
    input_path = Path(args.input)
    
    if not markup_path.exists():
        print(f"❌ Файл разметки не найден: {markup_path}")
        return
    
    if not input_path.exists():
        print(f"❌ Исходный файл не найден: {input_path}")
        return
    
    with open(markup_path, 'r', encoding='utf-8') as f:
        markup_text = f.read()
    
    with open(input_path, 'r', encoding='utf-8') as f:
        document_text = f.read()
    
    print(f"📄 Загружен markup: {markup_path.name}")
    print(f"📄 Загружен документ: {input_path.name} ({len(document_text)} символов)")
    
    # Парсим
    print("🔍 Извлекаем данные по разметке...")
    markup_parser = MarkupParser(markup_text, document_text)
    result = markup_parser.parse()
    
    # Добавляем метаинформацию
    result['file_info'] = {
        'source_file': input_path.name,
        'markup_file': markup_path.name,
        'parsing_method': 'markup_based'
    }
    
    # Определяем выходной файл
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_markup_parsed.json"
    
    # Сохраняем
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Результат сохранён: {output_path}")
    
    # Статистика
    print()
    print("📊 Извлечено:")
    print(f"   Метаданные:          {len(result['extracted_data']['metadata'])} полей")
    print(f"   Технические параметры: {len(result['extracted_data']['technical_params'])} полей")
    print(f"   Разделы:             {len(result['extracted_data']['sections'])} разделов")
    print(f"   Таблицы:             {len(result['extracted_data']['tables'])} таблиц")
    
    # Подразделы
    total_subsections = 0
    for section_data in result['extracted_data']['sections'].values():
        if isinstance(section_data, dict) and 'subsections' in section_data:
            total_subsections += len(section_data['subsections'])
    
    if total_subsections > 0:
        print(f"   Подразделы:          {total_subsections}")


if __name__ == "__main__":
    main()

