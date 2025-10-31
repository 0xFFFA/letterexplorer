#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regex-парсер с LLM-генерацией паттернов
Логика:
1. LLM анализирует текст и генерирует regex паттерны
2. Паттерны сохраняются в .regex файл для проверки
3. Применяем паттерны к тексту и извлекаем данные
4. Сохраняем результат в JSON
"""

import argparse
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class RegexGenerator:
    """Генератор regex паттернов с помощью LLM"""
    
    def __init__(self, ollama_url: str, model: str, token: Optional[str] = None, verify_ssl: bool = True):
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.token = token
        self.session = requests.Session()
        self.session.verify = verify_ssl
        if token:
            self.session.headers.update({'X-Access-Token': token})
    
    def call_ollama(self, prompt: str) -> Optional[str]:
        """Вызывает Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9
                }
            }
            
            response = self.session.post(f"{self.ollama_url}/api/generate", 
                                       json=payload, 
                                       timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"❌ Ошибка API: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка вызова Ollama: {e}")
            return None
    
    def generate_regex_patterns(self, text: str) -> Dict[str, Any]:
        """Генерирует regex паттерны для текста"""
        
        # Анализируем весь текст для понимания структуры
        full_text = text if len(text) < 15000 else text[:15000]
        
        prompt = f"""Ты эксперт по регулярным выражениям и парсингу технических документов.

Проанализируй ВЕСЬ текст технического документа и создай ПОЛНЫЙ набор регулярных выражений для извлечения ВСЕХ данных.

ТЕКСТ ДОКУМЕНТА:
{full_text}

ЗАДАЧА:
Создай ПОЛНЫЙ JSON с регулярными выражениями для извлечения ВСЕХ элементов документа:

1. МЕТАДАННЫЕ ДОКУМЕНТА
2. ТЕХНИЧЕСКИЕ ПАРАМЕТРЫ (ВСЕ, что найдёшь после "код ОЭМК", "экспортное наименование", "стандарт выплавки", "код вида продукции")
3. ТЕКСТОВЫЕ РАЗДЕЛЫ (найди ВСЕ разделы типа "1. ПРОИЗВОДСТВО НЛЗ:", "2. ПРОИЗВОДСТВО ПРОКАТА", "3. КОНТРОЛЬ И АТТЕСТАЦИЯ")
4. ВСЕ ТАБЛИЦЫ в документе (найди каждую "Таблица N" и создай паттерн)

ФОРМАТ JSON:

{{
  "metadata": {{
    "document_number": {{
      "pattern": "regex для извлечения номера документа (учитывай '--- Страница 1 ---')",
      "description": "краткое описание",
      "flags": ["MULTILINE"]
    }},
    "date": {{
      "pattern": "regex для даты в формате ММ.ГГГГ",
      "flags": ["MULTILINE"]
    }},
    "addressees": {{
      "pattern": "regex для всех адресатов (строки 'Начальнику ...')",
      "flags": ["MULTILINE"]
    }}
  }},
  
  "technical_params": {{
    "steel_grade": {{"pattern": "стали марки (\\\\d+)", "description": "..."}},
    "code_oemk": {{"pattern": "код ОЭМК\\\\s*([^\\\\n]+)", "flags": ["IGNORECASE"]}},
    "export_name": {{"pattern": "экспортное наименование\\\\s+([^\\\\n]+)", "flags": ["MULTILINE"]}},
    "analog": {{"pattern": "условный аналог\\\\s+([^\\\\n]+)", "flags": ["MULTILINE"]}},
    "production_code": {{"pattern": "код вида продукции\\\\s+(\\\\d+)", "flags": ["MULTILINE"]}},
    "standard": {{"pattern": "стандарт выплавки\\\\s+([^\\\\n]+)", "flags": ["MULTILINE"]}}
  }},
  
  "sections": {{
    "section_1_nlz": {{
      "pattern": "ПРОИЗВОДСТВО НЛЗ:([\\\\s\\\\S]+?)(?=2\\\\. ПРОИЗВОДСТВО ПРОКАТА|Таблица 3)",
      "description": "Раздел 1: Производство НЛЗ",
      "extraction_method": "structured_subsections",
      "subsection_pattern": "^(1\\\\.\\\\d+\\\\.?)\\\\s+([\\\\s\\\\S]+?)(?=^1\\\\.\\\\d+\\\\.|^2\\\\.|$)",
      "flags": ["DOTALL", "MULTILINE"]
    }},
    "section_2_rolling": {{
      "pattern": "2\\\\. ПРОИЗВОДСТВО ПРОКАТА([\\\\s\\\\S]+?)(?=3\\\\. КОНТРОЛЬ И АТТЕСТАЦИЯ)",
      "description": "Раздел 2: Производство проката",
      "extraction_method": "structured_subsections",
      "subsection_pattern": "^(2\\\\.\\\\d+\\\\.?)\\\\s+([\\\\s\\\\S]+?)(?=^2\\\\.\\\\d+\\\\.|^3\\\\.|$)",
      "flags": ["DOTALL", "MULTILINE"]
    }},
    "section_3_control": {{
      "pattern": "3\\\\. КОНТРОЛЬ И АТТЕСТАЦИЯ ПРОКАТА([\\\\s\\\\S]+?)(?=Таблица 6|Технический директор)",
      "description": "Раздел 3: Контроль и аттестация",
      "extraction_method": "structured_subsections",
      "subsection_pattern": "^(3\\\\.\\\\d+\\\\.?)\\\\s+([\\\\s\\\\S]+?)(?=^3\\\\.\\\\d+\\\\.|Технический директор|$)",
      "flags": ["DOTALL", "MULTILINE"]
    }},
    "signatures": {{
      "pattern": "Технический директор\\\\s*\\\\n([^\\\\n]+)",
      "flags": ["MULTILINE"]
    }},
    "distribution_plan": {{
      "pattern": "План рассылки:\\\\s*([^\\\\n]+)",
      "flags": ["MULTILINE"]
    }}
  }},
  
  "tables": [
    {{
      "table_name": "chemical_composition",
      "table_number": "1",
      "description": "Химический состав",
      "extraction_method": "structured",
      "structure": {{
        "elements": {{
          "pattern": "Таблица\\\\s*1\\\\s*\\\\n(С\\\\s*\\\\n\\\\s*Si\\\\s*\\\\nMn\\\\s*\\\\nS\\\\s*\\\\nCr\\\\s*\\\\nNi\\\\s*\\\\nCu\\\\s*\\\\nAl)",
          "flags": ["MULTILINE"]
        }},
        "recommended": {{
          "pattern": "Рекоменд:\\\\s*\\\\n([0-9,\\\\n]+?)\\\\nАттестат",
          "flags": ["MULTILINE"]
        }},
        "certificate": {{
          "pattern": "Аттестат\\\\.\\\\s*\\\\n([0-9,н\\\\.б\\\\s\\\\n]+?)\\\\nЭСПЦ",
          "flags": ["MULTILINE"]
        }},
        "espz": {{
          "pattern": "ЭСПЦ\\\\s*\\\\n([0-9,\\\\n]+?)\\\\nПримечания",
          "flags": ["MULTILINE"]
        }}
      }}
    }},
    {{
      "table_name": "temperature_regime",
      "table_number": "2",
      "description": "Температурный режим плавок",
      "extraction_method": "structured_table_2",
      "pattern": "Таблица 2\\\\s*\\\\n([\\\\s\\\\S]+?)\\\\n1\\\\.11\\\\.",
      "flags": ["DOTALL"]
    }},
    {{
      "table_name": "material_info",
      "table_number": "3",
      "description": "Информация о материале",
      "extraction_method": "key_value_table",
      "pattern": "Таблица 3\\\\s*\\\\n([\\\\s\\\\S]+?)\\\\n2\\\\.1\\\\.",
      "flags": ["DOTALL"]
    }},
    {{
      "table_name": "diameter_tolerances",
      "table_number": "4",
      "description": "Диаметр и допуски",
      "extraction_method": "structured_table_4",
      "pattern": "Таблица 4\\\\s*\\\\n([\\\\s\\\\S]+?)\\\\nРекомендуется прокатку",
      "flags": ["DOTALL"]
    }},
    {{
      "table_name": "hardenability_band",
      "table_number": "5",
      "description": "Полоса прокаливаемости",
      "extraction_method": "structured_table_5",
      "pattern": "Таблица 5\\\\s*\\\\n([\\\\s\\\\S]+?)\\\\n3\\\\.2\\\\.",
      "flags": ["DOTALL"]
    }},
    {{
      "table_name": "compression_degree",
      "table_number": "6",
      "description": "Степень обжатия",
      "extraction_method": "structured_table_6",
      "pattern": "Таблица 6\\\\.\\\\s*\\\\n([\\\\s\\\\S]+?)\\\\nТехнический директор",
      "flags": ["DOTALL"]
    }}
  ]
}}

КРИТИЧЕСКИ ВАЖНО:
1. Найди ВСЕ разделы документа (1., 2., 3., и т.д.)
2. Найди ВСЕ таблицы (Таблица 1, 2, 3, 4, 5, 6...)
3. Для каждого раздела создай паттерн с extraction_method: "structured_subsections" и subsection_pattern для разбиения на подпункты (1.1, 1.2, 2.1, 2.2...)
4. Для каждой таблицы определи правильный extraction_method:
   - "structured" - для таблиц с чётким форматом (как Таблица 1)
   - "structured_table_N" - для таблиц со специфической структурой
   - "key_value_table" - для таблиц с парами ключ-значение
5. Используй флаги MULTILINE и DOTALL где нужно
6. Все паттерны должны быть КОНТЕКСТНЫМИ (учитывай что идёт ДО и ПОСЛЕ)
7. В JSON используй двойные слеши: \\\\d, \\\\s, \\\\n, и т.д.

Верни ТОЛЬКО валидный JSON, без markdown блоков и дополнительного текста!
"""
        
        print("🤖 Генерируем regex паттерны с помощью LLM...")
        response = self.call_ollama(prompt)
        
        if not response:
            return {}
        
        # Извлекаем JSON из ответа
        try:
            # Убираем markdown блоки если есть
            if '```' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    response = response[json_start:json_end]
            
            # LLM может вернуть regex с разным экранированием, нормализуем
            import re as re_module
            
            # Функция для нормализации паттернов
            def normalize_pattern(match):
                field_name = match.group(1)
                pattern_value = match.group(2)
                
                # Если уже двойное экранирование (\\\\), оставляем как есть
                # Если одинарное (\\), удваиваем
                if '\\\\\\\\' not in pattern_value:  # Нет четверных слешей
                    # Заменяем двойные на временный маркер
                    pattern_value = pattern_value.replace('\\\\', '\x00DOUBLE\x00')
                    # Одинарные удваиваем
                    pattern_value = pattern_value.replace('\\', '\\\\')
                    # Возвращаем двойные обратно
                    pattern_value = pattern_value.replace('\x00DOUBLE\x00', '\\\\')
                
                return f'"{field_name}": "{pattern_value}"'
            
            # Нормализуем все поля pattern и subsection_pattern
            response = re_module.sub(r'"(pattern|subsection_pattern)":\s*"([^"]*)"', normalize_pattern, response)
            
            patterns = json.loads(response)
            return patterns
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"Ответ LLM:\n{response[:500]}...")
            return {}


class RegexParser:
    """Применяет regex паттерны к тексту"""
    
    def __init__(self, patterns: Dict[str, Any]):
        self.patterns = patterns
    
    def extract_with_pattern(self, text: str, pattern_info: Dict[str, Any]) -> Any:
        """Извлекает данные по одному паттерну"""
        pattern = pattern_info.get('pattern', '')
        flags_list = pattern_info.get('flags', [])
        
        if not pattern:
            return None
        
        # Формируем флаги
        flags = 0
        if 'MULTILINE' in flags_list:
            flags |= re.MULTILINE
        if 'IGNORECASE' in flags_list:
            flags |= re.IGNORECASE
        if 'DOTALL' in flags_list:
            flags |= re.DOTALL
        
        try:
            # Проверяем, есть ли именованные группы
            if '(?P<' in pattern:
                match = re.search(pattern, text, flags)
                if match:
                    return match.groupdict()
            else:
                # Ищем все совпадения
                matches = re.findall(pattern, text, flags)
                if matches:
                    # Если одно совпадение - возвращаем строку
                    # Если несколько - список
                    return matches[0] if len(matches) == 1 else matches
        except re.error as e:
            print(f"⚠️ Ошибка в regex паттерне: {e}")
            print(f"   Паттерн: {pattern}")
        
        return None
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """Извлекает метаданные"""
        metadata = {}
        
        if 'metadata' not in self.patterns:
            return metadata
        
        for key, pattern_info in self.patterns['metadata'].items():
            result = self.extract_with_pattern(text, pattern_info)
            if result:
                metadata[key] = result
        
        return metadata
    
    def extract_technical_params(self, text: str) -> Dict[str, Any]:
        """Извлекает технические параметры"""
        params = {}
        
        if 'technical_params' not in self.patterns:
            return params
        
        for key, pattern_info in self.patterns['technical_params'].items():
            result = self.extract_with_pattern(text, pattern_info)
            if result:
                params[key] = result
        
        return params
    
    def extract_table(self, text: str, table_info: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает одну таблицу"""
        table_data = {
            "table_name": table_info.get('table_name', 'unknown'),
            "table_number": table_info.get('table_number', ''),
            "description": table_info.get('description', ''),
            "columns": [],
            "rows": {},
            "data": {}
        }
        
        extraction_method = table_info.get('extraction_method', 'structured')
        
        # Метод 1: Структурированное извлечение (для Таблицы 1)
        if extraction_method == 'structured' and 'structure' in table_info:
            structure = table_info['structure']
            
            # Извлекаем элементы (заголовки столбцов)
            if 'elements' in structure:
                elements_result = self.extract_with_pattern(text, structure['elements'])
                if elements_result:
                    # Разбиваем на отдельные элементы
                    elements_text = elements_result if isinstance(elements_result, str) else str(elements_result)
                    table_data['columns'] = [elem.strip() for elem in elements_text.split() if elem.strip()]
            
            # Извлекаем данные для каждой строки
            for row_name in ['recommended', 'certificate', 'espz']:
                if row_name in structure:
                    row_result = self.extract_with_pattern(text, structure[row_name])
                    if row_result:
                        row_text = row_result if isinstance(row_result, str) else str(row_result)
                        # Разбиваем на значения
                        values = []
                        for val in row_text.split():
                            val = val.strip()
                            if val and (val[0].isdigit() or val.startswith('н')):
                                values.append(val)
                        table_data['rows'][row_name] = values
        
        # Метод 2: Простое извлечение с одним паттерном (для Таблицы 2)
        elif extraction_method == 'simple' and 'pattern' in table_info:
            result = self.extract_with_pattern(
                text,
                {
                    'pattern': table_info['pattern'],
                    'flags': table_info.get('flags', [])
                }
            )
            if result:
                # Результат - кортеж значений из групп
                if isinstance(result, tuple):
                    table_data['raw_data'] = list(result)
                else:
                    table_data['raw_data'] = result
        
        # Метод 3: Таблица 2 - температурный режим
        elif extraction_method == 'structured_table_2' and 'pattern' in table_info:
            result = self.extract_with_pattern(text, {
                'pattern': table_info['pattern'],
                'flags': table_info.get('flags', [])
            })
            if result:
                lines = result.strip().split('\n')
                # Первые 3 строки - заголовки, остальные - значения
                if len(lines) >= 6:
                    table_data['headers'] = [lines[0].strip(), lines[1].strip(), lines[2].strip()]
                    table_data['values'] = [[lines[3].strip(), lines[4].strip()],
                                           [lines[5].strip(), lines[6].strip()],
                                           [lines[7].strip(), lines[8].strip()]]
                else:
                    table_data['raw_text'] = result
        
        # Метод 4: Таблица 3 - ключ-значение
        elif extraction_method == 'key_value_table' and 'pattern' in table_info:
            result = self.extract_with_pattern(text, {
                'pattern': table_info['pattern'],
                'flags': table_info.get('flags', [])
            })
            if result:
                lines = [line.strip() for line in result.strip().split('\n') if line.strip()]
                table_data['data'] = {}
                # Парсим как пары ключ-значение
                i = 0
                while i < len(lines):
                    if i + 1 < len(lines):
                        key = lines[i]
                        value = lines[i+1]
                        # Если ключ выглядит как заголовок, а значение - как данные
                        if not value.startswith('N') and not value.startswith('Д'):
                            table_data['data'][key] = value
                            i += 2
                        else:
                            i += 1
                    else:
                        i += 1
        
        # Метод 5: Таблица 4 - диаметры и допуски
        elif extraction_method == 'structured_table_4' and 'pattern' in table_info:
            result = self.extract_with_pattern(text, {
                'pattern': table_info['pattern'],
                'flags': table_info.get('flags', [])
            })
            if result:
                lines = result.strip().split('\n')
                table_data['headers'] = []
                table_data['rows'] = []
                
                # Первые 5 строк - заголовки
                header_lines = []
                data_started = False
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    # Если строка содержит только цифры и знаки, это данные
                    if line.replace('+', '').replace(',', '').replace('.', '').replace('-', '').replace(' ', '').isdigit() or \
                       (line.isdigit() and int(line) >= 85):
                        data_started = True
                    
                    if not data_started:
                        header_lines.append(line)
                    else:
                        # Пытаемся разобрать строку данных
                        parts = line.split()
                        if len(parts) >= 3:
                            table_data['rows'].append(parts)
                
                table_data['headers'] = header_lines
        
        # Метод 6: Таблица 5 - прокаливаемость
        elif extraction_method == 'structured_table_5' and 'pattern' in table_info:
            result = self.extract_with_pattern(text, {
                'pattern': table_info['pattern'],
                'flags': table_info.get('flags', [])
            })
            if result:
                lines = result.strip().split('\n')
                table_data['data'] = {}
                
                # Собираем значения для каждой строки
                distances = []
                min_values = []
                max_values = []
                
                current_list = None
                for line in lines:
                    line = line.strip()
                    if 'Полоса прокаливаемости' in line:
                        continue
                    elif line == 'MM':
                        current_list = distances
                    elif line == 'min':
                        current_list = min_values
                    elif line == 'HRC':
                        continue
                    elif line == 'max':
                        current_list = max_values
                    elif line and current_list is not None:
                        # Разбиваем строку на числа
                        parts = line.split()
                        for part in parts:
                            if part.replace(',', '').replace('.', '').isdigit():
                                current_list.append(part)
                
                table_data['distances_mm'] = distances
                table_data['min_hrc'] = min_values
                table_data['max_hrc'] = max_values
        
        # Метод 7: Таблица 6 - степень обжатия
        elif extraction_method == 'structured_table_6' and 'pattern' in table_info:
            result = self.extract_with_pattern(text, {
                'pattern': table_info['pattern'],
                'flags': table_info.get('flags', [])
            })
            if result:
                lines = result.strip().split('\n')
                table_data['data'] = []
                
                # Парсим данные построчно
                current_row = []
                for line in lines:
                    line = line.strip()
                    if not line or line == ';':
                        continue
                    # Пропускаем заголовки
                    if 'Диаметр' in line or 'Степень' in line or 'проката' in line or 'обжатия' in line:
                        continue
                    
                    # Если строка содержит числа
                    parts = line.split()
                    for part in parts:
                        if part.replace(',', '').replace('.', '').isdigit():
                            current_row.append(part)
                            if len(current_row) == 2:
                                table_data['data'].append({
                                    'diameter_mm': current_row[0],
                                    'compression_degree': current_row[1]
                                })
                                current_row = []
        
        # Метод 8: Текстовый блок (fallback)
        elif extraction_method == 'text_block' and 'pattern' in table_info:
            result = self.extract_with_pattern(
                text,
                {
                    'pattern': table_info['pattern'],
                    'flags': table_info.get('flags', [])
                }
            )
            if result:
                table_data['raw_text'] = result if isinstance(result, str) else str(result)
        
        # Старый метод: header_pattern и value_patterns
        else:
            # Извлекаем заголовки столбцов
            if 'header_pattern' in table_info:
                header_result = self.extract_with_pattern(
                    text, 
                    {'pattern': table_info['header_pattern']}
                )
                if header_result:
                    if isinstance(header_result, list):
                        table_data['columns'] = header_result
                    else:
                        table_data['columns'] = [header_result]
            
            # Извлекаем названия строк
            if 'row_names' in table_info:
                table_data['rows'] = table_info['row_names']
            
            # Извлекаем значения
            if 'value_patterns' in table_info:
                for value_key, value_pattern in table_info['value_patterns'].items():
                    result = self.extract_with_pattern(
                        text,
                        {'pattern': value_pattern}
                    )
                    if result:
                        table_data['data'][value_key] = result
        
        return table_data
    
    def extract_tables(self, text: str) -> List[Dict[str, Any]]:
        """Извлекает все таблицы"""
        tables = []
        
        if 'tables' not in self.patterns:
            return tables
        
        for table_info in self.patterns['tables']:
            table_data = self.extract_table(text, table_info)
            tables.append(table_data)
        
        return tables
    
    def extract_sections(self, text: str) -> Dict[str, Any]:
        """Извлекает текстовые разделы документа"""
        sections = {}
        
        if 'sections' not in self.patterns:
            return sections
        
        for section_name, pattern_info in self.patterns['sections'].items():
            extraction_method = pattern_info.get('extraction_method', 'simple')
            
            # Извлекаем основной текст раздела
            result = self.extract_with_pattern(text, pattern_info)
            
            if not result:
                continue
            
            # Если нужно структурировать на подразделы
            if extraction_method == 'structured_subsections' and 'subsection_pattern' in pattern_info:
                subsections = {}
                subsection_pattern = pattern_info['subsection_pattern']
                
                # Ищем все подразделы (например, 1.1, 1.2, 2.1, 2.2...)
                import re as re_module
                matches = re_module.finditer(subsection_pattern, result, re_module.MULTILINE | re_module.DOTALL)
                
                for match in matches:
                    subsection_number = match.group(1).strip()
                    subsection_content = match.group(2).strip()
                    subsections[subsection_number] = subsection_content
                
                if subsections:
                    sections[section_name] = {
                        "full_text": result,
                        "subsections": subsections
                    }
                else:
                    # Если не удалось разбить на подразделы, сохраняем как есть
                    sections[section_name] = result
            else:
                sections[section_name] = result
        
        return sections
    
    def parse_document(self, text: str) -> Dict[str, Any]:
        """Парсит весь документ"""
        print("📄 Применяем regex паттерны к тексту...")
        
        result = {
            "metadata": self.extract_metadata(text),
            "technical_params": self.extract_technical_params(text),
            "sections": self.extract_sections(text),
            "tables": self.extract_tables(text)
        }
        
        return result


def save_regex_patterns(patterns: Dict[str, Any], output_file: Path):
    """Сохраняет паттерны в .regex файл"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(patterns, f, ensure_ascii=False, indent=2)
        print(f"💾 Regex паттерны сохранены: {output_file}")
        print(f"   Проверьте и отредактируйте паттерны при необходимости!")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")


def load_regex_patterns(regex_file: Path) -> Dict[str, Any]:
    """Загружает паттерны из .regex файла"""
    try:
        with open(regex_file, 'r', encoding='utf-8') as f:
            patterns = json.load(f)
        print(f"✅ Regex паттерны загружены: {regex_file}")
        return patterns
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return {}


def save_results(results: Dict[str, Any], output_file: Path):
    """Сохраняет результаты в JSON"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Результаты сохранены: {output_file}")
    except Exception as e:
        print(f"❌ Ошибка сохранения: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Regex-парсер с LLM-генерацией паттернов'
    )
    parser.add_argument('--file', '-f', required=True,
                       help='Путь к текстовому файлу для обработки')
    parser.add_argument('--mode', '-m', 
                       choices=['generate', 'apply', 'both'],
                       default='both',
                       help='Режим: generate (генерация regex), apply (применение), both (оба)')
    parser.add_argument('--regex-file', '-r',
                       help='Путь к .regex файлу (по умолчанию: имя_файла.regex)')
    parser.add_argument('--output', '-o',
                       help='Выходной JSON файл (по умолчанию: имя_файла_parsed.json)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='URL Ollama сервера')
    parser.add_argument('--model', required=True,
                       help='Название модели Ollama')
    parser.add_argument('--token',
                       help='Токен авторизации')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Отключить проверку SSL')
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"❌ Файл не найден: {input_file}")
        return
    
    # Определяем пути для файлов
    if args.regex_file:
        regex_file = Path(args.regex_file)
    else:
        regex_file = input_file.parent / f"{input_file.stem}.regex"
    
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_parsed.json"
    
    # Читаем текст
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Загружен файл: {input_file.name} ({len(text)} символов)")
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return
    
    patterns = {}
    
    # РЕЖИМ 1: Генерация regex паттернов
    if args.mode in ['generate', 'both']:
        print("\n" + "="*60)
        print("🔄 ЭТАП 1: Генерация regex паттернов")
        print("="*60)
        
        generator = RegexGenerator(
            args.ollama_url, 
            args.model, 
            args.token,
            verify_ssl=not args.no_ssl_verify
        )
        
        patterns = generator.generate_regex_patterns(text)
        
        if patterns:
            save_regex_patterns(patterns, regex_file)
            print("\n✅ Паттерны сгенерированы!")
            print(f"📝 Проверьте файл: {regex_file}")
            print("   При необходимости отредактируйте паттерны вручную.")
        else:
            print("\n❌ Не удалось сгенерировать паттерны")
            if args.mode == 'both':
                return
    
    # РЕЖИМ 2: Применение regex паттернов
    if args.mode in ['apply', 'both']:
        print("\n" + "="*60)
        print("🔄 ЭТАП 2: Применение regex паттернов")
        print("="*60)
        
        # Загружаем паттерны (если не сгенерировали только что)
        if args.mode == 'apply':
            if not regex_file.exists():
                print(f"❌ Файл с паттернами не найден: {regex_file}")
                print("   Сначала запустите с --mode generate")
                return
            patterns = load_regex_patterns(regex_file)
        
        if not patterns:
            print("❌ Нет паттернов для применения")
            return
        
        # Парсим документ
        regex_parser = RegexParser(patterns)
        results = regex_parser.parse_document(text)
        
        # Добавляем метаинформацию
        final_results = {
            "file_info": {
                "source_file": input_file.name,
                "regex_file": regex_file.name,
                "parsing_method": "regex_llm_generated"
            },
            "extracted_data": results
        }
        
        # Сохраняем результаты
        save_results(final_results, output_file)
        
        print("\n✅ Обработка завершена!")
        print(f"📁 Результат: {output_file}")


if __name__ == "__main__":
    main()

