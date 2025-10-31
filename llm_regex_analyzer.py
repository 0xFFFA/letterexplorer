#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM Regex Analyzer - гибридный подход
Логика:
1. LLM анализирует документ и находит данные
2. LLM генерирует regex паттерны для найденных данных
3. Сохраняем оба результата: данные + regex
4. Можем применить regex для валидации
"""

import argparse
import json
import requests
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMRegexAnalyzer:
    """Анализатор документов с генерацией regex через LLM"""
    
    def __init__(self, ollama_url: str, model: str, token: Optional[str] = None, verify_ssl: bool = True):
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.token = token
        self.session = requests.Session()
        self.session.verify = verify_ssl
        if token:
            self.session.headers.update({'X-Access-Token': token})
    
    def test_connection(self) -> bool:
        """Тестирует соединение с Ollama"""
        try:
            response = self.session.get(f"{self.ollama_url}/api/tags")
            if response.status_code == 200:
                print("✅ Соединение с Ollama установлено")
                return True
            else:
                print(f"❌ Ошибка соединения: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Ошибка соединения: {e}")
            return False
    
    def call_ollama(self, prompt: str, temperature: float = 0.1) -> Optional[str]:
        """Вызывает Ollama API"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "num_predict": 8000  # Увеличиваем лимит для больших JSON
                }
            }
            
            print(f"🤖 Отправляем запрос к LLM (модель: {self.model})...")
            response = self.session.post(f"{self.ollama_url}/api/generate", 
                                       json=payload, 
                                       timeout=300)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"❌ Ошибка API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка вызова Ollama: {e}")
            return None
    
    def load_instructions(self, instructions_file: Path) -> str:
        """Загружает инструкции из файла"""
        try:
            with open(instructions_file, 'r', encoding='utf-8') as f:
                instructions = f.read()
            print(f"📝 Загружены инструкции: {instructions_file.name}")
            return instructions
        except Exception as e:
            print(f"❌ Ошибка загрузки инструкций: {e}")
            return ""
    
    def analyze_document(self, text: str, instructions: str) -> Dict[str, Any]:
        """Анализирует документ с помощью LLM"""
        
        # Формируем полный промпт
        full_prompt = f"""{instructions}

════════════════════════════════════════════════════════════════
ТЕКСТ ДОКУМЕНТА ДЛЯ АНАЛИЗА:
════════════════════════════════════════════════════════════════

{text}

════════════════════════════════════════════════════════════════

ПРОАНАЛИЗИРУЙ этот документ согласно ВСЕМ инструкциям выше.
Для КАЖДОГО раздела:
1. Найди данные в тексте
2. Создай regex паттерны

Верни результат в формате JSON как указано в инструкциях.
ТОЛЬКО JSON, без markdown блоков!
"""
        
        print("🔄 Этап 1: Анализ документа и генерация regex паттернов...")
        print(f"   Размер документа: {len(text)} символов")
        print(f"   Размер инструкций: {len(instructions)} символов")
        
        response = self.call_ollama(full_prompt, temperature=0.1)
        
        if not response:
            print("❌ Не получен ответ от LLM")
            return {}
        
        # Парсим JSON из ответа
        try:
            # Убираем markdown блоки если есть
            if '```' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    response = response[json_start:json_end]
            
            result = json.loads(response)
            print("✅ Успешно распарсен JSON от LLM")
            return result
            
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга JSON: {e}")
            print(f"   Первые 500 символов ответа:\n{response[:500]}...")
            
            # Сохраняем raw ответ для отладки
            return {
                "error": "json_parse_error",
                "raw_response": response[:2000]
            }
    
    def validate_with_regex(self, text: str, regex_patterns: Dict[str, Any]) -> Dict[str, Any]:
        """Валидирует извлеченные данные с помощью regex"""
        print("🔄 Этап 2: Валидация данных с помощью regex...")
        
        validation_results = {}
        
        for section_name, section_data in regex_patterns.items():
            if 'regex_patterns' not in section_data:
                continue
            
            validation_results[section_name] = {}
            patterns = section_data['regex_patterns']
            
            for field_name, pattern_info in patterns.items():
                if not isinstance(pattern_info, dict) or 'pattern' not in pattern_info:
                    continue
                
                pattern = pattern_info['pattern']
                flags_list = pattern_info.get('flags', [])
                
                # Формируем флаги
                flags = 0
                if 'MULTILINE' in flags_list:
                    flags |= re.MULTILINE
                if 'IGNORECASE' in flags_list:
                    flags |= re.IGNORECASE
                if 'DOTALL' in flags_list:
                    flags |= re.DOTALL
                
                try:
                    # Пытаемся применить regex
                    matches = re.findall(pattern, text, flags)
                    
                    validation_results[section_name][field_name] = {
                        "pattern_valid": True,
                        "matches_found": len(matches) if matches else 0,
                        "first_match": matches[0] if matches else None
                    }
                    
                except re.error as e:
                    validation_results[section_name][field_name] = {
                        "pattern_valid": False,
                        "error": str(e)
                    }
        
        return validation_results
    
    def save_results(self, results: Dict[str, Any], output_file: Path):
        """Сохраняет результаты"""
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Результаты сохранены: {output_file}")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")
    
    def extract_regex_patterns_only(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает только regex паттерны из результата анализа"""
        regex_only = {}
        
        if 'sections' in analysis_result:
            for section_name, section_data in analysis_result['sections'].items():
                if isinstance(section_data, dict) and 'regex_patterns' in section_data:
                    regex_only[section_name] = section_data['regex_patterns']
        
        return regex_only
    
    def extract_data_only(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Извлекает только извлеченные данные из результата анализа"""
        data_only = {}
        
        if 'sections' in analysis_result:
            for section_name, section_data in analysis_result['sections'].items():
                if isinstance(section_data, dict) and 'extracted_data' in section_data:
                    data_only[section_name] = section_data['extracted_data']
        
        return data_only


def main():
    parser = argparse.ArgumentParser(
        description='LLM Regex Analyzer - анализ документов с генерацией regex'
    )
    parser.add_argument('--file', '-f', required=True,
                       help='Путь к текстовому файлу для обработки')
    parser.add_argument('--instructions', '-i', 
                       default='instructions_regex_generation.txt',
                       help='Файл с инструкциями для LLM')
    parser.add_argument('--output', '-o',
                       help='Выходной JSON файл (по умолчанию: имя_файла_analyzed.json)')
    parser.add_argument('--regex-output', '-r',
                       help='Файл для сохранения только regex (по умолчанию: имя_файла.regex.json)')
    parser.add_argument('--data-output', '-d',
                       help='Файл для сохранения только данных (по умолчанию: имя_файла.data.json)')
    parser.add_argument('--validate', action='store_true',
                       help='Валидировать результаты с помощью сгенерированных regex')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='URL Ollama сервера')
    parser.add_argument('--model', '-m', required=True,
                       help='Название модели Ollama')
    parser.add_argument('--token', '-t',
                       help='Токен авторизации')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Отключить проверку SSL')
    
    args = parser.parse_args()
    
    # Проверяем входной файл
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"❌ Файл не найден: {input_file}")
        return
    
    # Проверяем файл с инструкциями
    instructions_file = Path(args.instructions)
    if not instructions_file.exists():
        print(f"❌ Файл с инструкциями не найден: {instructions_file}")
        return
    
    # Определяем выходные файлы
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}_analyzed.json"
    
    if args.regex_output:
        regex_output_file = Path(args.regex_output)
    else:
        regex_output_file = input_file.parent / f"{input_file.stem}.regex.json"
    
    if args.data_output:
        data_output_file = Path(args.data_output)
    else:
        data_output_file = input_file.parent / f"{input_file.stem}.data.json"
    
    # Создаем анализатор
    analyzer = LLMRegexAnalyzer(
        args.ollama_url,
        args.model,
        args.token,
        verify_ssl=not args.no_ssl_verify
    )
    
    # Тестируем соединение
    if not analyzer.test_connection():
        return
    
    # Читаем файлы
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"📄 Загружен документ: {input_file.name} ({len(text)} символов)")
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return
    
    # Загружаем инструкции
    instructions = analyzer.load_instructions(instructions_file)
    if not instructions:
        return
    
    print("\n" + "="*70)
    print("🚀 НАЧИНАЕМ АНАЛИЗ")
    print("="*70)
    
    # Анализируем документ
    analysis_result = analyzer.analyze_document(text, instructions)
    
    if not analysis_result or 'error' in analysis_result:
        print("\n❌ Анализ завершился с ошибкой")
        # Сохраняем ошибку для отладки
        error_file = input_file.parent / f"{input_file.stem}_error.json"
        analyzer.save_results(analysis_result, error_file)
        return
    
    # Добавляем метаинформацию
    full_result = {
        "file_info": {
            "source_file": input_file.name,
            "instructions_file": instructions_file.name,
            "analysis_method": "llm_regex_hybrid",
            "model": args.model
        },
        "analysis_result": analysis_result
    }
    
    # Сохраняем полный результат
    analyzer.save_results(full_result, output_file)
    
    # Извлекаем и сохраняем только regex паттерны
    regex_patterns = analyzer.extract_regex_patterns_only(analysis_result)
    if regex_patterns:
        regex_result = {
            "file_info": {
                "source_file": input_file.name,
                "pattern_source": "llm_generated",
                "model": args.model
            },
            "patterns": regex_patterns
        }
        analyzer.save_results(regex_result, regex_output_file)
        print(f"📝 Regex паттерны сохранены отдельно: {regex_output_file}")
    
    # Извлекаем и сохраняем только данные
    extracted_data = analyzer.extract_data_only(analysis_result)
    if extracted_data:
        data_result = {
            "file_info": {
                "source_file": input_file.name,
                "extraction_method": "llm_analysis",
                "model": args.model
            },
            "extracted_data": extracted_data
        }
        analyzer.save_results(data_result, data_output_file)
        print(f"📊 Извлеченные данные сохранены отдельно: {data_output_file}")
    
    # Валидация с помощью regex (опционально)
    if args.validate and regex_patterns:
        print("\n" + "="*70)
        print("🔍 ВАЛИДАЦИЯ REGEX ПАТТЕРНОВ")
        print("="*70)
        
        validation_result = analyzer.validate_with_regex(text, analysis_result.get('sections', {}))
        
        validation_file = input_file.parent / f"{input_file.stem}_validation.json"
        analyzer.save_results({
            "file_info": {
                "source_file": input_file.name,
                "validation_method": "regex_matching"
            },
            "validation_results": validation_result
        }, validation_file)
        
        print(f"✅ Результаты валидации: {validation_file}")
    
    print("\n" + "="*70)
    print("✅ АНАЛИЗ ЗАВЕРШЕН!")
    print("="*70)
    print(f"📁 Полный результат: {output_file}")
    print(f"📝 Regex паттерны: {regex_output_file}")
    print(f"📊 Извлеченные данные: {data_output_file}")
    if args.validate:
        print(f"🔍 Валидация: {validation_file}")


if __name__ == "__main__":
    main()

