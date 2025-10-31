#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-based парсер с человеческими инструкциями
Человек пишет инструкции на естественном языке, LLM их выполняет
"""

import re
import json
import argparse
import requests
import urllib3
from pathlib import Path
from typing import Dict, List, Any, Optional

# Отключаем предупреждения SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMInstructionParser:
    """Парсер документов на основе LLM инструкций"""
    
    def __init__(self, ollama_url: str, model: str, token: Optional[str] = None, 
                 verify_ssl: bool = True):
        self.ollama_url = ollama_url.rstrip('/')
        self.model = model
        self.verify_ssl = verify_ssl
        
        # Настраиваем сессию
        self.session = requests.Session()
        if token:
            self.session.headers['X-Access-Token'] = token
    
    def parse_instructions_file(self, instructions_path: str) -> List[Dict[str, Any]]:
        """Парсит файл с инструкциями на задачи"""
        with open(instructions_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tasks = []
        
        # Разбиваем на задачи по маркеру [TASK: название]
        task_pattern = r'\[TASK:\s*(.+?)\](.*?)(?=\[TASK:|$)'
        matches = re.finditer(task_pattern, content, re.DOTALL)
        
        for match in matches:
            task_name = match.group(1).strip()
            task_content = match.group(2).strip()
            
            # Разбираем содержимое задачи
            description = ""
            output_key = task_name
            output_format = "json"
            
            # Ищем специальные директивы
            if '@OUTPUT_KEY:' in task_content:
                key_match = re.search(r'@OUTPUT_KEY:\s*(\S+)', task_content)
                if key_match:
                    output_key = key_match.group(1)
                    task_content = task_content.replace(key_match.group(0), '')
            
            if '@FORMAT:' in task_content:
                format_match = re.search(r'@FORMAT:\s*(\S+)', task_content)
                if format_match:
                    output_format = format_match.group(1)
                    task_content = task_content.replace(format_match.group(0), '')
            
            description = task_content.strip()
            
            tasks.append({
                'name': task_name,
                'output_key': output_key,
                'description': description,
                'format': output_format
            })
        
        return tasks
    
    def call_llm(self, prompt: str) -> str:
        """Вызов LLM через Ollama API"""
        try:
            url = f"{self.ollama_url}/api/generate"
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Низкая температура для точности
                    "num_predict": 4096
                }
            }
            
            response = self.session.post(
                url,
                json=payload,
                verify=self.verify_ssl,
                timeout=300
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                print(f"❌ Ошибка LLM API: {response.status_code}")
                print(f"   {response.text}")
                return ""
        
        except Exception as e:
            print(f"❌ Ошибка при вызове LLM: {e}")
            return ""
    
    def execute_task(self, task: Dict[str, Any], document_text: str) -> Any:
        """Выполняет одну задачу парсинга"""
        print(f"\n🔍 Выполняется задача: {task['name']}")
        
        # Формируем промпт
        prompt = f"""Ты - эксперт по извлечению данных из технических документов.

ДОКУМЕНТ:
```
{document_text[:15000]}  # Ограничиваем размер для LLM
```

ЗАДАЧА:
{task['description']}

ВАЖНО:
1. Отвечай ТОЛЬКО валидным JSON, без дополнительных объяснений
2. Если данные не найдены, верни пустой объект {{}}
3. Сохраняй точные значения из документа
4. Для чисел используй числовой тип, для текста - строки

ОТВЕТ (только JSON):"""

        # Вызываем LLM
        response = self.call_llm(prompt)
        
        if not response:
            print(f"   ⚠️  Пустой ответ от LLM")
            return {}
        
        # Извлекаем JSON из ответа
        try:
            # Ищем JSON в ответе (может быть обёрнут в markdown или текст)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                print(f"   ✅ Успешно извлечено: {len(str(result))} символов")
                return result
            else:
                print(f"   ⚠️  JSON не найден в ответе")
                print(f"   Ответ: {response[:200]}...")
                return {}
        
        except json.JSONDecodeError as e:
            print(f"   ❌ Ошибка парсинга JSON: {e}")
            print(f"   Ответ: {response[:500]}...")
            return {}
    
    def parse_document(self, document_path: str, instructions_path: str) -> Dict[str, Any]:
        """Парсит документ по инструкциям"""
        # Читаем документ
        with open(document_path, 'r', encoding='utf-8') as f:
            document_text = f.read()
        
        print(f"📄 Загружен документ: {Path(document_path).name}")
        print(f"   Размер: {len(document_text)} символов")
        
        # Читаем инструкции
        tasks = self.parse_instructions_file(instructions_path)
        print(f"📋 Загружено задач: {len(tasks)}")
        for task in tasks:
            print(f"   - {task['name']}")
        
        # Выполняем задачи
        result = {
            "file_info": {
                "source_file": Path(document_path).name,
                "instructions_file": Path(instructions_path).name,
                "parsing_method": "llm_instructions",
                "model": self.model
            },
            "extracted_data": {}
        }
        
        for task in tasks:
            task_result = self.execute_task(task, document_text)
            result['extracted_data'][task['output_key']] = task_result
        
        return result


def main():
    parser = argparse.ArgumentParser(
        description='LLM-based парсер с человеческими инструкциями'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Входной текстовый файл')
    parser.add_argument('--instructions', '-ins', required=True,
                       help='Файл с инструкциями для LLM')
    parser.add_argument('--output', '-o',
                       help='Выходной JSON файл (по умолчанию: имя_файла_llm_parsed.json)')
    parser.add_argument('--model', '-m', required=True,
                       help='Модель LLM (например: yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest)')
    parser.add_argument('--ollama-url', required=True,
                       help='URL Ollama сервера (например: https://server:443)')
    parser.add_argument('--token', '-t',
                       help='Токен авторизации для API')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Отключить проверку SSL сертификата')
    
    args = parser.parse_args()
    
    # Проверяем файлы
    input_path = Path(args.input)
    instructions_path = Path(args.instructions)
    
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        return
    
    if not instructions_path.exists():
        print(f"❌ Файл инструкций не найден: {instructions_path}")
        return
    
    # Определяем выходной файл
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_llm_parsed.json"
    
    print("="*80)
    print("🤖 LLM INSTRUCTION PARSER")
    print("="*80)
    
    # Инициализируем парсер
    llm_parser = LLMInstructionParser(
        ollama_url=args.ollama_url,
        model=args.model,
        token=args.token,
        verify_ssl=not args.no_ssl_verify
    )
    
    # Парсим документ
    result = llm_parser.parse_document(str(input_path), str(instructions_path))
    
    # Сохраняем результат
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Результат сохранён: {output_path}")
    print()
    print("📊 Извлечено разделов:")
    for key, value in result['extracted_data'].items():
        if isinstance(value, dict):
            print(f"   {key}: {len(value)} полей")
        elif isinstance(value, list):
            print(f"   {key}: {len(value)} элементов")
        else:
            print(f"   {key}: {type(value).__name__}")
    
    print("="*80)


if __name__ == "__main__":
    main()




