#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM-парсер для извлечения структурированных данных из текстовых файлов
Использует Ollama для обработки текста с помощью пользовательских промптов
"""

import argparse
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import re
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class LLMDocumentParser:
    """Парсер документов с использованием LLM через Ollama"""
    
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получает информацию о модели"""
        try:
            response = self.session.post(f"{self.ollama_url}/api/show", 
                                       json={"name": self.model})
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️ Не удалось получить информацию о модели: {response.status_code}")
                return {}
        except Exception as e:
            print(f"⚠️ Ошибка получения информации о модели: {e}")
            return {}
    
    def split_text_into_chunks(self, text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
        """Разбивает текст на чанки со скользящим окном"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Если это не последний чанк, ищем ближайший конец предложения
            if end < len(text):
                # Ищем последний символ конца предложения в пределах overlap
                search_start = max(start + chunk_size - overlap, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('\n', search_start, end)
                if sentence_end != -1:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Перекрытие для следующего чанка
            start = max(start + chunk_size - overlap, end)
            
            # Предотвращаем бесконечный цикл
            if start >= len(text):
                break
        
        return chunks
    
    def load_prompt_files(self, prompt_dir: Path) -> Dict[str, str]:
        """Загружает промпты из отдельных файлов"""
        prompts = {}
        
        # Основной промпт
        main_prompt_file = prompt_dir / "prompt_main.txt"
        if main_prompt_file.exists():
            prompts["main"] = main_prompt_file.read_text(encoding='utf-8')
        
        # Получаем типы блоков из основного промпта
        block_types = self.extract_block_types_from_main_prompt(prompts.get("main", ""))
        
        # Загружаем только те промпты, которые определены в main
        for block_type in block_types:
            if block_type == "unknown":
                continue  # Пропускаем unknown блоки
                
            prompt_file = prompt_dir / f"prompt_{block_type}.txt"
            if prompt_file.exists():
                prompts[block_type] = prompt_file.read_text(encoding='utf-8')
            else:
                print(f"⚠️ Файл промпта не найден: {prompt_file}")
        
        return prompts
    
    def extract_block_types_from_main_prompt(self, main_prompt: str) -> List[str]:
        """Извлекает типы блоков из основного промпта"""
        import re
        
        # Ищем все "block_type": "тип_блока" в промпте
        pattern = r'"block_type":\s*"([^"]+)"'
        matches = re.findall(pattern, main_prompt)
        
        # Убираем дубликаты и возвращаем список
        return list(set(matches))
    
    def get_main_prompt(self, prompts: Dict[str, str]) -> str:
        """Возвращает основной промпт для разбивки на блоки"""
        return prompts.get('main', '')
    
    def get_specialized_prompt(self, prompts: Dict[str, str], block_type: str) -> str:
        """Возвращает специализированный промпт для типа блока"""
        return prompts.get(block_type, '')
    
    def parse_blocks_from_main_response(self, response: str) -> List[Dict[str, Any]]:
        """Парсит ответ основного промпта и извлекает блоки"""
        try:
            # Убираем markdown код блоки если есть
            if '```' in response:
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    response = response[json_start:json_end]
            
            parsed = json.loads(response)
            return parsed.get('blocks', [])
        except json.JSONDecodeError as e:
            print(f"❌ Ошибка парсинга блоков: {e}")
            return []
    
    def call_ollama(self, prompt: str, context: str = "") -> Optional[str]:
        """Вызывает Ollama API"""
        try:
            # Объединяем промпт и контекст
            full_prompt = f"{prompt}\n\nКонтекст:\n{context}" if context else prompt
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # Низкая температура для более точных результатов
                    "top_p": 0.9
                }
            }
            
            response = self.session.post(f"{self.ollama_url}/api/generate", 
                                       json=payload, 
                                       timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            else:
                print(f"❌ Ошибка API: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Ошибка вызова Ollama: {e}")
            return None
    
    def process_chunk_with_smart_prompt(self, chunk: str, prompt_sections: Dict[str, str]) -> Dict[str, Any]:
        """Обрабатывает чанк с умным выбором промпта"""
        # Определяем тип чанка
        chunk_type = self.determine_chunk_type(chunk)
        print(f"  Тип чанка: {chunk_type}")
        
        # Получаем соответствующий промпт
        prompt = self.get_prompt_for_chunk_type(prompt_sections, chunk_type)
        
        if not prompt:
            print(f"  ⚠️ Промпт для типа '{chunk_type}' не найден")
            return {}
        
        print(f"  Применяем промпт для типа: {chunk_type}")
        
        # Вызываем Ollama с соответствующим промптом
        result = self.call_ollama(prompt, chunk)
        
        if result:
            try:
                # Пытаемся распарсить JSON результат
                parsed_result = json.loads(result)
                return parsed_result
            except json.JSONDecodeError:
                # Если не JSON, сохраняем как текст
                return {"raw_result": result}
        
        return {}
    
    def process_document_two_stage(self, content: str, prompts: Dict[str, str], file_path: Path) -> Dict[str, Any]:
        """Двухэтапная обработка документа"""
        print("🔄 Этап 1: Разбивка на логические блоки...")
        
        # Этап 1: Получаем основной промпт и разбиваем текст на блоки
        main_prompt = self.get_main_prompt(prompts)
        if not main_prompt:
            print("❌ Основной промпт не найден")
            return {}
        
        # Вызываем основной промпт для разбивки на блоки
        main_response = self.call_ollama(main_prompt, content)
        if not main_response:
            print("❌ Не удалось получить разбивку на блоки")
            return {}
        
        # Парсим блоки из ответа
        blocks = self.parse_blocks_from_main_response(main_response)
        if not blocks:
            print("❌ Не удалось распарсить блоки")
            return {}
        
        print(f"📦 Найдено блоков: {len(blocks)}")
        
        # Сохраняем промежуточный файл с результатами первого этапа
        intermediate_file = file_path.parent / f"{file_path.stem}.tmp.json"
        intermediate_data = {
            "file_info": {
                "filename": file_path.name,
                "stage": "blocks_extraction",
                "total_blocks": len(blocks)
            },
            "blocks": blocks
        }
        
        try:
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, ensure_ascii=False, indent=2)
            print(f"💾 Промежуточный файл сохранен: {intermediate_file}")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения промежуточного файла: {e}")
        
        # Этап 2: Обрабатываем каждый блок специализированным промптом
        print("🔄 Этап 2: Обработка блоков специализированными промптами...")
        
        final_results = {}
        unknown_blocks = []
        
        for i, block in enumerate(blocks):
            block_type = block.get('block_type', '')
            block_content = block.get('content', '')
            
            print(f"  📄 Обрабатываем блок {i+1}/{len(blocks)}: {block_type}")
            
            # Проверяем, является ли блок неизвестным
            if block_type == 'unknown':
                print(f"    ⚠️ Неизвестный блок найден, сохраняем в .err файл")
                unknown_blocks.append({
                    "block_index": i + 1,
                    "content": block_content,
                    "start_marker": block.get('start_marker', ''),
                    "end_marker": block.get('end_marker', '')
                })
                continue
            
            # Получаем специализированный промпт
            specialized_prompt = self.get_specialized_prompt(prompts, block_type)
            if not specialized_prompt:
                print(f"    ⚠️ Промпт для типа '{block_type}' не найден, сохраняем в .err файл")
                unknown_blocks.append({
                    "block_index": i + 1,
                    "block_type": block_type,
                    "content": block_content,
                    "start_marker": block.get('start_marker', ''),
                    "end_marker": block.get('end_marker', ''),
                    "reason": f"Промпт для типа '{block_type}' не найден"
                })
                continue
            
            # Обрабатываем блок
            block_result = self.call_ollama(specialized_prompt, block_content)
            if block_result:
                try:
                    # Пытаемся распарсить JSON результат
                    parsed_result = json.loads(block_result)
                    final_results.update(parsed_result)
                except json.JSONDecodeError:
                    # Если не JSON, сохраняем как текст
                    final_results[f"{block_type}_raw"] = block_result
            
            # Пауза между блоками
            time.sleep(0.5)
        
        # Сохраняем неизвестные блоки в .err файл
        if unknown_blocks:
            error_file = file_path.parent / f"{file_path.stem}.err"
            error_data = {
                "file_info": {
                    "filename": file_path.name,
                    "processing_stage": "unknown_blocks",
                    "total_unknown_blocks": len(unknown_blocks)
                },
                "unknown_blocks": unknown_blocks
            }
            
            try:
                with open(error_file, 'w', encoding='utf-8') as f:
                    json.dump(error_data, f, ensure_ascii=False, indent=2)
                print(f"⚠️ Неизвестные блоки сохранены в: {error_file}")
            except Exception as e:
                print(f"❌ Ошибка сохранения .err файла: {e}")
        
        return final_results
    
    def process_file(self, file_path: Path, prompt_dir: Path) -> Dict[str, Any]:
        """Обрабатывает один файл"""
        print(f"📄 Обрабатываем файл: {file_path.name}")
        
        # Читаем файл
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Ошибка чтения файла: {e}")
            return {}
        
        # Загружаем промпты из отдельных файлов
        prompts = self.load_prompt_files(prompt_dir)
        if not prompts:
            print("❌ Не удалось загрузить промпты")
            return {}
        
        print(f"📝 Загружено промптов: {len(prompts)}")
        for prompt_type in prompts.keys():
            print(f"  - {prompt_type}")
        
        # Используем двухэтапную обработку
        final_results = self.process_document_two_stage(content, prompts, file_path)
        
        # Формируем результат
        all_results = {
            "file_info": {
                "filename": file_path.name,
                "processing_method": "two_stage_separate_prompts",
                "total_prompts": len(prompts)
            },
            "extracted_data": final_results
        }
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Сохраняет результаты в JSON файл"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 Результаты сохранены: {output_path}")
        except Exception as e:
            print(f"❌ Ошибка сохранения: {e}")


def main():
    parser = argparse.ArgumentParser(description='LLM-парсер документов через Ollama')
    parser.add_argument('--file', '-f', required=True,
                       help='Путь к текстовому файлу для обработки')
    parser.add_argument('--prompt-dir', '-p', default='prompts',
                       help='Каталог с файлами промптов (по умолчанию: prompts)')
    parser.add_argument('--output', '-o', 
                       help='Выходной JSON файл (по умолчанию: имя_файла.json)')
    parser.add_argument('--ollama-url', default='http://localhost:11434',
                       help='URL Ollama сервера (по умолчанию: http://localhost:11434)')
    parser.add_argument('--model', '-m', required=True,
                       help='Название модели Ollama')
    parser.add_argument('--token', '-t',
                       help='Токен авторизации (если нужен)')
    parser.add_argument('--chunk-size', type=int, default=2000,
                       help='Размер чанка в символах (по умолчанию: 2000)')
    parser.add_argument('--overlap', type=int, default=200,
                       help='Размер перекрытия между чанками (по умолчанию: 200)')
    parser.add_argument('--no-ssl-verify', action='store_true',
                       help='Отключить проверку SSL сертификата')
    
    args = parser.parse_args()
    
    # Проверяем входные файлы
    input_file = Path(args.file)
    if not input_file.exists():
        print(f"❌ Файл не найден: {input_file}")
        return
    
    prompt_dir = Path(args.prompt_dir)
    if not prompt_dir.exists():
        print(f"❌ Каталог промптов не найден: {prompt_dir}")
        return
    
    # Создаем парсер
    parser_instance = LLMDocumentParser(args.ollama_url, args.model, args.token, 
                                      verify_ssl=not args.no_ssl_verify)
    
    # Тестируем соединение
    if not parser_instance.test_connection():
        return
    
    # Получаем информацию о модели
    model_info = parser_instance.get_model_info()
    if model_info:
        print(f"🤖 Модель: {args.model}")
        if 'parameter_size' in model_info:
            print(f"📊 Размер модели: {model_info['parameter_size']}")
    
    # Определяем выходной файл
    if args.output:
        output_file = Path(args.output)
    else:
        output_file = input_file.parent / f"{input_file.stem}.json"
    
    # Обрабатываем файл
    print(f"\n🚀 Начинаем обработку...")
    results = parser_instance.process_file(input_file, prompt_dir)
    
    if results:
        parser_instance.save_results(results, output_file)
        print(f"\n✅ Обработка завершена!")
        print(f"📁 Результат: {output_file}")
    else:
        print(f"\n❌ Обработка не удалась")


if __name__ == "__main__":
    main()
