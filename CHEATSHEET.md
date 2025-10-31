# 📋 Шпаргалка команд - Letter Explorer

## 🚀 Быстрый старт (3 команды)

```bash
# 1. OCR
python ocr_script.py --method easyocr

# 2. Парсинг с генерацией regex
python3 llm_regex_analyzer.py -f output/file.txt -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" --validate

# 3. Проверка результатов
cat output/file.data.json | jq '.extracted_data'
```

---

## 📸 OCR - Распознавание текста

### EasyOCR (рекомендуется)
```bash
python ocr_script.py --method easyocr
python ocr_script.py --method easyocr --gpu  # С GPU
```

### Tesseract
```bash
python ocr_script.py --method tesseract
```

### Кастомные каталоги
```bash
python ocr_script.py --method easyocr --input /path/input --output /path/output
```

---

## 🎯 Парсинг - Извлечение данных

### LLM Regex Analyzer ⭐ (ЛУЧШИЙ)
```bash
# Полный анализ + генерация regex + валидация
python3 llm_regex_analyzer.py \
  -f output/01-0530_6856.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --validate

# Без валидации
python3 llm_regex_analyzer.py -f file.txt -m "model"

# С кастомными инструкциями
python3 llm_regex_analyzer.py -f file.txt -m "model" -i my_instructions.txt
```

### Regex Parser (быстрое применение)
```bash
# Генерация regex (первый раз)
python3 regex_parser.py -f doc.txt -m "model" --mode generate

# Применение существующих regex
python3 regex_parser.py -f doc2.txt --regex-file doc.regex.json --mode apply

# Оба этапа сразу
python3 regex_parser.py -f doc.txt -m "model" --mode both
```

### LLM Parser (чистый LLM)
```bash
python3 llm_parser.py -f file.txt -m "model" --prompt-dir prompts/
```

---

## 🔍 Проверка результатов

### JSON форматирование
```bash
# Полный результат
cat output/file_analyzed.json | jq '.'

# Только данные
cat output/file.data.json | jq '.extracted_data'

# Только regex
cat output/file.regex.json | jq '.patterns'

# Валидация
cat output/file_validation.json | jq '.'
```

### Конкретные поля
```bash
# Метаданные документа
cat output/file.data.json | jq '.extracted_data.document_metadata'

# Параметры стали
cat output/file.data.json | jq '.extracted_data.steel_parameters'

# Химический состав
cat output/file.data.json | jq '.extracted_data.table_1_chemical_composition'
```

---

## 🔧 Ollama команды

### Управление сервером
```bash
# Запуск
ollama serve

# Проверка статуса
curl http://localhost:11434/api/tags

# Список моделей
ollama list
```

### Работа с моделями
```bash
# Загрузка модели
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest

# Другие модели
ollama pull llama3:8b
ollama pull llama3:70b
ollama pull mistral:latest

# Удаление модели
ollama rm model_name
```

---

## 🔄 Workflow для массовой обработки

### 1. Анализ первого документа
```bash
python3 llm_regex_analyzer.py \
  -f output/doc1.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --validate
```

### 2. Применение regex к остальным
```bash
# Один файл
python3 regex_parser.py \
  -f output/doc2.txt \
  --regex-file output/doc1.regex.json \
  --mode apply

# Пакетная обработка
for file in output/*.txt; do
  python3 regex_parser.py \
    -f "$file" \
    --regex-file output/doc1.regex.json \
    --mode apply
done
```

---

## 📁 Структура результатов

```
output/
├── file.txt                    # OCR результат
├── file_analyzed.json          # Полный анализ (данные + regex)
├── file.data.json              # Только извлеченные данные
├── file.regex.json             # Только regex паттерны
└── file_validation.json        # Валидация regex
```

---

## 🐛 Отладка

### Проверка соединения с Ollama
```bash
curl -X GET http://localhost:11434/api/tags
```

### Проверка модели
```bash
ollama list | grep yandex
```

### Логи
```bash
# Файл с ошибками
cat output/file_error.json

# Последние строки лога
tail -f /var/log/ollama.log  # если есть
```

### Повторный запуск с отладкой
```bash
python3 llm_regex_analyzer.py -f file.txt -m "model" --validate 2>&1 | tee debug.log
```

---

## 💡 Полезные комбинации

### OCR + Парсинг (полный pipeline)
```bash
# 1. OCR
python ocr_script.py --method easyocr --input input/ --output output/

# 2. Парсинг всех .txt файлов
for file in output/*.txt; do
  python3 llm_regex_analyzer.py -f "$file" -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
done
```

### Сравнение результатов разных методов
```bash
# LLM Regex Analyzer
python3 llm_regex_analyzer.py -f doc.txt -m "model" -d llm_regex_result.json

# LLM Parser
python3 llm_parser.py -f doc.txt -m "model" -o llm_result.json

# Сравнение
diff llm_regex_result.json llm_result.json
```

### Экспорт в CSV
```bash
# Установка jq если нет
sudo apt-get install jq

# Экспорт метаданных
cat output/*.data.json | jq -r '.extracted_data.document_metadata | [.document_number, .date] | @csv'

# Экспорт параметров стали
cat output/*.data.json | jq -r '.extracted_data.steel_parameters | [.steel_grade, .export_name] | @csv'
```

---

## ⚙️ Настройки и опции

### LLM Regex Analyzer
```bash
--file, -f           # Входной файл (обязательно)
--model, -m          # Модель Ollama (обязательно)
--instructions, -i   # Файл с инструкциями (по умолчанию: instructions_regex_generation.txt)
--output, -o         # Файл для полного результата
--regex-output, -r   # Файл для regex
--data-output, -d    # Файл для данных
--validate           # Включить валидацию
--ollama-url         # URL Ollama (по умолчанию: http://localhost:11434)
--token, -t          # Токен авторизации
--no-ssl-verify      # Отключить проверку SSL
```

### Regex Parser
```bash
--file, -f           # Входной файл
--model, -m          # Модель Ollama
--mode               # Режим: generate, apply, both
--regex-file         # Файл с regex паттернами
--output, -o         # Выходной файл
```

### OCR Script
```bash
--method, -m         # Метод: easyocr, tesseract
--input, -i          # Каталог с входными файлами
--output, -o         # Каталог для выходных файлов
--gpu                # Использовать GPU (только easyocr)
```

---

## 📊 Примеры результатов

### Метаданные
```json
{
  "document_number": "01-0530",
  "date": "04.2025",
  "addressees": ["Начальнику ЭСПЦ", "Начальнику СПЦ-1"]
}
```

### Параметры стали
```json
{
  "steel_grade": "6856",
  "code_oemk": "dэ7",
  "export_name": "42CrMoS4+H",
  "analog": "З8ХГМ"
}
```

### Regex паттерн
```json
{
  "pattern": "стали марки\\s+(\\d+)",
  "flags": ["MULTILINE"],
  "example_match": "6856",
  "description": "Марка стали"
}
```

---

## 🎓 Когда использовать что

| Задача | Инструмент |
|--------|-----------|
| Один документ, нужен быстрый результат | LLM Parser |
| Много похожих документов | LLM Regex Analyzer + Regex Parser |
| Нужны regex для переиспользования | LLM Regex Analyzer |
| Уже есть regex паттерны | Regex Parser (--mode apply) |
| Эксперимент с разными промптами | LLM Parser |
| Продакшн обработка | LLM Regex Analyzer → Regex Parser |

---

## 📚 Документация

- [README.md](README.md) - Главная документация
- [LLM_REGEX_ANALYZER_README.md](LLM_REGEX_ANALYZER_README.md) - Гибридный подход
- [LLM_PARSER_README.md](LLM_PARSER_README.md) - LLM парсер
- [QUICKSTART_REGEX_ANALYZER.md](QUICKSTART_REGEX_ANALYZER.md) - Быстрый старт

---

**💡 Совет дня:** Для массовой обработки документов используйте LLM Regex Analyzer один раз, затем Regex Parser для всех остальных. Это сэкономит часы времени!


