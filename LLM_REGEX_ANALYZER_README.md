# LLM Regex Analyzer - Гибридный подход

## 🎯 Концепция

**LLM Regex Analyzer** - это гибридный подход к парсингу технических документов, объединяющий преимущества LLM и регулярных выражений:

1. **LLM анализирует документ** и находит конкретные данные
2. **LLM генерирует regex паттерны** на основе найденных данных
3. **Regex сохраняются** для переиспользования на похожих документах
4. **Опциональная валидация** - проверка работоспособности regex

## 🔥 Преимущества

| Характеристика | LLM-парсер | Regex-парсер | **LLM Regex Analyzer** |
|----------------|------------|--------------|------------------------|
| Понимание контекста | ✅ Отлично | ❌ Слабо | ✅ Отлично |
| Скорость работы | ❌ Медленно | ✅ Быстро | ✅ Быстро (после генерации) |
| Переиспользование | ❌ Нет | ✅ Да | ✅ Да |
| Точность | ✅ Высокая | ⚠️ Средняя | ✅ Высокая |
| Стоимость (продакшн) | ❌ Высокая | ✅ Низкая | ✅ Низкая |
| Адаптивность | ✅ Да | ❌ Нет | ✅ Да |

## 📋 Требования

- Python 3.8+
- Ollama с загруженной моделью
- requests, urllib3

```bash
pip install requests urllib3
```

## 🚀 Быстрый старт

### Шаг 1: Запуск анализа

```bash
python3 llm_regex_analyzer.py \
  --file output/01-0530_6856.txt \
  --instructions instructions_regex_generation.txt \
  --model "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --ollama-url http://localhost:11434 \
  --validate
```

### Шаг 2: Проверка результатов

После выполнения создаются 3-4 файла:

1. **`*_analyzed.json`** - полный результат с данными и regex
2. **`*.regex.json`** - только regex паттерны (для переиспользования)
3. **`*.data.json`** - только извлеченные данные
4. **`*_validation.json`** - результаты валидации (если --validate)

## 📁 Структура результата

### Полный результат (`*_analyzed.json`)

```json
{
  "file_info": {
    "source_file": "01-0530_6856.txt",
    "instructions_file": "instructions_regex_generation.txt",
    "analysis_method": "llm_regex_hybrid",
    "model": "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
  },
  "analysis_result": {
    "sections": {
      "document_metadata": {
        "extracted_data": {
          "document_number": "01-0530",
          "date": "04.2025",
          "addressees": ["Начальнику ЭСПЦ", ...]
        },
        "regex_patterns": {
          "document_number": {
            "pattern": "^(\\d{2}-\\s*\\d{4})$",
            "flags": ["MULTILINE"],
            "context_before": "--- Страница 1 ---",
            "context_after": "\\d{2}\\.\\d{4}",
            "example_match": "01-0530",
            "description": "Номер документа"
          }
        }
      },
      "steel_parameters": { ... },
      "table_1_chemical_composition": { ... }
    }
  }
}
```

### Только Regex (`*.regex.json`)

```json
{
  "file_info": {
    "source_file": "01-0530_6856.txt",
    "pattern_source": "llm_generated"
  },
  "patterns": {
    "document_metadata": {
      "document_number": {
        "pattern": "^(\\d{2}-\\s*\\d{4})$",
        "flags": ["MULTILINE"],
        "description": "Номер документа"
      }
    }
  }
}
```

### Только данные (`*.data.json`)

```json
{
  "file_info": {
    "source_file": "01-0530_6856.txt",
    "extraction_method": "llm_analysis"
  },
  "extracted_data": {
    "document_metadata": {
      "document_number": "01-0530",
      "date": "04.2025",
      "addressees": [...]
    }
  }
}
```

## 🔧 Параметры запуска

```bash
python3 llm_regex_analyzer.py [OPTIONS]
```

### Обязательные параметры:

- `--file, -f` - Путь к текстовому файлу для анализа
- `--model, -m` - Название модели Ollama

### Опциональные параметры:

- `--instructions, -i` - Файл с инструкциями (по умолчанию: `instructions_regex_generation.txt`)
- `--output, -o` - Файл для полного результата (по умолчанию: `*_analyzed.json`)
- `--regex-output, -r` - Файл для regex паттернов (по умолчанию: `*.regex.json`)
- `--data-output, -d` - Файл для данных (по умолчанию: `*.data.json`)
- `--validate` - Включить валидацию regex паттернов
- `--ollama-url` - URL Ollama сервера (по умолчанию: `http://localhost:11434`)
- `--token, -t` - Токен авторизации для Ollama
- `--no-ssl-verify` - Отключить проверку SSL

## 📝 Инструкции для LLM

Файл `instructions_regex_generation.txt` содержит детальные инструкции для LLM по:

1. **Что искать** в документе
2. **Как создавать regex** для найденных данных
3. **Какой формат** использовать для результата

### Структура инструкций:

```
РАЗДЕЛ 1: МЕТАДАННЫЕ ДОКУМЕНТА
  [TASK: extract_document_metadata]
  - Номер документа
  - Дата
  - Адресаты

РАЗДЕЛ 2: ПАРАМЕТРЫ СТАЛИ
  [TASK: extract_steel_parameters]
  - Марка стали
  - Код ОЭМК
  - Экспортное наименование
  - ...

РАЗДЕЛ 3: ТАБЛИЦА 1 - ХИМИЧЕСКИЙ СОСТАВ
  [TASK: extract_table_1_chemical_composition]
  - Элементы
  - Рекомендуемые значения
  - Аттестатные значения
  - ...
```

## 🎨 Кастомизация инструкций

Вы можете создать свои инструкции для специфических типов документов:

```bash
# Копируем шаблон
cp instructions_regex_generation.txt my_custom_instructions.txt

# Редактируем под свои нужды
nano my_custom_instructions.txt

# Запускаем с кастомными инструкциями
python3 llm_regex_analyzer.py \
  --file my_document.txt \
  --instructions my_custom_instructions.txt \
  --model "..."
```

## 🔄 Workflow

### Первичный анализ (с генерацией regex):

```bash
python3 llm_regex_analyzer.py \
  --file document.txt \
  --model "model_name" \
  --validate
```

**Результат:**
- `document_analyzed.json` - полный анализ
- `document.regex.json` - regex паттерны
- `document.data.json` - извлеченные данные
- `document_validation.json` - валидация

### Переиспользование regex для похожих документов:

```bash
# Используйте regex_parser.py с сохраненными паттернами
python3 regex_parser.py \
  --file similar_document.txt \
  --regex-file document.regex.json \
  --mode apply
```

## 📊 Примеры использования

### Пример 1: Базовый анализ

```bash
python3 llm_regex_analyzer.py \
  -f output/01-0530_6856.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
```

### Пример 2: С валидацией

```bash
python3 llm_regex_analyzer.py \
  -f output/01-0530_6856.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --validate
```

### Пример 3: Кастомные выходные файлы

```bash
python3 llm_regex_analyzer.py \
  -f output/01-0530_6856.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  -o results/full_analysis.json \
  -r results/patterns.json \
  -d results/data_only.json
```

### Пример 4: С внешним Ollama сервером

```bash
python3 llm_regex_analyzer.py \
  -f output/01-0530_6856.txt \
  -m "llama3:latest" \
  --ollama-url https://my-ollama-server.com \
  --token "my-secret-token" \
  --no-ssl-verify
```

## 🐛 Отладка

### Проблема: LLM не возвращает валидный JSON

**Решение:**
1. Проверьте файл `*_error.json` с raw ответом LLM
2. Попробуйте другую модель (более мощную)
3. Упростите инструкции

### Проблема: Regex не работают

**Решение:**
1. Проверьте файл `*_validation.json`
2. Исправьте паттерны вручную в `*.regex.json`
3. Повторно запустите с `--validate`

### Проблема: Timeout при больших документах

**Решение:**
```python
# В llm_regex_analyzer.py увеличьте timeout:
response = self.session.post(..., timeout=600)  # 10 минут
```

## 🔄 Интеграция с другими парсерами

### С regex_parser.py:

```bash
# 1. Генерируем regex
python3 llm_regex_analyzer.py -f doc.txt -m "model" -r patterns.regex.json

# 2. Применяем к другим документам
python3 regex_parser.py -f doc2.txt --regex-file patterns.regex.json --mode apply
```

### С llm_parser.py (сравнение):

```bash
# Наш новый подход
python3 llm_regex_analyzer.py -f doc.txt -m "model" -d new_result.json

# Старый LLM подход
python3 llm_parser.py -f doc.txt -m "model" -o old_result.json

# Сравниваем результаты
diff new_result.json old_result.json
```

## 📈 Производительность

| Этап | Время | Можно кешировать |
|------|-------|------------------|
| Первый анализ (LLM) | ~60-180 сек | Нет |
| Генерация regex | ~0 сек (в составе анализа) | **Да** |
| Применение regex к новому документу | ~0.1-1 сек | **Да** |

**Вывод:** После первого анализа, regex можно переиспользовать для обработки тысяч похожих документов за секунды!

## 🎓 Советы и best practices

1. **Начните с малого** - сначала проанализируйте один образец документа
2. **Проверяйте regex** - используйте `--validate` для проверки
3. **Сохраняйте паттерны** - создавайте библиотеку regex для разных типов документов
4. **Кастомизируйте инструкции** - адаптируйте под свои документы
5. **Итеративно улучшайте** - при ошибках корректируйте инструкции и повторяйте

## 📚 Дополнительные ресурсы

- [Основной README](README.md)
- [LLM Parser README](LLM_PARSER_README.md)
- [Regex Parser (старый)](regex_parser.py)
- [Инструкции для LLM](instructions_regex_generation.txt)

## 🤝 Вклад

Этот подход экспериментальный и может быть улучшен. Идеи по улучшению:

1. Автоматическое тестирование regex на нескольких образцах
2. Машинное обучение для выбора лучших паттернов
3. UI для редактирования и отладки regex
4. Библиотека предобученных паттернов для разных типов документов

---

**Создано:** 31.10.2025  
**Лицензия:** MIT


