# 🚀 Быстрый старт: LLM Regex Analyzer

## За 3 минуты до первого результата

### 1️⃣ Проверка Ollama

```bash
# Убедитесь, что Ollama запущена
curl http://localhost:11434/api/tags

# Если не запущена:
ollama serve
```

### 2️⃣ Запуск анализа

```bash
cd /home/dev/letterexplorer

python3 llm_regex_analyzer.py \
  --file output/01-0530_6856.txt \
  --model "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --validate
```

### 3️⃣ Проверка результатов

```bash
# Полный анализ с данными и regex
cat output/01-0530_6856_analyzed.json | jq '.'

# Только regex паттерны
cat output/01-0530_6856.regex.json | jq '.patterns'

# Только извлеченные данные
cat output/01-0530_6856.data.json | jq '.extracted_data'

# Результаты валидации
cat output/01-0530_6856_validation.json | jq '.'
```

## 📋 Что происходит

1. **LLM читает документ** (output/01-0530_6856.txt)
2. **LLM читает инструкции** (instructions_regex_generation.txt)
3. **LLM находит данные** в документе:
   - Номер документа: 01-0530
   - Дата: 04.2025
   - Марка стали: 6856
   - Химический состав (Таблица 1)
   - Температуры, скорости, допуски
   - И многое другое...
4. **LLM создает regex** для каждого найденного элемента
5. **Система валидирует** regex на исходном документе
6. **Сохраняет 3 файла**:
   - Полный результат (данные + regex)
   - Только regex (для переиспользования)
   - Только данные (для использования)

## ⚡ Быстрые команды

### Минимальный запуск
```bash
./llm_regex_analyzer.py -f output/01-0530_6856.txt -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
```

### С валидацией
```bash
./llm_regex_analyzer.py -f output/01-0530_6856.txt -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" --validate
```

### С кастомными инструкциями
```bash
./llm_regex_analyzer.py \
  -f output/01-0530_6856.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  -i my_instructions.txt
```

### Для другого файла
```bash
./llm_regex_analyzer.py -f "output/01-0133 8720М.txt" -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest"
```

## 🎯 Ожидаемый результат

```
✅ Соединение с Ollama установлено
📝 Загружены инструкции: instructions_regex_generation.txt
📄 Загружен документ: 01-0530_6856.txt (27890 символов)

======================================================================
🚀 НАЧИНАЕМ АНАЛИЗ
======================================================================
🔄 Этап 1: Анализ документа и генерация regex паттернов...
   Размер документа: 27890 символов
   Размер инструкций: 15234 символов
🤖 Отправляем запрос к LLM (модель: yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest)...
✅ Успешно распарсен JSON от LLM
💾 Результаты сохранены: output/01-0530_6856_analyzed.json
📝 Regex паттерны сохранены отдельно: output/01-0530_6856.regex.json
📊 Извлеченные данные сохранены отдельно: output/01-0530_6856.data.json

======================================================================
🔍 ВАЛИДАЦИЯ REGEX ПАТТЕРНОВ
======================================================================
🔄 Этап 2: Валидация данных с помощью regex...
💾 Результаты сохранены: output/01-0530_6856_validation.json
✅ Результаты валидации: output/01-0530_6856_validation.json

======================================================================
✅ АНАЛИЗ ЗАВЕРШЕН!
======================================================================
📁 Полный результат: output/01-0530_6856_analyzed.json
📝 Regex паттерны: output/01-0530_6856.regex.json
📊 Извлеченные данные: output/01-0530_6856.data.json
🔍 Валидация: output/01-0530_6856_validation.json
```

## 🔧 Проблемы?

### Ошибка: "Connection refused"
```bash
# Проверьте, что Ollama запущена
ps aux | grep ollama

# Запустите Ollama
ollama serve &
```

### Ошибка: "Model not found"
```bash
# Проверьте доступные модели
ollama list

# Загрузите нужную модель
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest
```

### Ошибка: "JSON parse error"
```bash
# Проверьте файл с ошибкой
cat output/01-0530_6856_error.json

# Попробуйте другую модель (более мощную)
./llm_regex_analyzer.py -f output/01-0530_6856.txt -m "llama3:70b"
```

### Слишком долго работает
```bash
# Используйте более легкую модель
./llm_regex_analyzer.py -f output/01-0530_6856.txt -m "llama3:8b"
```

## 📖 Что дальше?

1. **Изучите результаты** в `*_analyzed.json`
2. **Проверьте regex** в `*.regex.json`
3. **Используйте данные** из `*.data.json`
4. **Переиспользуйте regex** для похожих документов:
   ```bash
   python3 regex_parser.py \
     --file another_document.txt \
     --regex-file output/01-0530_6856.regex.json \
     --mode apply
   ```

## 📚 Полная документация

- [LLM_REGEX_ANALYZER_README.md](LLM_REGEX_ANALYZER_README.md) - детальное описание
- [instructions_regex_generation.txt](instructions_regex_generation.txt) - инструкции для LLM

---

**Время работы:** ~60-180 секунд (зависит от модели)  
**После первого запуска:** regex можно переиспользовать для обработки сотен документов за секунды!


