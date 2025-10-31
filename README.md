# 📄 Letter Explorer - OCR и парсинг технических документов

Комплексное решение для распознавания (OCR) и извлечения структурированных данных из технических документов.

## 🎯 Основные компоненты

### 1️⃣ **`easyocr_script.py`** - Продвинутое OCR распознавание ⭐
Распознавание текста и таблиц из PDF используя:
- **EasyOCR** - для точного распознавания таблиц
- **Tesseract** - для быстрого извлечения текста
- **img2table** - для структурного выделения таблиц
- **Автокоррекция наклона** - исправляет перекошенные сканы
- **Markdown таблицы** - результат в удобном формате для LLM

**Особенности:**
- ✅ Сквозная нумерация таблиц по всему документу
- ✅ Таблицы с пустыми ячейками (корректно распознаются)
- ✅ Исключение областей таблиц из текста (нет дублирования)
- ✅ GPU поддержка для ускорения
- ✅ Настраиваемое DPI и порог уверенности

### 2️⃣ **`llm_regex_analyzer.py`** - Гибридный парсер 
Умный парсер, который:
- Анализирует документ с помощью LLM
- Извлекает данные из текста и Markdown-таблиц
- Генерирует regex паттерны для переиспользования
- Позволяет обрабатывать похожие документы за секунды

---

## Установка

1. Установите зависимости:
```bash
pip install -r requirements.txt
```

2. Для работы с Tesseract установите системную библиотеку:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-rus

# CentOS/RHEL
sudo yum install tesseract tesseract-langpack-rus
```

## Использование

### Базовое использование
```bash
# Обработка PDF (DPI=400, CPU)
python3 easyocr_script.py input/document.pdf

# С GPU ускорением
python3 easyocr_script.py input/document.pdf --gpu

# Настройка порога уверенности для сложных таблиц
python3 easyocr_script.py input/document.pdf --gpu --min-confidence 20
```

### Указание выходного каталога
```bash
python3 easyocr_script.py input/document.pdf --output output/custom_folder/
```

## Параметры

- `--gpu` - использование GPU для EasyOCR (по умолчанию: CPU)
- `--output, -o` - каталог для сохранения результатов
- `--dpi` - разрешение для конвертации PDF (по умолчанию: 400)
- `--min-confidence` - минимальная уверенность OCR для ячеек таблицы (по умолчанию: 30)

## Поддерживаемые форматы

- **Изображения**: JPG, JPEG, PNG, BMP, TIFF
- **PDF**: автоматическая конвертация в изображения

## Результат работы

После обработки создаётся:
```
output/document_easyocr/
├── document.txt         # Текст + Markdown таблицы
├── table1.csv          # Таблица 1 в CSV
├── table2.csv          # Таблица 2 в CSV
├── ...
├── page_1.png          # Изображения страниц (для отладки)
└── page_2.png
```

## Особенности

- **Markdown таблицы**: идеально для последующего анализа LLM
- **Пустые ячейки**: корректно обрабатываются и сохраняются
- **Автокоррекция наклона**: исправляет перекошенные сканы (detect_rotation=True)
- **Сквозная нумерация**: таблицы нумеруются по всему документу
- **GPU ускорение**: поддержка GPU для EasyOCR (параметр --gpu)
- **Высокое DPI**: 400 DPI по умолчанию для лучшего качества

---

# 📊 Парсинг технических документов

После OCR-обработки используйте **`llm_regex_analyzer.py`** - гибридный подход, объединяющий лучшее из LLM и regex.

## ⭐ LLM Regex Analyzer - Рекомендуемый подход

**Что делает:**
1. LLM анализирует документ и извлекает данные
2. LLM создает regex паттерны для этих данных
3. Вы получаете данные + regex для переиспользования

**Использование:**
```bash
python3 llm_regex_analyzer.py \
  -f output/document.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF" \
  -i instructions_regex_generation_v2.txt \
  --ollama-url "https://your-server:443" \
  -t "your-token" \
  --no-ssl-verify \
  --validate
```

**Результат:**
- `*_analyzed.json` - полный анализ (данные + regex)
- `*.data.json` - только извлеченные данные
- `*.regex.json` - regex паттерны для переиспользования
- `*_validation.json` - проверка работоспособности regex

**Преимущества:**
- ✅ Точность LLM + скорость regex
- ✅ Первый раз: 60-180 сек (анализ + генерация)
- ✅ Повторно: 0.1-1 сек (только regex)
- ✅ Regex переиспользуются для похожих документов
- ✅ Встроенная валидация

📚 [Подробная документация](LLM_REGEX_ANALYZER_README.md)  
🚀 [Быстрый старт за 3 минуты](QUICKSTART_REGEX_ANALYZER.md)  
🔬 [Альтернативные подходы](research/README.md) - эксперименты и исследования

---

## 🚀 Рекомендуемый workflow

### Шаг 1: OCR обработка с EasyOCR
```bash
python3 easyocr_script.py input/document.pdf --gpu --min-confidence 20
```

**Результат:**
- `output/document_easyocr/document.txt` - текст + Markdown таблицы
- `output/document_easyocr/table*.csv` - отдельные CSV файлы

### Шаг 2: Парсинг с LLM
```bash
python3 llm_regex_analyzer.py \
  -f output/document_easyocr/document.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --ollama-url "https://your-server:443" \
  -t "your-token" \
  --no-ssl-verify \
  --validate
```

**Результат:**
- ✅ Извлеченные данные в `*.data.json`
- ✅ Regex паттерны в `*.regex.json`
- ✅ Валидация в `*_validation.json`

### Шаг 3: Парсинг похожих документов (опционально)
```bash
# Используем regex для быстрой обработки
python3 research/regex_parser.py \
  --file output/another_doc.txt \
  --regex-file output/document.regex.json \
  --mode apply
```

**Результат:** Обработка за секунды вместо минут! 🚀

---

## 📁 Структура проекта

```
letterexplorer/
├── 🚀 ОСНОВНЫЕ СКРИПТЫ (для production)
│   ├── easyocr_script.py                  # OCR с EasyOCR + таблицы в Markdown ⭐
│   └── llm_regex_analyzer.py              # Гибридный парсер (LLM + regex) ⭐
│
├── 📂 КАТАЛОГИ
│   ├── input/                             # Входные PDF/изображения
│   ├── output/                            # Результаты OCR и парсинга
│   ├── prompts/                           # Промпты для парсеров
│   │   ├── prompt_main.txt
│   │   ├── prompt_chemical_composition.txt
│   │   └── ...
│   │
│   └── research/                          # 🔬 Экспериментальные подходы
│       ├── README.md                      # Документация исследований
│       ├── ocr_script.py                  # Старый OCR скрипт (EasyOCR/Tesseract)
│       ├── llm_parser.py                  # Чистый LLM парсер
│       ├── regex_parser.py                # Regex генератор
│       ├── ocr_trocr.py                   # TrOCR (трансформеры)
│       ├── ocr_with_tables.py             # OCR с таблицами
│       ├── markup_parser.py               # Markup-based парсинг
│       └── ... (другие эксперименты)
│
├── 📝 ИНСТРУКЦИИ ДЛЯ LLM
│   ├── instructions_regex_generation_v2.txt # Верхнеуровневые (рекомендуется) ⭐
│   ├── instructions_simple.txt            # Простые для начинающих
│   └── instructions_example.txt           # Примеры
│
├── 📚 ДОКУМЕНТАЦИЯ
│   ├── README.md                          # Этот файл
│   ├── LLM_REGEX_ANALYZER_README.md       # Детальная документация парсера
│   ├── QUICKSTART_REGEX_ANALYZER.md       # Быстрый старт (3 минуты)
│   ├── CHEATSHEET.md                      # Шпаргалка команд
│   └── QUICKSTART.md                      # Общий quickstart
│
├── requirements.txt                       # Зависимости Python
└── .gitignore                             # Git ignore правила
```

---

## 🎓 Примеры для разных задач

### Задача 1: Разовый анализ одного документа
```bash
# OCR с таблицами
python3 easyocr_script.py input/document.pdf --gpu --min-confidence 20

# Парсинг
python3 llm_regex_analyzer.py \
  -f output/document_easyocr/document.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --ollama-url "https://your-server:443" \
  -t "your-token" \
  --no-ssl-verify
```

### Задача 2: Анализ 100+ похожих документов
```bash
# Шаг 1: Анализируем первый документ (генерируем regex)
python3 llm_regex_analyzer.py \
  -f output/doc1.txt \
  -m "model" \
  --ollama-url "https://your-server:443" \
  -t "token" \
  --no-ssl-verify

# Шаг 2: Применяем regex к остальным 99 документам
# (используя скрипт из research/)
for file in output/doc*.txt; do
  python3 research/regex_parser.py \
    -f "$file" \
    --regex-file output/doc1.regex.json \
    --mode apply
done
```

### Задача 3: Экспериментирование с разными подходами
**Смотрите:** [`research/README.md`](research/README.md) - все экспериментальные скрипты и подходы

---

## 🔧 Требования

### Для OCR:
```bash
pip install easyocr pytesseract Pillow pdf2image
sudo apt-get install tesseract-ocr tesseract-ocr-rus
```

### Для парсинга:
```bash
pip install requests urllib3

# Установка Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Загрузка модели
ollama pull yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest
```

---

## 📚 Дополнительные ресурсы

**Основная документация:**
- [LLM Regex Analyzer - документация](LLM_REGEX_ANALYZER_README.md) ⭐
- [Быстрый старт за 3 минуты](QUICKSTART_REGEX_ANALYZER.md)
- [Шпаргалка команд](CHEATSHEET.md)
- [Инструкции для LLM](instructions_regex_generation_v2.txt)

**Дополнительно:**
- [Research - экспериментальные подходы](research/README.md) 🔬
- [Общий Quickstart](QUICKSTART.md)

---

## 🤝 Поддержка

При возникновении проблем:
1. Проверьте документацию к конкретному парсеру
2. Убедитесь, что Ollama запущена: `curl http://localhost:11434/api/tags`
3. Проверьте логи: `*_error.json`, `*_validation.json`

---

**Создано:** 2024-2025  
**Последнее обновление:** 31.10.2025
```
