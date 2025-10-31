# 📄 Letter Explorer - OCR и парсинг технических документов

Комплексное решение для распознавания (OCR) и извлечения структурированных данных из технических документов.

## 🎯 Основные компоненты

### 1️⃣ **`ocr_script.py`** - OCR распознавание текста
Распознавание текста с изображений и PDF файлов используя:
- **EasyOCR** - современная библиотека с отличной поддержкой русского языка
- **Tesseract** - классическая библиотека OCR

### 2️⃣ **`llm_regex_analyzer.py`** - Гибридный парсер ⭐ 
Умный парсер, который:
- Анализирует документ с помощью LLM
- Извлекает данные
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
python ocr_script.py --method easyocr
python ocr_script.py --method tesseract
```

### С GPU поддержкой
```bash
python ocr_script.py --method easyocr --gpu
```

### Указание каталогов
```bash
python ocr_script.py --method easyocr --input /path/to/input --output /path/to/output
```

## Параметры

- `--method, -m` - выбор OCR библиотеки (easyocr, tesseract)
- `--gpu` - использование GPU (если поддерживается библиотекой)
- `--input, -i` - каталог с входными файлами (по умолчанию: input)
- `--output, -o` - каталог для выходных файлов (по умолчанию: output)

## Поддерживаемые форматы

- **Изображения**: JPG, JPEG, PNG, BMP, TIFF
- **PDF**: автоматическая конвертация в изображения

## Структура проекта

```
letterexplorer/
├── input/          # Входные файлы
├── output/         # Результаты OCR
├── ocr_script.py   # Основной скрипт
└── requirements.txt # Зависимости
```

## Особенности

- **Русский язык**: все библиотеки настроены для работы с русским текстом
- **Рукописный текст**: Tesseract имеет специальные настройки для рукописного ввода
- **PDF поддержка**: автоматическая конвертация PDF в изображения с увеличением разрешения
- **Фильтрация**: результаты фильтруются по уровню уверенности (>50%)
- **GPU ускорение**: поддержка GPU для EasyOCR

## Примеры использования

```bash
# Обработка всех файлов в input/ с помощью EasyOCR
python ocr_script.py --method easyocr

# Обработка с GPU ускорением
python ocr_script.py --method easyocr --gpu

# Обработка конкретного каталога
python ocr_script.py --method tesseract --input /path/to/images --output /path/to/results
```

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

### Шаг 1: OCR обработка
```bash
python ocr_script.py --method easyocr --input input/ --output output/
```

### Шаг 2: Парсинг (первый документ)
```bash
python3 llm_regex_analyzer.py \
  --file output/01-0530_6856.txt \
  --model "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --validate
```

**Результат:**
- ✅ Извлеченные данные в `*.data.json`
- ✅ Regex паттерны в `*.regex.json`

### Шаг 3: Парсинг (остальные документы)
```bash
# Используем regex для быстрой обработки
python3 regex_parser.py \
  --file output/another_doc.txt \
  --regex-file output/01-0530_6856.regex.json \
  --mode apply
```

**Результат:** Обработка за секунды вместо минут! 🚀

---

## 📁 Структура проекта

```
letterexplorer/
├── 🚀 ОСНОВНЫЕ СКРИПТЫ (для production)
│   ├── ocr_script.py                      # OCR распознавание (EasyOCR/Tesseract)
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
# OCR
python3 ocr_script.py --method easyocr --input input/ --output output/

# Парсинг
python3 llm_regex_analyzer.py \
  -f output/doc.txt \
  -m "yandex/YandexGPT-5-Lite-8B-instruct-GGUF" \
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
