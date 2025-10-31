# OCR Скрипт для распознавания текста

Этот скрипт предназначен для распознавания текста с изображений и PDF файлов, используя две OCR библиотеки:
- **EasyOCR** - современная библиотека с хорошей поддержкой русского языка
- **Tesseract** - классическая библиотека с модулем для рукописного текста

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

После OCR-обработки текст можно парсить для извлечения структурированных данных.

## 🎯 Три подхода к парсингу

### 1. **LLM Parser** - Чистый LLM подход
Использует языковую модель для понимания и извлечения данных.

```bash
python3 llm_parser.py \
  --file output/01-0530_6856.txt \
  --model "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --prompt-dir prompts/
```

**Преимущества:**
- ✅ Хорошо понимает контекст
- ✅ Гибкий и адаптивный
- ✅ Работает с разными форматами

**Недостатки:**
- ❌ Медленный (~60-180 сек на документ)
- ❌ Дорогой для массовой обработки
- ❌ Нельзя переиспользовать

📚 [Подробная документация](LLM_PARSER_README.md)

---

### 2. **Regex Parser** - Регулярные выражения
LLM создает regex паттерны, затем они применяются.

```bash
# Генерация regex
python3 regex_parser.py \
  --file output/01-0530_6856.txt \
  --model "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --mode generate

# Применение regex
python3 regex_parser.py \
  --file output/another_doc.txt \
  --regex-file output/01-0530_6856.regex \
  --mode apply
```

**Преимущества:**
- ✅ Быстрый после генерации (~0.1-1 сек)
- ✅ Можно переиспользовать паттерны
- ✅ Дешевый в продакшене

**Недостатки:**
- ⚠️ Требует ручной корректировки паттернов
- ⚠️ Хуже работает с вариациями формата

---

### 3. **LLM Regex Analyzer** ⭐ РЕКОМЕНДУЕТСЯ
**Гибридный подход:** LLM анализирует документ И создает regex паттерны.

```bash
python3 llm_regex_analyzer.py \
  --file output/01-0530_6856.txt \
  --model "yandex/YandexGPT-5-Lite-8B-instruct-GGUF:latest" \
  --validate
```

**Преимущества:**
- ✅ Точность LLM + скорость regex
- ✅ Сразу получаете данные И regex
- ✅ Regex можно переиспользовать
- ✅ Встроенная валидация
- ✅ Лучший результат из всех подходов

**Что получаете:**
- `*_analyzed.json` - полный анализ с данными и regex
- `*.regex.json` - regex паттерны (для переиспользования)
- `*.data.json` - извлеченные данные (ready to use)
- `*_validation.json` - проверка работоспособности regex

📚 [Подробная документация](LLM_REGEX_ANALYZER_README.md)  
🚀 [Быстрый старт за 3 минуты](QUICKSTART_REGEX_ANALYZER.md)

---

## 📊 Сравнение подходов

| Характеристика | LLM Parser | Regex Parser | **LLM Regex Analyzer** |
|----------------|------------|--------------|------------------------|
| Понимание контекста | ✅ Отлично | ❌ Слабо | ✅ Отлично |
| Скорость (первый раз) | ❌ 60-180 сек | ⚠️ 60-180 сек | ⚠️ 60-180 сек |
| Скорость (повторно) | ❌ 60-180 сек | ✅ 0.1-1 сек | ✅ 0.1-1 сек |
| Переиспользование | ❌ Нет | ✅ Да | ✅ Да |
| Точность | ✅ Высокая | ⚠️ Средняя | ✅ Высокая |
| Стоимость (продакшн) | ❌ Высокая | ✅ Низкая | ✅ Низкая |
| Адаптивность | ✅ Да | ❌ Нет | ✅ Да |
| **ИТОГО** | Хорошо для разовых задач | Хорошо для массовых | **✅ ЛУЧШИЙ ВЫБОР** |

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

## 📁 Структура проекта (полная)

```
letterexplorer/
├── input/                          # Входные PDF/изображения
├── output/                         # Результаты OCR и парсинга
├── prompts/                        # Промпты для LLM Parser
│   ├── prompt_main.txt
│   ├── prompt_chemical_composition.txt
│   └── ...
│
# OCR скрипты
├── ocr_script.py                   # Основной OCR скрипт
├── ocr_with_tables.py              # OCR с распознаванием таблиц
│
# Парсинг скрипты
├── llm_parser.py                   # LLM парсер (чистый LLM)
├── regex_parser.py                 # Regex парсер (LLM → regex)
├── llm_regex_analyzer.py           # ⭐ Гибридный подход (РЕКОМЕНДУЕТСЯ)
│
# Инструкции для парсинга
├── instructions_example.txt        # Простые инструкции
├── instructions_regex_generation.txt # Инструкции для генерации regex
│
# Документация
├── README.md                       # Этот файл
├── LLM_PARSER_README.md           # Документация LLM Parser
├── LLM_REGEX_ANALYZER_README.md   # Документация LLM Regex Analyzer
├── QUICKSTART_REGEX_ANALYZER.md   # Быстрый старт за 3 минуты
│
└── requirements.txt               # Зависимости
```

---

## 🎓 Примеры для разных задач

### Задача 1: Разовый анализ одного документа
**Решение:** LLM Parser
```bash
python3 llm_parser.py -f doc.txt -m "model_name" --prompt-dir prompts/
```

### Задача 2: Анализ 100+ похожих документов
**Решение:** LLM Regex Analyzer + Regex Parser
```bash
# Шаг 1: Анализируем первый документ
python3 llm_regex_analyzer.py -f doc1.txt -m "model" --validate

# Шаг 2: Применяем regex к остальным 99
for file in doc*.txt; do
  python3 regex_parser.py -f "$file" --regex-file doc1.regex.json --mode apply
done
```

### Задача 3: Регулярная обработка новых документов
**Решение:** Cron + LLM Regex Analyzer
```bash
# crontab -e
0 */6 * * * /path/to/llm_regex_analyzer.py -f /path/to/new_docs/*.txt -m "model"
```

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

- [LLM Parser - документация](LLM_PARSER_README.md)
- [LLM Regex Analyzer - документация](LLM_REGEX_ANALYZER_README.md)
- [Быстрый старт за 3 минуты](QUICKSTART_REGEX_ANALYZER.md)
- [Инструкции для LLM](instructions_regex_generation.txt)

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
