# Fairytale Bot — diploma project / Сказочный бот — дипломный проект

[English](#english) | [Русский](#русский)

---

## English

Personalized bedtime fairy tales for children, in Russian. Step by step, the child chooses the hero and the hero's actions, and a language model (ruGPT3 fine-tuned with LoRA) writes the tale chapter by chapter. Channels: a Telegram bot and a web app (a shared FastAPI server).

**Status:** data preparation is complete. Next: model fine-tuning in Google Colab (T4 GPU), the server and the bot.

### Data

- Source: Russian Wikisource (public domain); the list is in `data/SOURCES.md`.
- 1714 texts collected; after cleaning and filtering, 1394 tales remain (1199 works), split into 4960 chapters.
- Every tale and chapter is annotated with the Claude Haiku model (tale card, mood, violence level, chapter summary).
- 1024 tales are suitable for training (no strong violence, non-tales or adult themes): 3303 examples (train 2810, val 226, test 267), about 3M tokens.
- The texts themselves are not included in the repository (samples only). They can be rebuilt with the scripts below.

### Data processing (run order, from the project root)

```bash
pip install -r requirements.txt
python training/collect_wikisource.py          # 1. download texts from Wikisource
python training/prepare_corpus.py              # 2. clean, assign a work key
python training/filter_corpus.py               # 3. remove non-tales and adult texts
python training/chunk_tales.py                 # 4. split into chapters, split train/val/test by work
python training/clean_chapters.py              # 5. remove edition headers, invisible characters, footnotes
python training/annotate_cards.py              # 6. annotation via the Claude API (key required in .env)
python training/build_dataset.py               # 7. build training examples in data/dataset/
python training/make_report_files.py           # 8. tables and statistics for reports
```

The API key goes into a `.env` file in the project root (`ANTHROPIC_API_KEY=...`); it is never committed to the repository.

### Training example format

One example is one chapter of a tale. The model learns to write the chapter text from a prompt (the loss is computed on the chapter text only). The prompt fields are in Russian: mood, characters, setting, goal, moral, previous chapters, the child's choice, chapter number.

```text
Настроение: adventure
Герои: Пак, жена-небожительница, олень
Место: Корея, горное озеро и небесный дом
Цель: Остаться с женой и жить счастливо в семье
Мораль: Главное в жизни — любить друг друга
Предыдущие главы: нет
Выбор ребёнка: Спрятать юбку девушки
Глава 1 из 4:
<chapter text></s>
```

Data samples: `data/dataset/sample.jsonl`, `data/dataset/examples.md`; statistics: `data/dataset/stats.md`. Tables and statistics on the corpus and annotation: `data/processed/report/`.

### Structure

- `training/` — data processing scripts (next: training and evaluation)
- `data/` — sources, samples, statistics (large files are in `.gitignore`)
- `server/`, `bot/`, `notebooks/` — server, Telegram bot, Colab notebooks (in progress)

### Rules

- Open data only; the source and license of every text are recorded in `data/SOURCES.md`.
- No datasets or code from internships were used.

---

## Русский

Персональные детские сказки на ночь на русском языке. Ребёнок шаг за шагом выбирает героя и действия, а языковая модель (ruGPT3, дообученная методом LoRA) пишет сказку по главам. Каналы: Telegram-бот и веб-приложение (общий сервер на FastAPI).

**Статус:** подготовка данных завершена, дальше дообучение модели в Google Colab (GPU T4), сервер и бот.

### Данные

- Источник: русская Викитека (общественное достояние), список в `data/SOURCES.md`.
- Собрано 1714 текстов, после очистки и отбора осталось 1394 сказки (1199 произведений), нарезано на 4960 глав.
- Все сказки и главы размечены моделью Claude Haiku (карточка сказки, настроение, жестокость, пересказ главы).
- Для обучения подходят 1024 сказки (без сильной жестокости, не-сказок и взрослых тем): 3303 примера (train 2810, val 226, test 267), около 3 млн токенов.
- Сами тексты в репозиторий не входят (только примеры). Их можно собрать заново скриптами ниже.

### Обработка данных (порядок запуска, из корня проекта)

```bash
pip install -r requirements.txt
python training/collect_wikisource.py          # 1. скачать тексты из Викитеки
python training/prepare_corpus.py              # 2. очистить, поставить ключ произведения
python training/filter_corpus.py               # 3. убрать не-сказки и взрослые тексты
python training/chunk_tales.py                 # 4. нарезать на главы, разбить на train/val/test по произведениям
python training/clean_chapters.py              # 5. убрать шапки изданий, невидимые символы, сноски
python training/annotate_cards.py              # 6. разметка через Claude API (нужен ключ в файле .env)
python training/build_dataset.py               # 7. собрать обучающие примеры в data/dataset/
python training/make_report_files.py           # 8. таблицы и статистика для отчётов
```

Ключ API кладётся в файл `.env` в корне проекта (`ANTHROPIC_API_KEY=...`), в репозиторий он не попадает.

### Формат обучающего примера

Один пример — одна глава сказки. Модель учится по подсказке писать текст главы (ошибка считается только на тексте главы):

```text
Настроение: adventure
Герои: Пак, жена-небожительница, олень
Место: Корея, горное озеро и небесный дом
Цель: Остаться с женой и жить счастливо в семье
Мораль: Главное в жизни — любить друг друга
Предыдущие главы: нет
Выбор ребёнка: Спрятать юбку девушки
Глава 1 из 4:
<текст главы></s>
```

Примеры данных: `data/dataset/sample.jsonl`, `data/dataset/examples.md`, статистика: `data/dataset/stats.md`. Таблицы и статистика по базе и разметке: `data/processed/report/`.

### Структура

- `training/` — скрипты обработки данных (дальше: обучение и оценка)
- `data/` — источники, примеры, статистика (большие файлы в `.gitignore`)
- `server/`, `bot/`, `notebooks/` — сервер, Telegram-бот, ноутбуки для Colab (в работе)

### Правила

- Только открытые данные; источник и лицензия каждого текста записываются в `data/SOURCES.md`.
- Датасет и код со стажировок не использовались.

---

© 2026 Alyona. All rights reserved. / Все права защищены.
