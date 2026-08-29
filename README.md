# I-Mage

**Локальный семантический поиск по личной коллекции фотографий**

## Зачем это нужно

Большие фотоархивы быстро превращаются в тысячи и сотни тысяч файлов, где имена вроде `IMG_4821.jpg` практически бесполезны для поиска. I-Mage добавляет к обычному файловому каталогу несколько способов поиска:

| Задача                                               | Как решает I-Mage                                          |
| ---------------------------------------------------- | ---------------------------------------------------------- |
| «Где фото с котом на диване?»                        | Семантический поиск по текстовому описанию (CLIP)          |
| «Покажи все снимки с машиной / собакой / ноутбуком»  | Поиск по классам объектов (YOLO, 80 классов COCO)          |
| «Найди все фото этого человека»                      | Поиск по лицу + автоматическая группировка людей           |
| «Найди фото с машиной на фоне пляжа»                 | Unified-поиск: CLIP + YOLO в одном запросе                 |
| «Каталог вырос — не хочу пересканировать всё заново» | Инкрементальная индексация медиатеки |

Каталог файлов остаётся источником правды на диске. SQLite хранит метаданные и результаты детекции, а Qdrant — векторные представления CLIP и ArcFace.

---

## ML-стек

Три ML-модуля отвечают за разные типы признаков и поиска. Они изолированы через единые абстракции `EmbeddingModel`, `ObjectsRetriever` и `FaceRecognizer`, поэтому конкретные реализации моделей можно заменять без изменения API и пайплайна индексации.

```text
                         ┌───────────────────────┐
                         │       Изображение     │
                         └───────────┬───────────┘
                                     │
                  ┌──────────────────┼──────────────────┐
                  │                  │                  │
                  ▼                  ▼                  ▼
          ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐
          │     CLIP      │  │    YOLOv8s    │  │   InsightFace    │
          │   ViT-B/32    │  │     COCO      │  │    buffalo_l     │
          │               │  │               │  │                  │
          │ image → 512d  │  │ image →       │  │ face detection   │
          │ embedding     │  │ bbox + label  │  │ + ArcFace 512d   │
          └───────┬───────┘  └───────┬───────┘  └────────┬─────────┘
                  │                  │                   │
                  ▼                  ▼                   ▼
          ┌───────────────┐  ┌───────────────┐  ┌──────────────────┐
          │ Qdrant        │  │    SQLite     │  │     Qdrant       │
          │   context     │  │  detections   │  │      faces       │
          │               │  │               │  │                  │
          │ CLIP vectors  │  │ label         │  │ ArcFace vectors  │
          │ 512d          │  │ confidence    │  │ 512d             │
          │               │  │ bbox          │  │ N / image        │
          └───────────────┘  └───────────────┘  └────────┬─────────┘
                  │                                      │
                  │                                      │
                  ▼                                      ▼
          ┌───────────────────┐              ┌─────────────────────┐
          │  Semantic Search  │              │       DBSCAN        │
          │                   │              │                     │
          │ text → CLIP       │              │ cosine distance     │
          │      → 512d       │              │                     │
          │      → Qdrant     │              │ ArcFace vectors     │
          └───────────────────┘              │        ↓            │
                                             │ face clusters       │
                                             └──────────┬──────────┘
                                                        │
                                                        ▼
                                             ┌─────────────────────┐
                                             │       Persons       │
                                             │                     │
                                             │ person ↔ face       │
                                             └─────────────────────┘
```

### Модели

| Модуль                   | Модель                                        | Задача                                                |
| ------------------------ | --------------------------------------------- | ----------------------------------------------------- |
| **Контекст / семантика** | `openai/clip-vit-base-patch32` (Transformers) | Общее векторное пространство для текста и изображений |
| **Объекты**              | YOLOv8s (Ultralytics)                         | Детекция объектов, 80 классов COCO                    |
| **Лица**                 | InsightFace `buffalo_l`                       | Детекция + эмбеддинги лиц (ArcFace)                   |

### ML-инженерия

* **Batch inference** — изображения обрабатываются пакетами при индексации для CLIP, YOLO и ArcFace.
* **GPU Scheduler** — единый priority queue планирует доступ к GPU: интерактивный поиск имеет приоритет над фоновой индексацией. Индексация захватывает GPU на один batch, после чего поиск может получить управление.
* **Идемпотентная индексация** — SHA-256 content hash определяет изменения файлов, а детерминированные UUID5 используются как point ID в Qdrant. Запись выполняется через `upsert`.
* **Разделение хранилищ** — CLIP и ArcFace находятся в разных коллекциях Qdrant, несмотря на одинаковую размерность 512. YOLO-детекции хранятся в SQLite как структурированные данные.
* **Кластеризация лиц** — DBSCAN с косинусной метрикой автоматически группирует лица. Группы можно объединять, разделять и переименовывать вручную.
* **Unified search** — текст кодируется CLIP, а упомянутые в запросе объекты извлекаются через словарь COCO-классов и дополнительно учитываются при ранжировании.
* **Gap indexing** — каждый ML-модуль имеет собственный статус индексации, поэтому повторный запуск обрабатывает только отсутствующие или инвалидированные результаты.
* **Реальный inference в тестах** — отдельный набор тестов запускает CLIP, YOLO и ArcFace на реальном железе, дополняя unit и integration tests.

---

## Архитектура

Приложение разделено на API, ML, индексацию, хранилища и файловый слой. FastAPI, indexer и ML-модели работают в одном Python-процессе, а Qdrant запускается отдельно через Docker.

```text
┌──────────────────────────────────────────────────────────────────┐
│  clients/          Web UI (HTML/JS)  ·  CLI (HTTP-клиент)        │
└────────────────────────────┬─────────────────────────────────────┘
                             │ REST / NDJSON stream
┌────────────────────────────▼─────────────────────────────────────┐
│  api/              FastAPI — поиск, индексация, люди, настройки  │
└──────┬───────────────────────────────┬───────────────────────────┘
       │ read                            │ write (indexer)
       ▼                                 ▼
┌──────────────┐                  ┌────────────────────────────────┐
│  vectors/    │                  │  indexing/                     │
│  Qdrant      │◄─────────────────│  executor, runner, scheduler   │
│  client      │                  │  background indexer, cluster   │
└──────────────┘                  └───────────────┬────────────────┘
       ▲                                          │
       │                                          ▼
┌──────┴───────┐                  ┌────────────────────────────────┐
│  db/         │◄─────────────────│  ml/                           │
│  SQLite      │                  │  embeddings · objects · faces  │
│  SQLAlchemy  │                  └────────────────────────────────┘
└──────────────┘
       ▲
       │ scan
┌──────┴───────┐
│  io_utils/   │  обход каталога, content hash, reveal в файловом менеджере
└──────────────┘
```

### Ключевые решения

* **SQLite — источник правды для метаданных**, Qdrant используется только для dense-векторов. Пути к файлам, статусы индексации, детекции, лица и персоны хранятся в SQLite.
* **Один Python-процесс** объединяет FastAPI, фоновые задачи и ML-модели. Тяжёлые операции выполняются через `asyncio.to_thread()`, чтобы не блокировать event loop.
* **WAL-режим SQLite** позволяет выполнять чтение из API параллельно с записью во время индексации.
* **Фоновая индексация** запускается периодически и выполняет только gap-run для настроенных модулей.
* **Streaming API** использует NDJSON для передачи результатов и прогресса в UI.
* **Модульные блокировки** предотвращают конфликтующие запуски `catalog / clip / yolo / faces / clustering`.
* **Thread-local SQLAlchemy sessions** дают каждому worker thread собственную сессию БД и изолируют транзакции.

### API (основное)

| Endpoint                | Назначение                              |
| ----------------------- | --------------------------------------- |
| `POST /search`          | Поиск по текстовому описанию            |
| `POST /search/class`    | Поиск по классу объекта (COCO)          |
| `POST /search/face`     | Поиск по лицу                           |
| `POST /search/unified`  | Комбинированный поиск (текст + объекты) |
| `POST /search/*/stream` | Поиск с прогрессом в NDJSON             |
| `GET /people`           | Список сгруппированных людей            |
| `POST /index/reconcile`, `POST /index/run/full/{module}`, … | Запуск индексации |
| `GET /index/status`     | Прогресс и статистика индексации        |
| `GET /health`           | Статус сервиса и Qdrant                 |

Полная документация доступна в Swagger UI после запуска: `http://127.0.0.1:8000/docs`.

---

## Индексация

Индексация разделена на reconcile каталога и независимые ML-модули. Изменения файлов определяются через content hash, а каждый модуль отслеживает собственный статус.

```text
scan_config.include_directories
        │
        ▼
collect_scoped_files()
        │
        ▼
reconcile_catalog()
        │
        ├── новые файлы
        ├── удалённые файлы
        └── изменённый content_hash
        │
        ▼
gap_paths = files WHERE module_status != DONE
        │
        ├── CLIP  → batch inference → Qdrant
        ├── YOLO  → batch inference → SQLite
        └── Faces → batch inference → SQLite + Qdrant
```

Reconcile и gap-run — **отдельные шаги**: gap обрабатывает только файлы, уже записанные в SQLite. Новые файлы на диске попадают в индекс после `POST /index/reconcile`.

### Инвалидация

При изменении содержимого файла:

```text
content_hash изменился
        │
        ├── сбрасываются module statuses
        ├── удаляются старые detections / faces
        ├── удаляются Qdrant points
        │
        ▼
gap indexing повторно обрабатывает файл
```

Для каждого ML-модуля запись выполняется после успешного inference. Например, для CLIP последовательность выглядит так:

```text
mark_running → encode → upsert Qdrant → mark_done
```

Для лиц Qdrant обновляется до `mark_done`, поэтому статус `done` не может означать наличие записи в SQLite без соответствующего вектора.

### Фоновый indexer

Фоновый indexer каждые `3600` секунд проверяет **расписание** (`background_indexer_enabled`, `schedule_interval_days`). Когда интервал наступил, запускается gap indexing для настроенных модулей — только по записям, уже есть в SQLite. Файловую систему он не сканирует; новые файлы требуют отдельного reconcile.

Фоновый запуск не конкурирует с активным manual run или reconcile.

---

## Поиск

### Семантический поиск

```text
text query
    │
    ▼
CLIP.encode_text()
    │
    ▼
Qdrant context search
    │
    ▼
image_id → file path
```

CLIP создаёт L2-нормализованный embedding размерности 512, после чего выполняется vector search по коллекции `context`.

Поиск использует уже построенный индекс — повторный inference изображений не требуется.

### Поиск по объектам

```text
"cat"
  │
  ▼
SQLite detections
  │
  ├── label = cat
  ├── image_yolo.status = done
  └── max confidence per image
  │
  ▼
results
```

YOLO запускается только при индексации. На этапе поиска используются сохранённые `label`, `confidence` и `bbox`.

### Поиск по лицу

```text
uploaded photo
      │
      ▼
ArcFace.analyze()
      │
      ▼
query embedding
      │
      ▼
Qdrant faces search
      │
      ▼
aggregate by image_id
      │
      ▼
best face score per image
```

В Web UI embedding и поиск разделены на два запроса, что позволяет передавать в streaming endpoint уже вычисленный вектор. Также доступен endpoint, объединяющий embedding и поиск.

### Unified search

Unified search объединяет семантический и объектный поиск:

```text
query
  │
  ├──► CLIP → Qdrant
  │
  └──► COCO keyword extraction → SQLite detections
                                      │
                                      ▼
                              merge + rank
```

Система извлекает из текста известные COCO-классы, выполняет CLIP-поиск и SQL-поиск по детекциям, после чего объединяет результаты.

YOLO **не запускается на этапе поиска**.

### Streaming

Эндпоинты `*/stream` используют `application/x-ndjson`. Сервер отправляет JSON-объекты по мере выполнения операции, а Web UI читает поток через `ReadableStream` и отображает текущий этап и прогресс.

---

## Люди и лица

После индексации лиц их embeddings используются для автоматической группировки:

```text
ArcFace
   │
   ▼
Qdrant faces + SQLite faces
   │
   ▼
DBSCAN (cosine distance)
   │
   ▼
persons
   │
   ├── rename
   ├── merge
   └── split
```

DBSCAN использует косинусную метрику и не создаёт `person` для singleton-лиц. После кластеризации связи между лицами и персонами сохраняются в `face_person_assignments`.

Merge и split помечаются как `manual_merge` / `manual_split`; переименованные персоны (`is_named`) не перезаписываются обычным regroup (`regroup=false`).

---

## Хранилища данных

### SQLite

SQLite (`data/i-mage.db`) хранит метаданные каталога и результаты структурированного ML-инференса. Используется WAL-режим, `busy_timeout=30000` и `foreign_keys=ON`.

| Таблица                                     | Содержимое                                     |
| ------------------------------------------- | ---------------------------------------------- |
| `images`                                    | Путь, UUID, content hash, mtime, size          |
| `image_yolo` / `image_clip` / `image_faces` | Статус индексации каждого ML-модуля            |
| `detections`                                | YOLO labels, confidence и bbox                 |
| `faces`                                     | Детекции лиц и metadata                        |
| `persons`                                   | Группы людей                                   |
| `face_person_assignments`                   | Связи лиц с персонами                          |
| `index_runs`                                | Состояние и прогресс индексации                |
| `app_settings`                              | Конфигурация сканирования и background indexer |

`image_id` — стабильный UUID и не зависит от пути к файлу. `face_id` и Qdrant point ID создаются детерминированно через UUID5.

### Qdrant

Qdrant используется только как vector store и запускается в server mode через Docker.

| Коллекция | Гранулярность         | Размерность   | Payload                                |
| --------- | --------------------- | ------------- | -------------------------------------- |
| `context` | 1 point на `image_id` | 512 (CLIP)    | `image_id`, `model_version`            |
| `faces`   | N points на фото      | 512 (ArcFace) | `face_id`, `image_id`, `model_version` |

Point ID детерминированы и операции записи выполняются через `upsert`.

YOLO-детекции в Qdrant не хранятся: labels, confidence и bbox используются как структурированные данные и эффективно ищутся через SQLite.

---

## Технологии

| Слой     | Стек                                                                                  |
| -------- | ------------------------------------------------------------------------------------- |
| API      | FastAPI, Uvicorn, Pydantic                                                            |
| ML       | PyTorch, Transformers, Ultralytics YOLO, InsightFace, OpenCV, scikit-learn            |
| Хранение | SQLite (SQLAlchemy 2), Qdrant                                                         |
| Инфра    | Docker Compose (Qdrant), conda-окружение `ml-env`                                     |
| Тесты    | pytest, httpx (~160 тестов: API, БД, индексация, векторы, concurrency, GPU scheduler) |

---

## Быстрый старт

### Требования

* Python 3.10+, conda
* Docker (для Qdrant)
* GPU с CUDA — опционально, но значительно ускоряет индексацию; на CPU проект также работает

### Запуск

```bash
# 1. Векторное хранилище
docker compose up -d

# 2. Окружение и зависимости
conda activate ml-env
pip install -r requirements.txt

# + установка torch, transformers, ultralytics, insightface, fastapi, uvicorn и др.
#   (основные ML-зависимости ставятся отдельно от requirements.txt)

# 3. Модели (скачиваются при первом запуске или заранее через scripts/)
#    CLIP — Hugging Face cache
#    YOLO — artifacts/yolov8s.pt
#    ArcFace — InsightFace buffalo_l

# 4. Сервер
./run-dev.sh          # разработка с hot-reload

# или

./run.sh              # production-like запуск

# 5. Открыть UI
# http://127.0.0.1:8000
```

### Переменные окружения

| Переменная                                                 | По умолчанию            | Описание                    |
| ---------------------------------------------------------- | ----------------------- | --------------------------- |
| `QDRANT_URL`                                               | `http://127.0.0.1:6333` | Адрес Qdrant                |
| `IMAGE_DB_PATH`                                            | `data/i-mage.db`        | Путь к SQLite               |
| `IMAGE_SEARCH_DIR`                                         | `data/flickr30k/images` | Каталог для индексации      |
| `FACE_SEARCH_DIR`                                          | `data/small_celeba`     | Каталог для лиц             |
| `HOST` / `PORT`                                            | `127.0.0.1` / `8000`    | Адрес API                   |
| `CLIP_BATCH_SIZE` / `YOLO_BATCH_SIZE` / `FACES_BATCH_SIZE` | 32 / 16 / 16            | Размер batch при индексации |
| `FACES_ANALYZE_WORKERS`                                    | 4                       | CPU workers для decode лиц  |

Каталоги сканирования и параметры background indexer также настраиваются через Web UI / API (`/settings`).

---

## Структура репозитория

```text
api/           REST API, схемы, search orchestration
clients/       Web UI и CLI
db/            SQLite: images, detections, faces, persons, index_runs
indexing/      Индексатор, executor, GPU scheduler, clustering
ml/            CLIP, YOLO, ArcFace — модели и сервисный слой
vectors/       Qdrant client, коллекции context / faces
io_utils/      Сканирование FS, content hash
tests/         Unit, integration, stress, real-model tests
docs/          План развития и техническая документация
```

---

## Ограничения и trade-offs

| Решение                                   | Причина                                                                        |
| ----------------------------------------- | ------------------------------------------------------------------------------ |
| Один Python-процесс (API + indexer)       | Проще локальный деплой и достаточно для текущего сценария                      |
| Qdrant только server mode                 | Embedded mode конфликтовал с `--reload` и multi-writer сценарием               |
| YOLO не в Qdrant                          | Labels и bbox — структурированные данные, для них достаточно SQL               |
| Gap indexing вместо полного re-scan       | Content hash и module status позволяют обрабатывать только изменившиеся данные |
| Search не запускает indexing              | Поиск остаётся read-only и не инициирует тяжёлый inference                     |
| GPU Scheduler с batch-level переключением | Фоновая индексация не должна полностью блокировать интерактивный поиск         |
| Singleton faces не создают persons        | Избегает большого количества одно-лицевых групп                                |
| In-process background indexer             | Меньше инфраструктуры; отдельный worker можно вынести в следующий этап         |

Система намеренно не включает multi-user authentication, облачную синхронизацию, filesystem watcher/inotify и обработку видео.

---

## Лицензия

Личный open-source проект. Уточняйте лицензию перед коммерческим использованием моделей (CLIP, YOLO, InsightFace имеют свои условия).
