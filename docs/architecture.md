# Техническое описание I-Mage

Документ описывает архитектуру, процессы, потоки данных, ML-компоненты и границу ответственности между клиентом (браузер) и сервером. Актуально для текущей реализации в репозитории.

Связанные материалы: [mvp.md](mvp.md), [indexing-plan.md](indexing-plan.md), [README.md](../README.md).

---

## 1. Обзор системы

I-Mage — **локальное** приложение для поиска по личной коллекции изображений. Все ML-модели, база метаданных и векторный индекс работают на машине пользователя. Данные **не отправляются** во внешние API (кроме одноразовой загрузки весов моделей с Hugging Face / Ultralytics / InsightFace при первом запуске).

Система решает три класса задач:

| Задача | ML-модуль | Хранилище результата |
|--------|-----------|----------------------|
| Семантический поиск по тексту | CLIP | Qdrant `context` |
| Поиск по объектам (COCO) | YOLOv8 | SQLite `detections` |
| Поиск и группировка по лицам | ArcFace (InsightFace) | Qdrant `faces` + SQLite `faces`, `persons` |

Каталог файлов — **источник правды на диске**. SQLite хранит метаданные и структурированные ML-результаты. Qdrant — только dense-векторы для CLIP и ArcFace.

---

## 2. Процессы и контейнеры

### 2.1. Топология развёртывания

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Хост (Linux / macOS / Windows)                                         │
│                                                                         │
│  ┌──────────────────────────────┐   ┌────────────────────────────────┐ │
│  │  Docker: qdrant               │   │  Python-процесс (uvicorn)       │ │
│  │  image: qdrant/qdrant:v1.18.2│   │  api.app:app                    │ │
│  │  ports: 6333 (HTTP), 6334    │◄──│  QDRANT_URL=http://127.0.0.1:6333│ │
│  │  volumes: data/qdrant_server/ │   │                                 │ │
│  └──────────────────────────────┘   │  ┌─────────────────────────────┐ │ │
│                                      │  │ FastAPI (async event loop)  │ │ │
│  ┌──────────────────────────────┐   │  │  · HTTP handlers            │ │ │
│  │  Браузер пользователя         │   │  │  · background_indexer_loop  │ │ │
│  │  http://127.0.0.1:8000        │──►│  │  · IndexExecutor tasks      │ │ │
│  └──────────────────────────────┘   │  └───────────┬─────────────────┘ │ │
│                                      │              │                   │ │
│                                      │  ┌───────────▼─────────────────┐ │ │
│                                      │  │ Worker threads (asyncio     │ │ │
│                                      │  │ .to_thread)               │ │ │
│                                      │  │  · индексация (CLIP/YOLO/   │ │ │
│                                      │  │    faces)                   │ │ │
│                                      │  │  · reconcile каталога       │ │ │
│                                      │  │  · DBSCAN-кластеризация     │ │ │
│                                      │  └───────────┬─────────────────┘ │ │
│                                      │              │                   │ │
│                                      │  ┌───────────▼─────────────────┐ │ │
│                                      │  │ ML-модели in-process        │ │ │
│                                      │  │  CLIP · YOLO · ArcFace      │ │ │
│                                      │  │  (PyTorch / ONNX на GPU/CPU)│ │ │
│                                      │  └─────────────────────────────┘ │ │
│                                      └────────────────────────────────┘ │
│                                                                         │
│  Файлы на диске:                                                        │
│    data/i-mage.db          — SQLite (WAL)                               │
│    data/qdrant_server/     — персистентные векторы Qdrant               │
│    IMAGE_SEARCH_DIR / …    — каталоги с фотографиями пользователя       │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2. Процессы

| Процесс | Как запускается | Роль |
|---------|-----------------|------|
| **Qdrant** | `docker compose up -d` | Сервер векторного поиска. Единственный writer векторов — Python-indexer. API только читает. |
| **Uvicorn / FastAPI** | `./run.sh` или `./run-dev.sh` | Единый Python-процесс: API, фоновый indexer, загрузка всех ML-моделей. |
| **Браузер** | Пользователь открывает `/` | Тонкий клиент: HTML/JS, все вычисления на сервере. |
| **CLI** (`clients/cli`) | `python -m clients.cli …` | Отладочный HTTP-клиент к локальному API, без собственной ML-логики. |

> **Важно:** отдельного worker-процесса для индексации **нет**. Индексация выполняется в том же процессе, что и API, в фоновых `asyncio.Task`, которые делегируют тяжёлую работу в `asyncio.to_thread()`. Это сознательное упрощение (этап 4 плана из [indexing-plan.md](indexing-plan.md) реализован in-process, а не как отдельный daemon).

### 2.3. Что не контейнеризовано

- Само приложение (FastAPI + ML) **не** упаковано в Docker — только Qdrant.
- ML-веса лежат в локальном кэше (Hugging Face, `artifacts/yolov8s.pt`, InsightFace models).
- SQLite — файл `data/i-mage.db` на хосте.

---

## 3. Архитектура слоёв

```
clients/          ← точки входа пользователя (Web UI, CLI)
    │
    ▼  HTTP / NDJSON
api/              ← FastAPI: маршруты, схемы, оркестрация
    │
    ├── api/search.py      — логика поиска (semantic, class, face, unified)
    ├── api/index.py       — запуск и статус индексации
    ├── api/people.py      — люди, merge/split, кластеризация
    ├── api/settings.py    — конфигурация scan
    └── api/keywords.py    — словарь COCO-классов для UI
    │
    ├──► indexing/         — пайплайн индексации
    ├──► ml/               — абстракции и реализации моделей
    ├──► db/               — SQLite через SQLAlchemy
    ├──► vectors/          — клиент Qdrant
    └──► io_utils/         — обход FS, content hash
```

### 3.1. Принцип разделения ответственности

| Слой | Ответственность | Не делает |
|------|-----------------|-----------|
| **clients/** | Отображение, ввод запроса, polling прогресса | ML, доступ к БД напрямую |
| **api/** | HTTP-контракт, валидация, маршрутизация | Прямой вызов `model.predict` (делегирует в `search.py` / `indexing/`) |
| **indexing/** | Scan, gap detection, batch inference, запись в stores | HTTP |
| **ml/** | Encode/detect/analyze, batch API моделей | Персистентность |
| **db/** | CRUD метаданных, статусы модулей, persons | Векторный поиск |
| **vectors/** | Upsert/search/delete points в Qdrant | Хранение bbox, labels |

---

## 4. Хранилища данных

### 4.1. SQLite (`data/i-mage.db`)

Режим **WAL** (`PRAGMA journal_mode=WAL`), `busy_timeout=30000`, `foreign_keys=ON`. Путь переопределяется через `IMAGE_DB_PATH`.

#### Основные таблицы

| Таблица | Содержимое |
|---------|------------|
| `images` | Каталог: `id` (UUID), `path`, `content_hash` (SHA-256), `mtime`, `size` |
| `image_yolo` / `image_clip` / `image_faces` | Статус индексации per module: `running` / `done` / `failed`, `model_version` |
| `detections` | YOLO: label, confidence, bbox (xyxy), FK → `images` |
| `faces` | Детекции лиц: bbox, `detection_score`, детерминированный `id` |
| `persons` | Группы людей (имя, `is_named`) |
| `face_person_assignments` | Привязка `face_id` → `person_id`, `source` (`auto_cluster`, `manual`, …) |
| `index_runs` | Прогресс прогонов: module, mode, phase, progress_done/total, status |
| `app_settings` | JSON `scan_config` (каталоги, фоновый indexer) |

#### Идентификаторы

- **`image_id`** — стабильный UUID, не зависит от пути. При переименовании файла обновляется `path`, `id` сохраняется.
- **`face_id`** — детерминированный `uuid5` от `(image_id, bbox, model_version)`.
- **`content_hash`** — триггер инвалидации: при изменении содержимого файла сбрасываются все module-статусы и связанные данные.

### 4.2. Qdrant

URL: `QDRANT_URL` (по умолчанию `http://127.0.0.1:6333`). Встроенный file-mode **удалён** — только server mode через Docker.

| Коллекция | Гранулярность | Размерность | Payload |
|-----------|---------------|-------------|---------|
| `context` | 1 point на `image_id` | 512 (CLIP) | `image_id`, `model_version` |
| `faces` | N points на фото (по лицу) | 512 (ArcFace) | `face_id`, `image_id`, `model_version` |

Point ID — детерминированные UUID5 (`vectors/ids.py`), операции — **upsert**. YOLO **не** хранится в Qdrant.

При удалении изображения из каталога: CASCADE в SQLite + `vector_store.delete_for_image(image_id)`.

### 4.3. Файловая система

Каталоги задаются в `scan_config.include_directories` (по умолчанию — `IMAGE_SEARCH_DIR`, `FACE_SEARCH_DIR`). Обход рекурсивный, фильтр по расширениям (`io_utils/fs.py`), опционально `ignore_globs`.

---

## 5. ML-модули

### 5.1. Абстракции (`ml/`)

```python
EmbeddingModel     # encode_text(), encode_image(), encode_images()
ObjectsRetriever   # detect(), detect_batch(), detect_labels()
FaceRecognizer     # analyze(), analyze_batch() → list[Face]
```

Реализации подменяемы без изменения API/indexing.

### 5.2. CLIP — семантический контекст

| Параметр | Значение |
|----------|----------|
| Класс | `CLIPEmbeddingModel` |
| Модель | `openai/clip-vit-base-patch32` |
| Библиотека | Transformers (`CLIPModel`, `CLIPProcessor`) |
| Устройство | CUDA если доступна, иначе CPU |
| Выход | L2-нормализованный вектор 512d |

**Использование:**
- **Индексация:** `encode_images(batch)` → upsert в Qdrant `context`.
- **Поиск:** `encode_text(query)` → `vector_store.search_context()`.

### 5.3. YOLO — объекты

| Параметр | Значение |
|----------|----------|
| Класс | `YoloObjectsRetriever` |
| Модель | `artifacts/yolov8s.pt` (YOLOv8s) |
| Библиотека | Ultralytics |
| Пороги | `conf=0.25`, `iou=0.45`, `imgsz=640` |
| Классы | 80 COCO |

**Использование:**
- **Индексация:** `detect_batch()` → строки в `detections`, статус `image_yolo=done`.
- **Поиск:** SQL `detections JOIN image_yolo WHERE label=…` — **без** повторного inference.

### 5.4. ArcFace — лица

| Параметр | Значение |
|----------|----------|
| Класс | `ArcFaceRecognizer` |
| Модель | InsightFace `buffalo_l` |
| Модули | detection + recognition |
| `det_size` | (640, 640) |
| Выход | L2-нормализованный embedding 512d per face |

**Использование:**
- **Индексация:** `analyze_batch()` → `faces` в SQLite + upsert в Qdrant `faces`.
- **Query embedding:** `analyze(uploaded_image)` → первое лицо → вектор запроса.
- **Поиск:** Qdrant `search_faces()` → агрегация по `image_id` (max score на фото).

**Fallback:** при недоступном Qdrant и `allow_bruteforce_fallback=True` возможен on-the-fly перебор каталога (`ml/faces/service.search_by_face`). В основном API-потоке используется **только индекс**.

### 5.5. Кластеризация лиц

| Параметр | Значение |
|----------|----------|
| Алгоритм | DBSCAN, metric=`cosine`, `eps = 1 - threshold` |
| Порог по умолчанию | 0.65 |
| Min cluster size | 2 (singletons не создают person) |
| Векторы | Читаются из Qdrant (`scroll_face_vectors`), не из SQLite |

После DBSCAN — запись `face_person_assignments` с `source=auto_cluster`. При regroup сохраняются stable person ID через matching с prior assignments.

### 5.6. Взаимодействие моделей при unified-поиске

Unified-поиск **не** запускает YOLO на лету. Он:

1. Извлекает COCO-метки из текста запроса (`api/keywords.py`: regex + словарь синонимов).
2. Кодирует текст через CLIP → Qdrant.
3. Для каждой метки — SQL-поиск по `detections`.
4. Объединяет результаты с rank_score: `bonus + max(clip_score, yolo_max)`, bonus=1 если совпали оба источника.

---

## 6. Потоки выполнения (threads и async)

### 6.1. Event loop (FastAPI / Uvicorn)

- **Main thread:** async handlers, `background_indexer_loop`, `IndexExecutor` tasks.
- **Worker threads:** `asyncio.to_thread()` для CPU/GPU-bound работы (индексация, reconcile, clustering).

### 6.2. Фоновые задачи при старте

```text
lifespan(app):
  1. Загрузка CLIP, YOLO, ArcFace (синхронно, ~десятки секунд)
  2. Подключение Qdrant, открытие Database
  3. fail_stale_runs() — помечает зависшие прогоны
  4. IndexExecutor(db, vector_store, models)
  5. asyncio.create_task(background_indexer_loop)  — tick каждые 3600 с
  6. yield — сервер принимает запросы
  shutdown:
  7. stop_event → await indexer_task
  8. executor.shutdown() — cooperative stop in-flight scans
  9. db.close(), vector_store.close()
```

### 6.3. IndexExecutor — координация индексации

```
IndexExecutor
  ├── _catalog_lock          — один reconcile каталога
  ├── _module_locks[yolo|clip|faces]  — один full run per module
  ├── _tasks[run_id]         — asyncio.Task per run
  ├── _stop_event            — shutdown signal
  └── GpuScheduler           — shared, singleton
```

Конфликты → HTTP 409 (`IndexRunConflictError`).

### 6.4. GpuScheduler

Один планировщик на процесс, priority queue:

| Приоритет | Значение | Примеры |
|-----------|----------|---------|
| `INTERACTIVE = 0` | Выше | `search:clip-text`, `search:face-query` |
| `INDEXING = 10` | Ниже | `index:clip`, `index:yolo`, `index:faces` |

Индексация захватывает GPU на один batch; между batch'ами интерактивный поиск может вклиниться.

Batch sizes (env): `CLIP_BATCH_SIZE=32`, `YOLO_BATCH_SIZE=16`, `FACES_BATCH_SIZE=16`, `FACES_ANALYZE_WORKERS=4` (CPU decode).

### 6.5. Database — thread-local sessions

`Database` использует `threading.local()` для SQLAlchemy Session per thread. Контекстный менеджер `with db:` → commit/rollback/close. Это позволяет worker threads безопасно писать в SQLite параллельно с read-heavy API (WAL).

---

## 7. Поток данных: индексация

### 7.1. Общая схема

```text
scan_config.include_directories
        │
        ▼
collect_scoped_files()          ← io_utils/scan.py
        │
        ▼
reconcile_catalog()             ← upsert images, remove missing, content_hash
        │
        ▼
run_scan(modules, mode)
        │
        ├── gap_paths = catalog paths WHERE module_status != DONE
        │
        └── для каждого модуля:
              batch inference → write stores → update progress
```

### 7.2. Режимы индексации

| Mode | Endpoint / триггер | Поведение |
|------|-------------------|-----------|
| `reconcile` | `POST /index/reconcile` | Только синхронизация FS ↔ SQLite (без ML) |
| `full` | `POST /index/run/full/{module}` | Gap-only: доиндексировать всё со статусом ≠ DONE |
| `gap` | `POST /index/run/background`, background loop | То же, но список modules из `background_modules` |
| `cluster` | `POST /people/cluster` | Только DBSCAN, без повторного ArcFace |

### 7.3. Порядок записи (безопасность)

**CLIP:**
```text
mark_running → encode → upsert Qdrant → mark_done
```

**YOLO:**
```text
detect_batch → replace detections → mark_done
```

**Faces:**
```text
analyze_batch → replace faces in SQLite → upsert Qdrant → delete removed face points → mark_done
```
(Qdrant **до** `mark_done`, чтобы не было status=done без векторов.)

**Full run faces** дополнительно вызывает `run_face_clustering()` после индексации.

### 7.4. Фоновый indexer

`background_indexer_loop`:
- Проверка каждые **3600 с** (`CHECK_INTERVAL_SECONDS`).
- Если `background_indexer_enabled` и прошло `schedule_interval_days` с `last_background_run_at` → `executor.start_background_gap()`.
- Не стартует при активном manual run или catalog lock.

### 7.5. Инвалидация при изменении файла

```text
content_hash изменился
  → delete detections, faces, module statuses
  → delete Qdrant points for image_id
  → gap re-index подхватит файл при следующем run
```

---

## 8. Поток данных: поиск

### 8.1. Семантический (`POST /search`)

```text
Browser: query text
    → API: run_search_by_description()
        → db.images.list_all() → paths
        → GpuScheduler.acquire(INTERACTIVE)
        → CLIP.encode_text(query)
        → vector_store.search_context(embedding, image_ids, k)
        → map image_id → path
    ← JSON { matches: [{ path, score }] }
```

Требует проиндексированный CLIP (`image_clip.status=done`) и доступный Qdrant.

### 8.2. По классу объекта (`POST /search/class`)

```text
Browser: label (e.g. "cat")
    → API: run_search_by_class()
        → db.detections.search_by_label(label, k)
            (JOIN image_yolo WHERE status=done, GROUP BY image, ORDER BY max confidence)
    ← JSON { matches: [{ path, confidence }] }
```

Требует проиндексированный YOLO. **Inference на поиске не выполняется.**

### 8.3. По лицу (`POST /search/face/*`)

Двухшаговый поток в Web UI:

```text
1. POST /search/face/embed  (multipart: photo)
       → ArcFace.analyze(upload) → { embedding, detection_score }

2. POST /search/face/stream  (JSON: { embedding, k, threshold })
       → vector_store.search_faces(embedding, scope_image_ids)
       → aggregate by image_id (best face score per photo)
    ← NDJSON stream → UI renders results
```

Альтернатива: `POST /search/face/upload` — embed + search одним запросом.

### 8.4. Unified (`POST /search/unified/stream`)

```text
query + optional labels[]
    → extract_labels_from_query()  // авто-метки из текста
    → CLIP search (Qdrant)
    → for each label: SQL detections search
    → merge + rank
    ← NDJSON { stage: done, matches: [...] }
```

### 8.5. Streaming (NDJSON)

Эндпоинты `*/stream` отдают `application/x-ndjson`: по одному JSON-объекту на строку. UI читает поток через `fetch` + `ReadableStream`, показывает прогресс (`stage`, `status`).

---

## 9. Поток данных: люди (persons)

```text
Faces index (ArcFace) → Qdrant faces + SQLite faces
        │
        ▼
POST /people/cluster  (или автоматически после full faces run)
        │
        ▼
run_face_clustering()
  1. list face_ids in scope
  2. scroll vectors from Qdrant
  3. DBSCAN → clusters (size ≥ 2)
  4. assign_faces → persons (source=auto_cluster)
        │
        ▼
GET /people  → UI: список групп с превью
PATCH /people/{id}  → rename
POST /people/merge  → объединить группы
POST /people/split  → разделить группу вручную
```

Ручные операции (`merge`, `split`, `rename`) помечаются `source=manual` и не перезаписываются при regroup auto-clusters (кроме явного `regroup=true` для unnamed groups).

---

## 10. Граница: клиент vs сервер

### 10.1. Что делает **пользователь / браузер**

| Действие | Где выполняется |
|----------|-----------------|
| Ввод текстового запроса | Браузер |
| Выбор/загрузка фото лица для поиска | Браузер (файл уходит на сервер multipart) |
| Отображение результатов, превью | Браузер (`<img src="/images/by-id/{id}">`) |
| Polling `/index/status` (~1 с при активной индексации) | Браузер |
| Настройка каталогов scan, фонового indexer | Браузер → `PUT /settings/scan` |
| Запуск reconcile / full index / cluster | Браузер → POST endpoints |
| Merge/split/rename людей | Браузер → `/people/*` |
| «Показать в файловом менеджере» | Браузер → `POST /files/reveal` → сервер вызывает `xdg-open` / аналог |

**Браузер не выполняет:** ML inference, embedding, vector search, работу с SQLite/Qdrant.

### 10.2. Что делает **сервер**

| Действие | Компонент |
|----------|-----------|
| Загрузка и удержание моделей в RAM/VRAM | `lifespan` → `app.state.model/objects_model/face_model` |
| Scan каталогов, content hash | `indexing/runner`, `io_utils` |
| Batch GPU inference | `indexing/runner._run_*_batches` |
| Запись в SQLite и Qdrant | `db/*`, `vectors/store.py` |
| Vector search | `vectors/store.py` |
| SQL search по detections | `db/detections.py` |
| DBSCAN, управление persons | `indexing/cluster.py`, `db/persons.py` |
| Приоритизация GPU | `indexing/gpu_scheduler.py` |
| Раздача статики UI | `GET /` → `clients/web/index.html` |
| Раздача файлов изображений | `GET /images/by-id/{id}`, `GET /images/{filename}` |

### 10.3. Что **не** делает система

- Нет аутентификации и multi-user.
- Нет облачной синхронизации каталога.
- Нет watch/inotify на FS (только periodic background + manual reconcile).
- Нет обработки видео.
- Нет отправки фото пользователя на внешние API при поиске (только локальный inference).

---

## 11. Возможности пользователя

### 11.1. Поиск (Web UI)

| Возможность | API | Условие работы |
|-------------|-----|----------------|
| Текстовый unified-поиск | `POST /search/unified/stream` | CLIP indexed + (для object boost) YOLO indexed |
| Подсказки COCO-классов | `GET /keywords` | Всегда |
| Поиск по лицу (upload) | `/search/face/embed` + `/search/face/stream` | Faces indexed, Qdrant up |
| Просмотр найденных фото | `/images/by-id/{id}` | Файл существует на диске |

### 11.2. Индексация и каталог

| Возможность | API |
|-------------|-----|
| Сверка каталога (FS → SQLite) | `POST /index/reconcile` |
| Полный gap-run модуля (yolo / clip / faces) | `POST /index/run/full/{module}` |
| Ручной background gap | `POST /index/run/background` |
| Прогресс и статистика | `GET /index/status?stats=true` |
| GPU queue snapshot | поле `gpu` в `/index/status` |
| Настройка каталогов и расписания | `GET/PUT /settings/scan` |

### 11.3. Люди

| Возможность | API |
|-------------|-----|
| Список сгруппированных людей | `GET /people` |
| Детали и лица персоны | `GET /people/{id}/faces` |
| Переименование | `PATCH /people/{id}` |
| Объединение групп | `POST /people/merge` |
| Разделение группы | `POST /people/split` |
| Запуск / перегруппировка DBSCAN | `POST /people/cluster?regroup=false\|true` |
| Лица с низкой уверенностью детекции | `GET /people/low-confidence` |

### 11.4. CLI (отладка)

```bash
python -m clients.cli health
python -m clients.cli search "кот на диване"
```

---

## 12. Конфигурация

### 12.1. Переменные окружения

| Переменная | Default | Назначение |
|------------|---------|------------|
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant server |
| `IMAGE_DB_PATH` | `data/i-mage.db` | SQLite |
| `IMAGE_SEARCH_DIR` | `data/flickr30k/images` | Стартовый каталог |
| `FACE_SEARCH_DIR` | `data/small_celeba` | Стартовой каталог лиц |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Uvicorn bind |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | `1` в run scripts | Не блокировать старт на Hub |
| `CLIP_BATCH_SIZE` / `YOLO_BATCH_SIZE` / `FACES_BATCH_SIZE` | 32 / 16 / 16 | Batch indexing |
| `FACES_ANALYZE_WORKERS` | 4 | CPU workers для decode лиц |

### 12.2. ScanConfig (SQLite `app_settings`)

```json
{
  "include_directories": ["/path/to/photos"],
  "ignore_globs": ["**/thumbnails/**"],
  "background_indexer_enabled": false,
  "schedule_interval_days": 7,
  "background_modules": ["yolo", "clip", "faces"],
  "last_background_run_at": "2026-08-29T12:00:00+00:00"
}
```

---

## 13. Диаграмма: полный жизненный цикл фото

```text
  [Новый файл на диске]
           │
           ▼
  POST /index/reconcile  ──►  images row (UUID, content_hash)
           │
           ▼
  POST /index/run/full/yolo  ──►  detections + image_yolo=done
           │
           ▼
  POST /index/run/full/clip  ──►  Qdrant context point + image_clip=done
           │
           ▼
  POST /index/run/full/faces ──►  faces rows + Qdrant face points
           │                      + image_faces=done
           ▼                      + auto DBSCAN → persons
  [Пользователь ищет]
           │
     ┌─────┴─────┬─────────────┐
     ▼           ▼             ▼
  unified     class        face upload
  CLIP+SQL    SQL only     Qdrant faces
     │           │             │
     └─────┬─────┴─────────────┘
           ▼
  GET /images/by-id/{id}  ──►  FileResponse с диска
```

---

## 14. Ограничения и trade-offs

| Решение | Причина |
|---------|---------|
| Один Python-проcess (API + indexer) | Проще деплой; достаточно для локального use case |
| Qdrant только server mode | Embedded mode ломался при `--reload` и multi-writer |
| YOLO не в Qdrant | Классы и bbox — structured data, эффективнее SQL |
| Gap indexing вместо full re-scan | Content hash + module status |
| Search не триггерит indexing | API read-only для search; indexer write async |
| Scope filter >1000 paths | Qdrant search global + Python filter (MatchAny медленный) |
| Singleton faces не в persons | Избежать миллионов одно-лицевых «людей» |

---

## 15. Структура модулей (справочник)

```
api/
  app.py           — FastAPI app, lifespan, search routes
  search.py        — search orchestration
  index.py         — index routes
  people.py        — persons API
  settings.py      — scan config
  keywords.py      — COCO keyword extraction
  schemas.py       — Pydantic models

indexing/
  executor.py      — async task coordinator
  runner.py        — scan, batch indexing
  background_indexer.py
  cluster.py       — face clustering job
  clip.py / yolo.py / faces.py  — single-image index helpers
  gpu_scheduler.py
  gap.py           — gap path detection

ml/
  embeddings/      — CLIP
  objects/         — YOLO
  faces/           — ArcFace + clustering util

db/
  database.py      — facade
  models.py        — ORM
  images.py, detections.py, faces.py, persons.py, index_runs.py, …

vectors/
  store.py         — Qdrant client
  ids.py           — deterministic point IDs
  config.py        — collection names, dims

clients/
  web/index.html   — SPA-like UI (vanilla JS)
  cli/__main__.py  — debug HTTP client
```

---

*Документ отражает код на момент написания. При расхождении с реализацией приоритет у исходников.*
