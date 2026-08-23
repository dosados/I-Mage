# План индексации (этапы 1–4)

Документ описывает поэтапный план перехода от текущего on-the-fly поиска к персистентному индексу с SQLite, Qdrant и отдельным процессом индексации.

**Scope:** этапы 1–4. Пятый этап (отдельные ML-worker'ы) сознательно не включён — см. [Что сознательно откладываем](#что-сознательно-откладываем).

Связанные документы: [mvp.md](mvp.md), [goal.md](goal.md).

---

## Принципы

Каждый этап должен:

1. **Не быть глобальным рефакторингом** — новый слой поверх существующего кода, старый путь остаётся fallback где нужно.
2. **Логически следовать из предыдущего** — без «скipping» (Qdrant до SQLite, multiprocess до отладки sync).
3. **Самостоятельно улучшать продукт** — измеримый выигрыш (скорость, надёжность, прогресс индексации).

Общая линия:

```text
каталог файлов → SQLite (метаданные + YOLO) → Qdrant (векторы) + jobs → два процесса (API | indexer)
```

---

## Этап 1 — Base (текущее состояние)

### Что есть

- On-the-fly поиск: scan FS → CLIP / YOLO / ArcFace на каждый запрос.
- Три режима: текст (`POST /search`), класс (`POST /search/class`), лицо (`POST /search/face`).
- Два каталога: `IMAGE_SEARCH_DIR`, `FACE_SEARCH_DIR`.
- Идентификация по `Path`; метаданные JSON датасетов API не использует.

### Что не меняем на этом этапе

- Интерфейсы `EmbeddingModel`, `FaceRecognizer`, `ObjectsRetriever` — станут backend'ом для indexer'а на следующих шагах.

### Критерий готовности к этапу 2

- Все три режима поиска стабильно работают через API (как сейчас).

---

## Этап 2 — SQLite: всё кроме векторов (in-process)

### Цель

Ввести **формат правды** для каталога и не-векторных ML-результатов. Индексация и поиск по-прежнему в одном процессе.

### Источник истины

```text
Файл на диске + строка images в SQLite — первичная правда
Производные данные (detections, faces_meta) — только из indexer pipeline
Векторы — пока не сохраняем (этап 3)
```

### Что хранить в SQLite

**Таблица `images`** — только каталог файлов

| Поле | Назначение |
|------|------------|
| `id` | Стабильный UUID (не path) |
| `path` | Абсолютный или канонический путь |
| `content_hash` | SHA256 содержимого файла |
| `mtime`, `size` | Для быстрого pre-check до hash |
| `created_at`, `updated_at` | Служебные |

**Таблицы заголовков модулей** — одна строка на фото (`image_yolo`, `image_faces`; позже `image_clip`)

| Поле | Назначение |
|------|------------|
| `image_id` | PK, FK → `images` |
| `status` | `running` \| `done` \| `failed` (нет строки = ещё не индексировали) |
| `model_version` | Версия модели |
| `indexed_at`, `last_error` | Служебные |

Строка создаётся при первой индексации модуля, не при scan каталога.

**Таблица `detections`** — результаты YOLO (N строк на `image_id`)

| Поле | Назначение |
|------|------------|
| `id` | PK |
| `image_id` | FK → `images` |
| `label` | Класс COCO |
| `confidence` | Для ранжирования |
| `bbox_x1` … `bbox_y2` | Bbox (xyxy) |

**Таблица `faces`** — результаты детекции лиц без embedding (N строк на `image_id`)

| Поле | Назначение |
|------|------------|
| `id` | PK |
| `image_id` | FK |
| `bbox_*`, `detection_score` | Метаданные лица |

Реализация: модуль `db/` (SQLAlchemy). `index_jobs` — позже, на этапе 4.

### Hash

- **Content hash (SHA256)** — решает, менялось ли содержимое; триггер переиндексации.
- **Path** — для отображения и `GET /images/{filename}`; при переименовании обновляем path, hash не меняется.
- `mtime` — только быстрый pre-filter; не заменяет content hash.

### Что запускать при индексации на этапе 2

**Запускать и сохранять:**

- Scan каталога → upsert `images`.
- YOLO → `detections` + `image_yolo.status=done`.

**Не запускать при индексации (или не сохранять результат):**

- CLIP `encode_image` — поиск по тексту остаётся **on-the-fly** до этапа 3.
- ArcFace embedding — поиск по лицу остаётся **on-the-fly** до этапа 3.

> **Важно:** не гонять CLIP/ArcFace при индексации, если векторы всё равно выбрасываются — это лишняя нагрузка без пользы.

### Стыковка с ML-сервисами

- Indexer вызывает существующие `detect_objects()` / `ObjectsRetriever` — без дублирования логики YOLO.
- Поиск по классу: **сначала SQL** (`detections` + JOIN `image_yolo`), **fallback** on-the-fly YOLO, если `image_yolo.status != done` или записей нет.

### In-process indexer

- Модуль `indexing/` (или аналог): `scan_catalog()`, `index_yolo(image_id)`.
- Запуск: CLI `python -m indexing.run --dir ...` или endpoint `POST /index/run` (синхронно, для отладки).

### Критерий готовности к этапу 3

- Scan стабилен; `image_id` используется везде вместо «голого» path как PK.
- `POST /search/class` быстрее on-the-fly на проиндексированном корпусе.
- Повторный scan не переиндексирует файлы с тем же `content_hash`.

---

## Этап 3 — Qdrant + index_jobs (in-process)

### Цель

Персистентные векторы для CLIP (контекст) и ArcFace (лица). Синхронизация SQLite ↔ Qdrant отлаживается **в одном процессе** до split на этапе 4.

### Qdrant: две коллекции

CLIP и ArcFace — **разные векторные пространства**, даже при одинаковой размерности (512). Одна коллекция на тип:

| Коллекция | Гранулярность | Payload (минимум) |
|-----------|---------------|-------------------|
| `context` | 1 point на `image_id` | `image_id`, `model_version` |
| `faces` | N points на фото (по лицу) | `face_id`, `image_id`, `model_version` |

**YOLO в Qdrant не класть** — классы и bbox остаются в SQLite.

### Point ID (идемпотентность)

- `context`: детерминированный id от `image_id` (+ `model_version` при смене модели).
- `faces`: детерминированный id от `face_id` (например hash от `image_id` + нормализованный bbox + `model_version`).
- Операции — **upsert**, не insert-with-random-id.

### Таблица `index_jobs`

| Поле | Назначение |
|------|------------|
| `id` | PK |
| `image_id` | FK |
| `module` | `yolo` \| `clip` \| `faces` |
| `status` | `pending` \| `running` \| `done` \| `failed` |
| `model_version` | Версия модели для этого job |
| `attempts`, `last_error` | Retry и диагностика |
| `updated_at` | Служебное |

Плюс поля на `images`: `clip_status`, `faces_status` (аналогично `yolo_status`).

### Порядок записи (безопасный шаг)

```text
1. ML → результат
2. Upsert в Qdrant / SQLite
3. Только при успехе → index_jobs.status = done, images.*_status = done
```

При сбое между 2 и 3 job остаётся `failed`/`running` — retry безопасен (upsert идемпотентен).

### Поиск

| Режим | Источник |
|-------|----------|
| Текст | `encode_text(query)` + Qdrant `context` search → `image_id` → path из SQLite |
| Лицо | Query embedding + Qdrant `faces` search → агрегация по `image_id` (лучший score на фото) → path из SQLite |
| Класс | SQLite `detections` (как на этапе 2) |

**Агрегация лиц:** Qdrant возвращает top-k **лиц**; для UI нужен top-k **фото** — post-aggregate по `image_id` (max score), как в текущем `search_by_face`.

### SearchBackend (абстракция)

Ввести слой, чтобы этап 4 не менял контракт API:

```text
api/search.py
    ↓
SearchBackend (protocol)
    ├── LiveSearchBackend       # fallback, on-the-fly
    └── IndexedSearchBackend    # SQLite + Qdrant
```

Политика для не проиндексированных файлов (зафиксировать явно):

- только indexed (быстро, неполно), или
- indexed + live fallback (полнее, медленнее), или
- UI показывает «индексируется N фото».

### In-process indexer

Расширить `indexing/runner`: для каждого `image_id` (или job) — `clip`, `faces`, `yolo` (yolo уже с этапа 2).

Qdrant — отдельный процесс-сервер (`localhost:6333`); Python — клиент. ML-модели — in-process с indexer'ом.

### Критерий готовности к этапу 4

- Text/face search из индекса быстрее on-the-fly на целевом корпусе.
- Partial failure отрабатывается: видно `clip=done`, `faces=failed`, retry только `faces`.
- Reconciliation: файл удалён → tombstone в SQLite + delete points в Qdrant по `image_id`.

---

## Этап 4 — Два процесса: API и Indexer

### Цель

API не блокируется длительной индексацией. Indexer в фоне сканирует каталог, сравнивает hash, обрабатывает новые/изменённые файлы, пишет в SQLite и Qdrant.

### Роли процессов

```text
┌─────────────────────────┐         ┌──────────────────────────────┐
│  Process 1: API         │         │  Process 2: Indexer            │
│  (uvicorn)              │         │  (python -m indexing.worker)   │
│                         │         │                                │
│  • POST /search, /face  │  read   │  • scan directory              │
│  • POST /search/class   │ ──────► │  • content_hash vs SQLite      │
│  • GET /images          │ SQLite  │  • claim index_jobs            │
│                         │ Qdrant  │  • ML → SQLite + Qdrant        │
│  • POST /index/trigger  │         │  • update job status           │
│    (enqueue only)       │ enqueue │                                │
│  • GET /index/status    │ ──────► │                                │
└─────────────────────────┘  jobs   └──────────────────────────────┘
         в SQLite
```

### Что API **не** делает

- Не вызывает indexer по RPC на **каждый** search-запрос.
- Не запускает CLIP/YOLO/ArcFace для наполнения индекса (только read path + enqueue).

Search **читает** уже записанное; indexer **пишет** асинхронно.

### Что делает Indexer

1. Scan `IMAGE_SEARCH_DIR` / `FACE_SEARCH_DIR` (или один корень с полем `source`).
2. Быстрый pre-check (`mtime`/`size`) → при подозрении на изменение — `content_hash`.
3. Новые/изменённые → upsert `images` + создать `index_jobs` (`pending`).
4. Loop: claim job → ML → write stores → `status=done`.
5. Удалённые файлы → reconciliation (tombstone + delete Qdrant points).

### Координация: очередь через SQLite

Предпочтительно **не** отдельный HTTP между процессами для каждой операции:

- API: `INSERT INTO index_jobs ...` или `POST /index/trigger` → enqueue.
- Indexer: атомарный `claim` pending jobs (`UPDATE ... RETURNING` или эквивалент).

Один **writer** в SQLite — indexer. API: read-heavy + редкий enqueue.

### SQLite

- Режим **WAL** — параллельное чтение API и запись indexer'а.
- Избегать двух активных writer'ов (конкуренция, `database is locked`).

### Qdrant

- **Запись** — только indexer.
- **Search** — API (read-only клиент).

Sync SQLite ↔ Qdrant завершается **до** `job=done`; search не должен зависеть от «фонового merge» после ответа пользователю.

### Endpoints (ориентир)

| Method | Endpoint | Процесс | Назначение |
|--------|----------|---------|------------|
| `POST` | `/index/trigger` | API | Поставить scan/reindex в очередь |
| `GET` | `/index/status` | API | Прогресс: pending/done/failed по jobs |
| — | worker loop | Indexer | Scan + claim + ML + write |

### Критерий готовности этапа 4

- Индексация 10k+ фото не блокирует `/search`.
- Новый файл появляется в выдаче после indexer (или по политике fallback).
- Перезапуск API не останавливает indexer; перезапуск indexer продолжает с `pending` jobs.

---

## Сводная таблица этапов

| Этап | Хранилище | Индексация | Поиск |
|------|-----------|------------|-------|
| 1 | FS | — | On-the-fly |
| 2 | SQLite | YOLO + catalog, in-process | Class: SQL (+ fallback); text/face: on-the-fly |
| 3 | SQLite + Qdrant | + CLIP + faces, in-process, `index_jobs` | Indexed (+ fallback по политике) |
| 4 | То же | Отдельный worker-процесс | API read-only; indexer write |

---

## Идемпотентность (сквозные правила)

1. **Стабильные id:** `image_id`, `face_id` — не path.
2. **Content hash** — триггер invalidate / новые jobs при изменении файла.
3. **`model_version`** в jobs и Qdrant payload — полный re-embed при смене модели.
4. **Upsert** в Qdrant и upsert строк в SQLite — повтор job безопасен.
5. **Порядок:** сначала write в store, потом `job=done`.
6. **Reconciliation** (периодически или после scan): FS ↔ SQLite ↔ Qdrant.

---

## Что сознательно откладываем

| Идея | Почему |
|------|--------|
| **Этап 5:** третий процесс (ML worker отдельно от «таблицы») | Избыточно, пока один indexer не упирается в bottleneck; риск гонок «ML шлёт всё подряд» без job queue |
| Watch FS (inotify) | После стабильного scan + worker |
| Гибридный поиск (текст + класс + лицо в одном запросе) | После того как каждый режим читает из индекса |
| Staging tables + merge двух пайплайнов | Дублирует `index_jobs` + upsert |
| Авторизация, multi-user | Вне scope локального приложения |

### Если позже понадобится этап 5

Разделять **по модулям** (`clip-worker`, `yolo-worker`, `faces-worker`), а не «ML vs база». Все worker'ы берут work из `index_jobs WHERE status=pending` — **не** обходят каталог сами. Критерий введения: один indexer не успевает, GPU простаивает из-за I/O, нужны параллельные ML worker'ы.

---

## Структура модулей (ориентир)

```text
db/
  models.py        # ORM: images, image_yolo, image_faces, detections, faces
  images.py        # каталог, reconcile
  image_yolo.py    # статус YOLO
  image_faces.py   # статус faces
  detections.py    # результаты YOLO
  faces.py         # результаты лиц
  database.py      # фасад Database
```

---

## Чеклист перехода между этапами

| Переход | Критерий |
|---------|----------|
| 1 → 2 | API стабилен; решена схема `images` + `detections` |
| 2 → 3 | Class search из SQL; batch indexer гоняет YOLO без ручного вмешательства |
| 3 → 4 | Text/face из Qdrant; partial retry работает; sync SQLite↔Qdrant предсказуем |
| 4 done | API и indexer живут отдельно; индекс растёт в фоне |
