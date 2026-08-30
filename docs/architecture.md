# Architecture

I-Mage is a local image-search application. The browser communicates with a
FastAPI server; the server loads the ML models, manages indexing, and reads
from local storage. User images remain on the machine. Model weights may be
downloaded from their upstream providers when they are not already cached.

## Components

```text
Browser (clients/web)
        |
        | HTTP / NDJSON
        v
FastAPI application (api)
  |       |          |
  |       |          +-- indexing: catalogue reconciliation and batch jobs
  |       +------------- ML: CLIP, YOLOv8, and ArcFace
  |
  +-- SQLite: catalogue, indexing state, detections, people
  +-- Qdrant: CLIP and face vectors
  +-- local image directories
```

### Web client

`clients/web/index.html` is a static HTML/JavaScript client served by
`GET /`. It calls the REST endpoints and consumes progress-enabled operations
as newline-delimited JSON (NDJSON).

### API and runtime

`api.app` creates one FastAPI process. At startup it loads CLIP, YOLOv8, and
ArcFace; opens SQLite; connects to Qdrant; and starts the background-indexer
loop. CPU/GPU-bound catalogue and inference work runs in worker threads so
that the event loop can continue serving requests.

### ML services

| Service | Purpose | Persisted result |
| --- | --- | --- |
| CLIP (`openai/clip-vit-base-patch32`) | Text-to-image semantic search | One 512-dimensional vector per image in Qdrant `context` |
| YOLOv8s | Object detection for COCO labels | Labels, confidence, and boxes in SQLite |
| InsightFace ArcFace (`buffalo_l`) | Face detection and face similarity | Face metadata in SQLite and one vector per face in Qdrant `faces` |

### Storage

SQLite (`IMAGE_DB_PATH`, default `data/i-mage.db`) is the metadata source of
truth. It stores image paths and hashes, per-module indexing statuses, YOLO
detections, detected faces, people assignments, run history, and scan
settings. SQLite runs in WAL mode to allow reads while indexing writes.

Qdrant is a separate Docker service. It stores only dense CLIP and ArcFace
vectors. Vector point IDs are deterministic, and writes use upserts, making
retries safe. Its persistent data lives in `data/qdrant_server/`.

The image directories themselves remain the source files. The configured
directories are scanned recursively; changing a file's content hash invalidates
its derived results and queues it for reindexing.

## Indexing flow

```text
configured image directories
        |
        v
catalogue reconciliation --> SQLite images table
        |
        v
gap or full module run
  |         |          |
  CLIP      YOLO       Faces
  |         |          |
Qdrant    SQLite    SQLite + Qdrant
```

Reconciliation and model indexing are separate operations. Reconciliation
adds, updates, and removes catalogue records. A full run processes every
eligible image for one module; a gap run only processes missing or invalidated
results. The optional background indexer runs gap indexing on its configured
schedule and does not scan the filesystem by itself.

## Search flow

- Semantic search encodes text with CLIP, searches Qdrant `context`, and maps
  image IDs back to local paths in SQLite.
- Object search queries the persisted YOLO detections in SQLite; it does not
  run YOLO for each request.
- Face search encodes an uploaded face with ArcFace, searches Qdrant `faces`,
  and aggregates face matches by image.
- Unified search combines CLIP results with detected COCO labels extracted
  from the query or supplied explicitly.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `QDRANT_URL` | `http://127.0.0.1:6333` | Qdrant HTTP endpoint |
| `IMAGE_DB_PATH` | `data/i-mage.db` | SQLite database path |
| `IMAGE_SEARCH_DIR` | `data/flickr30k/images` | Default image directory |
| `FACE_SEARCH_DIR` | `data/small_celeba` | Default face directory |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | API bind address |

Scan directories, ignore patterns, and background-indexer options can also be
changed through the UI or `PUT /settings/scan`.
