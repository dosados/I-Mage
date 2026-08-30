# Getting started

## Requirements

- Python 3.10 or later and Conda
- Docker with Docker Compose
- The `ml-env` Conda environment, including PyTorch, Transformers,
  Ultralytics, InsightFace, FastAPI, and Uvicorn

A CUDA-capable GPU is optional. The application can run on CPU, although
indexing will take longer.

## Start the application

Run these commands from the repository root:

```bash
docker compose up -d
./run-dev.sh
```

`run-dev.sh` activates `ml-env` and starts Uvicorn with source reload enabled.
For a normal server without reload, use:

```bash
./run.sh
```

Open the application at <http://127.0.0.1:8000>. The interactive API reference
is available at <http://127.0.0.1:8000/docs>.

## Verify the services

In a second terminal, check the API:

```bash
curl http://127.0.0.1:8000/health
```

The expected response includes `"status": "ok"` and `"qdrant": "ok"`.
To inspect Qdrant's container:

```bash
docker compose ps
```

## Models and first start

The server loads CLIP, YOLOv8s, and ArcFace during startup. The shell scripts
default to offline mode for Hugging Face and Transformers, so the required
models must already be present in the local caches for an offline start. If
they are not cached, temporarily allow downloads:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 ./run.sh
```

YOLO and InsightFace may likewise download their weights on their first use.

## Index your photos

Use the web UI to set scan directories and start reconciliation followed by
the required module runs. Alternatively, use the API:

```bash
curl -X POST http://127.0.0.1:8000/index/reconcile
curl -X POST http://127.0.0.1:8000/index/run/full/clip
curl -X POST http://127.0.0.1:8000/index/run/full/yolo
curl -X POST http://127.0.0.1:8000/index/run/full/faces
```

Monitor progress with:

```bash
curl http://127.0.0.1:8000/index/status
```

Avoid editing Python files while a run is active under `run-dev.sh`: hot reload
stops the server and interrupts that run.

## Stop the services

Stop Uvicorn with `Ctrl+C`. Stop Qdrant when it is no longer needed:

```bash
docker compose down
```
