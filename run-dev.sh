#!/usr/bin/env bash
# Dev-запуск с авто-перезагрузкой, но БЕЗ слежки за data/ (БД, WAL).
#
# Требуется запущенный Qdrant-сервер: `docker compose up -d`.
# Слежка ведётся только за исходниками (*.py, *.html). Правки кода перезапускают
# сервер — не делайте этого во время активной индексации, иначе прогон прервётся.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml-env

exec uvicorn api.app:app \
  --host "$HOST" \
  --port "$PORT" \
  --reload \
  --reload-dir api \
  --reload-dir db \
  --reload-dir indexing \
  --reload-dir ml \
  --reload-dir vectors \
  --reload-dir io_utils \
  --reload-include "*.html" \
  --reload-exclude "data/*" \
  --log-level info \
  "$@"
