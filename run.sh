#!/usr/bin/env bash
# Запуск I-Mage без авто-перезагрузки.
#
# ВАЖНО: перед запуском поднимите Qdrant-сервер: `docker compose up -d`.
# Qdrant работает только как сервер (встроенный режим удалён).
#
# Не используйте `uvicorn --reload` во время активной индексации: перезапуск
# сервера при правке .py прервёт прогон ("interrupted by server restart").
# Для разработки используйте run-dev.sh.
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"

# Адрес Qdrant-сервера (docker compose up -d).
export QDRANT_URL="${QDRANT_URL:-http://127.0.0.1:6333}"
# Don't block startup on Hugging Face Hub reachability when models are cached.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ml-env

exec uvicorn api.app:app --host "$HOST" --port "$PORT" --log-level info "$@"
