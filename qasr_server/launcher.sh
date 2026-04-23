#!/bin/bash
# Container entrypoint. Idempotent; safe to restart.
set -e

# Ensure FastAPI/uvicorn present (base image may not ship them).
python3 -c "import fastapi, uvicorn" 2>/dev/null || \
  pip install --no-cache-dir --break-system-packages fastapi 'uvicorn[standard]' >/dev/null

exec uvicorn qasr_server.server:app \
  --host "${QASR_HOST:-0.0.0.0}" \
  --port "${QASR_PORT:-8000}" \
  --log-level "${QASR_LOG_LEVEL:-info}"
