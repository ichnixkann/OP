#!/usr/bin/env bash
# Launch the Voice Assistant: starts the Python backend, waits for it to be
# ready, then starts the Electron app.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${VOICE_ASSISTANT_PORT:-8765}"
HOST="127.0.0.1"

echo "==> Starting backend on ${HOST}:${PORT}"
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi
"$PY" -m uvicorn backend.server:app --host "${HOST}" --port "${PORT}" &
BACKEND_PID=$!

cleanup() {
  echo "==> Shutting down"
  kill "${BACKEND_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Waiting for backend to be ready"
for i in $(seq 1 30); do
  if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
    echo "==> Backend ready"
    break
  fi
  sleep 1
  if [ "$i" -eq 30 ]; then
    echo "!! Backend did not become ready in 30s"
    exit 1
  fi
done

echo "==> Starting Electron app"
if command -v npx >/dev/null 2>&1; then
  npx electron .
else
  echo "!! electron not found. Install with: npm install"
  exit 1
fi
