#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
API_PORT="${API_PORT:-8001}"
WEB_PORT="${WEB_PORT:-3000}"
RESET_NEXT_CACHE="${RESET_NEXT_CACHE:-1}"

kill_port_if_busy() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "[dev] releasing port :$port (pid: $pids)"
    kill $pids 2>/dev/null || true
    sleep 1
  fi
}

cleanup() {
  if [[ -n "${API_PID:-}" ]] && kill -0 "$API_PID" 2>/dev/null; then
    kill "$API_PID" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID:-}" ]] && kill -0 "$WEB_PID" 2>/dev/null; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
}

trap cleanup EXIT INT TERM

kill_port_if_busy "$API_PORT"
kill_port_if_busy "$WEB_PORT"

if [[ "$RESET_NEXT_CACHE" == "1" ]]; then
  echo "[dev] clearing Next cache (.next)"
  rm -rf "$ROOT_DIR/frontend/.next"
fi

echo "[dev] starting backend on :${API_PORT}"
echo "[dev] runtime-config: http://127.0.0.1:${API_PORT}/api/runtime-config"
(
  cd "$ROOT_DIR"
  source ".venv/bin/activate"
  uvicorn api_server:app --host 0.0.0.0 --port "$API_PORT" --reload
) &
API_PID=$!

echo "[dev] starting frontend on :${WEB_PORT} (api: :${API_PORT})"
(
  cd "$ROOT_DIR/frontend"
  NEXT_PUBLIC_API_URL="http://localhost:${API_PORT}" npm run dev -- --port "$WEB_PORT"
) &
WEB_PID=$!

while true; do
  if ! kill -0 "$API_PID" 2>/dev/null; then
    wait "$API_PID" 2>/dev/null || true
    break
  fi
  if ! kill -0 "$WEB_PID" 2>/dev/null; then
    wait "$WEB_PID" 2>/dev/null || true
    break
  fi
  sleep 1
done
