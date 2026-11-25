#!/bin/bash
# Double-click this file (macOS) to start Transcriber and open it in your browser.
# Keep the Terminal window open while you work; close it (or press Ctrl+C) to stop the app.

cd "$(dirname "$0")" || exit 1
PORT="${TRANSCRIBER_PORT:-8765}"
URL="http://127.0.0.1:${PORT}"

pause() { echo; read -n 1 -s -r -p "Press any key to close this window."; echo; }

if [ ! -x .venv/bin/transcriber ]; then
  echo "Transcriber is not installed in this folder yet."
  echo
  echo "Run these once in Terminal, from this folder:"
  echo "  uv venv --python 3.12 && uv pip install -e \".[asr,mlx]\"    (Apple Silicon)"
  echo "  uv venv --python 3.12 && uv pip install -e \".[asr]\"        (other computers)"
  echo "See the README for details."
  pause
  exit 1
fi

# Already running? Just open it, instead of failing with "address already in use".
if lsof -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  # Match the app's own page title, not just the word, so another program's page can't be mistaken for it.
  if curl -s --max-time 3 "$URL/" | grep -q "· Transcriber</title>"; then
    echo "Transcriber is already running. Opening it in your browser..."
    open "$URL"
    sleep 1
    exit 0
  fi
  echo "Port $PORT is in use by another program, so Transcriber cannot start there."
  echo "Close that program, or start Transcriber on another port, for example:"
  echo "  TRANSCRIBER_PORT=8800 ./Transcriber.command"
  pause
  exit 1
fi

source .venv/bin/activate
transcriber serve --port "$PORT"
status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "Transcriber stopped with an error (code $status)."
  pause
fi
exit "$status"
