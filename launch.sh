#!/bin/zsh
APP_DIR="/Users/m4/.gemini/antigravity/scratch/medical-epub-translator"
cd "$APP_DIR" || exit 1
if ! lsof -i :8000 -sTCP:LISTEN >/dev/null 2>&1; then
    nohup "$APP_DIR/.venv/bin/uvicorn" backend.main:app --host 0.0.0.0 --port 8000 >/tmp/medical_epub_translator.log 2>&1 &
    sleep 2
fi
if [ -d "/Applications/Google Chrome.app" ]; then
    open -na "Google Chrome" --args --app="http://127.0.0.1:8000"
else
    open "http://127.0.0.1:8000"
fi
