#!/bin/bash
set -a; [ -f .env ] && source .env; set +a
echo "AURUM v17 — http://localhost:8000"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
