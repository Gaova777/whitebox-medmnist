#!/usr/bin/env bash
# Shortcuts for this project (make isn't installed on the machine it was built on).
set -euo pipefail
cd "$(dirname "$0")"

case "${1:-help}" in
  test)       uv run --group mlops pytest -q && uv run ruff check src scripts tests ;;
  benchmark)  for ds in pneumoniamnist bloodmnist; do uv run scripts/train.py $ds --size 28; done ;;
  train)      for ds in pneumoniamnist bloodmnist; do uv run scripts/train.py $ds --size 128; done ;;
  sanity)     for ds in pneumoniamnist bloodmnist; do uv run scripts/sanity.py $ds --size 128; done ;;
  drift)      for ds in pneumoniamnist bloodmnist; do uv run --group mlops scripts/drift.py $ds --size 128; done ;;
  gate)       uv run --group mlops scripts/quality_gate.py --size 128 ;;
  api)        uv run uvicorn whitebox_medmnist.entrypoints.api:app --port 8000 ;;
  web)        cd web && PATH=$HOME/.local/opt/node/bin:$PATH npx ng serve ;;
  mlflow)     uv run --group mlops mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000 ;;
  *)          echo "usage: ./run.sh test|benchmark|train|sanity|drift|gate|api|web|mlflow" ;;
esac
