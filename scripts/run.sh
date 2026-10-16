#!/usr/bin/env bash
set -euo pipefail

MODEL=${1:-"GradientBoosting"}
DATA=${2:-"data/roboBohr.csv"}
SEED=${3:-42}
OUTPUT=${4:-"results"}

echo "=============================================="
echo "ML Pipeline: Atomization Energy Prediction"
echo "=============================================="
echo "Model  : ${MODEL}"
echo "Data   : ${DATA}"
echo "Seed   : ${SEED}"
echo "Output : ${OUTPUT}"
echo "=============================================="

mkdir -p "${OUTPUT}"

# ВАЖНО: используем 'uv run', чтобы гарантированно взять Python с установленными библиотеками
uv run python -m src.main \
    --model "${MODEL}" \
    --data  "${DATA}" \
    --seed  "${SEED}" \
    --output "${OUTPUT}"

echo "[OK] Pipeline завершён успешно."
