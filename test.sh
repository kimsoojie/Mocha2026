#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-${CARE_PD_DATASET:-}}"
PY="${PYTHON:-python3}"
if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: $0 /path/to/CARE-PD-Skeleton-6 [true|all]" >&2
  exit 2
fi
LABEL_SUBSET="${2:-true}"
"${PY}" "${ROOT}/code/evaluate.py" \
  --dataset "${DATASET_DIR}" --cache-dir "${ROOT}/data/cache" \
  --model-a "${ROOT}/output/final/model_a.pth" \
  --model-b "${ROOT}/output/final/model_b.pth" \
  --template "${ROOT}/assets/template" --label-subset "${LABEL_SUBSET}" \
  --output "${ROOT}/output/test_metrics.json"

