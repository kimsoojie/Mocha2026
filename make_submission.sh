#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
MODEL_A="${ROOT}/output/final/model_a.pth"
MODEL_B="${ROOT}/output/final/model_b.pth"
[[ -f "${MODEL_A}" && -f "${MODEL_B}" ]] || { echo "Run train.sh first." >&2; exit 2; }
"${PY}" "${ROOT}/code/build_best_mocha_score_effectiveprior_submission.py" \
  --template-zip "${ROOT}/assets/Best_submission.zip" \
  --model-a "${MODEL_A}" --model-b "${MODEL_B}" \
  --effective-counts 1244 1071 1093 1045 --logit-tau 0 \
  --prior-kind none --prior-source none --pseudo-weight 1.0 --prior-sample-count 4453 \
  --output-zip "${ROOT}/output/best_s7_all_clipcap_score_noadj.zip" \
  --work-dir "${ROOT}/output/submission_build" --score-tau 1.96 --score-scale 1.0 \
  --source-finetune-epochs 15 --source-lr-backbone 2e-6 --source-lr-head 7e-5 \
  --source-lr-aux 1e-4 --source-sample-counts 2643 1166 511 133 \
  --training-method "Four source-specific dual-GRU fine-tunes with frozen CLIP caption contrast and reconstruction; SCORE merge; no logit adjustment" \
  --caption-source "CARE-PD llm_captions" --caption-encoder "frozen CLIP ViT-B/32" \
  --caption-matched 2915 --caption-missing 1538 \
  --selection-note "Reproduced self-contained best_s7_all_clipcap_score_noadj pipeline."
unzip -tq "${ROOT}/output/best_s7_all_clipcap_score_noadj.zip"
echo "Submission: ${ROOT}/output/best_s7_all_clipcap_score_noadj.zip"

