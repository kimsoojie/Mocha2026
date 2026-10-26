#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-${CARE_PD_DATASET:-}}"
PY="${PYTHON:-python3}"
DEVICE="${DEVICE:-auto}"
EPOCHS="${EPOCHS:-15}"
BATCH_SIZE="${BATCH_SIZE:-96}"

if [[ -z "${DATASET_DIR}" ]]; then
  echo "Usage: $0 /path/to/CARE-PD-Skeleton-6" >&2
  echo "   or: CARE_PD_DATASET=/path/to/dataset $0" >&2
  exit 2
fi
DATASET_DIR="$(cd "${DATASET_DIR}" && pwd)"
if [[ -d "${DATASET_DIR}/llm_captions" ]]; then
  CAPTION_DIR="${DATASET_DIR}/llm_captions"
elif [[ -d "${DATASET_DIR}/CARE-PD-Skeleton-6/llm_captions" ]]; then
  CAPTION_DIR="${DATASET_DIR}/CARE-PD-Skeleton-6/llm_captions"
else
  echo "Missing llm_captions in ${DATASET_DIR}" >&2
  exit 2
fi

mkdir -p "${ROOT}/output/sources" "${ROOT}/output/final" "${ROOT}/logs" "${ROOT}/tmp"
export TMPDIR="${ROOT}/tmp"
export MPLCONFIGDIR="${ROOT}/tmp/matplotlib"

if [[ "${DEVICE}" == auto ]]; then
  DEVICE="$(${PY} -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')"
fi
echo "dataset=${DATASET_DIR} captions=${CAPTION_DIR} device=${DEVICE}"

tags=(pdgam bmclab tsdupd 3dgait)
names=(PD-GaM BMCLab T-SDU-PD 3DGait)
for i in "${!tags[@]}"; do
  tag="${tags[$i]}"
  name="${names[$i]}"
  "${PY}" -u "${ROOT}/code/train_best_mocha_s7_trueonly_dualdecoder_clipcaption_source.py" \
    --cache "${ROOT}/data/cache/${tag}_motion_smpl_recon_cache.pt" \
    --template "${ROOT}/assets/template" \
    --caption-dir "${CAPTION_DIR}" \
    --clip-model "${ROOT}/assets/clip-vit-base-patch32" \
    --source-name "${name}" \
    --output-dir "${ROOT}/output/sources/${tag}" \
    --label-subset all --epochs "${EPOCHS}" --batch-size "${BATCH_SIZE}" \
    --lr-backbone 2e-6 --lr-head 7e-5 --lr-aux 1e-4 --weight-decay 1e-4 \
    --caption-weight 1.0 --recon-a-weight 0.05 --recon-b-weight 0.02 \
    --temperature 0.07 --seed 42 | tee "${ROOT}/logs/train_${tag}.log"
done

"${PY}" -u "${ROOT}/code/score_merge_recon_supcon_checkpoints.py" \
  --base-motion-encoder "${ROOT}/assets/template/weights/final_ensemble/model_a.pth" \
  --base-classifier "${ROOT}/assets/template/weights/final_ensemble/model_b.pth" \
  --source-motion-encoders \
    "${ROOT}/output/sources/pdgam/model_a.pth" "${ROOT}/output/sources/bmclab/model_a.pth" \
    "${ROOT}/output/sources/tsdupd/model_a.pth" "${ROOT}/output/sources/3dgait/model_a.pth" \
  --source-classifiers \
    "${ROOT}/output/sources/pdgam/model_b.pth" "${ROOT}/output/sources/bmclab/model_b.pth" \
    "${ROOT}/output/sources/tsdupd/model_b.pth" "${ROOT}/output/sources/3dgait/model_b.pth" \
  --source-names pdgam bmclab tsdupd 3dgait \
  --output-motion-encoder "${ROOT}/output/final/model_a.pth" \
  --output-classifier "${ROOT}/output/final/model_b.pth" \
  --metrics "${ROOT}/output/final/score_metrics.json" \
  --tau 1.96 --scale 1.0 --device "${DEVICE}" | tee "${ROOT}/logs/merge.log"

echo "Final model: ${ROOT}/output/final/model_a.pth + model_b.pth"

