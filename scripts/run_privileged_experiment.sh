#!/usr/bin/env bash
# One self-contained run of the privileged-anchor experiment matrix.
#
#   bash scripts/run_privileged_experiment.sh <dataset> <method>
#   DATASET=<dataset> METHOD=<method> bash scripts/run_privileged_experiment.sh
#
# dataset: openthoughts | numinamath     (reads ${DATA_ROOT}/train/privileged-<dataset>-19200.parquet)
# method:  direct_opd | opsd_answer | opsd_solution | ours_answer | ours_solution
#
# Presets only fill variables the caller left unset, so any train_justrl_qwen.sh variable can still be overridden.
# DRY_RUN=1 prints the resolved command without starting Ray.
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

DATASET=${1:-${DATASET:-}}
METHOD=${2:-${METHOD:-}}
if [ -z "${DATASET}" ] || [ -z "${METHOD}" ]; then
  echo "usage: $0 <openthoughts|numinamath> <direct_opd|opsd_answer|opsd_solution|ours_answer|ours_solution>" >&2
  exit 2
fi

case "${DATASET}" in
  openthoughts|numinamath) ;;
  *) echo "unknown dataset: ${DATASET}" >&2; exit 2 ;;
esac

case "${METHOD}" in
  direct_opd)
    preset_mode=none;     preset_scale=1.0; preset_adaptive=True;  preset_coef=2.5 ;;
  opsd_answer)
    preset_mode=answer;   preset_scale=0.0; preset_adaptive=False; preset_coef=1.0 ;;
  opsd_solution)
    preset_mode=solution; preset_scale=0.0; preset_adaptive=False; preset_coef=1.0 ;;
  ours_answer)
    preset_mode=answer;   preset_scale=1.0; preset_adaptive=True;  preset_coef=2.5 ;;
  ours_solution)
    preset_mode=solution; preset_scale=1.0; preset_adaptive=True;  preset_coef=2.5 ;;
  *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

export PRIVILEGED_MODE=${PRIVILEGED_MODE:-${preset_mode}}
export DELTA_REWARD_SCALE=${DELTA_REWARD_SCALE:-${preset_scale}}
export ADAPTIVE_KL=${ADAPTIVE_KL:-${preset_adaptive}}
export KL_LOSS_COEF=${KL_LOSS_COEF:-${preset_coef}}

DATA_ROOT=${DATA_ROOT:-datasets}
export TRAIN_DATASET=${TRAIN_DATASET:-"${DATA_ROOT}/train/privileged-${DATASET}-19200.parquet"}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"qwen3_1p7b_${DATASET}_${METHOD}"}

train_path=${TRAIN_DATASET}
case "${train_path}" in
  /*) ;;
  *) train_path="${REPO_ROOT}/${train_path}" ;;
esac
if [ ! -f "${train_path}" ]; then
  echo "training parquet not found: ${train_path} (run scripts/prepare_privileged_math.py first)" >&2
  exit 1
fi

exec bash "${REPO_ROOT}/scripts/train_justrl_qwen.sh"
