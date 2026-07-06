# Setup

This repository contains the Direct-OPD training code and small validation sets. It does not include model weights or the main training parquet.

## Environment

```bash
conda create -n direct-opd python=3.12
conda activate direct-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
cd ..
```

The launch script uses `PYTHON_BIN=/usr/bin/python3.12` by default. Override it if your Python binary is elsewhere:

```bash
PYTHON_BIN="$(which python)" bash scripts/train_justrl_qwen.sh
```

## Required Files

By default, `scripts/train_justrl_qwen.sh` expects:

```text
models/
  Qwen3-1.7B/
  JustRL-DeepSeek-1.5B/
  DeepSeek-R1-Distill-Qwen-1.5B/

datasets/
  train/dapo_math_17k.parquet
  eval/aime24.parquet
  eval/aime25.parquet
  eval/hmmt_feb.parquet
```

The `datasets/eval/` files are included in this repository. The model weights and training parquet should be placed or symlinked locally.

## Launch

```bash
bash scripts/train_justrl_qwen.sh
```

Use W&B logging if needed:

```bash
LOGGER="['console','wandb']" bash scripts/train_justrl_qwen.sh
```

Outputs are written to `checkpoints/` and `logs/` by default. To redirect storage:

```bash
MODEL_ROOT=/path/to/models \
DATA_ROOT=/path/to/datasets \
OUTPUT_ROOT=/path/to/checkpoints \
LOG_ROOT=/path/to/logs \
bash scripts/train_justrl_qwen.sh
```

## Useful Overrides

The launch script is controlled by environment variables. Common ones:

```bash
ACTOR_MODEL_PATH=/path/to/student \
REWARD_MODEL_PATH=/path/to/post_rl_teacher \
TEACHER_REF_MODEL_PATH=/path/to/pre_rl_teacher \
TRAIN_DATASET=/path/to/train.parquet \
TOTAL_TRAINING_STEPS=300 \
GPUS_PER_NODE=8 \
NUM_NODES=1 \
bash scripts/train_justrl_qwen.sh
```

Validation files default to:

```text
datasets/eval/aime24.parquet
datasets/eval/aime25.parquet
datasets/eval/hmmt_feb.parquet
```

## Notes

- `MANAGE_RAY=True` starts and stops a local Ray head inside the script.
- Checkpoints are saved under `${OUTPUT_ROOT}/${EXPERIMENT_NAME}`.
- Validation generations are written under `${CHECKPOINT_DIR}/outputs/validation_log/`.
