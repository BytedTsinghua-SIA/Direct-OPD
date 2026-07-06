# Direct-OPD

This repository contains a compact Direct-OPD reproduction codebase built on a patched `verl` tree. It includes launch scripts for the main Qwen3-1.7B experiments, small validation parquets, and preflight checks for local model/data paths.

Large files are intentionally not stored here. Model weights, checkpoints, logs, W&B files, and long validation dumps should live outside the repository.

## Repository Layout

```text
verl/              patched verl implementation
scripts/           training and preflight entrypoints
datasets/eval/     small validation parquets used by the training loop
docs/              short reproduction notes
```

## Environment

The code was used with Python 3.12 and the local `verl` install flow:

```bash
conda create -n direct-opd python=3.12
conda activate direct-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
```

If you use W&B logging, log in before launching training:

```bash
wandb login
```

## Required Local Assets

The scripts default to the paths used on the reproduction server:

```text
models:
  /GenSIvePFS/users/hhchi/models_local

training data:
  /GenSIvePFS/users/hhchi/benchmarks/data

outputs:
  /sia-thu/chihaohan/direct-opd-repro
```

You can override these with environment variables:

```bash
MODEL_ROOT=/path/to/models \
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/checkpoints \
LOG_ROOT=/path/to/logs \
bash scripts/direct_opd_justrl_qwen3_1p7b_adaptive_kl_len2k.sh
```

See [docs/reproduction.md](docs/reproduction.md) for the exact model and dataset paths used for the main runs.

## Main Scripts

JustRL-distill teacher:

```bash
bash scripts/direct_opd_justrl_qwen3_1p7b_adaptive_kl_len2k.sh
```

QuestA teacher:

```bash
bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

QuestA preflight only, without starting Ray or training:

```bash
PREFLIGHT_ONLY=True bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

Online W&B logging:

```bash
LOGGER="['console','wandb']" \
bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

## Tests

```bash
cd verl
PYTHONPATH=. pytest tests/test_delta_opd_reward.py tests/test_aime_average_metrics.py tests/test_validation_balancing.py -q
```

## Notes

- `datasets/eval/*.parquet` are small validation files and are kept in the repo.
- Do not commit `models/`, `checkpoint/`, `outputs/`, `logs/`, `wandb/`, or generated validation dumps.
- For QuestA runs, use the correct OpenMath reference model directory. The script checks for the ModelScope layout before launch.
