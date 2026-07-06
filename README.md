# Direct-OPD

Direct-OPD is a compact reproduction repository for token-level direct on-policy distillation with `verl`.

## Quick Start

Prepare the environment:

```bash
conda create -n direct-opd python=3.12
conda activate direct-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
cd ..
```

Place or link the required model weights and training data:

```text
models/
  Qwen3-1.7B/
  JustRL-DeepSeek-1.5B/
  DeepSeek-R1-Distill-Qwen-1.5B/

datasets/
  train/dapo_math_17k.parquet
```

Launch the main experiment:

```bash
bash scripts/train_justrl_qwen.sh
```

Enable W&B if needed:

```bash
LOGGER="['console','wandb']" bash scripts/train_justrl_qwen.sh
```

Outputs are written to `checkpoints/` and `logs/` by default. Use environment variables to point at a different storage location:

```bash
MODEL_ROOT=/path/to/models \
DATA_ROOT=/path/to/datasets \
OUTPUT_ROOT=/path/to/checkpoints \
LOG_ROOT=/path/to/logs \
bash scripts/train_justrl_qwen.sh
```

## Repository Layout

```text
datasets/eval/        validation parquets used during training
docs/                 reproduction notes
scripts/              launch scripts
verl/                 patched verl source tree
```

## Notes

- The repository does not include model weights or the training parquet.
- `datasets/eval/` contains the small validation files used by the training script.
- Detailed launch options are documented in `docs/`.
