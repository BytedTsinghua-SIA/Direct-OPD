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
  train/skywork-or1-math-dapo-original.parquet
  eval/aime24.parquet
  eval/aime25.parquet
  eval/hmmt_feb.parquet
```

The evaluation parquets are included. Download the model weights separately and prepare the training parquet as described below.

### Evaluation Data

The included validation parquets were prepared from these public benchmarks:

- `datasets/eval/aime24.parquet`: [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)
- `datasets/eval/aime25.parquet`: [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)
- `datasets/eval/hmmt_feb.parquet`: [MathArena/hmmt_feb_2025](https://huggingface.co/datasets/MathArena/hmmt_feb_2025)

Each file contains 30 problems normalized to the VERL evaluation schema used by the launch script.

### Training Data

Download the 105,055-row
[Skywork-OR1 math parquet](https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/blob/main/data/math-00000-of-00001.parquet)
and save it as `datasets/raw/skywork-or1-math.parquet`. Then run the converter from the repository root:

```bash
mkdir -p datasets/raw datasets/train

curl -L --fail \
  https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/resolve/main/data/math-00000-of-00001.parquet \
  -o datasets/raw/skywork-or1-math.parquet

python scripts/prepare_skywork_math.py \
  --input datasets/raw/skywork-or1-math.parquet \
  --output datasets/train/skywork-or1-math-dapo-original.parquet
```

The converter applies the DAPO-style prompt described in the
[Direct-OPD appendix](https://arxiv.org/html/2607.05394#A1) and validates the generated parquet.
`dapo-original` refers to this prompt format; the underlying problems still come from Skywork-OR1.

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

## Privileged-Anchor Experiments

These runs compare Direct-OPD, OPSD and Direct-OPD with a privileged KL anchor, where the frozen reference sees the
problem plus its final answer or worked solution. Only the reference prompt changes; the student and both teachers
still see the plain problem.

### Data

Each dataset is filtered (answer present and consistent with the solution's `\boxed{}`, no multiple-choice or proof
problems, one `\boxed{}`, decontaminated against `datasets/eval`, student prompt <= 1024 and privileged prompt <= 2048
tokens) and sampled to 19,200 rows (300 steps x 64 prompts, one pass) with a fixed seed and order.

```bash
mkdir -p datasets/raw/openthoughts datasets/raw/numinamath
for i in 0 1; do
  curl -L --fail -o datasets/raw/openthoughts/$i.parquet \
    https://huggingface.co/api/datasets/siyanzhao/Openthoughts_math_30k_opsd/parquet/default/train/$i.parquet
done
for i in 0 1 2 3 4; do
  curl -L --fail -o datasets/raw/numinamath/$i.parquet \
    https://huggingface.co/api/datasets/AI-MO/NuminaMath-CoT/parquet/default/train/$i.parquet
done

python scripts/prepare_privileged_math.py --source openthoughts_opsd \
  --input 'datasets/raw/openthoughts/*.parquet' \
  --output datasets/train/privileged-openthoughts-19200.parquet

# NuminaMath: excludes every OpenThoughts problem, samples 65% olympiads / 20% aops_forum / 10% synthetic_amc / 5% math
python scripts/prepare_privileged_math.py --source numinamath_cot \
  --input 'datasets/raw/numinamath/*.parquet' \
  --exclude 'datasets/raw/openthoughts/*.parquet' \
  --output datasets/train/privileged-numinamath-19200.parquet
```

`--tokenizer` defaults to `models/Qwen3-1.7B`.

### Runs

Every run is independent: `bash scripts/run_privileged_experiment.sh <dataset> <method>`.

| method | privileged anchor | Delta reward | KL coefficient |
|---|---|---|---|
| `direct_opd` | none | yes | adaptive from 2.5 in [0.5, 2.5] |
| `opsd_answer` / `opsd_solution` | answer / solution | no (`DELTA_REWARD_SCALE=0`) | fixed 1.0 |
| `ours_answer` / `ours_solution` | answer / solution | yes | adaptive from 2.5 in [0.5, 2.5] |

`dataset` is `openthoughts` or `numinamath`. Presets only fill unset variables, so e.g. `KL_LOSS_COEF=0.8` still
overrides. `DRY_RUN=1` prints the resolved command without starting Ray.

Slurm (one job per node, since each run starts and force-stops its own Ray head); submit from the repository root:

```bash
mkdir -p logs
for dataset in openthoughts numinamath; do
  for method in direct_opd opsd_answer opsd_solution ours_answer ours_solution; do
    sbatch --job-name="${dataset}_${method}" --export=ALL,DATASET=${dataset},METHOD=${method} \
      scripts/slurm/submit_privileged.sbatch
  done
done
```

Fill in the partition, account and time limit in `scripts/slurm/submit_privileged.sbatch` for your cluster.

## Notes

- `MANAGE_RAY=True` starts and stops a local Ray head inside the script.
- Checkpoints are saved under `${OUTPUT_ROOT}/${EXPERIMENT_NAME}`.
- Validation generations are written under `${CHECKPOINT_DIR}/outputs/validation_log/`.
