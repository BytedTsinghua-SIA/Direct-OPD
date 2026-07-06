# Direct-OPD

Direct-OPD is a lightweight reproduction repository for token-level direct on-policy distillation experiments on math reasoning models. It packages a patched `verl` training stack, the launch scripts used for the Qwen3-1.7B main runs, and the small validation sets needed by the training-time evaluator.

The repository is intentionally small. Model weights, checkpoints, logs, W&B runs, and generated validation dumps are kept outside git.

## Highlights

- Direct-OPD training on top of a patched `verl` PPO trainer.
- Ready-to-run Qwen3-1.7B launch scripts for the JustRL and QuestA teacher settings.
- Built-in preflight checks for model paths, validation data, and the QuestA/OpenMath reference pairing.
- Small AIME/HMMT validation parquets included for reproducible training-time eval.
- Focused tests for Delta-OPD reward computation and validation metric aggregation.

## Quick Start

Clone or enter the repository:

```bash
cd Direct-OPD
```

Check that the local QuestA resources are usable without starting training:

```bash
PREFLIGHT_ONLY=True bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

Run the QuestA Direct-OPD experiment:

```bash
LOGGER="['console','wandb']" \
bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

Run the JustRL Direct-OPD experiment:

```bash
LOGGER="['console','wandb']" \
bash scripts/direct_opd_justrl_qwen3_1p7b_adaptive_kl_len2k.sh
```

All path defaults can be overridden with environment variables:

```bash
MODEL_ROOT=/path/to/models \
DATA_ROOT=/path/to/data \
OUTPUT_ROOT=/path/to/checkpoints \
LOG_ROOT=/path/to/logs \
bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

## Installation

The reproduction environment used Python 3.12.

```bash
conda create -n direct-opd python=3.12
conda activate direct-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
```

For online W&B logging:

```bash
wandb login
```

## Experiments

| Script | Actor | Teacher | Reference teacher |
| --- | --- | --- | --- |
| `scripts/direct_opd_justrl_qwen3_1p7b_adaptive_kl_len2k.sh` | Qwen3-1.7B | JustRL-DeepSeek-1.5B | DeepSeek-R1-Distill-Qwen-1.5B |
| `scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh` | Qwen3-1.7B | QuestA-Nemotron-1.5B | OpenMath-Nemotron-1.5B |

Shared training settings:

```text
algorithm.adv_estimator=token_reward_direct
rollout.reward_mode=delta_opd
n_responses=4
log_prob_top_k=16
top_k_strategy=only_stu
reward_weight_mode=student_p
max_prompt_length=1024
max_response_length=2048
validation n=32
validation temperature=0.7
validation top_p=0.95
```

Full path defaults are documented in [docs/reproduction.md](docs/reproduction.md).

## Data and Models

The included validation files are:

```text
datasets/eval/aime24_once.parquet
datasets/eval/aime25_once.parquet
datasets/eval/hmtt_feb.parquet
```

Default local roots on the reproduction server:

```text
models:
  /GenSIvePFS/users/hhchi/models_local

training data:
  /GenSIvePFS/users/hhchi/benchmarks/data

outputs:
  /sia-thu/chihaohan/direct-opd-repro
```

For QuestA runs, the OpenMath reference should point to the ModelScope layout:

```text
/GenSIvePFS/users/hhchi/models_local/nvidia/OpenMath-Nemotron-1.5B.ms/model.safetensors
```

The QuestA script checks this before launching Ray.

## Repository Layout

```text
.
|-- datasets/eval/     small validation parquets
|-- docs/              reproduction notes
|-- scripts/           main launch scripts and preflight checks
`-- verl/              patched verl source tree
```

## Tests

```bash
cd verl
PYTHONPATH=. pytest \
  tests/test_delta_opd_reward.py \
  tests/test_aime_average_metrics.py \
  tests/test_validation_balancing.py \
  -q
```

Expected result in the prepared environment:

```text
14 passed
```

## Output Hygiene

Keep generated artifacts out of the repository:

```text
models/
checkpoint/
checkpoints/
outputs/
logs/
wandb/
validation_log/
```

The `.gitignore` is set up for this layout. Commit only source code, launch scripts, small validation data, and concise documentation.

## Troubleshooting

- Run `PREFLIGHT_ONLY=True ...questa...sh` before a QuestA training run.
- If W&B is required, run `wandb login` first and use `LOGGER="['console','wandb']"`.
- If a previous Ray job is still alive, stop it with `ray stop --force` before restarting.
- If `delta_opd/weighted_reward_mean` stays exactly zero, check that the teacher and reference teacher paths are different and that the OpenMath reference is the `.ms` directory above.
