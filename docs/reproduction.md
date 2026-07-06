# Reproduction Notes

This document records the minimal information needed to reproduce the main Direct-OPD runs without depending on temporary notes from the original working directory.

## Experiments

The main entrypoints are:

```text
scripts/direct_opd_justrl_qwen3_1p7b_adaptive_kl_len2k.sh
scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

Both scripts train a Qwen3-1.7B actor with token-level Direct-OPD reward:

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

## Default Local Paths

The scripts are path-configurable, but default to the server layout used during reproduction.

JustRL run:

```text
actor:
  /GenSIvePFS/users/hhchi/models_local/Qwen/Qwen3-1.7B
teacher:
  /GenSIvePFS/users/hhchi/models_local/hbx/JustRL-DeepSeek-1.5B
teacher_ref:
  /GenSIvePFS/users/hhchi/models_local/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
```

QuestA run:

```text
actor:
  /GenSIvePFS/users/hhchi/models_local/Qwen/Qwen3-1.7B
teacher:
  /GenSIvePFS/users/hhchi/models_local/foreverlasting1202/QuestA-Nemotron-1.5B
teacher_ref:
  /GenSIvePFS/users/hhchi/models_local/nvidia/OpenMath-Nemotron-1.5B.ms
```

Training data:

```text
/GenSIvePFS/users/hhchi/benchmarks/data/BytedTsinghua-SIA/DAPO-Math-17k/data/dapo-math-17k.parquet
```

Validation data:

```text
datasets/eval/aime24_once.parquet
datasets/eval/aime25_once.parquet
datasets/eval/hmtt_feb.parquet
```

Outputs:

```text
/sia-thu/chihaohan/direct-opd-repro/checkpoints
/sia-thu/chihaohan/direct-opd-repro/logs
/sia-thu/chihaohan/direct-opd-repro/reports
```

## Preflight

Run this before a QuestA experiment:

```bash
PREFLIGHT_ONLY=True bash scripts/direct_opd_questa_qwen3_1p7b_adaptive_kl_len2k.sh
```

The check verifies:

- model and dataset paths exist
- the OpenMath reference path points to the ModelScope layout containing `model.safetensors`
- teacher and teacher reference are not the same path
- validation parquets contain prompts and ground-truth answers

The Qwen3 actor tokenizer has vocab size `151669`; the QuestA/OpenMath tokenizer has vocab size `151665`. This is expected for the current model combination, so the QuestA script records the mismatch but does not block on it by default.

## Output Hygiene

Keep large or generated files outside the repository:

```text
models/
checkpoint/
checkpoints/
outputs/
logs/
wandb/
validation_log/
```

The repository only keeps the small validation parquets required to run the training-time eval protocol.
