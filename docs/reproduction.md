# Reproduction Notes

The main supported entrypoint is:

```bash
bash scripts/train_justrl_qwen.sh
```

It runs Direct-OPD with:

```text
actor: Qwen3-1.7B
teacher: JustRL-DeepSeek-1.5B
teacher_ref: DeepSeek-R1-Distill-Qwen-1.5B
train: datasets/train/dapo_math_17k.parquet
eval: datasets/eval/aime24.parquet, datasets/eval/aime25.parquet, datasets/eval/hmmt_feb.parquet
```

The training script is configured through environment variables. The most commonly changed ones are:

```text
MODEL_ROOT
DATA_ROOT
OUTPUT_ROOT
LOG_ROOT
LOGGER
TOTAL_TRAINING_STEPS
SAVE_FREQ
TEST_FREQ
```

The default layout is intentionally local and portable:

```text
models/
datasets/
checkpoints/
logs/
```

For large runs, point `OUTPUT_ROOT` and `LOG_ROOT` to external storage.
