<h1 align="center">Weak-to-Strong Generalization via Direct On-Policy Distillation</h1>

<p align="center">
  Shiyuan Feng* · Huan-ang Gao*‡ · Haohan Chi* · Hanlin Wu · Zhilong Zhang · Zheng Jiang · Bingxiang He · Wei-Ying Ma · Ya-Qin Zhang · Hao Zhou†
</p>

<p align="center">
  <a href="https://bytedtsinghua-sia.github.io/Direct-OPD/">Project Page</a> ·
  <a href="https://bytedtsinghua-sia.github.io/Direct-OPD/assets/w2s-opd.pdf">Paper</a> ·
  <a href="https://github.com/BytedTsinghua-SIA/Direct-OPD">Code</a> ·
  <a href="./docs/setup.md">Setup</a> ·
  <a href="./scripts/train_justrl_qwen.sh">Training Script</a>
</p>

<p align="center">
  * Equal contribution · ‡ Project Lead · † Corresponding author
</p>

Direct-OPD transfers the policy shift learned by a small RL teacher to a stronger student, instead of asking the student to imitate the teacher's final distribution. The key signal is the teacher's change before and after RL:

```text
Delta_T(y | x) = log pi_T(y | x) - log pi_Tref(y | x)
```

This log-ratio acts as a dense token-level implicit reward on states visited by the student. The repository contains the training code used for the JustRL-to-Qwen Direct-OPD experiment, built on a patched `verl` codebase.

## Highlights

- Transfers RL-discovered directions from a small teacher pair to stronger student models.
- Keeps the student on-policy: rollouts come from the current student, while teacher/reference log-ratios score candidate tokens.
- Avoids directly matching the post-RL teacher distribution, which can import the weak teacher's capacity ceiling.
- Includes validation data for AIME 2024, AIME 2025, and HMMT February.

## Quick Start

Install the environment and dependencies:

```bash
conda create -n direct-opd python=3.12
conda activate direct-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
cd ..
```

Prepare model weights and the training parquet, then launch:

```bash
bash scripts/train_justrl_qwen.sh
```

The default script expects:

```text
models/Qwen3-1.7B/
models/JustRL-DeepSeek-1.5B/
models/DeepSeek-R1-Distill-Qwen-1.5B/
datasets/train/dapo_math_17k.parquet
```

For custom paths, logging, and launch options, see [docs/setup.md](./docs/setup.md).

## Repository Layout

```text
datasets/eval/        validation parquets used by the training script
scripts/              experiment launch scripts
verl/                 patched verl source tree
docs/setup.md         setup, path, and launch notes
```

## Reference

```bibtex
@misc{feng2026directopd,
  title  = {Weak-to-Strong Generalization via Direct On-Policy Distillation},
  author = {Shiyuan Feng and Huan-ang Gao and Haohan Chi and Hanlin Wu and Zhilong Zhang and Zheng Jiang and Bingxiang He and Wei-Ying Ma and Ya-Qin Zhang and Hao Zhou},
  year   = {2026}
}
```
