<h1 align="center">Weak-to-Strong Generalization via Direct On-Policy Distillation</h1>

<p align="center">
  Shiyuan Feng* · Huan-ang Gao*‡ · Haohan Chi* · Hanlin Wu · Zhilong Zhang · Zheng Jiang · Bingxiang He · Wei-Ying Ma · Ya-Qin Zhang · Hao Zhou†
</p>

<p align="center">
  <sup>1</sup>SIA-Lab of Tsinghua AIR and ByteDance Seed<br>
  <sup>2</sup>Institute for AI Industry Research (AIR), Tsinghua University<br>
  <sup>3</sup>Department of Computer Science and Technology, Tsinghua University<br>
  <sup>4</sup>Peking University
</p>

<p align="center">
  * Equal contribution · ‡ Project Lead · † Corresponding author
</p>

<p align="center">
  <a href="https://bytedtsinghua-sia.github.io/Direct-OPD/index.html">
    <img src="https://img.shields.io/badge/Project-Page-0A8AA0?style=flat-square&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="#">
    <img src="https://img.shields.io/badge/arXiv-coming%20soon-B31B1B?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/BytedTsinghua-SIA">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>
</p>

Direct-OPD transfers the policy shift learned by a small RL teacher to a stronger student, instead of asking the student to imitate the teacher's final distribution. The key signal is the teacher's change before and after RL:

$$
\Delta_T(y \mid x)=\log \pi_T(y \mid x)-\log \pi_{T_{\mathrm{ref}}}(y \mid x)
$$

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
