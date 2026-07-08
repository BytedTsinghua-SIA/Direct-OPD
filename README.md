<h1 align="center">Weak-to-Strong Generalization via Direct On-Policy Distillation</h1>

<p align="center">
  <b>Transfer RL-discovered policy shifts from weak teachers to stronger students.</b>
</p>

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
    <img src="https://img.shields.io/badge/Project-Page-0A8AA0?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2607.05394">
    <img src="https://img.shields.io/badge/arXiv-2607.05394-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://huggingface.co/collections/BytedTsinghua-SIA/direct-opd">
    <img src="https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">
  </a>
</p>

Direct-OPD transfers the policy shift learned by a small RL teacher to a stronger student, instead of asking the student to imitate the teacher's final distribution. Given a pre-RL teacher reference $\pi_{T_{\mathrm{ref}}}$ and a post-RL teacher $\pi_T$, Direct-OPD reads the teacher's RL-induced change as a dense token-level implicit reward:

$$
\Delta_T(y \mid x)=\log \pi_T(y \mid x)-\log \pi_{T_{\mathrm{ref}}}(y \mid x)
$$

The student remains on-policy: rollouts come from the current student, while the teacher/reference log-ratio scores the candidate tokens the student actually considers. This repository contains the training code used for the JustRL-to-Qwen Direct-OPD experiment, built on a patched `verl` codebase. Model releases and related artifacts are collected on [Hugging Face](https://huggingface.co/collections/BytedTsinghua-SIA/direct-opd).

## Highlights

- Weak RL teachers can improve stronger students by transferring the direction learned during RL.
- Direct-OPD avoids matching the post-RL teacher endpoint, which can import the weak teacher's capacity ceiling.
- Small-model RL followed by Direct-OPD reduces the need to rediscover credit assignment on larger models.
- Policy shifts learned by different RL processes can be applied sequentially to the same student.

## Quick Start

Install dependencies:

```bash
conda create -n direct-opd python=3.12
conda activate direct-opd

cd verl
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify pyarrow transformers
cd ..
```

Place or symlink model weights and the training parquet under the default paths described in [docs/setup.md](./docs/setup.md), then launch:

```bash
bash scripts/train_justrl_qwen.sh
```

For custom model paths, logging, and launch options, see [docs/setup.md](./docs/setup.md).

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
  year   = {2026},
  eprint = {2607.05394},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url    = {https://arxiv.org/abs/2607.05394}
}
```
