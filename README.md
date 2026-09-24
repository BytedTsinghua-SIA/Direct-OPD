<h1 align="center">Privileged-Anchor Direct-OPD</h1>

<p align="center">
  <b>Does a KL anchor that knows the answer transfer an RL policy shift better than a conservative one?</b>
</p>

<p align="center">
  An experiment built on
  <a href="https://github.com/BytedTsinghua-SIA/Direct-OPD">Direct-OPD</a>
  (<a href="https://arxiv.org/abs/2607.05394">arXiv:2607.05394</a>)
  and on-policy self-distillation
  (<a href="https://arxiv.org/abs/2601.18734">OPSD, arXiv:2601.18734</a>).
</p>

This repository extends the Direct-OPD training code with one change: the frozen student that anchors the KL term can
see privileged information $z$, either the final answer or a worked solution. The student, both teachers and the
reward are unchanged. Direct-OPD and OPSD run from the same code, data and settings as baselines.

## Method

Notation:
- $p_\theta$ is the trainable student and $y \sim p_\theta(\cdot \mid x)$ is its on-policy rollout.
- $p_S$ is the frozen initial student.
- $\pi_{T,\mathrm{pre}}$ and $\pi_{T,\mathrm{post}}$ are the teacher before and after RL.

Direct-OPD transfers the teacher's RL shift as a dense token-level reward:

$$
\Delta_T(y \mid x)=\log \pi_{T,\mathrm{post}}(y \mid x)-\log \pi_{T,\mathrm{pre}}(y \mid x)
$$

The four methods differ only in whether they use this reward and in which frozen model the KL term pulls toward:

| Method | Objective (minimise) | RL-shift reward | KL anchor |
|---|---|:---:|---|
| OPD | $D_{\mathrm{KL}}(p_\theta \Vert q_T)$ | – | external teacher $q_T$ |
| OPSD | $D_{\mathrm{KL}}(p_\theta(\cdot \mid x) \Vert p_S(\cdot \mid x,z))$ | – | privileged self |
| Direct-OPD | $-\mathbb{E}[\Delta_T] + \alpha\, D_{\mathrm{KL}}(p_\theta(\cdot \mid x) \Vert p_S(\cdot \mid x))$ | ✓ | ordinary self |
| **Ours** | $-\mathbb{E}[\Delta_T] + \alpha\, D_{\mathrm{KL}}(p_\theta(\cdot \mid x) \Vert p_S(\cdot \mid x,z))$ | ✓ | **privileged self** |

The optimal policy for ours is $p^\ast \propto p_S(y \mid x,z)\,\exp(\Delta_T/\alpha)$, the privileged self tilted
toward the teacher's RL shift. It becomes OPSD as $\alpha \to \infty$ and pure $\Delta_T$ maximisation as
$\alpha \to 0$.

Unlike Direct-OPD, the KL term is non-zero from the first step, so the anchor acts as a second teacher as well as a
regulariser.

### What changes in the code

Only the reference forward pass changes. Its prompt becomes the student's prompt with a privileged block appended:

| z | Appended to the user turn |
|---|---|
| answer | `The final answer to this problem is: {answer}` + "Please solve this problem using your own approach below." |
| solution | `Here is a reference solution: {solution}` + "After understanding the reference solution, please try to solve this problem using your own approach below." |

The student's sampled response tokens are appended unchanged, and log-probs are read only over those tokens. The
student and both teachers still see the plain problem.

OPSD is the same anchor with the $\Delta_T$ reward scaled to 0. The advantages are then exactly zero and only the KL
loss trains.

## Experiment

### Models

| Role | Model |
|---|---|
| Student $p_\theta$, KL anchor $p_S$ | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) |
| Post-RL teacher $\pi_{T,\mathrm{post}}$ | [hbx/JustRL-DeepSeek-1.5B](https://huggingface.co/hbx/JustRL-DeepSeek-1.5B) |
| Pre-RL teacher $\pi_{T,\mathrm{pre}}$ | [deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B) |

### Datasets

Both datasets have worked solutions and final answers. Skywork-OR1, used in the original Direct-OPD, has answers only.

| | Primary | Secondary |
|---|---|---|
| Source | [siyanzhao/Openthoughts_math_30k_opsd](https://huggingface.co/datasets/siyanzhao/Openthoughts_math_30k_opsd), the OPSD paper's split | [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) |
| Answer | `Answer` column | last `\boxed{}` in the solution |
| Raw rows | 29,434 | 250,340 (olympiads, aops_forum, synthetic_amc, math) |
| After filtering | 23,755 | 125,822 (every OpenThoughts problem removed) |
| Used | **19,200** | **19,200**, sampled 65% olympiads / 20% aops_forum / 10% synthetic_amc / 5% math, following [arXiv:2608.18271](https://arxiv.org/abs/2608.18271) |

Filters, applied in [`scripts/prepare_privileged_math.py`](./scripts/prepare_privileged_math.py):
- the answer is present and single-line, and matches the solution's `\boxed{}` after normalisation
- no multiple-choice letter answers and no proof problems
- exactly one `\boxed{}` in the solution
- each problem appears only once
- decontaminated against AIME24/AIME25/HMMT with normalised substring and 13-gram overlap
- student prompt ≤ 1024 tokens and privileged prompt ≤ 2048 tokens (Qwen3 chat template)

19,200 = 300 steps × 64 prompts, so each run makes exactly one pass. Every method sees the same problems in the same
fixed order. The two training sets share no problems.

### Run matrix

There are 5 runs per dataset, 10 in total. Each is an independent job.

| `METHOD` | Privileged anchor | $\Delta_T$ reward | KL coefficient |
|---|---|:---:|---|
| `direct_opd` | none | ✓ | adaptive |
| `opsd_answer` | answer | – | fixed 1.0 |
| `opsd_solution` | worked solution | – | fixed 1.0 |
| `ours_answer` | answer | ✓ | adaptive |
| `ours_solution` | worked solution | ✓ | adaptive |

The adaptive rule, from Direct-OPD, is $\alpha \leftarrow \mathrm{clip}(\alpha(1+0.01\,\mathrm{sgn}(\bar r)), 0.5, 2.5)$,
starting at 2.5. OPSD has no $\Delta_T$ reward for this rule to follow, so it uses a fixed value inside the paper's
fixed range of [0.8, 2].

### Hyperparameters

These follow Direct-OPD Appendix A, Tables 2–3.

| | Value |
|---|---|
| Global batch / mini-batch | 64 / 64 prompts, 4 rollouts each |
| Max prompt / response | 1024 / 2048 tokens (privileged anchor prompt ≤ 2048) |
| Rollout temperature / top-p | 1.0 / 1.0 |
| Learning rate | 1e-6 |
| Steps | 300 |
| $\Delta_T$ reward | student top-16 support, weighted by student probability |
| KL loss | `low_var_kl` (k3) on sampled tokens, identical in every run |
| Evaluation | AIME24 + AIME25 (HMMT reported as an extra), 32 samples, temperature 0.7, top-p 0.95, max 31,744 tokens |

On student-sampled tokens, the `low_var_kl` estimator has a forward-KL gradient. Every run uses it, so the anchor is
the only thing that differs between methods. For a reverse-KL ablation, set `KL_LOSS_TYPE=low_var_kl+` or `k2`.

## Quick Start

These commands assume a single node with 4× H100, which is the default. Run them from the repository root.

### 1. Environment

```bash
git clone https://github.com/oldpilluwu/Direct-OPD.git && cd Direct-OPD
conda create -n direct-opd python=3.12 -y && conda activate direct-opd
cd verl && USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh && cd ..
pip install math-verify pyarrow transformers pandas huggingface_hub pytest
export PYTHON_BIN="$(which python)"   # the launch script otherwise uses /usr/bin/python3.12
```

### 2. Models (~10 GB)

```bash
for m in Qwen/Qwen3-1.7B hbx/JustRL-DeepSeek-1.5B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B; do
  huggingface-cli download "$m" --local-dir "models/${m#*/}"
done
```

### 3. Data (~1.5 GB download, 10–15 min on CPU)

```bash
mkdir -p datasets/raw/openthoughts datasets/raw/numinamath logs
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

python scripts/prepare_privileged_math.py --source numinamath_cot \
  --input 'datasets/raw/numinamath/*.parquet' \
  --exclude 'datasets/raw/openthoughts/*.parquet' \
  --output datasets/train/privileged-numinamath-19200.parquet
```

The prep is deterministic. The printed SHA256 should begin `36922143…` for OpenThoughts and `c375c250…` for
NuminaMath.

### 4. Tests and a dry run

```bash
(cd verl && pytest tests/test_privileged_ref.py tests/test_delta_opd_reward.py)
DRY_RUN=1 bash scripts/run_privileged_experiment.sh openthoughts ours_solution
```

### 5. Smoke test (3 steps, no validation or checkpoints)

```bash
TOTAL_TRAINING_STEPS=3 VAL_BEFORE_TRAIN=False TEST_FREQ=-1 SAVE_FREQ=-1 EXPERIMENT_NAME=smoke_ours \
  bash scripts/run_privileged_experiment.sh openthoughts ours_solution
TOTAL_TRAINING_STEPS=3 VAL_BEFORE_TRAIN=False TEST_FREQ=-1 SAVE_FREQ=-1 EXPERIMENT_NAME=smoke_opsd \
  bash scripts/run_privileged_experiment.sh openthoughts opsd_answer
```

In `logs/smoke_*.log`, check that:
- the runs finish with no out-of-memory errors
- `privileged_ref/prompt_len_mean` is logged
- `actor/kl_loss` is above 0 at step 1
- `actor/pg_loss` is 0 for OPSD

### 6. Full runs

The runs are independent, and each one uses the whole node.

**Single node, one run after another:**

```bash
nohup bash -c 'for d in openthoughts numinamath; do
  for m in direct_opd opsd_answer opsd_solution ours_answer ours_solution; do
    bash scripts/run_privileged_experiment.sh $d $m
  done
done' > logs/all_runs.out 2>&1 &
```

**Slurm, one job per run.** Add `--partition` / `--account` / `--time` if your cluster requires them.

```bash
for d in openthoughts numinamath; do
  for m in direct_opd opsd_answer opsd_solution ours_answer ours_solution; do
    sbatch --job-name=${d}_${m} --export=ALL,PYTHON_BIN=$(which python),DATASET=$d,METHOD=$m \
      scripts/slurm/submit_privileged.sbatch
  done
done
```

Outputs are written to `checkpoints/qwen3_1p7b_<dataset>_<method>/` and `logs/qwen3_1p7b_<dataset>_<method>.log`.

### Hardware notes

- **Other GPU counts:** set both `GPUS_PER_NODE=N` and `sbatch --gpus-per-node=N`. These values are not linked. N must
  divide 256 sequences per step and 32 validation samples, so use 1, 2, 4, 8, 16 or 32. The algorithm does not
  change; only the time per step does.
- **One run per node:** each run starts and force-stops its own local Ray head. The Slurm template therefore uses
  `--exclusive`, and multi-node runs are not supported.
- **Memory:** models run in fp32, as in the paper. If a smoke run runs out of memory, lower
  `GPU_MEMORY_UTILIZATION` (default 0.45) and use the same value for every run.
- **Overrides:** every setting in [`scripts/train_justrl_qwen.sh`](./scripts/train_justrl_qwen.sh) is an environment
  variable. Method presets only fill values you leave unset, so `KL_LOSS_COEF=0.8 bash
  scripts/run_privileged_experiment.sh …` still takes effect.

## Repository Layout

```text
scripts/prepare_privileged_math.py      filter + sample the OpenThoughts / NuminaMath training sets
scripts/run_privileged_experiment.sh    one run: <dataset> <method> -> presets -> train_justrl_qwen.sh
scripts/slurm/submit_privileged.sbatch  Slurm template for one run
scripts/train_justrl_qwen.sh            Direct-OPD launcher (env-var configured; DRY_RUN=1 prints the command)
verl/verl/trainer/ppo/privileged_ref.py builds the privileged anchor prompts, aligned to the student's responses
verl/verl/trainer/ppo/ray_trainer.py    reference-forward hook and Delta-reward scale
verl/tests/test_privileged_ref.py       CPU tests for alignment, templates, zero-reward OPSD and data prep
datasets/eval/                          AIME24, AIME25, HMMT Feb validation parquets
docs/setup.md                           setup, path and launch notes (including the original Skywork setup)
```

The original Direct-OPD Skywork-OR1 experiment still runs with `bash scripts/train_justrl_qwen.sh`. See
[docs/setup.md](./docs/setup.md).

## Acknowledgements and References

This code is a fork of [BytedTsinghua-SIA/Direct-OPD](https://github.com/BytedTsinghua-SIA/Direct-OPD), which is built
on [verl](https://github.com/volcengine/verl). If you use it, please cite Direct-OPD:

```bibtex
@article{feng2026weak,
  title={Weak-to-Strong Generalization via Direct On-Policy Distillation},
  author={Feng, Shiyuan and Gao, Huan-ang and Chi, Haohan and Wu, Hanlin and Zhang, Zhilong and Jiang, Zheng and He, Bingxiang and Ma, Wei-Ying and Zhang, Ya-Qin and Zhou, Hao},
  journal={arXiv preprint arXiv:2607.05394},
  year={2026}
}
```

Related work used in this experiment:

- Self-Distilled Reasoner: On-Policy Self-Distillation for Large Language Models ([arXiv:2601.18734](https://arxiv.org/abs/2601.18734)). OPSD, and the source of the OpenThoughts training split.
- Rethinking Privileged Information in On-Policy Self-Distillation ([arXiv:2608.18271](https://arxiv.org/abs/2608.18271)). The NuminaMath-CoT source mix and the answer-vs-solution comparison.
