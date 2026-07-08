# Direct-OPD `ENFORCE_EAGER=False` Debug 交接 Prompt

下面这段可以直接给另一个服务器上的 agent 使用。目标是先 debug vLLM CUDA graph 模式，也就是 `ENFORCE_EAGER=False`，debug 阶段不要做训练前 validation；debug 成功后再启动正式 run，正式 run 需要做训练前 validation。

---

你现在接手 Direct-OPD 主实验复现的一个系统效率问题。请在不破坏现有环境的前提下，debug `ENFORCE_EAGER=False`，并在 debug 成功后启动正式复现实验。

## 背景

实验目标是复现论文主实验 3.1 中的 JustRL-distill -> Qwen3-1.7B Direct-OPD 设置。

核心实验配置应保持为：

```text
actor/base model: Qwen/Qwen3-1.7B
reward teacher: JustRL-DeepSeek-1.5B
teacher reference: DeepSeek-R1-Distill-Qwen-1.5B
rollout group size: 4
reward mode: delta_opd
adv estimator: token_reward_direct
log_prob_top_k: 16
top_k_strategy: only_stu
reward_weight_mode: student_p
max prompt length: 1024
max response length: 2048
total training steps: 300
save freq: 20
test freq: 20
validation n: 32
validation temperature: 0.7
validation top_p: 0.95
```

本机已有一个能跑通但效率较低的 run。它使用了 `ENFORCE_EAGER=True`，原因是 final repo smoke 在 `ENFORCE_EAGER=False` 下触发了 vLLM CUDA graph / torch compile 报错。

已知当前服务器信息：

```text
repo: Direct-OPD-final
repo commit: e8792b16ccbfc55dad534f9f9d76d889f4f2ecd5
python env: opd_verl_py312_20260507
python: 3.12.13
torch: 2.8.0+cu128
cuda used by torch: 12.8
vllm: 0.11.0
```

成功但较慢的当前 run：

```text
experiment: direct_opd_final_main_justrl_qwen3_1p7b_300s_20260706_141808
tmux: opd_final_full_direct_opd_final_main_justrl_qwen3_1p7b_300s_20260706_141808
wandb: https://wandb.ai/ahyd3775-tsinghua-university/Direct-OPD/runs/krs88xsw
log: /sia-thu/chihaohan/direct-opd-repro/logs/direct_opd_final_main_justrl_qwen3_1p7b_300s_20260706_141808.log
checkpoint dir: /sia-thu/chihaohan/direct-opd-repro/checkpoints/direct_opd_final_main_justrl_qwen3_1p7b_300s_20260706_141808
```

这个 run 没有训练前 validation：

```text
VAL_BEFORE_TRAIN=False
ENFORCE_EAGER=True
WANDB_MODE=online
LOGGER="['console','wandb']"
```

它的 delta 信号正常，不是之前 QuestA 错误模型导致的 delta=0 问题。例如早期 step：

```text
delta_opd/weighted_reward_mean ~= -0.011 到 -0.013
teacher/entropy_minus_ref ~= 0.13 到 0.15
```

效率对比：

```text
旧成功 run, ENFORCE_EAGER=False:
  mean timing_s/step ~= 83.9s
  300 step 纯训练时间约 7.0h

当前 final run, ENFORCE_EAGER=True:
  mean timing_s/step ~= 124.1s
  300 step 纯训练时间约 10.3h
```

慢的主要部分：

```text
                  eager=True 当前     eager=False 旧 run
timing_s/gen      ~41.4s             ~32.9s
timing_s/reward   ~23.9s             ~24.1s
timing_s/ref      ~5.4s              ~5.2s
timing_s/update_actor ~49.1s         ~17.6s
timing_s/step     ~124.1s            ~83.9s
```

旧成功 run 来自旧工作树：

```text
repo: Direct-OPD
commit: 55a2dbfd7e564abec516a3a5260e8673b18b4a7f
log: /sia-thu/chihaohan/direct-opd-repro/logs/direct_opd_main_justrl_qwen3_1p7b_300s_wandb_20260629_140316.log
```

当前 final repo 的非 eager smoke 报错日志：

```text
log: /sia-thu/chihaohan/direct-opd-repro/logs/final_repo_smoke_justrl_qwen_20260706_110526.log
setting: ENFORCE_EAGER=False
```

关键错误：

```text
RuntimeError: CUDA driver error: invalid argument
```

关键栈：

```text
vllm/compilation/cuda_graph.py
vllm/compilation/cuda_piecewise_backend.py
torch/_inductor/runtime/static_cuda_launcher.py
```

同一时期的 eager smoke 可以跑：

```text
log: /sia-thu/chihaohan/direct-opd-repro/logs/final_repo_smoke_justrl_qwen_eager_20260706_110854.log
setting: ENFORCE_EAGER=True
```

当前判断：问题大概率不是 Direct-OPD reward 逻辑，而是 final repo 在 vLLM 0.11 + torch 2.8 下走 CUDA graph / compile 路径时触发了兼容性问题。也可能和 final repo 相比旧 Direct-OPD 的代码/config 差异有关。

## 你的任务

### 1. 先做 debug，不要训练前 validation

debug 阶段目标是让 `ENFORCE_EAGER=False` 能稳定完成至少一个短 smoke run。

要求：

```text
ENFORCE_EAGER=False
VAL_BEFORE_TRAIN=False
WANDB_MODE=disabled 或只用 console logger
TOTAL_TRAINING_STEPS=1 或 2
SAVE_FREQ 设大一些，避免无意义 checkpoint
TEST_FREQ 设大一些，避免触发 validation
```

debug 阶段不要跑训练前 val，也不要跑 step validation。先确认非 eager 能不能进训练 step、完成 rollout、reward、actor update。

如果失败，请定位具体原因，优先检查：

```text
1. Direct-OPD-final 与旧 Direct-OPD 在 verl/workers/rollout/vllm_rollout 相关代码的差异
2. rollout 配置中 vLLM compile / cudagraph / enforce_eager / max_model_len / gpu_memory_utilization / tensor_parallel_size 的差异
3. torch compile 是否可以关闭但保留 CUDA graph
4. vLLM 0.11 是否有可配置项绕过 piecewise compile backend
5. 是否是 Ray 残留进程或旧 worker 污染，debug 前应 ray stop --force 并确认 GPU 空闲
```

不要通过改训练目标、reward、teacher/ref 模型、数据、rollout group size 来“修复”这个问题。debug 只应该动系统执行层或 final repo 与旧 repo 的非算法差异。

### 2. debug 成功后，启动正式 run

正式 run 需要：

```text
ENFORCE_EAGER=False
VAL_BEFORE_TRAIN=True
WANDB_MODE=online
LOGGER="['console','wandb']"
TOTAL_TRAINING_STEPS=300
SAVE_FREQ=20
TEST_FREQ=20
```

正式 run 应该在 tmux 中启动，输出日志、checkpoint、wandb dir 放到大盘或服务器指定输出目录，不要写进 repo。

正式 run 的第一个检查点：

```text
1. 训练前 validation 已经开始并完成
2. validation 参数为 n=32, temperature=0.7, top_p=0.95
3. step 1 后 delta_opd/weighted_reward_mean 不是长期 0
4. teacher/entropy_minus_ref 不是长期 0
5. timing_s/step 接近旧成功 run 的量级，理想情况下约 80-90s/step，而不是 120s/step 以上
```

正式 run 启动后请持续盯到：

```text
1. W&B online run 链接可访问
2. 训练前 val 完成
3. 至少完成 step 1 和 step 2
4. delta 信号正常
5. GPU 利用率稳定
```

### 3. 需要输出的结论

debug 结束后请给出中文报告，至少包括：

```text
1. `ENFORCE_EAGER=False` 报错是否复现
2. 如果修复了，实际改了什么，为什么这不改变算法设置
3. debug smoke 的日志路径
4. 正式 run 的 tmux session、日志路径、checkpoint 路径、W&B 链接
5. 正式 run 是否做了训练前 validation
6. 前几个 step 的 timing_s/step、delta_opd/weighted_reward_mean、teacher/entropy_minus_ref
7. 与旧成功 run 的训练效率对比
```

## 注意事项

不要破坏现有环境。需要安装包时优先新建环境或 clone 环境，不要原地升级已有可用环境。

不要把大文件、checkpoint、wandb、logs 写进 repo。repo 里只放代码、脚本、小数据和简洁文档。

如果遇到并发读取、模型缓存、下载不完整等问题，把临时模型和缓存放在服务器的模型目录或大盘目录，不要放系统 `/tmp`。

如果另一个服务器没有完全相同路径，请用环境变量传入模型、数据、输出目录，不要把路径硬编码进 repo。

