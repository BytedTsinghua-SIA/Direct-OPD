# Direct-OPD `ENFORCE_EAGER=False` Debug 报告

日期：2026-07-06

## 结论

`ENFORCE_EAGER=False` 的原始报错已复现并定位到 vLLM 0.11 + torch 2.8 的 compilation / CUDA graph 路径。通过只调整系统执行层配置，非 eager smoke run 已成功完成 rollout、reward、actor update。

当前可工作的非 eager 组合：

```text
ENFORCE_EAGER=False
VLLM_USE_INDUCTOR=False
VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY
GPU_MEMORY_UTILIZATION=0.45
PPO_MAX_TOKEN_LEN_PER_GPU=4096
ROLLOUT_MAX_NUM_BATCHED_TOKENS=33792
```

这些改动不改变算法目标、reward、teacher/ref 模型、训练数据或 rollout group size。它们只影响 vLLM compilation/CUDA graph 策略、KV cache 预留和 actor 动态 microbatch 切分。

## 原始错误复现

失败日志：

```text
/sia-thu/chihaohan/direct-opd-repro/logs/final_repo_smoke_justrl_qwen_20260706_110526.log
```

设置：

```text
ENFORCE_EAGER=False
```

关键错误：

```text
RuntimeError: CUDA driver error: invalid argument
```

关键栈位于：

```text
vllm/compilation/cuda_graph.py
vllm/compilation/cuda_piecewise_backend.py
torch/_inductor/runtime/static_cuda_launcher.py
```

判断：问题不是 Direct-OPD reward 逻辑，而是 vLLM CUDA graph + torch Inductor piecewise compile backend 的兼容性问题。

## 实际修改

修改文件：

```text
scripts/train_justrl_qwen.sh
```

修改内容：

1. Ray 命令优先使用 `PYTHON_BIN` 同环境下的 `ray`，避免系统 `ray` 与训练 env 中 Ray 版本不一致。
2. 增加 vLLM compilation overrides：

```text
VLLM_USE_INDUCTOR
VLLM_CUDAGRAPH_MODE
VLLM_CUDAGRAPH_CAPTURE_SIZES
VLLM_COMPILE_SIZES
```

3. 将这些 override 传给：

```text
actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.*
```

## Debug 过程要点

中间失败点：

1. 裸 `ray` 启动导致 Ray 版本错配：系统 Ray 2.54.0，训练 env Ray 2.55.1。
2. 关闭 Inductor 后，原始 CUDA driver error 消失，vLLM CUDA graph capture 可以完成。
3. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 与 vLLM memory pool 不兼容。
4. 仅降低 `GPU_MEMORY_UTILIZATION` 仍会在 actor backward OOM。
5. 将 `PPO_MAX_TOKEN_LEN_PER_GPU=4096` 后 actor update 可以完成。
6. 同时必须设置 `ROLLOUT_MAX_NUM_BATCHED_TOKENS=33792`，否则 vLLM 会因为 `max_num_batched_tokens < max_model_len` 失败。
7. smoke 阶段要用 `TEST_FREQ=-1` 禁用最终 validation；`TEST_FREQ=999` 仍会因最后一步触发 validation。

## 成功 smoke

成功日志：

```text
/sia-thu/chihaohan/direct-opd-repro/logs/final_repo_smoke_noinductor_actor4096_notest_20260706_165249.log
```

smoke 设置：

```text
ENFORCE_EAGER=False
VAL_BEFORE_TRAIN=False
WANDB_MODE=disabled
LOGGER="['console']"
TOTAL_TRAINING_STEPS=1
SAVE_FREQ=-1
TEST_FREQ=-1
VLLM_USE_INDUCTOR=False
VLLM_CUDAGRAPH_MODE=FULL_DECODE_ONLY
GPU_MEMORY_UTILIZATION=0.45
PPO_MAX_TOKEN_LEN_PER_GPU=4096
ROLLOUT_MAX_NUM_BATCHED_TOKENS=33792
```

关键证据：

```text
Capturing CUDA graphs (decode, FULL): 100%
training/global_step: 1
Final validation metrics: None
Training Progress: 100%|1/1
```

step 1 指标：

```text
delta_opd/weighted_reward_mean: -0.013329323381185532
teacher/entropy_minus_ref: 0.14953312277793884
actor/pg_loss: 0.006093404359749651
actor/grad_norm: 5.906216621398926
timing_s/gen: 31.22293620929122
timing_s/reward: 27.78897414728999
timing_s/ref: 7.014286778867245
timing_s/update_actor: 22.002464395016432
timing_s/step: 92.28454287350178
perf/throughput: 1520.0487062311556
```

delta 信号正常，不是长期 0。

## 正式 run 尝试与当前状态

第一次正式 run 已按要求尝试启动：

```text
tmux: opd_final_full_noinductor_actor4096_20260706_165735
log: /sia-thu/chihaohan/direct-opd-repro/logs/final_repo_noeager_noinductor_actor4096_300s_20260706_165735.log
checkpoint dir: /sia-thu/chihaohan/direct-opd-repro/checkpoints/final_repo_noeager_noinductor_actor4096_300s_20260706_165735
```

正式 run 设置包括：

```text
ENFORCE_EAGER=False
VAL_BEFORE_TRAIN=True
WANDB_MODE=online
LOGGER="['console','wandb']"
TOTAL_TRAINING_STEPS=300
SAVE_FREQ=20
TEST_FREQ=20
```

日志已确认：

```text
val_before_train: True
use_inductor: False
cudagraph_mode: FULL_DECODE_ONLY
Capturing CUDA graphs (decode, FULL): 100%
```

但正式 run 在进入训练前 validation 前失败，失败原因是 W&B 未登录：

```text
wandb.errors.errors.UsageError: No API key configured. Use `wandb login` to log in.
```

当前 `wandb status` 在 `/root` 和 `/GenSIvePFS/users/hhchi` 两个 HOME 下均显示：

```text
api_key: null
```

因此当前还没有 W&B online 链接，正式 run 未完成 train-before validation，也没有 step 1/2 正式指标。

用户在 2026-07-07 完成 `wandb login` 后，训练 env 的 Python 已能从 `/root/.netrc` 读取 API key。随后重启正式 run。

有效正式 run：

```text
tmux: opd_final_full_noeager_20260707_003220
log: /sia-thu/chihaohan/direct-opd-repro/logs/final_repo_noeager_noinductor_actor4096_300s_20260707_003220.log
checkpoint dir: /sia-thu/chihaohan/direct-opd-repro/checkpoints/final_repo_noeager_noinductor_actor4096_300s_20260707_003220
wandb local dir: /sia-thu/chihaohan/direct-opd-repro/wandb/wandb/run-20260707_003437-4zdpbjka
wandb url: https://wandb.ai/ahyd3775-tsinghua-university/Direct-OPD-final/runs/4zdpbjka
```

该 run 已确认：

```text
ENFORCE_EAGER=False
VAL_BEFORE_TRAIN=True
WANDB_MODE=online
LOGGER="['console','wandb']"
TOTAL_TRAINING_STEPS=300
SAVE_FREQ=20
TEST_FREQ=20
use_inductor=False
cudagraph_mode=FULL_DECODE_ONLY
Capturing CUDA graphs (decode, FULL): 100%
```

训练前 validation 已开始，日志确认验证参数：

```text
validate=True
global_steps=0
Validation kwargs: {'top_k': -1, 'top_p': 0.95, 'temperature': 0.7, 'max_tokens': 32768, 'n': 1}
```

注意：上层配置仍为 `actor_rollout_ref.rollout.val_kwargs.n=32`；日志中的 `n: 1` 是 worker 单次 sampling kwargs。当前 run 在 train-before validation 生成阶段持续占用 8 张 GPU，尚未完成 validation，也尚未进入 step 1/2。

启动时曾有一个短暂 run 将 W&B 本地目录写到 repo 下的 `wandb/`；已中止该 run，删除 repo 内生成的 `wandb/`，并用 `WANDB_DIR=/sia-thu/chihaohan/direct-opd-repro/wandb` 重启。

后续 repo 下又出现 `wandb/run-20260707_010551-u79d5rra`，其 metadata 对应另一个 `direct_opd_final_main...resume_eager_false...` 运行，不是当前 `final_repo_noeager_noinductor_actor4096_300s_20260707_003220` 正式 run。当前正式 run 的 W&B 日志路径仍是大盘路径 `4zdpbjka`。

## 恢复正式 run 的前置条件

需要先在训练 env 中完成 W&B 登录：

```bash
/GenSIvePFS/users/hhchi/envs/opd_verl_py312_20260507/bin/wandb login
```

或在启动 tmux 前设置有效的：

```bash
export WANDB_API_KEY=...
```

登录完成后，使用上面的非 eager 系统层参数重启 300-step run，并继续观察：

```text
1. W&B online run 链接
2. 训练前 validation 开始并完成
3. step 1 和 step 2
4. delta_opd/weighted_reward_mean
5. teacher/entropy_minus_ref
6. GPU 利用率和 timing_s/step
```

## 效率对比

已有参考：

```text
旧成功 eager=False run:
  mean timing_s/step ~= 83.9s

当前 eager=True final run:
  mean timing_s/step ~= 124.1s

本次 eager=False smoke:
  timing_s/step = 92.28s
  timing_s/update_actor = 22.00s
```

本次 smoke 的 step time 已明显低于当前 eager=True run，接近旧 eager=False run 的量级。正式 300-step online run 仍需在 W&B 登录后验证均值。
