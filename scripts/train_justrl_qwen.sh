#!/usr/bin/env bash
set -euo pipefail
set -x

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/verl:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=${RAY_memory_usage_threshold:-0.99}
export TORCH_NCCL_BLOCKING_WAIT=${TORCH_NCCL_BLOCKING_WAIT:-1}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-7200}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-true}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3.12}
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES:-1}

MODEL_ROOT=${MODEL_ROOT:-models}
DATA_ROOT=${DATA_ROOT:-datasets}
OUTPUT_ROOT=${OUTPUT_ROOT:-checkpoints}
LOG_ROOT=${LOG_ROOT:-logs}

PROJECT_NAME=${PROJECT_NAME:-Direct-OPD}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-justrl_qwen3_1p7b}
ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-"${MODEL_ROOT}/Qwen3-1.7B"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"${MODEL_ROOT}/JustRL-DeepSeek-1.5B"}
TEACHER_REF_MODEL_PATH=${TEACHER_REF_MODEL_PATH:-"${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B"}
TRAIN_DATASET=${TRAIN_DATASET:-"${DATA_ROOT}/train/dapo_math_17k.parquet"}
TEST_DATASET=${TEST_DATASET:-"['${REPO_ROOT}/datasets/eval/aime24.parquet','${REPO_ROOT}/datasets/eval/aime25.parquet','${REPO_ROOT}/datasets/eval/hmmt_feb.parquet']"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"${OUTPUT_ROOT}/${EXPERIMENT_NAME}"}
OUTPUTS_DIR=${OUTPUTS_DIR:-"${CHECKPOINT_DIR}/outputs"}
TRAIN_LOG=${TRAIN_LOG:-"${LOG_ROOT}/${EXPERIMENT_NAME}.log"}

ADV_ESTIMATOR=${ADV_ESTIMATOR:-token_reward_direct}
GRPO_OUTCOME_WEIGHT=${GRPO_OUTCOME_WEIGHT:-1.0}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESP_LENGTH=${MAX_RESP_LENGTH:-2048}
MAX_VAL_RESP_LENGTH=${MAX_VAL_RESP_LENGTH:-32768}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))}
MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-128}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}
TEMPERATURE=${TEMPERATURE:-1.0}
TEACHER_TEMPERATURE=${TEACHER_TEMPERATURE:-1.0}
REPETITION_PENALTY=${REPETITION_PENALTY:-1.0}
N_RESPONSES=${N_RESPONSES:-4}
LOG_PROB_TOP_K=${LOG_PROB_TOP_K:-16}
TOP_K_STRATEGY=${TOP_K_STRATEGY:-only_stu}
REWARD_WEIGHT_MODE=${REWARD_WEIGHT_MODE:-student_p}
KL_LOSS_COEF=${KL_LOSS_COEF:-2.5}
ADAPTIVE_KL_LOSS_REWARD_KEY=${ADAPTIVE_KL_LOSS_REWARD_KEY:-delta_opd/weighted_reward_mean}
ADAPTIVE_KL_LOSS_EPS=${ADAPTIVE_KL_LOSS_EPS:-0.01}
ADAPTIVE_KL_LOSS_MIN_COEF=${ADAPTIVE_KL_LOSS_MIN_COEF:-0.5}
ADAPTIVE_KL_LOSS_MAX_COEF=${ADAPTIVE_KL_LOSS_MAX_COEF:-2.5}
MODEL_DTYPE=${MODEL_DTYPE:-fp32}
LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
PARALLEL_SIZE=${PARALLEL_SIZE:-1}
OPTIM_LR=${OPTIM_LR:-1e-6}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
FREE_CACHE_ENGINE=${FREE_CACHE_ENGINE:-True}
ENFORCE_EAGER=${ENFORCE_EAGER:-False}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-1024}
ENABLE_CHUNKED_PREFILL=${ENABLE_CHUNKED_PREFILL:-True}
ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING:-True}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
NUM_NODES=${NUM_NODES:-1}
LOGGER=${LOGGER:-"['console']"}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
LOG_VAL_GENERATIONS=${LOG_VAL_GENERATIONS:-2}
VAL_N=${VAL_N:-32}
SAVE_FREQ=${SAVE_FREQ:-20}
TEST_FREQ=${TEST_FREQ:-20}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-2}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
IS_PLOT=${IS_PLOT:-True}
MANAGE_RAY=${MANAGE_RAY:-True}

PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-$(( MAX_MODEL_LEN > 32768 ? MAX_MODEL_LEN : 32768 ))}
ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU=${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-16384}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-${PPO_MAX_TOKEN_LEN_PER_GPU}}
mkdir -p "${LOG_ROOT}" "${CHECKPOINT_DIR}" "${OUTPUTS_DIR}/validation_log"

if [ "${MANAGE_RAY}" = "True" ]; then
  ray stop --force || true
  ray start --head --disable-usage-stats
  sleep 5
fi

set +e
"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator="${ADV_ESTIMATOR}" \
  algorithm.grpo_outcome_weight="${GRPO_OUTCOME_WEIGHT}" \
  data.shuffle=False \
  data.train_files="${TRAIN_DATASET}" \
  data.val_files="${TEST_DATASET}" \
  data.train_batch_size="${MINI_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESP_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.return_raw_chat=True \
  actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_activation_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${OPTIM_LR}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${PARALLEL_SIZE}" \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
  actor_rollout_ref.actor.adaptive_kl_loss_coef=True \
  actor_rollout_ref.actor.adaptive_kl_loss_reward_key="${ADAPTIVE_KL_LOSS_REWARD_KEY}" \
  actor_rollout_ref.actor.adaptive_kl_loss_eps="${ADAPTIVE_KL_LOSS_EPS}" \
  actor_rollout_ref.actor.adaptive_kl_loss_min_coef="${ADAPTIVE_KL_LOSS_MIN_COEF}" \
  actor_rollout_ref.actor.adaptive_kl_loss_max_coef="${ADAPTIVE_KL_LOSS_MAX_COEF}" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN_PER_GPU}" \
  +actor_rollout_ref.rollout.reward_mode=delta_opd \
  +actor_rollout_ref.rollout.log_prob_top_k="${LOG_PROB_TOP_K}" \
  +actor_rollout_ref.rollout.top_k_strategy="${TOP_K_STRATEGY}" \
  +actor_rollout_ref.rollout.reward_weight_mode="${REWARD_WEIGHT_MODE}" \
  +actor_rollout_ref.rollout.teacher_temperature="${TEACHER_TEMPERATURE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.free_cache_engine="${FREE_CACHE_ENGINE}" \
  actor_rollout_ref.rollout.enforce_eager="${ENFORCE_EAGER}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${ENABLE_CHUNKED_PREFILL}" \
  actor_rollout_ref.rollout.enable_prefix_caching="${ENABLE_PREFIX_CACHING}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.n="${N_RESPONSES}" \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.val_kwargs.max_tokens="${MAX_VAL_RESP_LENGTH}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
  actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  reward_model.enable=True \
  +reward_model.reward_kwargs.enable_format_reward=False \
  reward_model.model.path="${REWARD_MODEL_PATH}" \
  reward_model.model.input_tokenizer=null \
  reward_model.model.use_remove_padding=True \
  reward_model.model.fsdp_config.param_offload=False \
  +reward_model.model.dtype="${MODEL_DTYPE}" \
  reward_model.micro_batch_size_per_gpu=24 \
  +teacher_ref_reward_model.enable=True \
  +teacher_ref_reward_model.model.path="${TEACHER_REF_MODEL_PATH}" \
  +teacher_ref_reward_model.model.input_tokenizer=null \
  +teacher_ref_reward_model.model.use_remove_padding=True \
  +teacher_ref_reward_model.model.fsdp_config.param_offload=False \
  +teacher_ref_reward_model.model.dtype="${MODEL_DTYPE}" \
  +teacher_ref_reward_model.micro_batch_size_per_gpu=24 \
  custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/ttrl_math/__init__.py" \
  custom_reward_function.name=reward_func \
  trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
  trainer.log_val_generations="${LOG_VAL_GENERATIONS}" \
  trainer.logger="${LOGGER}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.validation_data_dir="${OUTPUTS_DIR}/validation_log/${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.nnodes="${NUM_NODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${CHECKPOINT_DIR}" \
  trainer.is_plot="${IS_PLOT}" \
  2>&1 | tee -a "${TRAIN_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

if [ "${MANAGE_RAY}" = "True" ]; then
  ray stop --force || true
fi
exit "${STATUS}"
