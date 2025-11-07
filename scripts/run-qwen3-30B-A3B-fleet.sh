#!/bin/bash

set -ex

# for rerun the task (optional cleanup like other scripts)
pkill -9 sglang || true
sleep 2 || true
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-30B-A3B.sh"

CKPT_ARGS=(
   --hf-checkpoint /workspace/models/Qwen3-VL-30B-A3B-Thinking # THIS WILL CHANGE TO HF MODEL PATH
   --ref-load /workspace/models/megatron_ckpts/Qwen3-VL-30B-A3B-Thinking-tp1-pp1 # THIS WILL CHANGE TO HF MODEL PATH
   --load /root/Qwen3-VL-30B-A3B-slime/
   --save /root/Qwen3-VL-30B-A3B-slime/
   --save-interval 20
)

ROLLOUT_ARGS=(
  --prompt-data /root/fleet_tasks.jsonl
  --input-key prompt
  --metadata-key metadata
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --global-batch-size 1
  --num-rollout 50
  --rollout-max-response-len 16384
)

FLEET_ARGS=(
  --custom-generate-function-path examples.fleet-env.generate_with_fleet.generate
  --custom-rm-path examples.fleet-env.fleet_rm.custom_rm
  --fleet-env amazon
  --max-tool-turns 4
)

DEBUG_ARGS=(
   --save-debug-rollout-data /workspace/rollouts/{wandb_run_id}/
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime /root/aime-2024/aime-2024.jsonl
   --n-samples-per-eval-prompt 16
   --eval-max-response-len 16384
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --context-parallel-size 1
   #--expert-model-parallel-size 8
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 4

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 20480
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98

   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-qwen3
   --wandb-group qwen3-4B-4xgpu
   --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.9
   --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
   --actor-num-gpus-per-node 1 # for debugging
   --sglang-disable-cuda-graph
   --debug-rollout-only
   --sglang-tool-call-parser qwen25
   --sglang-grammar-backend xgrammar

)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash
)

# launch the master node of ray in container
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"32\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"SGLANG_TOOL_STRICT_LEVEL\": \"1\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --use-slime-router \
   ${DEBUG_ARGS[@]} \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${FLEET_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}
