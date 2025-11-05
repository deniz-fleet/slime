#!/bin/bash

set -ex

# for rerun the task (optional cleanup like other scripts)
pkill -9 sglang || true
sleep 2 || true
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true

# prevent stdout/stderr buffering
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Resolve script dir and source model args (Qwen2.5-VL-7B)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-7B.sh"

CKPT_ARGS=(
  --hf-checkpoint /root/Qwen2.5-VL-7B-Instruct
  --use-torch-fsdp2
  --no-gradient-accumulation-fusion
  --ref-load /root/Qwen2.5-VL-7B_torch_dist
  --load /root/Qwen2.5-VL-7B_slime
  --save /root/Qwen2.5-VL-7B_slime
  --save-interval 20
)

ROLLOUT_ARGS=(
  --prompt-data /root/fleet_tasks.jsonl
  --input-key prompt
  --metadata-key metadata
  --rollout-batch-size 2
  --n-samples-per-prompt 2
  --global-batch-size 4
  --num-rollout 50
  --rollout-max-response-len 1024
)

FLEET_ARGS=(
  --custom-generate-function-path examples/fleet-env/generate_with_fleet.py:generate
  --custom-rm-path examples/fleet-env/fleet_rm.py:custom_rm
  --fleet-env amazon
  --max-tool-turns 4
)

PERF_ARGS=(
  --use-dynamic-batch-size
  --max-tokens-per-gpu 2048
  --tensor-model-parallel-size 2
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1

  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 4
)

SGLANG_ARGS=(
   --rollout-num-gpus 2
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.5
   --sglang-max-running-requests 128
   --sglang-disable-cuda-graph
)
OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
)


WANDB_ARGS=(
   --use-wandb
   --wandb-project slime-dev-qwen3
   --wandb-group qwen3-4B-4xgpu
   --wandb-key ${WANDB_KEY}
)

# 3) Launch ray head (single-node default)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    "NCCL_NVLS_ENABLE": "'"${HAS_NVLINK}"'"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 8 \
  --colocate \
  ${SGLANG_ARGS[@]} \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${FLEET_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${WANDB_ARGS[@]} \
  ${PERF_ARGS[@]}


