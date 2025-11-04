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

# Resolve script dir and source model args (Qwen3-VL-8B)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen3-vl-8B.sh"

CKPT_ARGS=(
  --hf-checkpoint /root/Qwen3-VL-8B-Thinking
  --ref-load /root/Qwen3-VL-8B_torch_dist
  --load /root/Qwen3-VL-8B_slime
  --save /root/Qwen3-VL-8B_slime
  --save-interval 20
)

ROLLOUT_ARGS=(
  --prompt-data /root/fleet_tasks.jsonl
  --input-key prompt
  --metadata-key metadata
  --rollout-batch-size 4
  --n-samples-per-prompt 1
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
  # Adjust parallelism as needed; leave simple by default
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
)

# 3) Launch ray head (single-node default)
export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON='{
  "env_vars": {
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "'"${HAS_NVLINK}"'"
  }
}'

ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 \
  ${MODEL_ARGS[@]} \
  ${CKPT_ARGS[@]} \
  ${ROLLOUT_ARGS[@]} \
  ${FLEET_ARGS[@]} \
  ${OPTIMIZER_ARGS[@]} \
  ${PERF_ARGS[@]}


