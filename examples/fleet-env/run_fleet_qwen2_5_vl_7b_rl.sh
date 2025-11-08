#!/bin/bash

set -ex

# 0) Env
export PYTHONBUFFERED=16
export FLEET_API_KEY=${FLEET_API_KEY:-""}

# 1) Download model and export Fleet tasks
hf download Qwen/Qwen2.5-VL-7B-Instruct --local-dir /root/Qwen2.5-VL-7B-Instruct

python examples/fleet-env/export_fleet_tasks.py \
  --fleet-env hubspot \
  --out /root/fleet_tasks.jsonl

# 2) Convert HF to torch dist (for RL)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
source "${SLIME_ROOT}/scripts/models/qwen2.5-7B.sh"

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
  ${MODEL_ARGS[@]} \
  --hf-checkpoint /root/Qwen2.5-VL-7B-Instruct \
  --save /root/Qwen2.5-VL-7B_torch_dist \
  --ckpt-format torch_dcp \
  --use-torch-fsdp2 \
  --no-gradient-accumulation-fusion

# 3) Run RL with separate train/rollout GPUs
bash scripts/run-qwen2.5-vl-7B-fleet.sh


