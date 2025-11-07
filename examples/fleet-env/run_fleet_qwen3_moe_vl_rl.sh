#!/bin/bash

set -ex

# 0) Env
export PYTHONBUFFERED=16
export FLEET_API_KEY=${FLEET_API_KEY:-""}

# # 1) Download model and export Fleet tasks
# hf download Qwen/Qwen3-VL-8B-Thinking --local-dir /root/Qwen3-VL-8B-Thinking

# python examples/fleet-env/export_fleet_tasks.py \
#   --fleet-env amazon \
#   --out /root/fleet_tasks.jsonl

# 2) Convert HF to torch dist (for RL)
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"
source "${SLIME_ROOT}/scripts/models/qwen3-30B-A3B.sh"

# PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
#   ${MODEL_ARGS[@]} \
#   --hf-checkpoint /root/Qwen3-VL-8B-Thinking \
#   --save /root/Qwen3-VL-8B_torch_dist

# 3) Run RL with separate train/rollout GPUs
bash scripts/run-qwen3-vl-moe-fleet.sh


