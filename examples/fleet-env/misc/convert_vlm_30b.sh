#!/usr/bin/env bash
# HF → Megatron (dist ckpt) converter for Qwen/Qwen3-VL-30B-A3B-Thinking
# Single GPU, bf16, outputs to /workspace/data/models/megatron_ckpts/...

set -euo pipefail
PAI_PATCH_DIR="/workspace/code/Pai-Megatron-Patch"
MEGATRON_BACKEND_DIR="${PAI_PATCH_DIR}/backends/megatron/Megatron-LM-250624"  # vendored, has core+training
# ----- converter entry (define AFTER PAI_PATCH_DIR!) -----
CONVERT_PY="${PAI_PATCH_DIR}/toolkits/distributed_checkpoints_convertor/impl/convert.py"

# ----- data I/O (under /workspace/data) -----
HF_DIR="/workspace/models/Qwen3-VL-30B-A3B-Thinking"
OUT_DIR="/workspace/models/Qwen3-VL-30B-A3B-Thinking-tp4-pp1"

----- repo roots -----
mkdir -p "${PAI_PATCH_DIR}"
git clone https://github.com/alibaba/Pai-Megatron-Patch.git "${PAI_PATCH_DIR}"
cd "${PAI_PATCH_DIR}"
git submodule update --init --recursive
cd ../..


# ----- tiny deps: pin hub for transformers in the container; no HF snapshot here -----
python - <<'PY'
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "huggingface_hub>=0.34.0,<1.0", "typer-slim"])
PY

# ----- safety checks -----
[ -d "${PAI_PATCH_DIR}" ] || { echo "Missing ${PAI_PATCH_DIR}"; exit 1; }
[ -d "${MEGATRON_BACKEND_DIR}" ] || { 
  echo "Vendored Megatron dir empty. Initializing submodules..."
  ( cd "${PAI_PATCH_DIR}" && git submodule update --init --recursive )
}
[ -f "${MEGATRON_BACKEND_DIR}/megatron/core/process_groups_config.py" ] || {
  echo "Megatron submodule not populated: ${MEGATRON_BACKEND_DIR}"; exit 1; }

# needs both core and training
[ -d "${MEGATRON_BACKEND_DIR}/megatron/training" ] || {
  echo "Vendored Megatron lacks 'training' at ${MEGATRON_BACKEND_DIR}/megatron/training"; exit 1; }

[ -f "${CONVERT_PY}" ] || { echo "Missing converter: ${CONVERT_PY}"; exit 1; }
[ -f "${HF_DIR}/config.json" ] || { echo "Missing HF model at ${HF_DIR}. Download it there first."; exit 1; }

mkdir -p "${OUT_DIR}"

# ----- print the PYTHONPATH we intend to use -----
echo "Using PYTHONPATH:"
echo "  1) ${MEGATRON_BACKEND_DIR}"
echo "  2) ${PAI_PATCH_DIR}"

# ----- run conversion (single proc, bf16) -----
echo "==> Converting HF → Megatron (Qwen3-VL-30B-A3B, single GPU, bf16)"
env -i PATH="$PATH" CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  TORCH_DISTRIBUTED_DEBUG=DETAIL \
  NCCL_DEBUG=INFO \
  CUDA_DEVICE_MAX_CONNECTIONS=1 \
  PYTHONNOUSERSITE=1 PYTHONPATH="${MEGATRON_BACKEND_DIR}:${PAI_PATCH_DIR}" \
  python -m torch.distributed.run --nproc_per_node=8 \
    "${CONVERT_PY}" \
    --model-type GPT \
    --load-dir "${HF_DIR}" \
    --save-dir "${OUT_DIR}" \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model "${HF_DIR}" \
    --hf-dir "${HF_DIR}" \
    --target-ckpt-format torch_dist \
    --synchronizer qwen3_vl \
    --pretrain-script qwen3_vl.pretrain_qwen \
    --auto-model AutoModelForImageTextToText \
    --no-load-optim --no-load-rng --logging-level 1 \
    --bf16 --use-gpu \
    --sequence-parallel \
    --tensor-model-parallel-size 4 --pipeline-model-parallel-size 1 --expert-model-parallel-size 2 \
    --micro-batch-size 1 --global-batch-size 4  --train-iters 1 \
    --normalization RMSNorm --swiglu --disable-bias-linear --seq-length 1 \
    --attention-backend auto --position-embedding-type mrope --group-query-attention \
    --kv-channels 128 --qk-layernorm --max-position-embeddings 262144 --padded-vocab-size 151936 \
    --mrope-section 24 20 20 \
    --num-layers 48 --hidden-size 2048 --ffn-hidden-size 6144 --moe-ffn-hidden-size 768 \
    --num-attention-heads 32 --untie-embeddings-and-output-weights \
    --moe-grouped-gemm --moe-router-score-function softmax --moe-token-dispatcher-type alltoall \
    --moe-router-topk 8 --moe-layer-freq 1 --num-experts 128 --num-query-groups 4

echo; echo "==> Done. Megatron checkpoint at: ${OUT_DIR}"
ls -lah "${OUT_DIR}" || true