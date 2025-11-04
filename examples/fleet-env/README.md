Quickstart (Fleet MCP + Qwen3‑VL RL)

1) Setup
```
cd slime
pip install -e .
pip install fleet-python
export FLEET_API_KEY="sk_your_key_here"
```

2) RL‑only pipeline (download → export tasks → convert → run)
```
# One‑shot convenience script
bash examples/fleet-env/run_fleet_qwen3_vl_8b_rl.sh
```

What it does
- Download HF model `Qwen/Qwen3-VL-8B-Thinking` locally (for consistent runs).
- Export Fleet tasks to JSONL via API:
  ```bash
  python examples/fleet-env/export_fleet_tasks.py --fleet-env amazon --out /root/fleet_tasks.jsonl
  ```
- Convert HF → TorchDist for RL:
  ```bash
  source scripts/models/qwen3-vl-8B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3-VL-8B-Thinking \
    --save /root/Qwen3-VL-8B_torch_dist
  ```
- Launch training with separate GPUs for train vs rollout:
  ```bash
  bash scripts/run-qwen3-vl-8B-fleet.sh
  ```

Custom hooks used
- Generate (multi‑turn Fleet MCP tool loop): `examples/fleet-env/generate_with_fleet.py:generate`
- Reward (verify_detailed_async): `examples/fleet-env/fleet_rm.py:custom_rm`

Model reference
- Qwen/Qwen3‑VL‑8B‑Thinking: https://huggingface.co/Qwen/Qwen3-VL-8B-Thinking


