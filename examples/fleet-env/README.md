Quickstart (Fleet MCP + Qwen3‑VL RL)

1) Setup
```
cd slime
pip install fleet-python
pip install -e .
pip install -U transformers
export FLEET_API_KEY=sk_BdFKC2l8LqWzhQ36AdRB4GQHBYsBbODvrIAfXvxn3G4
export WANDB_KEY=f66373399824f8c1942490b0862a664b8afc8802
```

```
git checkout -- examples/fleet-env/run_fleet_qwen2_5_vl_7b_rl.sh
chmod +x examples/fleet-env/run_fleet_qwen2_5_vl_7b_rl.sh
./examples/fleet-env/run_fleet_qwen2_5_vl_7b_rl.sh
```

2) RL‑only pipeline (download → export tasks → convert → run)
```
# One‑shot convenience script
bash examples/fleet-env/run_fleet_qwen3_vl_8b_rl.sh
```

```
# One‑shot convenience script
chmod +x examples/fleet-env/run_fleet_qwen3_moe_vl_rl.sh
bash examples/fleet-env/run_fleet_qwen3_moe_vl_rl.sh
```

What it does
- Download HF model `Qwen/Qwen2.5-VL-7B-Instruct` locally (for consistent runs).
- Export Fleet tasks to JSONL via API:
  ```bash
  python examples/fleet-env/export_fleet_tasks.py --fleet-env amazon --out /root/fleet_tasks.jsonl
  ```
- Convert HF → TorchDist for RL:
  ```bash
  source scripts/models/qwen2.5-7B.sh
  PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen2.5-VL-7B-Instruct \
    --save /root/Qwen2.5-VL-7B_torch_dist
  ```
- Launch training with separate GPUs for train vs rollout:
  ```bash
  bash scripts/run-qwen3-vl-8B-fleet.sh
  ```

Custom hooks used
- Generate (multi‑turn Fleet MCP tool loop): `examples/fleet-env/generate_with_fleet.py:generate`
- Reward (verify_detailed_async): `examples/fleet-env/fleet_rm.py:custom_rm`

Model reference
- Qwen/Qwen2.5‑VL‑7B‑Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct


