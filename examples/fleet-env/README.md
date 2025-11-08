Quickstart (Fleet MCP + Qwen3‑VL)

1) Setup
Starting inside the [slime container](https://github.com/deniz-fleet/slime/tree/main/docker)
```
git clone https://github.com/deniz-fleet/slime.git
git checkout deniz/fleet-integration
pip install fleet-python
pip install -U transformers
pip install -U sglang
pip install -e .

export FLEET_API_KEY=<your_fleet_api_key>
export WANDB_KEY=<your_wandb_api_key>
```

2) Download models (or convert them with different parallelism configurations)
```
# Convert HF → Megatron (tp4-pp1), then start the RL run
# (This takes a loooong time)
bash examples/fleet-env/misc/convert_vlm_30b.sh
```
or download the existing converted Megatron checkpoint.
```
aws s3 sync s3://rl-training-tests/Qwen3-VL-30B-A3B-Thinking-tp4-pp1/ /workspace/models/Qwen3-VL-30B-A3B-Thinking-tp4-pp1
aws s3 sync s3://rl-training-tests/Qwen3-VL-30B-A3B-Thinking /workspace/models/Qwen3-VL-30B-A3B-Thinking &
```


3) Pipeline (export tasks → convert → run)
```
bash examples/fleet-env/run_fleet_qwen3_moe_vl_rl.sh
```

Brief note on `scripts/run-qwen3-30B-A3B-fleet.sh`
- The script is comprised of RL flags, Megatron flags, and SGLang flags.
- Checkpoint paths should match your downloads:
  ```bash
  --hf-checkpoint /workspace/models/Qwen3-VL-30B-A3B-Thinking
  --ref-load /workspace/models/Qwen3-VL-30B-A3B-Thinking-tp1-pp1
  ```
- Custom paths for multi‑turn env interaction and verifiers:
  ```bash
  --custom-generate-function-path examples.fleet-env.generate_with_fleet.generate
  --custom-rm-path examples.fleet-env.fleet_rm.custom_rm
  ```
- Choose the Fleet environment:
  ```bash
  --fleet-env amazon
  ```
- Save rollouts for debugging:
  ```bash
  --save-debug-rollout-data /workspace/rollouts/{wandb_run_id}/
  ```
- Rollout‑only mode (no training step):
  ```bash
  --debug-rollout-only
  ```

What it does
- Download HF model locally (for consistent runs).
- Export Fleet tasks to JSONL via API:
  ```bash
  python examples/fleet-env/export_fleet_tasks.py --fleet-env amazon --out /root/fleet_tasks.jsonl
  ```

Custom hooks used to interact with Fleet environment
- Generate (multi‑turn Fleet MCP tool loop): `examples/fleet-env/generate_with_fleet.py:generate`
- Reward (verify_detailed_async): `examples/fleet-env/fleet_rm.py:custom_rm`

Model reference
- Qwen/Qwen2.5‑VL‑7B‑Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen/Qwen3-VL-30B-A3B-Thinking: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking


For more information, see the slime Usage Guide: `https://thudm.github.io/slime/get_started/usage.html`

