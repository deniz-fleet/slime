from typing import Any, Optional

import fleet

from slime.utils.types import Sample


async def custom_rm(args, sample: Sample, **kwargs) -> float:
    """Custom reward function for Fleet tasks.

    Looks up the Fleet task by key, spins an environment, and runs the task's
    verifier against the model's final answer (sample.response).
    """
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_key: Optional[str] = meta.get("task_key") or meta.get("task_id") or getattr(args, "task_key", None)
    env_key: Optional[str] = meta.get("env_key") or getattr(args, "fleet_env", None)

    if not task_key or not env_key:
        raise ValueError("Missing task_key or env_key (--fleet-env); include in sample.metadata or args")

    # Fetch task and create env aligned to task config
    task = fleet.get_task(task_key)
    print(f"evaluating task {task_key} with env {env_key}")
    env = await task.make(region=None, image_type="mcp", ttl_seconds=1800)

    try:
        # Verify and extract a numeric score
        detailed = await task.verify_detailed_async(env, final_answer=sample.response or None)
        score = getattr(detailed, "score", None)
        if isinstance(score, (int, float)):
            return float(score)

        # Fallbacks in case the response model differs
        is_success = getattr(detailed, "is_success", None)
        if isinstance(is_success, bool):
            return 1.0 if is_success else 0.0

        # Last resort: treat truthiness as success
        return 1.0 if detailed else 0.0
    finally:
        try:
            await env.close()
        except Exception:
            pass


