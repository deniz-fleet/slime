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
    env_key_override: Optional[str] = meta.get("env_key") or getattr(args, "fleet_env", None)

    if not task_key:
        raise ValueError("Missing task_key; include in sample.metadata['task_key'] or --task-key")

    # Fetch task and create env aligned to task config using Fleet async APIs
    tasks = await fleet.load_tasks_async(keys=[task_key])
    if not tasks:
        raise ValueError(f"No Fleet task found for key: {task_key}")
    task = tasks[0]

    env_key_to_use: Optional[str] = env_key_override or getattr(task, "env_key", None)
    data_key: Optional[str] = getattr(task, "data_key", None)
    env_variables: Optional[dict] = getattr(task, "env_variables", None)

    if not env_key_to_use:
        raise ValueError("Unable to determine env_key from task or args")

    print(f"evaluating task {task_key} with env {env_key_to_use}")
    env = await fleet.env.make_async(
        env_key=env_key_to_use,
        data_key=data_key,
        env_variables=env_variables,
        ttl_seconds=10800,
    )

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


