from typing import Any, Optional
import json

import fleet

from slime.utils.types import Sample


async def custom_rm(args, sample: Sample, **kwargs) -> float:
    """Custom reward function for Fleet tasks.

    Looks up the Fleet task by key, spins an environment, and runs the task's
    verifier against the model's final answer (sample.response).
    """
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    task_key: Optional[str] = meta.get("task_key")
    env_key: Optional[str] = meta.get("env_key")

    if not task_key:
        raise ValueError("Missing task_key; include in sample.metadata['task_key'] or --task-key")

    # Fetch task and create env aligned to task config using Fleet async APIs
    tasks = await fleet.load_tasks_async(keys=[task_key])
    if not tasks:
        raise ValueError(f"No Fleet task found for key: {task_key}")
    task = tasks[0]

    data_key: Optional[str] = getattr(task, "data_key", None)
    env_variables: Optional[dict] = getattr(task, "env_variables", None)

    if not env_key:
        raise ValueError("Unable to determine env_key from task or args")

    env = await fleet.env.make_async(
        env_key=env_key,
        data_key=data_key,
        env_variables=env_variables,
        ttl_seconds=10800,
    )

    try:
        # Prepare final answer (transcript) for verifier
        final_answer: Optional[str] = meta.get("final_answer")
        # Verify and extract a numeric score
        print(f"{final_answer=}")
        detailed = await task.verify_detailed_async(env, transcript=final_answer)
        print(
            f"evaluating task {task_key} with env {env_key}",
            f"final answer: {final_answer}",
            f"detailed verification response: {detailed}",
        )

        return detailed.result


    finally:
        try:
            await env.close()
        except Exception:
            pass


