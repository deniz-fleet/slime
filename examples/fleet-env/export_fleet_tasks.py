import json
from argparse import ArgumentParser
from typing import Any, Dict

import fleet


def export_tasks(env_key: str, out_path: str, version: str | None = None,
                 data_id: str | None = None, data_version: str | None = None) -> None:
    """Export Fleet tasks to a JSONL file for training.

    Each line contains a record with fields expected by the RL/SFT pipeline.
    Minimal schema:
      - prompt: task prompt text
      - task_key: Fleet task key
      - env_key: environment key (env_id[:version])
      - version, data_id, data_version: copied from task (if present)
      - rm_type: fixed to "fleet" so the custom RM is selected
      - metadata: passthrough of task.metadata (story, etc.)
    """
    tasks = fleet.load_tasks(
        env_key=env_key, version=version, data_id=data_id, data_version=data_version
    )

    def _record(task) -> Dict[str, Any]:
        meta = dict(task.metadata or {})
        # Ensure RM can resolve keys from sample.metadata
        meta.setdefault("task_key", task.key)
        meta.setdefault("env_key", task.env_key)
        return {
            "prompt": task.prompt,
            "task_key": task.key,
            "env_key": task.env_key,
            "version": task.version,
            "data_id": task.data_id,
            "data_version": task.data_version,
            "rm_type": "fleet",
            "metadata": meta,
        }

    with open(out_path, "w") as f:
        for t in tasks:
            f.write(json.dumps(_record(t)) + "\n")


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--fleet-env", required=True, help="Fleet environment key, e.g. 'amazon' or 'amazon:vX'.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--version", default=None)
    parser.add_argument("--data-id", default=None)
    parser.add_argument("--data-version", default=None)
    args = parser.parse_args()

    export_tasks(
        env_key=args.fleet_env,
        out_path=args.out,
        version=args.version,
        data_id=args.data_id,
        data_version=args.data_version,
    )


if __name__ == "__main__":
    main()


