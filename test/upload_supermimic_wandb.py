"""Upload Light SuperMimic training metrics and Sim-to-Sim validation summary to W&B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="SuperMimic training directory")
    parser.add_argument("--sim2sim-report", type=Path, default=None, help="Optional Isaac Sim2Sim JSON report")
    parser.add_argument("--project", default="humanoid-rl", help="W&B project name")
    parser.add_argument("--name", default="light_supermimic_walk1_t4_fixed_20261118", help="W&B run name")
    args = parser.parse_args()

    import wandb

    resolved = json.loads((args.run_dir / "resolved.json").read_text())
    latest = json.loads((args.run_dir / "latest.json").read_text())
    config = dict(resolved["settings"])
    config["latest_checkpoint"] = latest

    run = wandb.init(
        project=args.project,
        name=args.name,
        config=config,
        job_type="train_and_sim2sim",
    )
    assert run is not None

    with (args.run_dir / "metrics.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            step = int(row["env_steps"])
            wandb.log(row, step=step)

    if args.sim2sim_report is not None and args.sim2sim_report.is_file():
        sim2sim = json.loads(args.sim2sim_report.read_text())
        for key in (
            "num_clips_evaluated",
            "completed_clips",
            "completion_rate",
            "fall_rate",
            "mean_survived_steps",
            "mean_survival_ratio",
            "mean_joint_pos_rmse_rad",
            "mean_joint_pos_rmse_deg",
            "mean_root_height_err_m",
        ):
            run.summary[f"isaac_sim2sim/{key}"] = sim2sim[key]
        run.summary["checkpoint_sha256"] = sim2sim["checkpoint_sha256"]

    url = run.url
    run.finish()
    print(json.dumps({"wandb_url": url}))


if __name__ == "__main__":
    main()
