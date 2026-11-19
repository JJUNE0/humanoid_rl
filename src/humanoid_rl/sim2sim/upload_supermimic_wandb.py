"""Upload Light SuperMimic training metrics and Sim-to-Sim validation summary to W&B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time


def _drain_complete_lines(metrics_path: Path, offset: int, last_step: int, wandb_mod) -> tuple[int, int, int]:
    """Log only newline-terminated JSON rows starting from byte offset."""
    if not metrics_path.is_file():
        return offset, last_step, 0
    logged = 0
    with metrics_path.open("rb") as f:
        f.seek(offset)
        while True:
            pos = f.tell()
            raw = f.readline()
            if not raw:
                break
            if not raw.endswith(b"\n"):
                f.seek(pos)
                break
            offset = f.tell()
            text = raw.decode("utf-8").strip()
            if not text:
                continue
            row = json.loads(text)
            step = int(row["env_steps"])
            if step > last_step:
                wandb_mod.log(row, step=step)
                last_step = step
                logged += 1
    return offset, last_step, logged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="SuperMimic training directory")
    parser.add_argument("--sim2sim-report", type=Path, default=None, help="Optional Isaac Sim2Sim JSON report")
    parser.add_argument("--project", default="humanoid-rl", help="W&B project name")
    parser.add_argument("--name", default="light_supermimic_walk1_t4_fixed_20261118", help="W&B run name")
    parser.add_argument("--run-id", default=None, help="Optional W&B run id to resume")
    parser.add_argument("--follow", action="store_true", help="Stream metrics.jsonl live until complete.json appears")
    parser.add_argument("--poll-seconds", type=float, default=2.0, help="Polling interval for --follow mode")
    args = parser.parse_args()

    import wandb

    resolved_path = args.run_dir / "resolved.json"
    while args.follow and not resolved_path.is_file():
        time.sleep(args.poll_seconds)

    resolved = json.loads(resolved_path.read_text())
    config = dict(resolved["settings"])
    latest_path = args.run_dir / "latest.json"
    if latest_path.is_file():
        config["latest_checkpoint"] = json.loads(latest_path.read_text())

    meta_path = args.run_dir / "wandb_run.json"
    run_id = args.run_id
    if run_id is None and meta_path.is_file():
        run_id = json.loads(meta_path.read_text()).get("run_id")

    init_kwargs = dict(
        project=args.project,
        name=args.name,
        config=config,
        job_type="train_and_sim2sim",
    )
    if run_id:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    run = wandb.init(**init_kwargs)
    assert run is not None

    meta = {
        "wandb_url": run.url,
        "run_id": run.id,
        "project": args.project,
        "name": args.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"status": "following" if args.follow else "uploading", **meta}), flush=True)

    metrics_path = args.run_dir / "metrics.jsonl"
    complete_path = args.run_dir / "complete.json"
    offset = 0
    last_step = -1

    while True:
        offset, last_step, logged = _drain_complete_lines(metrics_path, offset, last_step, wandb)
        if args.follow and logged > 0:
            print(json.dumps({"logged_rows": logged, "last_step": last_step}), flush=True)
        if not args.follow or complete_path.is_file():
            # Final drain after complete.json lands
            offset, last_step, _ = _drain_complete_lines(metrics_path, offset, last_step, wandb)
            break
        time.sleep(args.poll_seconds)

    if latest_path.is_file():
        run.summary["latest_checkpoint"] = json.loads(latest_path.read_text())

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
            if key in sim2sim:
                run.summary[f"isaac_sim2sim/{key}"] = sim2sim[key]
        if "checkpoint_sha256" in sim2sim:
            run.summary["checkpoint_sha256"] = sim2sim["checkpoint_sha256"]

    url = run.url
    run.finish()
    print(json.dumps({"status": "finished", "wandb_url": url, "last_step": last_step}), flush=True)


if __name__ == "__main__":
    main()
