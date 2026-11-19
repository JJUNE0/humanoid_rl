"""Import saved metrics or follow an existing training process without restarting it."""

import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from humanoid_rl.algorithm.flashsac.tracking import WandbLogger, experiment_identity


def pending_rows(accumulator, counts):
    """Use per-tag cursors so a late evaluation at an already logged step survives."""
    rows, updated = {}, dict(counts)
    for tag in accumulator.Tags()["scalars"]:
        events = accumulator.Scalars(tag)
        start = counts.get(tag, 0)
        if start > len(events):
            raise ValueError(f"TensorBoard history shrank for {tag}")
        for event in events[start:]:
            if not math.isfinite(event.value):
                raise ValueError(f"Nonfinite saved scalar: {tag} at {event.step}")
            row = rows.setdefault(event.step, {"source/wall_time_unix_s": event.wall_time})
            row["source/wall_time_unix_s"] = max(row["source/wall_time_unix_s"], event.wall_time)
            row[tag] = event.value
        updated[tag] = len(events)
    return sorted(rows.items()), updated


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", type=Path, required=True)
    cli.add_argument("--project", required=True)
    cli.add_argument("--entity")
    cli.add_argument("--name")
    cli.add_argument("--follow", action="store_true")
    cli.add_argument("--status-file", type=Path)
    cli.add_argument("--poll-seconds", type=float, default=10)
    cli.add_argument("--mode", choices=("online", "offline"), default="online")
    cli.add_argument("--video", type=Path)
    args = cli.parse_args()
    if args.follow and args.status_file is None:
        cli.error("--follow needs the training driver's --status-file")
    if not 1 <= args.poll_seconds <= 60:
        cli.error("--poll-seconds must be between 1 and 60")
    directory = args.run_dir.resolve()
    config = json.loads((directory / "config.json").read_text())
    identity = experiment_identity(config, directory)
    storage = directory / "wandb_import"
    storage.mkdir(exist_ok=True)
    receipt_path = directory / "wandb_sync.json"
    # Prevent accidental concurrent writers for this source, including retries.
    lock = (storage / "sync.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    saved = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    if saved and (saved["project"], saved.get("entity")) != (args.project, args.entity):
        raise ValueError("This TensorBoard source was already assigned another W&B destination")
    accumulator = EventAccumulator(str(directory), size_guidance={"scalars": 0}, purge_orphaned_data=False)
    logger = WandbLogger(config, directory, project=args.project, entity=args.entity, name=args.name,
                         mode=args.mode, collect_system_stats=False,
                         job_type="tensorboard-follow" if args.follow else "tensorboard-backfill", storage=storage)
    counts = saved.get("tag_counts", {}) if logger.run.resumed else {}
    receipt = logger.metadata() | {"source": str(directory), "follow": args.follow,
                                   "state": "syncing", "tag_counts": counts}
    atomic_json(receipt_path, receipt)
    logger.run.save(str(directory / "config.json"), base_path=str(directory), policy="now")
    logger.run.save(str(directory / "events.out.tfevents.*"), base_path=str(directory),
                    policy="live" if args.follow else "now")
    evaluation = directory / "evaluation.jsonl"
    if evaluation.exists():
        logger.run.save(str(evaluation), base_path=str(directory), policy="live" if args.follow else "now")
    exit_code, source_state = 0, "completed"

    def sync_once():
        nonlocal counts
        accumulator.Reload()
        rows, counts = pending_rows(accumulator, counts)
        for step, metrics in rows:
            logger.log(metrics, step)
        if rows:
            receipt["last_env_steps"] = max(receipt.get("last_env_steps", 0), rows[-1][0])
            receipt["last_vector_steps"] = receipt["last_env_steps"] / config["num_envs"]
            logger.run.summary["source_max_env_steps"] = receipt["last_env_steps"]
            logger.run.summary["source_max_vector_steps"] = receipt["last_vector_steps"]
            print(f"Synced {len(rows)} rows; source vector steps={receipt['last_vector_steps']:g}; {logger.run.url}", flush=True)
        receipt.update(tag_counts=counts, scalar_points=sum(counts.values()), checked_at_unix_s=time.time())
        atomic_json(receipt_path, receipt)

    try:
        while True:
            sync_once()
            if not args.follow:
                break
            source = json.loads(args.status_file.read_text())
            source_state = source["state"]
            if source_state in ("completed", "failed", "cancelled"):
                # The writer may have flushed between our previous read and
                # the driver's final state update. Read it once more before finish.
                sync_once()
                exit_code = 0 if source_state == "completed" else 1
                break
            try:
                os.kill(source["train_pid"], 0)
            except ProcessLookupError:
                source_state, exit_code = "interrupted", 1
                sync_once()
                break
            time.sleep(args.poll_seconds)
        if args.video and not saved.get("video_uploaded"):
            import wandb
            logger.log({"rollout/final": wandb.Video(str(args.video.resolve()), format="mp4"),
                        "rollout/seed": 42, "rollout/duration_s": 30}, receipt.get("last_env_steps", 0))
            receipt["video_uploaded"] = str(args.video.resolve())
        logger.run.summary["source_training_state"] = source_state
        receipt.update(state="completed" if exit_code == 0 else "failed", source_training_state=source_state)
    except BaseException:
        exit_code = 1
        receipt["state"] = "failed"
        raise
    finally:
        logger.finish(exit_code=exit_code)
        atomic_json(receipt_path, receipt)
        lock.close()


if __name__ == "__main__":
    main()
