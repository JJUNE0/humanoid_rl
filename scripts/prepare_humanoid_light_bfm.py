"""Build a boundary-safe CMU + LAFAN manifest from geometrically screened runs.

Input archives are never rewritten. Invalid frames/quarter-frame intervals split
the sequences; neither interpolation over a rejected interval nor tail padding
is used. Passing this screen is not a dynamic feasibility certificate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from humanoid_light_collision import MeshCollisionIndex, body_collision_pairs
from optimize_humanoid_light_cmu_landing import checked_mesh_distance, foot_geometry
from robots.humanoid.humanoid_light.config import JOINT_NAMES, get_spec


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def valid_runs(good, min_frames, guard_frames=2):
    """Return half-open, original-frame intervals without crossing invalid data."""
    changes = np.diff(np.r_[False, good, False].astype(np.int8))
    result = []
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        start = int(start)+(guard_frames if start > 0 else 0)
        stop = int(stop)-(guard_frames if stop < len(good) else 0)
        if stop-start >= min_frames:
            result.append([start, stop])
    return result


def screen(job):
    path = ROOT/job["path"]
    digest = sha256(path)
    cache = ROOT/job["cache"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256((digest+job["screen_version"]).encode()).hexdigest()
    if cache.is_file():
        previous = json.loads(cache.read_text())
        if previous.get("signature") == signature:
            return previous
    started = time.monotonic()
    with np.load(path, allow_pickle=False) as archive:
        q, qd = archive["qpos"], archive["qvel"]
        names, fps = tuple(archive["joint_names"].tolist()), float(archive["frequency"])
        split_points = archive["split_points"]
    if names != ("root",)+JOINT_NAMES or not np.isclose(fps, 50):
        raise ValueError(f"Incompatible Humanoid Light motion: {path}")
    if not np.isfinite(q).all() or not np.isfinite(qd).all():
        raise ValueError(f"Nonfinite states: {path}")
    model = get_spec().compile()
    meshes, *_ = foot_geometry(model)
    pairs, _ = body_collision_pairs(model, meshes, checked_mesh_distance)
    index = MeshCollisionIndex(model, meshes, pairs, checked_mesh_distance)
    data = mujoco.MjData(model)
    limits = json.loads((ROOT/"src/robots/humanoid/humanoid_light/assets/control.json").read_text())["joints"]
    speed = np.array([limits[name]["velocity_limit"] for name in JOINT_NAMES])
    good = (np.isfinite(q).all(1) & np.isfinite(qd).all(1)
            & (np.abs(np.linalg.norm(q[:, 3:7], axis=1)-1) < 1e-4)
            & (q[:, 7:] >= model.jnt_range[1:, 0]-1e-6).all(1)
            & (q[:, 7:] <= model.jnt_range[1:, 1]+1e-6).all(1)
            & (np.abs(qd[:, 6:]) <= speed*1.001).all(1))
    tested = floor_bad = self_bad = 0

    def clear(pose):
        nonlocal tested, floor_bad, self_bad
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        index.update(data)
        tested += 1
        if index.minimum_height(data) < -.001:
            floor_bad += 1
            return False
        if any(distance <= .00005 for *_, distance, witness, certificate in index.query(data, .0001)):
            self_bad += 1
            return False
        return True

    # Check all stored poses, then all three quarter-frame interpolants of the
    # surviving intervals. Testing never joins different source split_points.
    for frame in np.flatnonzero(good):
        if not clear(q[frame]):
            good[frame] = False
    interval_good = good[:-1] & good[1:]
    for boundary in split_points[1:-1]:
        interval_good[int(boundary)-1] = False
    for start, stop in zip(split_points[:-1], split_points[1:]):
        if stop-start < 2:
            continue
        times = np.arange(start, stop)
        rotation = Slerp(times, Rotation.from_quat(q[start:stop, 3:7], scalar_first=True))
        intervals = np.flatnonzero(interval_good[start:stop-1])+start
        if len(intervals):
            sub_times = (intervals[:, None]+np.array([.25, .5, .75])).ravel()
            quaternions = rotation(sub_times).as_quat(scalar_first=True).reshape(-1, 3, 4)
            for frame, quats in zip(intervals, quaternions):
                for fraction, quaternion in zip((.25, .5, .75), quats):
                    pose = (1-fraction)*q[frame]+fraction*q[frame+1]
                    pose[3:7] = quaternion
                    if not clear(pose):
                        interval_good[frame] = False
                        break
    # Remove both endpoints of a bad interval. This is conservative and leaves
    # original velocities intact at accepted boundaries.
    bad = ~interval_good
    good[:-1] &= ~bad
    good[1:] &= ~bad
    runs = []
    for start, stop in zip(split_points[:-1], split_points[1:]):
        runs.extend([[a+int(start), b+int(start)] for a, b in valid_runs(
            good[start:stop], job["min_frames"], guard_frames=2)])
    result = {key: job[key] for key in ("path", "source", "subject", "split", "name")}
    result.update(sha256=digest, signature=signature, frames=len(q), runs=runs,
                  accepted_frames=sum(b-a for a, b in runs), tested_poses=tested,
                  floor_failures=floor_bad, body_failures=self_bad,
                  seconds=time.monotonic()-started)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2)+"\n")
    tmp.replace(cache)
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ROOT/"outputs/humanoid_light_bfm")
    cli.add_argument("--workers", type=int, default=6)
    cli.add_argument("--min-frames", type=int, default=25, help="At least 0.5 s per original contiguous run")
    cli.add_argument("--limit", type=int, help="Per-source limit for development")
    args = cli.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    version = sha256(__file__)+sha256(ROOT/"scripts/humanoid_light_collision.py")
    version += sha256(ROOT/"scripts/optimize_humanoid_light_cmu_landing.py")
    version += sha256(ROOT/"src/robots/humanoid/humanoid_light/assets/humanoid_light.xml")
    version += str(args.min_frames)
    jobs = []
    for source, folder in (("cmu", "humanoid_light_cmu_full"), ("lafan", "humanoid_light_lafan_full")):
        directory = ROOT/"outputs"/folder
        with (directory/"sequence_index.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        if args.limit:
            rows = rows[:args.limit]
        for row in rows:
            if source == "cmu" and row.get("source_timing_verified", "").lower() not in ("true", "1"):
                continue
            name = row["name"]
            subject = int(row["subject"]) if source == "cmu" else int(name.rsplit("_subject", 1)[1])
            split = "validation" if (subject % 10 == 0 if source == "cmu" else subject == 5) else "train"
            path = directory/"motions"/(name+".npz")
            jobs.append(dict(path=str(path.relative_to(ROOT)), source=source, subject=subject,
                             split=split, name=name, min_frames=args.min_frames,
                             cache=str((out/"screens"/source/(name+".json")).relative_to(ROOT)),
                             screen_version=version))
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(screen, job): job for job in jobs}
        for future in as_completed(futures):
            record = future.result(); records.append(record)
            if len(records) % 25 == 0 or len(records) == len(jobs):
                print(json.dumps(dict(screened=len(records), total=len(jobs),
                    accepted_frames=sum(row["accepted_frames"] for row in records))), flush=True)
    splits = {"train": [], "validation": []}
    for row in sorted(records, key=lambda item: (item["source"], item["name"])):
        for start, stop in row["runs"]:
            splits[row["split"]].append({key: row[key] for key in ("path", "sha256", "source", "subject", "name")} |
                                        dict(start_frame=start, stop_frame=stop))
    manifest = dict(version=1, robot="humanoid_light", fps=50, screen_version=version,
                    min_frames=args.min_frames, splits=splits, dynamic_feasibility_verified=False,
                    screening="original mesh hulls; stored poses + quarter-frame SLERP; 1 mm floor tolerance; 50 um body clearance",
                    notes=["Invalid intervals split clips; original files and qvel are preserved.",
                           "Subject-disjoint validation; CMU subject IDs divisible by 10 and LAFAN subject 5.",
                           "CMU subject IDs can refer to repeat performers; this is not a performer-disjoint certificate.",
                           "Timing-unverified CMU clips excluded. Props/partners/terrain are not reconstructed."])
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    stats = {split: {source: dict(segments=sum(r["source"] == source for r in rows),
                    files=len({r["path"] for r in rows if r["source"] == source}),
                    frames=sum(r["stop_frame"]-r["start_frame"] for r in rows if r["source"] == source))
                    for source in ("cmu", "lafan")} for split, rows in splits.items()}
    (out/"summary.json").write_text(json.dumps(stats, indent=2)+"\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
