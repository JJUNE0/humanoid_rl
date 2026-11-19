"""Copy pinned H1 assets and motion datasets into this source tree.

The resulting runtime files are regular local files: no symlinks, Hugging Face
cache paths or third-party source paths are needed to load them.
"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
H1 = ROOT / "src/robots/humanoid/h1"
DATASETS = ROOT / "src/humanoid_retarget/datasets"
MODEL_REVISION = "1aeea8b21cb160ff9f24d9ee8198661c9591f245"
LAFAN_REVISION = "94084601bacdf9cc3764b5c73daaeccae6035fac"
HF_REVISION = "75479db3e07a86a4477982ca8a706b8a45828974"
HF_REPO = "robfiras/loco-mujoco-datasets"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "humanoid-rl-asset-setup"}),
        timeout=60,
    )


def get_json(url):
    with request(url) as response:
        return json.load(response)


def download(url, destination, expected_size=None):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and expected_size and destination.stat().st_size == expected_size:
        return destination
    temporary = destination.with_name(destination.name + ".part")
    try:
        for attempt in range(3):
            try:
                with request(url) as response, temporary.open("wb") as stream:
                    shutil.copyfileobj(response, stream, length=1024 * 1024)
                if expected_size is not None and temporary.stat().st_size != expected_size:
                    raise ValueError(f"Unexpected download size: {destination}")
                with temporary.open("rb") as stream:
                    if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                        raise ValueError(f"Downloaded an LFS pointer instead of data: {destination}")
                temporary.replace(destination)
                return destination
            except Exception:
                if attempt == 2:
                    raise
    finally:
        temporary.unlink(missing_ok=True)


def record(path, relative_to, **extra):
    return {"path": path.relative_to(relative_to).as_posix(),
            "bytes": path.stat().st_size, "sha256": sha256(path), **extra}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def copy_model():
    repository = "robfiras/loco-mujoco-models"
    tree = get_json(f"https://api.github.com/repos/{repository}/git/trees/{MODEL_REVISION}?recursive=1")["tree"]
    selected = [entry for entry in tree if entry["type"] == "blob" and any(
        entry["path"].startswith(f"loco_mujoco_models/{prefix}/")
        for prefix in ("unitree_h1", "common"))]

    def fetch(entry):
        source = PurePosixPath(entry["path"])
        relative = source.relative_to("loco_mujoco_models/unitree_h1") if "unitree_h1" in source.parts else source.relative_to("loco_mujoco_models")
        url = f"https://raw.githubusercontent.com/{repository}/{MODEL_REVISION}/{source}"
        path = H1 / relative
        # h1.xml has two intentionally rewritten local include paths; restore
        # the original before applying those changes when running setup again.
        download(url, path, entry["size"])
        return record(path, H1, source_url=url, source_git_blob=entry["sha"])

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, selected))
    xml = H1 / "h1.xml"
    original = xml.read_text()
    xml.write_text(original.replace('file="../common/', 'file="common/'))
    copyright_notice = original.split("<!--", 1)[1].split("Modified by", 1)[0].strip()
    (H1 / "LICENSE.Unitree").write_text(copyright_notice + "\n")
    shutil.copyfile(ROOT / "third_party/loco-mujoco/LICENSE", H1 / "LICENSE.LocoMuJoCo")
    for entry in records:
        path = H1 / entry["path"]
        entry["source_sha256"] = entry["sha256"]
        entry.update(sha256=sha256(path), bytes=path.stat().st_size)
    write_json(H1 / "manifest.json", {
        "robot": "Unitree H1 (19 actuated joints)", "repository": repository,
        "revision": MODEL_REVISION,
        "local_changes": ["h1.xml include paths ../common/ -> common/; model dynamics unchanged"],
        "files": records,
    })
    print(f"H1: copied {len(records)} model/asset files", flush=True)


def copy_raw_lafan():
    root = DATASETS / "lafan1"
    repository = "ubisoft/ubisoft-laforge-animation-dataset"
    raw = f"https://raw.githubusercontent.com/{repository}/{LAFAN_REVISION}"
    download(f"{raw}/license.txt", root / "LICENSE")
    download(f"{raw}/README.md", root / "SOURCE_README.md")
    archive_url = f"https://media.githubusercontent.com/media/{repository}/{LAFAN_REVISION}/lafan1/lafan1.zip"
    archive = download(archive_url, root / ".downloads/lafan1.zip", 144051503)
    records = []
    with zipfile.ZipFile(archive) as zipped:
        for item in sorted(zipped.infolist(), key=lambda item: item.filename):
            if not item.filename.lower().endswith(".bvh") or "__MACOSX" in item.filename:
                continue
            path = root / "raw" / PurePosixPath(item.filename).name
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(item) as source, path.open("wb") as target:
                shutil.copyfileobj(source, target)
            frames = frame_time = None
            with path.open() as stream:
                for line in stream:
                    if line.startswith("Frames:"):
                        frames = int(line.split(":", 1)[1])
                    elif line.startswith("Frame Time:"):
                        frame_time = float(line.split(":", 1)[1])
                        break
            if frames is None or frame_time is None:
                raise ValueError(f"Missing BVH motion header: {path}")
            records.append(record(path, root, frames=frames, fps=1/frame_time))
    write_json(root / "raw_manifest.json", {
        "dataset": "LAFAN1", "representation": "original human BVH", "repository": repository,
        "revision": LAFAN_REVISION, "archive_url": archive_url,
        "archive_sha256": sha256(archive), "license": "CC-BY-NC-ND-4.0",
        "sequence_count": len(records), "frame_count": sum(item["frames"] for item in records),
        "files": records,
    })
    print(f"LAFAN1 raw: copied {len(records)} BVH sequences", flush=True)


def copy_hf_motions(source_prefix, destination, name):
    import numpy as np

    entries = get_json(f"https://huggingface.co/api/datasets/{HF_REPO}/tree/{HF_REVISION}/{source_prefix}")
    entries = sorted([entry for entry in entries if entry["type"] == "file" and entry["path"].endswith(".npz")], key=lambda entry: entry["path"])

    def fetch(entry):
        url = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/{entry['path']}?download=true"
        path = download(url, destination / PurePosixPath(entry["path"]).name, entry["size"])
        with np.load(path, allow_pickle=False) as motion:
            qpos, qvel = motion["qpos"], motion["qvel"]
            frequency, joint_names = float(motion["frequency"]), motion["joint_names"].tolist()
            if qpos.ndim != 2 or qpos.shape[1] != 26 or qvel.shape != (len(qpos), 25):
                raise ValueError(f"Unexpected H1 motion shape: {path}")
            if not np.isfinite(qpos).all() or not np.isfinite(qvel).all():
                raise ValueError(f"Nonfinite H1 motion: {path}")
        return record(path, destination, source_url=url, frames=len(qpos), fps=frequency,
                      joint_names=joint_names, qpos_dim=26, qvel_dim=25)

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, entries))
    write_json(destination / "manifest.json", {
        "dataset": name, "robot": "UnitreeH1", "representation": "retargeted robot qpos/qvel",
        "repository": HF_REPO, "revision": HF_REVISION, "license": "CC-BY-NC-ND-4.0",
        "sequence_count": len(records), "frame_count": sum(item["frames"] for item in records),
        "duration_seconds": sum(item["frames"]/item["fps"] for item in records), "files": records,
    })
    print(f"{name}: copied {len(records)} H1 motions", flush=True)


def copy_g1_motions():
    source = ROOT / "third_party/BFM-Zero/humanoidverse/data"
    destination = DATASETS / "lafan1/retargeted/g1"
    records = []
    for name in ("lafan_29dof.pkl", "lafan_29dof_10s-clipped.pkl"):
        original = source / name
        with original.open("rb") as stream:
            if stream.read(80).startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"The user must download BFM-Zero LFS data first: {original}")
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / name
        shutil.copyfile(original, path)
        if sha256(path) != sha256(original) or path.stat().st_ino == original.stat().st_ino:
            raise ValueError(f"G1 motion must be an independent, exact copy: {path}")
        records.append(record(path, destination, source_repository="LeCAR-Lab/BFM-Zero",
                              source_path=f"humanoidverse/data/{name}"))
    write_json(destination / "manifest.json", {
        "dataset": "LAFAN1", "robot": "Unitree G1 29 DoF",
        "representation": "BFM-Zero joblib motion dictionaries; not H1 motions",
        "files": records,
    })
    print("LAFAN1 G1: copied both existing BFM-Zero motion files", flush=True)


def main():
    copy_model()
    # Independent downloads can run concurrently; each writes its own directory.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(copy_raw_lafan),
                   pool.submit(copy_hf_motions, "Lafan1/mocap/UnitreeH1", DATASETS/"lafan1/retargeted/h1", "LAFAN1"),
                   pool.submit(copy_hf_motions, "DefaultDatasets/mocap/UnitreeH1", DATASETS/"loco_mujoco/mocap/h1", "LocoMuJoCo default mocap")]
        for future in futures:
            future.result()
    copy_g1_motions()
    root = DATASETS / "loco_mujoco"
    download(f"https://huggingface.co/datasets/{HF_REPO}/raw/{HF_REVISION}/README.md", root/"SOURCE_README.md")
    download("https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode.txt", root/"LICENSE")
    print("All runtime assets and motions are stored locally.", flush=True)


if __name__ == "__main__":
    main()
