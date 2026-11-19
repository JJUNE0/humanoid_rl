"""Source snapshots include working changes and exclude training data/secrets."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

SOURCE_SUFFIXES = {".py", ".json", ".toml", ".lock", ".yaml", ".yml", ".md", ".txt",
                   ".patch", ".sh", ".xml", ".urdf", ".stl", ".obj", ".dae", ".mtl", ".png"}
SKIP_DIRS = {".git", ".venv", ".venv_super_mimic", "__pycache__", "artifacts", ".downloads",
             ".pytest_cache", ".ruff_cache", "node_modules"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance(root):
    root = Path(root)
    saved = root / ".source.json"
    if saved.is_file() and not (root / ".git").exists():
        return json.loads(saved.read_text())
    versions = {}
    for relative in (".", "third_party/BFM-Zero", "third_party/FlashSAC", "third_party/supermimic", "third_party/mjlab_research"):
        directory = root / relative
        if not (directory / ".git").exists():
            continue
        def git(*args):
            return subprocess.check_output(["git", "-C", str(directory), *args], text=True, timeout=10).strip()
        try:
            versions[relative] = dict(revision=git("rev-parse", "HEAD"),
                                      tracked_changes=bool(git("status", "--porcelain", "--untracked-files=no")))
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            versions[relative] = {"revision": None}
    return {"repositories": versions}


def source_files(root):
    root = Path(root).resolve()
    prefixes = ("src", "configs", "environments", "patches", "scripts", "docs",
                "third_party/BFM-Zero/humanoidverse/agents",
                "third_party/BFM-Zero/humanoidverse/utils",
                "third_party/BFM-Zero/humanoidverse/config",
                "third_party/FlashSAC/flash_rl")
    prefixes += ("third_party/supermimic/src", "third_party/mjlab_research/src")
    files = set()
    for prefix in prefixes:
        directory = root / prefix
        if not directory.exists():
            continue
        for current, children, names in os.walk(directory):
            children[:] = [name for name in children if name not in SKIP_DIRS and not name.startswith(".")
                           and not name.endswith(".egg-info") and not (Path(current) / name).is_symlink()]
            for name in names:
                path = Path(current) / name
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    continue
                if path.suffix.lower() in SOURCE_SUFFIXES or path.name.upper().startswith(("LICENSE", "COPYING")):
                    files.add(path)
    for name in ("humanoid.py", "train.py", "play.py", "pyproject.toml", "uv.lock", ".python-version", "README.md",
                 "third_party/BFM-Zero/humanoidverse/__init__.py",
                 "third_party/BFM-Zero/LICENSE", "third_party/FlashSAC/LICENSE"):
        path = root / name
        if path.is_file():
            files.add(path)
    for repository in ("supermimic", "mjlab_research"):
        for name in ("pyproject.toml", "uv.lock", "LICENSE", "README.md"):
            path = root / "third_party" / repository / name
            if path.is_file():
                files.add(path)
    return sorted(files)


def describe(root):
    saved = root / ".source.json"
    if saved.exists() and not (root / ".git").exists():
        return json.loads(saved.read_text())
    files = {path.relative_to(root).as_posix(): sha256(path) for path in source_files(root)}
    return provenance(root) | {"files": files, "source_sha256": hashlib.sha256(
        json.dumps(files, sort_keys=True).encode()).hexdigest()}


def snapshot(root, output):
    files = source_files(root)
    if not files:
        raise ValueError("No source files found")
    fingerprints = {}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(root).as_posix()
            # Read once so the recorded hash describes exactly the packaged code.
            contents = path.read_bytes()
            fingerprints[relative] = hashlib.sha256(contents).hexdigest()
            archive.writestr(relative, contents)
        metadata = provenance(root) | {"files": fingerprints}
        metadata["source_sha256"] = hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest()
        archive.writestr(".source.json", json.dumps(metadata, indent=2))
    return metadata
