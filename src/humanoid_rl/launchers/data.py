"""Enumerate portable experiment inputs without rewriting their contents."""

from pathlib import Path

from humanoid_rl.launchers.config import read_json
from humanoid_rl.launchers.source import sha256


def collect_inputs(plan, *, verify=True):
    root = Path(plan.paths["data_root"]).resolve()
    files, digests = {}, {}

    def add(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Dataset path escapes data_root: {path}")
        if not path.is_file():
            raise ValueError(f"Missing dataset file: {path}")
        relative = path.relative_to(root).as_posix()
        if verify:
            if relative not in digests:
                digests[relative] = sha256(path)
            if expected and expected != digests[relative]:
                raise ValueError(f"Screened motion checksum differs: {path}")
        files[relative] = path

    for item in plan.inputs:
        path = Path(item["path"])
        kind = item.get("kind", "file")
        if kind == "directory":
            if not path.is_dir():
                raise ValueError(f"Missing input directory: {path}")
            for child in sorted(path.rglob("*")):
                if child.is_file() and not any(p.startswith(".") for p in child.relative_to(path).parts):
                    add(child)
        elif kind in ("file", "motion_manifest"):
            add(path)
            if kind == "motion_manifest":
                for records in read_json(path)["splits"].values():
                    for record in records:
                        relative = Path(record["path"])
                        if relative.is_absolute():
                            raise ValueError("Portable motion manifests require relative paths")
                        add(Path(item.get("root", root)) / relative, record["sha256"])
        else:
            raise ValueError(f"Unknown input kind: {kind}")
    return files, digests
