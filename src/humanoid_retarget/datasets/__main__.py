"""Print the local motion inventory without contacting any external service."""

import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    sources = (
        ("LAFAN1 human BVH", "lafan1/raw_manifest.json"),
        ("LAFAN1 H1", "lafan1/retargeted/h1/manifest.json"),
        ("LAFAN1 G1 (two export variants)", "lafan1/retargeted/g1/manifest.json"),
        ("LocoMuJoCo default H1", "loco_mujoco/mocap/h1/manifest.json"),
    )
    for name, relative in sources:
        manifest = root / relative
        if not manifest.is_file():
            print(f"{name}: not installed")
            continue
        value = json.loads(manifest.read_text())
        base = root/"lafan1" if relative.endswith("raw_manifest.json") else manifest.parent
        present = sum((base/item["path"]).is_file() for item in value["files"])
        size = sum(item["bytes"] for item in value["files"])
        frequencies = sorted({round(item["fps"], 3) for item in value["files"] if "fps" in item})
        rate = f", {frequencies} Hz" if frequencies else ""
        print(f"{name}: {present}/{len(value['files'])} files, {size/2**20:.1f} MiB{rate}")


if __name__ == "__main__":
    main()
