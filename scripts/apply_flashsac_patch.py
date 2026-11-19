"""Apply the small, tracked mjlab integration patch to the FlashSAC submodule."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "FlashSAC"
PATCH = ROOT / "patches" / "flashsac-mjlab.patch"


def main():
    if not (SOURCE / "flash_rl").is_dir():
        raise SystemExit("Run git submodule update --init --recursive first")
    command = ["git", "-C", str(SOURCE), "apply"]
    if (
        subprocess.run(
            command + ["--reverse", "--check", str(PATCH)], capture_output=True
        ).returncode
        == 0
    ):
        print("FlashSAC mjlab patch is already applied.")
        return
    check = subprocess.run(
        command + ["--check", str(PATCH)], capture_output=True, text=True
    )
    if check.returncode:
        raise SystemExit(
            "Patch does not match this checkout; review local changes.\n" + check.stderr
        )
    subprocess.run(command + [str(PATCH)], check=True)
    print("Applied FlashSAC mjlab patch.")


if __name__ == "__main__":
    main()
