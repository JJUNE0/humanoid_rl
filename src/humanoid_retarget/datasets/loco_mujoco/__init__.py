"""Local copies of LocoMuJoCo's default retargeted mocap motions."""

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
H1_DIR = DATASET_DIR / "mocap/h1"


def h1_sequences():
    return sorted(H1_DIR.glob("*.npz"))
