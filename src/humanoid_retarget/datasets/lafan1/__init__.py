"""Local copies of original LAFAN1 BVH and H1/G1 retargeted motion data."""

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parent
RAW_DIR = DATASET_DIR / "raw"
H1_DIR = DATASET_DIR / "retargeted/h1"
G1_DIR = DATASET_DIR / "retargeted/g1"


def raw_sequences():
    return sorted(RAW_DIR.glob("*.bvh"))


def h1_sequences():
    return sorted(H1_DIR.glob("*.npz"))
