"""Locate the checked-out FlashSAC source without installing its simulator extras."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
FLASH_ROOT = ROOT / "third_party" / "FlashSAC"


def enable_flashsac():
    if not (FLASH_ROOT / "flash_rl" / "agents" / "flashSAC" / "agent.py").exists():
        raise RuntimeError(
            "FlashSAC source missing. Run git submodule update --init --recursive"
        )
    if str(FLASH_ROOT) not in sys.path:
        sys.path.insert(0, str(FLASH_ROOT))
