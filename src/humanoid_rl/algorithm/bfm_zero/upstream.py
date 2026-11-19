"""Import the original BFM-Zero learning core without its Isaac dependencies."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
BFM_ROOT = ROOT/"third_party/BFM-Zero"


def enable_bfm_zero():
    if not (BFM_ROOT/"humanoidverse/agents/fb_cpr_aux/agent.py").is_file():
        raise RuntimeError("BFM-Zero source missing; initialize third_party/BFM-Zero")
    if str(BFM_ROOT) not in sys.path:
        sys.path.insert(0, str(BFM_ROOT))
