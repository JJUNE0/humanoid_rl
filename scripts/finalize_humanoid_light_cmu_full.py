"""Automatically verify and render the full CMU batch when conversion ends."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/humanoid_light_cmu_full'


def state(stage,**values):
    path=OUT/'finalization.json';temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(dict(stage=stage,**values),indent=2)+'\n');temporary.replace(path)
    print(stage,flush=True)


def main():
    os.nice(5);state('waiting_for_conversion',pid=os.getpid())
    while True:
        manifest=json.loads((OUT/'manifest.json').read_text())
        if manifest['run_complete']:
            if manifest['failed_sequences']:
                state('waiting_for_processing_failure_retry',failed_sequences=manifest['failed_sequences'])
            else:break
        time.sleep(10)
    try:
        state('validating_outputs')
        subprocess.run([sys.executable,str(ROOT/'scripts/summarize_humanoid_light_cmu_full.py'),str(OUT)],check=True)
        state('rendering_showcase')
        subprocess.run([sys.executable,str(ROOT/'scripts/render_humanoid_light_cmu_full.py'),'--dataset',str(OUT)],check=True)
        state('complete',summary='summary.json',sequence_index='sequence_index.csv',
              video='previews/showcase/cmu_showcase.mp4')
    except Exception as exc:
        state('failed',error=str(exc));raise


if __name__=='__main__':main()
