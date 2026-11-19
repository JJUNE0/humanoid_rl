"""Repeat the full scan, then check flagged conditions sequentially."""

from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/humanoid_light_nan_scan"
PYTHON=ROOT/".venv/bin/python"


def write_status(data):
    p=OUT/"validation_status.json";t=p.with_suffix(".tmp")
    t.write_text(json.dumps(data,indent=2)+"\n");t.replace(p)


def main():
    first=json.loads((OUT/"scan_5ms/summary.json").read_text())
    if first["state"]!="complete":raise RuntimeError("Initial scan must finish first")
    status=dict(state="running",started=datetime.now().isoformat())

    def stage(name,options):
        directory=OUT/name;directory.mkdir(exist_ok=True)
        status.update(stage=name,stage_started=datetime.now().isoformat());write_status(status)
        cmd=[str(PYTHON),"-u",str(ROOT/"scripts/scan_humanoid_light_nan.py"),"--output",str(directory),*options]
        with (directory/"scan.log").open("a") as f:
            subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)

    try:
        stage("repeat_5ms",[])
        union={r["key"]:r for r in json.loads((OUT/"scan_5ms/candidates.json").read_text())}
        union.update({r["key"]:r for r in json.loads((OUT/"repeat_5ms/candidates.json").read_text())})
        path=OUT/"union_candidates.json";path.write_text(json.dumps(list(union.values()),indent=2)+"\n")
        stage("check_2p5ms",["--candidates",str(path),"--dt",".0025"])
        stage("check_nj1024",["--candidates",str(path),"--njmax","1024"])
        subprocess.run([str(PYTHON),str(ROOT/"scripts/summarize_humanoid_light_nan_scan.py"),
                        "--root",str(OUT)],cwd=ROOT,check=True)
        status.update(state="complete",finished=datetime.now().isoformat());write_status(status)
    except BaseException as error:
        status.update(state="stopped",error=str(error),finished=datetime.now().isoformat());write_status(status)
        raise


if __name__=="__main__":main()
