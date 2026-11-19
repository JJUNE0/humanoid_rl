"""Join baseline, repeated scan and parameter checks into window/file reports."""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def load(folder):
    p=folder/"results.jsonl"
    return {r["key"]:r for r in map(json.loads,p.read_text().splitlines())} if p.exists() else {}


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root",type=Path,default=Path("outputs/humanoid_light_nan_scan"))
    args=cli.parse_args();root=args.root
    primary=load(root/"scan_5ms")
    repeated=load(root/"repeat_5ms")
    half=load(root/"check_2p5ms")
    capacity=load(root/"check_nj1024")
    if not repeated or len(primary)!=len(repeated):
        raise RuntimeError("Both full 5 ms scans must finish before summarization")
    joined=[]
    per_file=defaultdict(list)
    for key,row in repeated.items():
        first=primary[key]
        candidate=row["outcome"]!="finite" or first["outcome"]!="finite"
        item={k:row[k] for k in ("key","path","source","split","frame","mode","horizon")}
        item.update(first_outcome=first["outcome"],repeat_outcome=row["outcome"],
                    repeated_nan=first["outcome"]==row["outcome"]=="nan",
                    repeated_instability=first["outcome"] in ("nan","divergence") and row["outcome"] in ("nan","divergence"),
                    max_joint_speed_rad_s=row["max_joint_speed_rad_s"],
                    max_qacc=row["max_qacc"],max_nefc=row.get("max_nefc"),
                    overflow_flags=row.get("overflow_flags"),
                    first_failure_time_s=first.get("failure_time_s"),
                    repeat_failure_time_s=row.get("failure_time_s"),
                    outcome_2p5ms=half.get(key,{}).get("outcome"),
                    outcome_nj1024=capacity.get(key,{}).get("outcome"))
        per_file[row["path"]].append(item)
        if candidate:joined.append(item)
    joined.sort(key=lambda r:(not r["repeated_nan"],not r["repeated_instability"],r["mode"]!="reference",r["path"],r["frame"]))
    write_csv(root/"suspect_windows.csv",joined)
    files=[]
    for path,rows in per_file.items():
        bad=[r for r in rows if r["first_outcome"]!="finite" or r["repeat_outcome"]!="finite"]
        if not bad:continue
        files.append(dict(path=path,source=rows[0]["source"],split=rows[0]["split"],
                          tested_windows=len(rows),flagged_conditions=len(bad),
                          repeated_nan=sum(r["repeated_nan"] for r in bad),
                          repeated_instability=sum(r["repeated_instability"] for r in bad),
                          nan_in_both_scans=any(r["first_outcome"]=="nan" for r in bad) and any(r["repeat_outcome"]=="nan" for r in bad),
                          reference_nan=sum(r["repeat_outcome"]=="nan" and r["mode"]=="reference" for r in bad),
                          random_nan=sum(r["repeat_outcome"]=="nan" and r["mode"]=="random" for r in bad),
                          capacity_overflow=sum(bool(r["overflow_flags"]) for r in bad),
                          finite_at_2p5ms=sum(r["outcome_2p5ms"]=="finite" for r in bad),
                          finite_at_nj1024=sum(r["outcome_nj1024"]=="finite" for r in bad)))
    files.sort(key=lambda r:(-r["reference_nan"],-r["repeated_nan"],-r["repeated_instability"],-r["random_nan"],r["path"]))
    write_csv(root/"suspect_files.csv",files)
    summary=dict(files_tested=len(per_file),tests_per_full_scan=len(repeated),
                 first_outcomes=dict(Counter(r["outcome"] for r in primary.values())),
                 repeat_outcomes=dict(Counter(r["outcome"] for r in repeated.values())),
                 repeat_outcomes_by_mode={m:dict(Counter(r["outcome"] for r in repeated.values() if r["mode"]==m)) for m in ("reference","random")},
                 suspect_conditions=len(joined),suspect_files=len(files),
                 repeated_nan_conditions=sum(r["repeated_nan"] for r in joined),
                 repeated_nan_files=sum(r["repeated_nan"]>0 for r in files),
                 repeated_unstable_conditions=sum(r["repeated_instability"] for r in joined),
                 repeated_unstable_files=sum(r["repeated_instability"]>0 for r in files),
                 nan_in_both_scans_files=sum(r["nan_in_both_scans"] for r in files),
                 checked_2p5ms=len(half),outcomes_2p5ms=dict(Counter(r["outcome"] for r in half.values())),
                 checked_nj1024=len(capacity),outcomes_nj1024=dict(Counter(r["outcome"] for r in capacity.values())),
                 notes=["Conditional finite-horizon physics tests; flags do not certify bad motion data or long-run safety.",
                        "Accepted train and validation intervals only. Three start positions per ten-second chunk where possible.",
                        "No artificial lying resets. Reference position tracking and deterministic full-range random commands are separate protocols.",
                        "Candidate-only parameter checks retain world positions and action seeds, but change cohort composition; GPU reductions/contact ordering can change outcomes.",
                        "Training inputs/manifests have not been changed or filtered by these results."])
    manifest=Path("outputs/humanoid_light_bfm/manifest.json")
    summary["manifest_sha256"]=hashlib.sha256(manifest.read_bytes()).hexdigest()
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
