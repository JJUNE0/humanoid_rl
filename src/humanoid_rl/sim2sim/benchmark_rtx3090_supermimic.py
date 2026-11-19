"""Grid-search optimal (num_envs, batch_size, updates_per_step, compile) on local RTX 3090.

Preserves exact SuperMimic UTD ratio (replay_ratio = 16, policy_ratio = 32) starting from
the Stage-1 checkpoint, requires >= 3.5 GiB free VRAM safety margin, and outputs JSON results.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from humanoid_rl.launchers.config import load_named, resolve, write_json

RESUME_CKPT = (
    REPO_ROOT
    / "runs/light_supermimic_lafan_all_t4/t4_lafan_all_stage1_20261118/training/checkpoints/step_5632000"
)
OUT_DIR = REPO_ROOT / "outputs/sim2sim/bench_3090_grid"

# All candidates preserve exact UTD: (updates_per_step * 2 * batch_size * 8) / num_envs == 16.0
CANDIDATES = [
    {"name": "e1024_b1024_u1", "num_envs": 1024, "batch_size": 1024, "updates": 1, "compile": False},
    {"name": "e2048_b1024_u2", "num_envs": 2048, "batch_size": 1024, "updates": 2, "compile": False},
    {"name": "e2048_b2048_u1", "num_envs": 2048, "batch_size": 2048, "updates": 1, "compile": False},
    {"name": "e3072_b1536_u2", "num_envs": 3072, "batch_size": 1536, "updates": 2, "compile": False},
    {"name": "e4096_b2048_u2", "num_envs": 4096, "batch_size": 2048, "updates": 2, "compile": False},
]


def run_candidate(base_settings: dict, initial_steps: int, cand: dict) -> dict:
    name = cand["name"]
    c_dir = OUT_DIR / name
    c_dir.mkdir(parents=True, exist_ok=True)
    cfg = deepcopy(base_settings)
    num_envs = cand["num_envs"]
    measure_iters = 25
    total_iters = cfg["training"]["warmup_steps"] + 8 + measure_iters
    cfg["training"].update(
        num_envs=num_envs,
        updates_per_step=cand["updates"],
        compile=cand["compile"],
        cuda_graphs=True,
        muon_dtype="bfloat16",
        total_env_steps=initial_steps + num_envs * total_iters,
        save_every=2**60,
        log_every=20,
    )
    cfg["runner"]["wm"]["batch_size"] = cand["batch_size"]
    cfg["runner"]["policy"]["batch_size"] = cand["batch_size"]

    cfg_path = c_dir / "settings.json"
    write_json(cfg_path, cfg)

    cmd = [
        sys.executable,
        "-m",
        "humanoid_rl.algorithm.supermimic.train",
        "--config",
        str(cfg_path),
        "--output",
        str(c_dir / "run"),
        "--resume",
        str(RESUME_CKPT),
        "--benchmark",
    ]
    print(f"\n=== Benchmarking {name} ===", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    (c_dir / "stdout.log").write_text(proc.stdout + "\n" + proc.stderr)

    perf_path = c_dir / "run/performance.json"
    res = dict(cand, exit_code=proc.returncode)
    if proc.returncode == 0 and perf_path.is_file():
        perf = json.loads(perf_path.read_text())
        res.update(
            env_steps_per_second=round(perf["env_steps_per_second"], 2),
            gpu_free_mib=round(perf["gpu_free_mib"], 1),
            gpu_peak_torch_mib=round(perf["gpu_peak_torch_mib"], 1),
            simulator_seconds=round(perf["simulator_seconds"], 2),
            learner_seconds=round(perf["learner_seconds"], 2),
            cold_start_seconds=round(perf["cold_start_and_warmup_seconds"], 2),
            safe=perf["gpu_free_mib"] >= 3500.0,
        )
        print(
            f"[OK] {name}: {res['env_steps_per_second']} steps/s | free_vram={res['gpu_free_mib']} MiB | safe={res['safe']}",
            flush=True,
        )
    else:
        res["safe"] = False
        tail = "\n".join((proc.stdout + "\n" + proc.stderr).splitlines()[-8:])
        res["error_tail"] = tail
        print(f"[FAIL] {name} (code={proc.returncode}):\n{tail}", flush=True)
    return res


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plan = resolve(
        load_named("light_supermimic_lafan_all", "experiments"),
        load_named("local", "runtimes"),
        run_id="grid_3090",
    )
    init_steps = json.loads((RESUME_CKPT / "run.json").read_text())["progress"]["env_steps"]

    results = []
    for cand in CANDIDATES:
        r = run_candidate(plan.settings, init_steps, cand)
        results.append(r)

    safe_results = [r for r in results if r.get("safe")]
    if safe_results:
        best = max(safe_results, key=lambda x: x["env_steps_per_second"])
        # Also test torch.compile=True on the winning shape!
        comp_cand = dict(
            name=best["name"] + "_compiled",
            num_envs=best["num_envs"],
            batch_size=best["batch_size"],
            updates=best["updates"],
            compile=True,
        )
        r_comp = run_candidate(plan.settings, init_steps, comp_cand)
        results.append(r_comp)
        if r_comp.get("safe") and r_comp["env_steps_per_second"] > best["env_steps_per_second"]:
            best = r_comp
    else:
        best = None

    summary = {"candidates": results, "selected": best}
    out_file = OUT_DIR / "summary.json"
    out_file.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== FINAL WINNER ===")
    print(json.dumps(best, indent=2), flush=True)


if __name__ == "__main__":
    main()
