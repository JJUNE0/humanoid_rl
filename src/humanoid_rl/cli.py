"""One front door for experiments, method environments, and execution locations."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

from humanoid_rl.launchers.config import ROOT, load_named, read_json, resolve
from humanoid_rl.methods import METHODS


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List available methods, task combinations and profiles")
    listing.add_argument("--json", action="store_true")
    setup = commands.add_parser("setup", help="Install one method's locked environment")
    setup.add_argument("--method", required=True, choices=METHODS)
    setup.add_argument("--task", help="Select a task-specific environment (e.g. Light SuperMimic)")
    setup.add_argument("--dry-run", action="store_true")
    for name, help_text in (("launch", "Run locally, submit to Colab, or export a bundle"),
                            ("train", "Train on the current machine using a runtime profile")):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--experiment", required=True, help="Name in configs/experiments, or a JSON path")
        command.add_argument("--runtime", default="local", help="local, colab, or a runtime JSON path")
        command.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                             help="Override method settings, e.g. training.num_envs=64 (applied last)")
        command.add_argument("--run-id", help="Unique run identifier (UTC timestamp by default)")
        command.add_argument("--resume", type=Path, help="Method checkpoint; continues into a new run folder")
        command.add_argument("--python", help="Override the selected method interpreter")
        command.add_argument("--device")
        for option in ("data-root", "cache-root", "run-root", "persistent-root"):
            command.add_argument("--" + option, help="Override the runtime path")
        command.add_argument("--dry-run", action="store_true", help="Print a plan without creating files or running a trainer")
        if name == "launch":
            command.add_argument("--output", type=Path, help="Colab bundle destination")
            command.add_argument("--bundle-data", action="store_true", help="Include declared input files in the Colab bundle")
            command.add_argument("--submit", action="store_true", help="Run directly through the Colab CLI, without a notebook")
            command.add_argument("--colab-gpu", choices=("T4", "L4", "G4", "A100", "H100"),
                                 help="GPU for a new dedicated Colab session (default: T4)")
            command.add_argument("--results-root", type=Path, help="Local results directory for --submit (default: runs)")
    return cli


def inventory():
    return {
        "methods": {name: {"environment": method.environment, "tasks": method.tasks,
                           "task_environments": method.task_environments,
                           "description": method.description} for name, method in METHODS.items()},
        "experiments": {path.stem: {key: value for key, value in read_json(path).items()
                                    if key in ("method", "task", "robot", "description")}
                        for path in sorted((ROOT / "configs/experiments").glob("*.json"))},
        "runtimes": [path.stem for path in sorted((ROOT / "configs/runtimes").glob("*.json"))],
    }


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        if args.command == "list":
            data = inventory()
            if args.json:
                print(json.dumps(data, indent=2))
            else:
                for name, method in data["methods"].items():
                    print(f"{name} [{method['environment']}]: {', '.join(method['tasks'])}")
                    print(f"  {method['description']}")
                    for task, environment in method.get("task_environments", {}).items():
                        print(f"  {task}: environment={environment}")
                print("\nExperiments: " + ", ".join(data["experiments"]))
                print("Runtimes: " + ", ".join(data["runtimes"]))
            return 0
        if args.command == "setup":
            from humanoid_rl.launchers.environment import setup, setup_commands
            if args.dry_run:
                print(json.dumps(setup_commands(ROOT, args.method, args.task), indent=2))
            else:
                setup(ROOT, args.method, args.task)
            return 0
        experiment = load_named(args.experiment, "experiments")
        runtime = load_named(args.runtime, "runtimes")
        submit = args.command == "launch" and args.submit
        if submit and runtime["name"] != "colab":
            raise ValueError("--submit requires --runtime colab")
        if args.command == "launch" and (args.colab_gpu or args.results_root) and not submit:
            raise ValueError("--colab-gpu and --results-root require --submit")
        if submit:
            if args.persistent_root:
                raise ValueError("--submit retrieves results locally; use --results-root instead of --persistent-root")
            runtime["persistent_root"] = None
        for name in ("device", "data_root", "cache_root", "run_root", "persistent_root"):
            if getattr(args, name) is not None:
                runtime[name] = getattr(args, name)
        plan = resolve(experiment, runtime, overrides=args.set, run_id=args.run_id, resume=args.resume, python=args.python)
        export = args.command == "launch" and runtime["name"] == "colab"
        if args.command == "launch" and not export and (args.bundle_data or args.output):
            raise ValueError("--bundle-data and --output apply to Colab export only")
        if export and args.python:
            raise ValueError("Colab creates its own method environment; omit --python when exporting")
        if args.dry_run:
            action = "submit_colab" if submit else "export_colab" if export else "train"
            details = {"colab_gpu": args.colab_gpu or "T4",
                       "results_root": str(args.results_root or ROOT / "runs")} if submit else {}
            print(json.dumps(plan.snapshot() | {"action": action} | details, indent=2))
            return 0
        if export:
            destination = args.output or ROOT / "runs/colab" / (experiment["name"] + "_" + plan.run_id)
            if submit:
                from humanoid_rl.launchers.colab_submit import execute as submit_to_colab
                return submit_to_colab(plan, destination, gpu=args.colab_gpu or "T4", results_root=args.results_root)
            from humanoid_rl.launchers.colab import prepare
            result = prepare(plan, destination, bundle_data=args.bundle_data)
            print(f"Colab bundle: {result}\nOpen {result / 'train_colab.ipynb'} after uploading the folder to Drive.")
            print("Prepared files only; no remote job has been submitted.")
            return 0
        from humanoid_rl.launchers.local import execute
        return execute(plan)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"humanoid: {error}", file=sys.stderr)
        return 2
    except subprocess.CalledProcessError as error:
        print(f"humanoid: command exited with status {error.returncode}", file=sys.stderr)
        return error.returncode or 1


if __name__ == "__main__":
    raise SystemExit(main())
