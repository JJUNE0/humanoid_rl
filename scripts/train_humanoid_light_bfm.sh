#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$task_root"
if [[ ! -f outputs/humanoid_light_bfm/manifest.json ]]; then
  echo 'Prepare the CMU + LAFAN manifest first:' >&2
  echo '.venv/bin/python scripts/prepare_humanoid_light_bfm.py --workers 2' >&2
  exit 1
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
exec .venv/bin/python -u train.py --algorithm bfm_zero --task Humanoid-Light-Motion-Prior "$@"
