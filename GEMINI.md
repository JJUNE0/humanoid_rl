# GEMINI.md — Antigravity (`agy`) Multi-Model Routing Policy

This project-level rule supplements `AGENTS.md` specifically for Google Antigravity (`agy`).
By explicit user directive, override the default single-model (`inherit`) worker behavior and route tasks across model tiers by complexity while keeping the main conversation on **Gemini 4 Carbon (High)**.

---

## 1. 3-Tier Model Assignment

### Tier 1 — Complex / Coordinator (`Gemini 4 Carbon (High)`)
- **Role**: Main interactive agent (Coordinator) and high-complexity solver.
- **Scope**:
  - System architecture, cross-module dataflow design, and final integration decisions.
  - Physics, kinematics, retargeting, reward/termination math, `LightDynamics` / coupled-PD control, and Sim-to-Real / safety-critical logic.
  - Root-cause diagnosis of subtle numerical (`NaN`/non-finite) or RL training instabilities.
- **Tool Mapping**:
  - Handled directly by the main conversation agent, or pinned to `model: "Gemini 4 Carbon (High)"` when an independent complex `swarm` worker is required (`Model: "pro"` or `"inherit"` in `invoke_subagent` / `run_workflow`).

### Tier 2 — Medium Complexity (`Gemini 4 Argon (High)`)
- **Role**: Scoped implementation, test authoring, and verification workers.
- **Scope**:
  - Implementing well-specified modules, adapters, CLI/viewer scripts, and refactors.
  - Writing and running unit/integration tests under `/test/` or `tests/` and executing `VALIDATION_SIM` rollouts.
  - Independent code-review and diff-verification (`verifier` workers in `swarm`).
- **Tool Mapping**:
  - When spawning `swarm` workers (`action: "add"`), pin `model: "Gemini 4 Argon (High)"` (or `"Gemini 4 Argon (Medium)"` for routine test runs).

### Tier 3 — Simple / High-Volume (`Gemini 4 Hydrogen (Medium)`)
- **Role**: Fast codebase exploration, log summarization, and boilerplate tasks.
- **Scope**:
  - Searching files, tracing symbol references, and surveying configs/checkpoints.
  - Parsing and summarizing training/evaluation logs (`metrics.jsonl`, `progress.json`) and drafting `/docs/exp/YYYY-MM-DD/` experiment summaries.
  - Simple formatting, schema checks, and low-risk housekeeping.
- **Tool Mapping**:
  - **Single-shot read-only delegation (preferred for quick lookups)**: Call `invoke_subagent` with `TypeName: "research"` (or `"self"`) and `Model: "flash"` or `"flash_lite"`.
  - **Parallel / Swarm delegation**: Spawn `swarm` workers with `model: "Gemini 4 Hydrogen (Medium)"` (or `"Gemini 4 Hydrogen (Low)"` for trivial checks).

---

## 2. Delegation & Safety Rules

1. **Right-size orchestration**:
   - Trivial single-file reads or 1-step checks: execute directly in the main turn without spawning extra agents.
   - Isolated exploration or log digestion: use `invoke_subagent` (`Model: "flash"` / `"flash_lite"`) to keep the main context clean.
   - Multi-part implementation + verification: use `swarm` with explicit per-worker `model` pins (`Argon` for implementation/verification, `Hydrogen` for reconnaissance/summaries) and `workspace: "share"` for concurrent editors.
2. **Propagate `AGENTS.md` invariants to every worker**:
   - Every delegated subagent or `swarm` brief must enforce `AGENTS.md`: state the Execution Domain, never work on `main`/`master`, place ad-hoc debug/test scripts in `/test/`, log experiments in `/docs/exp/YYYY-MM-DD/`, never touch unrelated working-tree changes, and never run physical robot commands automatically.
3. **Quota fallback**:
   - If a pinned tier runs out of quota, switch the waiting worker via `swarm(action="model", ...)` to an available adjacent tier and inform the user.
