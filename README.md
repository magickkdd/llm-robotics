# UR20 Screw Removal

Standalone MuJoCo simulation and LLM-controlled UR20 benchmark for seating a hex socket on an M6 battery screw and removing it.

## Install

Use Python 3.10 or newer, then install the project and test tools:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Set the API key and base URL for the profile you plan to use in `.env`. Set `MUJOCO_PYTHON` only when the simulator needs a separate interpreter. On headless Linux, set `MUJOCO_GL=egl`.

## Run

Run a local simulator smoke check and the mocked policy contract:

```bash
MUJOCO_GL=egl python scripts/rollout/run_mock.py
python scripts/rollout/check_policy.py
```

The single-episode runner opens the MuJoCo viewer by default; pass `--no-mujoco-viewer` for headless execution. The LLM receives only `agentview` and `wristview`; the remaining scene cameras are retained for internal contact and seating evaluation.

Run one paid LLM-controlled episode:

```bash
MUJOCO_GL=egl python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna \
  --state-id 0 \
  --execute-paid-rollouts
```

The runner creates a unique timestamped directory under `outputs/` automatically. Use `--output` to choose a base name; an existing directory receives a `_run02`, `_run03`, and so on suffix instead of being overwritten.

Run all 10 deterministic initial states with `scripts/rollout/run_benchmark.py`. The runner requires `--execute-paid-rollouts` explicitly and writes a separate result directory for each episode. Force feedback, compliant approach, spiral seating, and threaded extraction are opt-in.

## Project layout

- `src/ur20_screw/simulation/`: MuJoCo worker, robot adapter, and simulation timing.
- `src/ur20_screw/vision/`: camera-based screw and tool estimates.
- `src/ur20_screw/control/`: policy tool-to-action mapping and task-specific control contracts.
- `src/ur20_screw/policy/`: tool schema, policy client, dashboard, and API budget accounting.
- `src/ur20_screw/evaluation/`: rollout artifact encoding and evaluation support.
- `scripts/`: grouped scene, vision, control, evaluation, and rollout commands.
- `assets/ur20_screw_station/`: scene, UR20 model, workcell models, meshes, and textures.
- `tests/`: UR20 behavior, policy, budget, and artifact tests.
- `outputs/`: ignored local experiment records; preserved during source cleanup.

The scene couples screw rotation and axial travel with an ideal helical constraint. It evaluates visual alignment, socket engagement, and closed-loop extraction; it does not model manufactured thread friction or battery-case deformation.
