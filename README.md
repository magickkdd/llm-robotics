# Guarded Action-Stream Control for UR20 Screw Removal

Research code for a MuJoCo UR20 benchmark that uses an LLM tool-calling policy to align a hex socket, engage an M6 screw, and extract it. This repository studies guarded action-stream fusion: grouping up to four consecutive free-space waypoints into one policy action while returning control at contact, low visual confidence, or an XY alignment phase change.

The implementation preserves the existing per-waypoint robot motion limits and task-specific contact, seating, and extraction controllers. The reported evaluation replays archived decisions; it does not run an online LLM comparison.

## Results

| Metric | Native | Fused | Change |
|---|---:|---:|---:|
| Policy turns | 56 | 26 | -53.6% |
| Archived policy latency summed under replay protocol | 1397.2 s | 561.9 s | -59.8% |
| Projected replay + archived latency | 1926.0 s | 1087.7 s | -43.5% |
| Simulation replay wall time | 528.8 s | 525.8 s | -0.6% |
| Task successes | 1/2 | 1/2 | unchanged in these traces |

Both modes executed the same 53 robot actions, 2071 environment steps, and 135980 physics steps. The projected time assumes that a fused group incurs only the archived response latency of its first call; it is not a measured online end-to-end latency. The two archived traces share `state_id=0`, and replay holds their policy decisions fixed. These results demonstrate fewer policy turns in this replay, but do not establish online speedup or success-rate non-inferiority.

Per-trace values are in [`results/action_stream_replay.json`](results/action_stream_replay.json). The full study, including protocol, definitions, analysis, and limitations, is in [`docs/action_stream_ur20_study.md`](docs/action_stream_ur20_study.md). Broader historical benchmark records and their interpretation are in [`RESULTS.md`](RESULTS.md).

## Method

The optional `move_sequence` policy tool accepts one to four absolute Cartesian waypoints. Each waypoint is mapped and executed separately through the existing robot adapter. The adapter ends the sequence when axial contact is detected, visual confidence falls below 0.7, or the XY alignment lock changes. Contact search, socket seating, and screw turning remain separate task controllers and are not batched into waypoint sequences.

The benchmark task uses a 10 mm hex socket and an M6 screw in a MuJoCo scene. Success in the paired replay requires verified engagement and at least 21 mm of measured axial extraction from the start of socket turning. The scene uses an ideal helical constraint and does not represent manufactured thread friction or physical UR20 performance.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env
```

Configure the policy API key and base URL in `.env` only when running a live policy episode. On headless Linux, set `MUJOCO_GL=egl`.

## Reproduce the paired replay

The replay uses the two committed source summaries and makes no policy API calls:

```bash
MUJOCO_GL=egl PYTHONPATH=src python scripts/evaluation/replay_action_stream.py \
  --output outputs/action_stream_replay_run1 \
  --traces archive/rollouts/20261109_183017_agentview_primary/summary.json \
           archive/rollouts/20261109_191326_wristview_primary/summary.json
```

The output path must not already exist. The runner writes per-mode summaries, trajectory and force-trace artifacts, and an aggregate `results.json`. The tracked aggregate used in this repository is a compact copy of that result file.

## Run the simulator

Run a local mock and policy contract check:

```bash
MUJOCO_GL=egl python scripts/rollout/run_mock.py
python scripts/rollout/check_policy.py
```

To run one live policy episode, the API call and rollout cost require explicit opt-in:

```bash
MUJOCO_GL=egl python scripts/rollout/run_episode.py \
  --model openai/gpt-6-luna --profile luna --state-id 0 \
  --execute-paid-rollouts --fused-action-stream
```

The rollout runner writes timestamped records under `outputs/`. The `--fused-action-stream` flag enables guarded waypoint grouping; native mode remains the default. Force feedback, compliant approach, spiral seating, and threaded extraction are separate opt-in controls.

## Repository map

- `src/ur20_screw/simulation/`: MuJoCo worker, robot adapter, and simulation timing.
- `src/ur20_screw/vision/`: image-based screw and tool estimates.
- `src/ur20_screw/control/`: action mapping and task-specific control contracts.
- `src/ur20_screw/policy/`: tool schema, policy client, and API budget accounting.
- `scripts/evaluation/`: archived-decision replay and evaluation utilities.
- `scripts/rollout/`: live and mocked rollout entry points.
- `assets/ur20_screw_station/`: UR20 scene, robot model, meshes, and textures.
- `archive/rollouts/`: source rollout records used by the paired replay.

## Documentation

- [Action-stream fusion study](docs/action_stream_ur20_study.md)
- [System technical report](docs/technical_report.md)
- [Historical benchmark results](RESULTS.md)

## License and hardware scope

This repository does not declare a root-level license. Check the licenses of bundled assets, including `assets/ur20_screw_station/ur20/LICENSE`, before redistribution. The simulation force limits, clearances, and contact parameters are not validated operating limits for a physical UR20. No physical-robot evaluation is reported.
