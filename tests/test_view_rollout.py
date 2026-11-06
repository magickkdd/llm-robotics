from scripts.view_rollout import (
    _dashboard_data,
    _rollout_entries,
    _socket_opening_dashboard_data,
    render_dashboard,
)


def test_dashboard_renders_multifield_rollout_filters():
    html = render_dashboard({
        "video": None,
        "rollouts": [{
            "url": "run/report.html",
            "label": "run · model · scene · success",
            "model": "openai/gpt-6-luna",
            "date": "2026-11-04",
            "scene": "ur20_screw_station",
            "outcome": "success",
            "selected": True,
        }],
    })

    assert 'data-filter="model"' in html
    assert 'data-filter="date"' in html
    assert 'data-filter="scene"' in html
    assert 'data-filter="outcome"' in html
    assert 'data-model="gpt-6-luna"' in html
    assert '<option value="gpt-6-luna">gpt-6-luna</option>' in html
    assert "new URLSearchParams(location.search)" in html
    assert "location.href=picker.value+(query?`?${query}`:'')" in html
    assert "point.t" in html
    assert "progress*maxStep" in html
    assert "const floorCorners" in html
    assert "const currentPoint=visible[visible.length-1]" in html
    assert "currentPoint.z.toFixed(3)" in html
    assert "stroke-dasharray':'4 4'" in html
    assert "matching" in html


def test_rollout_catalog_skips_archived_directories(tmp_path):
    archived_inspect = tmp_path / "_archive_terra_bad_run" / "run" / "inspect"
    archived_inspect.mkdir(parents=True)
    (archived_inspect / "placeholder.json").write_text("{}", encoding="utf-8")

    assert _rollout_entries(tmp_path) == []


def test_ur20_root_inspect_directory_loads_recorded_cartesian_actions(tmp_path):
    run_dir = tmp_path / "ur20_sol_run"
    actions_dir = run_dir / "inspect" / "actions"
    actions_dir.mkdir(parents=True)
    (actions_dir / "trial.jsonl").write_text(
        '{"kind":"header","labels":["x","y","z"]}\n'
        '{"t":0,"action":[0.1,0.2,0.3]}\n'
        '{"t":1,"action":[0.11,0.2,0.29]}\n',
        encoding="utf-8",
    )
    log = {"samples": [{"trial_metadata": [{"actions": "actions/trial.jsonl"}]}]}

    data = _dashboard_data(run_dir, run_dir, log, {})

    assert data["trajectory"] == [
        {"x": 0.1, "y": 0.2, "z": 0.3, "t": 0},
        {"x": 0.11, "y": 0.2, "z": 0.29, "t": 1},
    ]


def test_socket_opening_viewer_uses_world_frame_ground_truth(tmp_path):
    (tmp_path / "trajectory.jsonl").write_text(
        '{"event":"start_pose","sim_time_s":0,"socket_opening_center_world_m":[0.8,1.0,0.153]}\n'
        '{"event":"socket_opening_sample","sim_time_s":0.002,"socket_opening_center_world_m":[0.801,1.0,0.153]}\n',
        encoding="utf-8",
    )

    data = _socket_opening_dashboard_data(tmp_path)
    html = render_dashboard(data)

    assert data["trajectory"] == [
        {"x": 0.8, "y": 1.0, "z": 0.153, "t": 0.0},
        {"x": 0.801, "y": 1.0, "z": 0.153, "t": 0.002},
    ]
    assert "Socket opening center" in html
    assert "World-frame MuJoCo ground truth" in html
