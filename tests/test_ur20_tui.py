import io
import json

from scripts.ur20_tui import UR20Dashboard


def test_dashboard_parses_rollout_events_without_changing_non_tty_logs():
    stream = io.StringIO()
    dashboard = UR20Dashboard(stream)
    events = [
        "[UR20] model=openai/gpt-6-luna state=00 limits=10 calls/600 sim steps output=outputs/run",
        "[episode 0 | call 3/10] waiting for the model...",
        "[agent] << tool_call move_joints(" + json.dumps({
            "note": "Align above the screw.",
            "targets": {"shoulder_pan": -2.4},
        }) + ")",
        "[UR20 step 2/600 | physics 50] socket_engaged=False ever_engaged=False "
        "screw_lift=0.0000m rotation=0.0deg socket_error(axial=0.02m, radial=0.01m) success=False",
    ]
    for event in events:
        dashboard.write(event + "\n")

    assert dashboard.model == "openai/gpt-6-luna"
    assert dashboard.state_id == "00"
    assert dashboard.calls == 3
    assert dashboard.steps == 2
    assert dashboard.current_action == "move_joints"
    assert dashboard.targets == "shoulder_pan=-2.4"
    assert dashboard.note == "Align above the screw."
    assert "[UR20 step 2/600" in stream.getvalue()


def test_dashboard_tracks_success_state():
    dashboard = UR20Dashboard(io.StringIO())
    dashboard.write(
        "[UR20 step 25/600 | physics 625] socket_engaged=True ever_engaged=True "
        "screw_lift=0.0210m rotation=130.0deg socket_error(axial=0m, radial=0m) success=True\n"
    )
    dashboard.finish(True)

    assert dashboard.socket_engaged
    assert dashboard.engaged_seen
    assert dashboard.screw_lift == 0.021
    assert dashboard.rotation == 130.0
    assert dashboard.task_status == "SUCCESS"
