import io
import json
import os
import time
import re
from unittest.mock import patch

from ur20_screw.policy.tui import UR20Dashboard


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


def test_dashboard_shows_observation_input_and_separates_policy_stages():
    dashboard = UR20Dashboard(io.StringIO())
    dashboard.write("[agent] >> input: Current observation. | state[tip]: [0.1, 0.2, 0.3]\n")
    dashboard.write('[agent] << tool_call move_joints({"targets":{"z":0.1},"note":"approach"})\n')
    dashboard.write("[agent] -- approved after delta clamp\n")

    assert "state[tip]" in dashboard.observation
    assert "move_joints" in dashboard.policy_output
    assert "approved after delta clamp" == dashboard.approval


def test_dashboard_summarizes_repeated_instruction_in_live_input():
    dashboard = UR20Dashboard(io.StringIO())
    dashboard.write(
        "[agent] >> input: Current observation. | Instruction: "
        "Seat the socket securely onto the M6 external-hex screw head. "
        "Align above the screw and continue carefully.\n"
    )

    assert "Current observation" in dashboard.input_summary
    assert "held constant" in dashboard.input_summary
    assert "Seat the socket securely" not in dashboard.input_summary


def test_tty_dashboard_uses_fixed_colored_screen_and_filters_raw_transcript():
    class TtyStream(io.StringIO):
        def isatty(self):
            return True

    stream = TtyStream()
    with patch("ur20_screw.policy.tui.shutil.get_terminal_size", return_value=os.terminal_size((80, 24))):
        dashboard = UR20Dashboard(stream)
        try:
            dashboard.write("[UR20] model=openai/gpt-6-luna state=00 limits=10 calls/600 sim steps output=x\n")
            dashboard.write("[agent] goal: this full goal should not be repeated in the event panel\n")
            dashboard.write(
                "[agent] >> step 0: 2 camera(s), state[joint_pos]: shoulder_pan=-2.4 "
                "state[sites]: [0.88, 1.0, 0.77, 0.80, 1.0, 0.28, 0.80, 1.0, 0.32] "
                "state[task_state]: [0.0, 0.0, 0.0]\n"
            )
            dashboard.write("[episode 0 | call 1/10] waiting for the model...\n")
            time.sleep(0.2)
            dashboard.finish(False)
        finally:
            dashboard.close()

    output = stream.getvalue()
    assert "\033[?1049h" in output
    assert "\033[2J\033[H" in output
    assert "LOCATE" in output and "APPROACH" in output and "SEAT" in output
    assert "CALLS" in output and "Screw lift" not in output
    assert "this full goal should not be repeated" not in output
    assert "tip=(0.800, 1.000, 0.280)" in output
    assert "\033[38;5;45m" in output
    assert "\033[38;5;220m" in output
    assert "\033[38;5;207m" in output
    last_frame = output.rsplit("\033[2J\033[H", 1)[-1].split("\033[J", 1)[0]
    assert len(last_frame.splitlines()) <= 23


def test_dashboard_fits_and_aligns_panels_at_multiple_terminal_sizes():
    class TtyStream(io.StringIO):
        def isatty(self):
            return True

    for columns, lines in ((132, 42), (100, 30), (80, 24), (54, 18), (40, 14)):
        stream = TtyStream()
        with patch("ur20_screw.policy.tui.shutil.get_terminal_size",
                   return_value=os.terminal_size((columns, lines))):
            dashboard = UR20Dashboard(stream)
            try:
                dashboard.model = "openai/gpt-6.1-terra"
                dashboard.state_id = "00"
                dashboard.estimated_cost = "$0.40"
                dashboard.current_action = "move_joints"
                dashboard.targets = "shoulder_pan=-2.408, shoulder_lift=-1.02, elbow=1.1511"
                dashboard.note = "That lift adjustment moved the socket farther above the head in the wrist image. I’ll reverse it and restore the wrist’s vertical orientation while correcting the pan slightly."
                dashboard.observation = "agentview + wrist · tool tip and screw visible"
                dashboard.policy_output = "move_joints · shoulder_pan=-2.4"
                dashboard.approval = "approved · delta clamped"
                dashboard._render()
            finally:
                dashboard.close()
        frame = stream.getvalue().rsplit("\033[2J\033[H", 1)[-1].split("\033[J", 1)[0]
        visible_lines = [re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", row) for row in frame.splitlines()]
        widths = [UR20Dashboard._display_width(row) for row in visible_lines]
        assert all(width <= columns for width in widths), (columns, widths)
        box_rows = [row for row in visible_lines if row.startswith(("┌", "│", "└"))]
        for row in box_rows:
            assert row.count("│") in (0, 2, 4), (columns, row)
        if columns >= 80 and lines >= 18:
            assert "RUN / SIM STATE" in frame and "LIVE DATA FLOW" in frame
