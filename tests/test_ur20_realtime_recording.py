import base64
from unittest.mock import Mock

from PIL import Image

from ur20_screw.simulation.robot import UR20ScrewEmbodiment
from ur20_screw.simulation.timing import wait_for_simulated_time


def test_simulation_waits_for_real_time_deadline_without_viewer(monkeypatch):
    monkeypatch.setattr("ur20_screw.simulation.timing.time.perf_counter", lambda: 10.0)
    sleep = Mock()
    monkeypatch.setattr("ur20_screw.simulation.timing.time.sleep", sleep)

    wait_for_simulated_time(10.0, 0.05)

    sleep.assert_called_once()
    assert abs(sleep.call_args.args[0] - 0.05) < 1e-9


def test_ur20_records_each_observation_as_a_paired_jpeg_frame(tmp_path):
    embodiment = UR20ScrewEmbodiment(output_dir=tmp_path)
    png = tmp_path / "sample.png"
    Image.new("RGB", (2, 2), (12, 34, 56)).save(png)
    encoded = base64.b64encode(png.read_bytes()).decode("ascii")

    embodiment._record_video_frame({"images": {"agentview": encoded, "wristview": encoded}})

    frame_path = embodiment.video_frames_dir / "ur20_video_000000.jpg"
    assert frame_path.is_file()
    assert Image.open(frame_path).size == (4, 2)
    assert embodiment.video_frame_index == 1
