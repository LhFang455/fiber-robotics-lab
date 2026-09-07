import json

import numpy as np
import pytest

from fiber_robotics_sim import demo_extensions, eskin


def parameters(event="即将滑移", **overrides):
    return dict(event=event, sample_rate_hz=100, duration_s=4.0,
                normal_force_n=12.0, slip_threshold=.35, temperature_c=25.0,
                noise_ratio=0.0, seed=7, **overrides)


@pytest.mark.parametrize("event", ["稳定按压", "载荷爬升", "横向滑动", "即将滑移", "热物体", "温漂"])
def test_dynamic_replay_preserves_every_recorded_sample_and_final_decision(event):
    params = parameters(event)
    expected = eskin.simulate_dynamic_skin_event(**params)
    payload = demo_extensions.dynamic_demo(params)
    frames = payload["frames"]
    assert len(frames) == len(expected["time_s"])
    for field, source in [("time", "time_s"), ("normal", "normal_force_n"),
                          ("shear", "shear_force_n"), ("x", "centroid_x_mm"),
                          ("temperature", "temperature_c"), ("ratio", "shear_ratio"),
                          ("speed", "centroid_velocity_mm_s")]:
        np.testing.assert_array_equal([f[field] for f in frames], expected[source])
    current = frames[payload["initial_index"]]
    assert current["alert"] == expected["alert"]
    assert current["peak_ratio"] == expected["peak_shear_ratio"]
    assert current["peak_speed"] == expected["peak_centroid_speed_mm_s"]
    assert payload["playback_duration_ms"] == pytest.approx(current["time"] * 1000)
    assert current["time"] < params["duration_s"]
    assert frames[0]["alert"] is None
    json.dumps(payload, allow_nan=False)


def test_replay_does_not_show_future_alert_and_thermal_event_is_not_slip():
    replay = demo_extensions.dynamic_demo(parameters())
    early = [f for f in replay["frames"] if f["time"] < .48 * 4]
    assert not any(f["alert"] for f in early)
    assert replay["frames"][-1]["alert"]
    thermal = demo_extensions.dynamic_demo(parameters("热物体"))
    assert thermal["frames"][-1]["temperature"] == 41
    assert not any(f["alert"] for f in thermal["frames"])


def test_window_rule_allows_peaks_at_different_times(monkeypatch):
    raw = eskin.simulate_dynamic_skin_event(**parameters())
    raw["shear_ratio"][:] = .1
    raw["centroid_velocity_mm_s"][:] = 0
    raw["shear_ratio"][100] = .6
    raw["centroid_velocity_mm_s"][200] = 3
    monkeypatch.setattr(eskin, "simulate_dynamic_skin_event", lambda **kwargs: raw)
    frames = demo_extensions.dynamic_demo(parameters())["frames"]
    assert not frames[100]["alert"]
    assert frames[200]["alert"]
    assert frames[200]["ratio"] < .35
    assert "不要求同一时刻" in frames[200]["caption"]


def test_highest_sampling_rate_keeps_short_events_and_seeded_noise():
    params = parameters()
    params.update(sample_rate_hz=200, duration_s=8.0, noise_ratio=.08)
    payload = demo_extensions.dynamic_demo(params)
    assert len(payload["frames"]) == 1600
    assert payload == demo_extensions.dynamic_demo(params)
    assert all(f["time"] < 8 for f in payload["frames"])
    json.dumps(payload, allow_nan=False)
