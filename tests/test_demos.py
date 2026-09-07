import json

import numpy as np
import pytest

from fiber_robotics_sim import demos


FOOT = dict(terrain="平地", load_n=180.0, phase_percent=55.0,
            support="支撑期", temperature_c=10.0, noise_nm=0.0, seed=7,
            failed_zone=None, drift_nm=0.0)
SHAPE = dict(curvature_per_m=8.0, direction_deg=35.0, twist_per_m=0.0,
             length_mm=150.0, core_radius_um=125.0, temperature_c=10.0,
             noise_nm=0.0, core_temperature_gradient_c=0.0, seed=7)
SKIN = dict(sensor_count=8, touch_points=[(24.0, 30.0, 6.0)],
            skin_width_mm=80.0, skin_height_mm=60.0,
            receptive_width_mm=18.0, temperature_c=25.0, noise_nm=0.0, seed=7)


def test_foot_scan_conserves_load_and_moves_cop_from_heel_to_forefoot():
    result = demos.foot_demo(FOOT)
    first, last = result["frames"][0], result["frames"][-1]
    assert first["loads"] == pytest.approx([0, 0, 0, 60, 60, 60])
    assert last["loads"] == pytest.approx([60, 60, 60, 0, 0, 0])
    assert first["cop"] == pytest.approx([1.45, .42])
    assert last["cop"] == pytest.approx([1.45, 1.42])
    assert all(sum(frame["loads"]) == pytest.approx(180) for frame in result["frames"])
    assert result["frames"][result["initial_index"]]["phase"] == 55


def test_foot_zero_load_hides_undefined_cop_and_failed_channel_retains_true_load():
    zero = demos.foot_demo({**FOOT, "load_n": 0.0})
    assert all(frame["cop"] is None and not frame["reliable"] for frame in zero["frames"])
    broken = demos.foot_demo({**FOOT, "failed_zone": 1, "temperature_c": 0.0})
    last = broken["frames"][-1]
    assert last["true_loads"][0] == pytest.approx(60)
    assert last["loads"][0] == pytest.approx(0)
    assert not last["reliable"]


@pytest.mark.parametrize("count", [4, 8, 16])
def test_skin_current_frame_preserves_contacts_and_force_and_unloaded_frame_has_no_centroid(count):
    result = demos.skin_demo({**SKIN, "sensor_count": count})
    current = result["frames"][result["initial_index"]]
    assert current["touches"] == [[24.0, 30.0, 6.0]]
    assert current["force"] == pytest.approx(6.0)
    assert current["estimated_force"] == pytest.approx(6.0)
    assert len(current["signals"]) == count
    assert result["frames"][0]["centroid"] is None
    assert result["frames"][-1]["centroid"] is None
    json.dumps(result, allow_nan=False)


def test_skin_slide_stays_inside_patch_and_zero_load_has_no_false_location():
    result = demos.skin_demo({**SKIN, "touch_points": [(80.0, 60.0, 0.0), (0.0, 0.0, 0.0)], "noise_nm": .001})
    for frame in result["frames"]:
        assert frame["centroid"] is None
        for x, y, force in frame["touches"]:
            assert 0 <= x <= 80 and 0 <= y <= 60 and force == 0
    json.dumps(result, allow_nan=False)


def test_zero_curvature_shape_stays_straight_and_coincident_without_noise():
    result = demos.shape_demo({**SHAPE, "curvature_per_m": 0.0})
    for frame in result["frames"]:
        assert np.asarray(frame["truth"])[-1] == pytest.approx([0, 0, 150])
        assert np.asarray(frame["estimate"]) == pytest.approx(np.asarray(frame["truth"]))
        assert frame["rmse"] == pytest.approx(0)
    json.dumps(result, allow_nan=False)


def test_shape_temperature_gradient_produces_visible_error_and_target_curvature_is_reached():
    result = demos.shape_demo({**SHAPE, "core_temperature_gradient_c": 10.0})
    target = result["frames"][result["initial_index"]]
    assert target["curvature"] == pytest.approx(8.0)
    assert target["rmse"] > 1.0
    assert target["tip_error"] > 1.0
    assert not np.allclose(target["truth"], target["estimate"])
    json.dumps(result, allow_nan=False)


def test_noisy_zero_estimated_load_does_not_hide_true_cop():
    result = demos.foot_demo({**FOOT, "load_n": 5.0, "support": "摆动期", "phase_percent": 0.0,
                             "temperature_c": 0.0, "noise_nm": .02, "seed": 8})
    first = result["frames"][0]
    assert first["cop"] is None
    assert first["true_cop"] == pytest.approx([1.45, .42])


def test_skin_contact_count_ignores_an_unloaded_second_press():
    result = demos.skin_demo({**SKIN, "touch_points": [(24, 30, 0), (58, 22, 4)]})
    current = result["frames"][result["initial_index"]]
    assert dict(current["metrics"])["接触数量"] == "1"
