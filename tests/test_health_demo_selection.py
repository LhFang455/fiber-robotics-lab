import numpy as np
from fiber_robotics_sim import demos, experiments


def test_selected_channel_mapping_and_diagnosis_unchanged():
    p = dict(
        load_n=80.0,
        anomaly_position_mm=320.0,
        anomaly_severity=0.7,
        sensor_count=6,
        temperature_c=15.0,
        noise_nm=0.0,
        seed=7,
    )
    args = {**p, "inspection_channel": 4}
    d = demos.health_demo(args)
    r = experiments.run_health_experiment(p)["results"]
    np.testing.assert_allclose(d["sensors"], r["sensor_positions_mm"])
    np.testing.assert_allclose(
        d["frames"][d["initial_index"]]["signals"], r["wavelength_shifts_nm"]
    )
    assert "ARM-N06-05" in d["subtitle"]
    assert d["frames"][-1]["suspected"] == r["suspected_location_mm"]
    assert args["inspection_channel"] == 4
    assert demos.health_demo({**p, "inspection_channel": 99})["inspection_channel"] == 5
    healthy = demos.health_demo({**p, "anomaly_severity": 0, "inspection_channel": 1})
    assert all(
        f["truth_location"] is None and f["suspected"] is None
        for f in healthy["frames"]
    )
    assert demos.health_demo(p)["inspection_channel"] is None
