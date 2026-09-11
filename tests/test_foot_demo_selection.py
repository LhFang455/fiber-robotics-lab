import numpy as np
from fiber_robotics_sim import demos, experiments


def test_selected_zone_phase_and_failed_loads():
    p = dict(
        load_n=180.0,
        terrain="平地",
        phase_percent=57.0,
        support="支撑期",
        temperature_c=10.0,
        noise_nm=0.0,
        seed=7,
        failed_zone=None,
        drift_nm=0.0,
    )
    args = {**p, "inspection_channel": 2}
    d = demos.foot_demo(args)
    f = d["frames"][d["initial_index"]]
    r = experiments.run_foot_experiment(p)["results"]
    assert f["phase"] == 57
    assert d["inspection_channel"] == 2
    np.testing.assert_allclose(f["loads"], r["estimated_zone_loads_n"])
    np.testing.assert_allclose(f["cop"], r["estimated_cop_xy"])
    assert all(f["valid_loads"])
    failed = demos.foot_demo({**args, "failed_zone": 3})
    assert all(not f["valid_loads"][2] for f in failed["frames"])
    assert "无有效观测" in failed["frames"][-1]["caption"]
    assert args["inspection_channel"] == 2
    assert demos.foot_demo(p)["inspection_channel"] is None
