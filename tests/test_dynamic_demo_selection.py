import numpy as np
from fiber_robotics_sim import demo_extensions, eskin


def test_exact_selected_sample_and_cumulative_vs_final():
    p = dict(
        event="横向滑动",
        sample_rate_hz=100,
        duration_s=4,
        normal_force_n=12,
        slip_threshold=0.35,
        temperature_c=25,
        noise_ratio=0,
        seed=7,
    )
    r = eskin.simulate_dynamic_skin_event(**p)
    for index in (0, 100, 300, 399):
        args = {**p, "inspection_index": index}
        d = demo_extensions.dynamic_demo(args)
        f = d["frames"][d["initial_index"]]
        assert d["initial_index"] == index
        for key, source in (
            ("time", "time_s"),
            ("x", "centroid_x_mm"),
            ("normal", "normal_force_n"),
            ("shear", "shear_force_n"),
        ):
            assert np.isclose(f[key], r[source][index])
        assert args["inspection_index"] == index
    d = demo_extensions.dynamic_demo({**p, "inspection_index": 100})
    assert not d["frames"][100]["alert"]
    assert d["frames"][-1]["alert"] == r["alert"]
    assert r["alert"]
    assert "未定义" in d["frames"][0]["caption"]
    assert (
        demo_extensions.dynamic_demo({**p, "inspection_index": 999})["initial_index"]
        == 399
    )
    assert demo_extensions.dynamic_demo(p)["initial_index"] == 399
