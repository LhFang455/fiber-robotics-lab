import numpy as np
from fiber_robotics_sim import demos, experiments


def test_selected_node_error_current_frame_and_input_preserved():
    p = dict(
        curvature_per_m=8.0,
        direction_deg=35.0,
        twist_per_m=0.0,
        length_mm=150.0,
        core_radius_um=125.0,
        temperature_c=20.0,
        noise_nm=0.0,
        core_temperature_gradient_c=4.0,
        seed=7,
    )
    args = {**p, "inspection_node": 120}
    d = demos.shape_demo(args)
    r = experiments.run_shape_experiment(p)["results"]
    f = d["frames"][d["initial_index"]]
    np.testing.assert_allclose(f["truth"][120], r["true_centerline_xyz_mm"][120])
    np.testing.assert_allclose(
        f["estimate"][120], r["estimated_centerline_xyz_mm"][120]
    )
    assert np.isclose(f["selected_error_mm"], r["point_error_mm"][120])
    assert d["inspection_node"] == args["inspection_node"] == 120
    assert demos.shape_demo({**p, "inspection_node": 999})["inspection_node"] == 160
    assert demos.shape_demo(p)["inspection_node"] is None
