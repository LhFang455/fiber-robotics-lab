import numpy as np
from fiber_robotics_sim import demo_eskin, eskin


def test_selection_values_clamping_and_original_model_preserved():
    p = dict(
        scenario="双点接触",
        sparse_size=4,
        output_size=16,
        peak_pressure_kpa=80,
        bandwidth=0.16,
        noise_kpa=0.5,
        seed=7,
    )
    original = eskin.simulate_pressure_reconstruction(**p)
    args = {**p, "inspection_node": [7, 99]}
    demo = demo_eskin.pressure_demo(args)
    selected = demo["inspection_node"]
    assert selected["id"] == "GRID-N16-R08-C16"
    assert selected["truth"] == original["truth_kpa"][7, 15]
    assert selected["reconstruction"] == original["reconstruction_kpa"][7, 15]
    np.testing.assert_array_equal(
        demo["reconstruction"], original["reconstruction_kpa"]
    )
    assert args["inspection_node"] == [7, 99]
    assert demo_eskin.pressure_demo(p)["inspection_node"] is None
