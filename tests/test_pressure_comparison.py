import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import pressure_comparison as comparison


def test_controlled_truth_samples_and_convex_bound():
    for scenario in ("单点接触", "双点接触", "边缘接触", "滑动前兆"):
        cases = comparison.compare(scenario, 80, 16)
        for c in cases:
            r = c["result"]
            np.testing.assert_array_equal(
                r["truth_kpa"], cases[0]["result"]["truth_kpa"]
            )
            assert (
                r["reconstruction_kpa"].max() <= r["sparse_samples_kpa"].max() + 1e-10
            )
        for i in (0, 2):
            np.testing.assert_array_equal(
                cases[i]["result"]["sparse_samples_kpa"],
                cases[i + 1]["result"]["sparse_samples_kpa"],
            )
        fig, row, col = comparison.profile(cases)
        assert (
            cases[0]["result"]["truth_kpa"][row, col]
            == cases[0]["result"]["truth_kpa"].max()
        )
        assert len(fig.data) == 5


def test_expansion_does_not_change_experiment_parameters():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    before = [
        app.slider(key=k).value
        for k in ("eskin_kernel_bandwidth", "eskin_pressure_noise")
    ]
    app.checkbox(key="pressure_compare_enabled").check().run()
    assert not app.exception
    assert before == [
        app.slider(key=k).value
        for k in ("eskin_kernel_bandwidth", "eskin_pressure_noise")
    ]
    assert any("四组中全场 RMSE" in x.value for x in app.markdown)
