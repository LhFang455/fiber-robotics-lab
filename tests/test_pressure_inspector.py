import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import eskin, pressure_inspector as inspector


@pytest.mark.parametrize("n,m,b", [(4, 16, 0.06), (8, 32, 0.3)])
def test_contributions_reproduce_existing_reconstruction(n, m, b):
    result = eskin.simulate_pressure_reconstruction(
        "双点接触", n, m, bandwidth=b, noise_kpa=0.5
    )
    for row, col in [(0, 0), (m // 2, m // 2), (m - 1, m - 1)]:
        weights, values = inspector.contribution(result, row, col, b)
        assert np.isclose(weights.sum(), 1)
        assert np.isclose(values.sum(), result["reconstruction_kpa"][row, col])
    space, profile, bars = inspector.figures(result, 2, 3, b)
    assert len(space.data[1].x) == n * n
    assert space.data[2].customdata[3] == 3
    assert profile.data[1].customdata[3] == 3
    assert len(bars.data[0].y) == n * n


def test_missing_values_remain_missing():
    result = eskin.simulate_pressure_reconstruction("单点接触", 4, 16)
    result["reconstruction_kpa"][2, 3] = np.nan
    result["sparse_samples_kpa"][0, 0] = np.nan
    _, profile, bars = inspector.figures(result, 2, 3, 0.16)
    assert profile.data[1].y[3] is None
    assert bars.data[0].y[0] is None


def test_grid_change_keeps_selection_valid():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.select_slider(key="eskin_output_size").set_value(32).run()
    app.selectbox(key="pressure_inspect_col").set_value(31).run()
    app.select_slider(key="eskin_output_size").set_value(16).run()
    assert not app.exception
    assert app.selectbox(key="pressure_inspect_col").value == 15
    assert any("GRID-N16" in x.value for x in app.caption)
