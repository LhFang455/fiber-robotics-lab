import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import skin_comparison as comparison


def test_same_load_centroid_and_four_corner_ambiguity():
    cases, _, _ = comparison.compare(80, 60, 18, 10, 32)
    for c in cases:
        assert (
            c["single"]["true_total_force_n"] == c["double"]["true_total_force_n"] == 10
        )
        np.testing.assert_allclose(
            c["single"]["true_centroid_mm"], c["double"]["true_centroid_mm"]
        )
        np.testing.assert_allclose(
            c["single"]["sensor_positions_mm"], c["double"]["sensor_positions_mm"]
        )
    _, single, double = comparison.compare(80, 60, 18, 10, 32)
    signals = comparison.figures(cases[0], single, double, 80, 60)[1]
    assert signals.layout.yaxis.range[0] == 0
    assert signals.layout.yaxis.range[1] > 0.03
    assert cases[0]["maximum"] < 1e-12
    assert cases[1]["maximum"] > 1e-4


def test_coincident_hypotheses_and_contact_bounds():
    cases, _, _ = comparison.compare(40, 30, 6, 6, 0)
    for c in cases:
        np.testing.assert_allclose(
            c["single"]["compensated_shift_nm"], c["double"]["compensated_shift_nm"]
        )
    _, _, contacts = comparison.compare(40, 30, 6, 6, 32)
    assert all(0 <= x <= 40 and 0 <= y <= 30 for x, y, _ in contacts)


def test_page_toggle_and_zero_load():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    before = app.selectbox(key="eskin_contact_mode").value
    app.checkbox(key="skin_compare_enabled").check().run()
    assert not app.exception
    assert app.selectbox(key="eskin_contact_mode").value == before
    app.slider(key="eskin_touch1_force").set_value(0).run()
    assert not app.exception
    assert any("当前总载荷为零" in x.value for x in app.info)
