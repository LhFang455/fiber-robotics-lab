import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import eskin, dynamic_comparison as comparison


def test_fixed_generation_and_monotonic_threshold_scan():
    p = dict(
        event="即将滑移",
        sample_rate_hz=50,
        duration_s=4,
        normal_force_n=12,
        temperature_c=25,
        noise_ratio=0.03,
        seed=7,
    )
    groups = comparison.records(p)
    saved = [x.copy() for x in groups]
    for peaks in groups:
        counts = [
            comparison.decisions(peaks, t).sum()
            for t in (0.15, 0.25, 0.35, 0.45, 0.55, 0.7)
        ]
        assert all(a >= b for a, b in zip(counts, counts[1:]))
    for a, b in zip(saved, groups):
        np.testing.assert_array_equal(a, b)
    original = eskin.simulate_dynamic_skin_event(**p, slip_threshold=0.35)
    np.testing.assert_allclose(
        groups[1][0],
        [original["peak_shear_ratio"], original["peak_centroid_speed_mm_s"]],
    )
    assert comparison.decisions(groups[1], 0.35)[0] == original["alert"]


def test_strict_thresholds_and_zero_noise():
    assert comparison.decisions(
        np.array([[0.35, 3], [0.5, 2], [0.5, 3]]), 0.35
    ).tolist() == [False, False, True]
    groups = comparison.records(dict(event="稳定按压", noise_ratio=0, seed=7))
    np.testing.assert_array_equal(*groups)


def test_page_independent_threshold():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    before = app.slider(key="eskin_slip_threshold").value
    app.checkbox(key="dynamic_compare_enabled").check().run()
    app.slider(key="dynamic_compare_threshold").set_value(0.55).run()
    assert not app.exception
    assert app.slider(key="eskin_slip_threshold").value == before
    assert any("当前阈值 0.55" in x.value for x in app.info)
