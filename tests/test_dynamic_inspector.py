import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import eskin, dynamic_inspector as inspector


def test_peaks_match_original_offline_rule():
    for event in ("稳定按压", "横向滑动", "热物体", "即将滑移"):
        r = eskin.simulate_dynamic_skin_event(event, noise_ratio=0)
        a, b = inspector.peak_indices(r, 4)
        assert r["time_s"][a] >= 0.8
        assert np.isclose(r["shear_ratio"][a], r["peak_shear_ratio"])
        assert np.isclose(
            abs(r["centroid_velocity_mm_s"][b]), r["peak_centroid_speed_mm_s"]
        )
        assert r["alert"] == bool(
            r["shear_ratio"][a] > 0.35 and abs(r["centroid_velocity_mm_s"][b]) > 2
        )


def test_undefined_ratio_and_missing_are_not_zero():
    r = eskin.simulate_dynamic_skin_event("横向滑动", noise_ratio=0)
    assert np.isnan(inspector.series(r, "剪切比", 12)[0])
    r["temperature_c"][17] = np.nan
    fig = inspector.timeline(r, "温度", 17, 4, 12, 0.35)
    assert fig.data[0].y[17] is None
    assert fig.data[0].hoverinfo == "skip"
    assert 17 in fig.data[1].customdata


def test_sampling_change_and_peak_navigation():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.slider(key="dynamic_inspect_index").set_value(399).run()
    app.select_slider(key="eskin_dynamic_sample_rate").set_value(20).run()
    assert not app.exception
    assert app.slider(key="dynamic_inspect_index").value == 79
    app.button(key="dynamic_jump_speed").click().run()
    assert not app.exception
    assert app.slider(key="dynamic_inspect_index").value >= 16
