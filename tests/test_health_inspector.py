import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import experiments, health_inspector as inspector

PARAMS = dict(
    load_n=80.0,
    anomaly_position_mm=320.0,
    anomaly_severity=0.7,
    sensor_count=6,
    temperature_c=15.0,
    noise_nm=0.0,
    seed=7,
)


def test_channel_values_and_diagnosis_scale_match_model():
    r = experiments.run_health_experiment(PARAMS)
    c = inspector.channels(r)
    np.testing.assert_allclose([v["strain"] for v in c], r["results"]["strain_ue"])
    assert np.isclose(max(v["excess"] for v in c) / 800, r["results"]["damage_index"])
    assert c[0]["id"] == "ARM-N06-01"
    r["results"]["wavelength_shifts_nm"][0] = np.nan
    c = inspector.channels(r)
    assert c[0]["wavelength"] is None
    assert all(v["excess"] is None for v in c)


def test_only_valid_localization_shows_interval():
    baseline = experiments.run_health_experiment(dict(PARAMS, anomaly_severity=0.0))
    space, _ = inspector.figures(baseline, 0, True)
    assert len(space.layout.shapes) == 1
    active = experiments.run_health_experiment(PARAMS)
    assert active["results"]["localization_valid"]
    space, _ = inspector.figures(active, 2, False)
    assert len(space.layout.shapes) == 3
    assert space.data[0].customdata[2] == 2


def test_array_density_preserves_valid_selection():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.selectbox(key="health_inspect_channel").set_value(5).run()
    app.select_slider(key="arm_fbg_count").set_value(4).run()
    assert not app.exception
    assert app.selectbox(key="health_inspect_channel").value == 3
    assert any("ARM-N04-04 → FBG 4" in c.value for c in app.caption)
