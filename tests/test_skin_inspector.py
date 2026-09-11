import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import eskin, skin_inspector as inspector


def result(touches=((24.0, 30.0, 6.0), (58.0, 22.0, 4.0))):
    return eskin.simulate_fbg_skin(
        sensor_count=8, touch_points=touches, temperature_c=36.0, noise_nm=0.0
    )


def test_channel_mapping_and_weighted_contact_response():
    r = result()
    c = inspector.channel(r, 3)
    assert c["id"] == "SKIN-N08-04"
    np.testing.assert_allclose(c["position"], r["sensor_positions_mm"][3])
    assert np.isclose(c["raw"] - c["compensated"], r["temperature_shift_nm"])
    np.testing.assert_allclose(r["receptive_fields"].sum(axis=1), 1)
    assert np.isclose(
        c["compensated"], 0.012 * np.dot([6.0, 4.0], r["receptive_fields"][:, 3])
    )
    space, profile = inspector.figures(r, 3, 80, 60, True)
    assert space.data[0].customdata[3] == 3
    assert profile.data[1].customdata[3] == 3


def test_zero_load_and_missing_are_not_fake_measurements():
    r = result(((24.0, 30.0, 0.0),))
    space, _ = inspector.figures(r, 0, 80, 60, True)
    assert len(space.data) == 1
    r["measured_shift_nm"][0] = np.nan
    assert inspector.channel(r, 0)["raw"] is None
    assert inspector.figures(r, 0, 80, 60, True)[1].data[0].y[0] is None


def test_skin_page_density_and_selection():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.selectbox(key="skin_inspect_channel").set_value(7).run()
    app.select_slider(key="eskin_fbg_sensor_count").set_value(4).run()
    assert not app.exception
    assert app.selectbox(key="skin_inspect_channel").value == 3
    assert any("SKIN-N04-04 → FBG 4" in c.value for c in app.caption)
