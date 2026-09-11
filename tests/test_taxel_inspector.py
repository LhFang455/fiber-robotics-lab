import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import eskin, taxel_inspector as inspector


def result(**kw):
    return eskin.simulate_triaxial_taxel(
        fx_n=2, fy_n=-1.5, fz_n=8, temperature_c=38, **kw
    )


def test_forward_inverse_and_reference_identity():
    r = result(noise_pf=0.01, reference_match=0.8)
    forward, inverse = inspector.contributions(r)
    np.testing.assert_allclose(inverse.sum(axis=1), r["corrected_estimate_n"])
    np.testing.assert_allclose(r["active_pf"] - r["reference_pf"], r["corrected_pf"])
    ideal = result(noise_pf=0, reference_match=1)
    np.testing.assert_allclose(
        inspector.contributions(ideal)[0].sum(axis=1), ideal["corrected_pf"]
    )
    assert forward[1, 0] < 0


def test_missing_and_linked_channel_ids():
    r = result()
    r["active_pf"][2] = np.nan
    space, profile = inspector.figures(r, 2)
    assert profile.data[0].y[2] is None
    assert space.data[0].hovertext[2] == "缺测"
    for trace in (*space.data, *profile.data):
        assert trace.customdata[2] == 2


def test_page_selection():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    app.selectbox(key="taxel_inspect_channel").set_value(4).run()
    assert not app.exception
    assert any("TAXEL-C5：" in x.value for x in app.markdown)
