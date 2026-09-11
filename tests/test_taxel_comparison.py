import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import taxel_comparison as comparison


def test_paired_noise_and_residual_identity():
    p = dict(fx_n=2, fy_n=-1.5, fz_n=8, temperature_c=38, noise_pf=0.02, seed=7)
    cases = comparison.compare(p)
    for start in (0, 3):
        for c in cases[start : start + 3]:
            r = c["result"]
            np.testing.assert_array_equal(
                r["active_pf"], cases[start]["result"]["active_pf"]
            )
    for a, b in zip(cases[:3], cases[3:]):
        np.testing.assert_allclose(
            b["result"]["corrected_pf"] - a["result"]["corrected_pf"],
            cases[3]["result"]["corrected_pf"] - cases[0]["result"]["corrected_pf"],
            atol=1e-12,
        )
    assert cases[2]["result"]["corrected_mae_n"] < 1e-12
    for c in cases[:3]:
        r = c["result"]
        np.testing.assert_allclose(
            r["corrected_pf"] - r["sensitivity_matrix"] @ r["forces_true_n"],
            (1 - c["match"]) * r["common_mode_pf"],
            atol=1e-12,
        )


def test_zero_common_mode_no_false_matching_benefit():
    cases = comparison.compare(dict(fx_n=2, fy_n=0, fz_n=8, noise_pf=0.02, seed=7))
    np.testing.assert_array_equal(
        cases[3]["result"]["corrected_pf"], cases[5]["result"]["corrected_pf"]
    )


def test_page_does_not_overwrite_matching():
    app = AppTest.from_file("app.py", default_timeout=30).run()
    before = app.slider(key="eskin_reference_match").value
    app.checkbox(key="taxel_compare_enabled").check().run()
    assert not app.exception
    assert app.slider(key="eskin_reference_match").value == before
    assert any("校正残差 =" in x.value for x in app.info)
