import json

import numpy as np
import pytest

from fiber_robotics_sim import eskin
from fiber_robotics_sim import demo_eskin


def taxel_parameters(**overrides):
    params = dict(fx_n=2.0, fy_n=-1.5, fz_n=8.0, curvature_per_m=3.0,
                  strain_fraction=.001, temperature_c=38.0, noise_pf=.01,
                  reference_match=.98, seed=7)
    return {**params, **overrides}


@pytest.mark.parametrize("overrides", [{}, {"fx_n": -5., "fy_n": 5., "fz_n": 0.},
                                      {"fx_n": 0., "fy_n": 0., "fz_n": 0., "noise_pf": 0.},
                                      {"temperature_c": 0., "reference_match": .8, "noise_pf": .1}])
def test_taxel_current_frame_matches_original_true_raw_and_corrected(overrides):
    params = taxel_parameters(**overrides)
    expected = eskin.simulate_triaxial_taxel(**params)
    payload = demo_eskin.taxel_demo(params)
    final = payload["frames"][payload["initial_index"]]
    for field, key in [("truth", "forces_true_n"), ("raw", "raw_estimate_n"),
                       ("corrected", "corrected_estimate_n")]:
        np.testing.assert_array_equal(final[field], expected[key])
    assert final["raw_mae"] == expected["raw_mae_n"]
    assert final["corrected_mae"] == expected["corrected_mae_n"]
    assert payload["force_max"] > 0
    json.dumps(payload, allow_nan=False)


def test_taxel_sweep_separates_force_loading_from_common_mode():
    params = taxel_parameters(noise_pf=0., reference_match=1.)
    frames = demo_eskin.taxel_demo(params)["frames"]
    np.testing.assert_array_equal(frames[0]["truth"], [0, 0, 0])
    np.testing.assert_allclose(frames[60]["truth"], [2, -1.5, 8])
    assert frames[60]["temperature"] == 25
    assert frames[60]["common_mode"] == 0
    assert frames[-1]["common_mode"] > 0
    assert frames[-1]["corrected_mae"] < 1e-12
    assert frames[-1]["raw_mae"] > .1


@pytest.mark.parametrize("scenario", ["单点接触", "双点接触", "边缘接触", "滑动前兆"])
@pytest.mark.parametrize("size,output", [(4,16),(8,32)])
def test_pressure_preserves_full_fields_samples_and_final_metrics(scenario, size, output):
    params = dict(scenario=scenario, sparse_size=size, output_size=output,
                  peak_pressure_kpa=80., bandwidth=.16, noise_kpa=.5, seed=7)
    expected = eskin.simulate_pressure_reconstruction(**params)
    payload = demo_eskin.pressure_demo(params)
    for field, key in [("truth", "truth_kpa"), ("samples", "sparse_samples_kpa"),
                       ("reconstruction", "reconstruction_kpa")]:
        np.testing.assert_array_equal(payload[field], expected[key])
    final = payload["frames"][payload["initial_index"]]
    assert final["read_count"] == size**2
    assert final["row_count"] == output
    assert final["rmse"] == expected["rmse_kpa"]
    assert final["signals"][-1] == np.max(expected["reconstruction_kpa"])
    assert payload["pressure_max"] == max(np.max(expected[k]) for k in ["truth_kpa", "sparse_samples_kpa", "reconstruction_kpa"])
    json.dumps(payload, allow_nan=False)


def test_pressure_does_not_invent_partial_reconstruction_or_early_error_metrics():
    payload = demo_eskin.pressure_demo(dict(scenario="边缘接触", sparse_size=4, output_size=16,
        peak_pressure_kpa=20., bandwidth=.06, noise_kpa=5., seed=19))
    frames = payload["frames"]
    assert frames[0]["read_count"] == 0
    assert frames[0]["signals"][1:] == [None, None]
    for frame in frames:
        if frame["row_count"] < 16:
            assert frame["rmse"] is None
            assert frame["signals"][-1] is None
        if frame["read_count"] < 16:
            assert frame["row_count"] == 0
    assert "不是算法迭代" in payload["boundary"]
