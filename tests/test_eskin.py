import numpy as np
import pytest

from fiber_robotics_sim.eskin import (
    repeat_dynamic_event,
    simulate_dynamic_skin_event,
    simulate_fbg_skin,
    simulate_pressure_reconstruction,
    simulate_triaxial_taxel,
)


def test_triaxial_taxel_is_deterministic_and_reference_correction_reduces_common_mode_error():
    kwargs = dict(
        fx_n=2.0, fy_n=-1.5, fz_n=8.0, curvature_per_m=3.0,
        strain_fraction=0.001, temperature_c=38.0, noise_pf=0.0,
        reference_match=1.0, seed=11,
    )
    result = simulate_triaxial_taxel(**kwargs)
    repeated = simulate_triaxial_taxel(**kwargs)

    assert result["sensitivity_matrix"].shape == (5, 3)
    assert result["active_pf"].shape == (5,)
    assert result["reference_pf"].shape == (5,)
    assert result["corrected_pf"].shape == (5,)
    np.testing.assert_allclose(result["corrected_estimate_n"], repeated["corrected_estimate_n"])
    assert result["matrix_rank"] == 3
    assert np.isfinite(result["condition_number"])
    assert result["corrected_mae_n"] < result["raw_mae_n"]
    assert result["units"]["force"] == "N"


def test_triaxial_reference_mismatch_and_noise_are_reproducible():
    kwargs = dict(
        fx_n=-3.0, fy_n=1.0, fz_n=5.0, curvature_per_m=1.5,
        strain_fraction=0.0005, temperature_c=30.0, noise_pf=0.02,
        reference_match=0.93, seed=21,
    )
    first = simulate_triaxial_taxel(**kwargs)
    second = simulate_triaxial_taxel(**kwargs)
    np.testing.assert_allclose(first["active_pf"], second["active_pf"])
    assert first["corrected_mae_n"] >= 0.0


def test_fbg_skin_supports_approved_layouts_and_preserves_total_load_after_compensation():
    result = simulate_fbg_skin(
        sensor_count=8,
        touch_points=[(24.0, 30.0, 6.0), (58.0, 22.0, 4.0)],
        skin_width_mm=80.0,
        skin_height_mm=60.0,
        receptive_width_mm=18.0,
        temperature_c=40.0,
        noise_nm=0.0,
        seed=9,
    )

    assert result["sensor_positions_mm"].shape == (8, 2)
    assert result["measured_shift_nm"].shape == (8,)
    assert result["compensated_shift_nm"].shape == (8,)
    assert result["estimated_total_force_n"] == pytest.approx(10.0)
    assert result["load_error_n"] == pytest.approx(0.0, abs=1e-10)
    assert result["location_error_mm"] >= 0.0
    assert result["interpretation"] == "双点接触按载荷质心评价，不声称唯一分离两个接触点。"


@pytest.mark.parametrize("sensor_count", [4, 8, 16])
def test_fbg_skin_is_deterministic_for_every_supported_sensor_count(sensor_count):
    kwargs = dict(
        sensor_count=sensor_count, touch_points=[(35.0, 25.0, 7.0)],
        skin_width_mm=80.0, skin_height_mm=60.0, receptive_width_mm=20.0,
        temperature_c=31.0, noise_nm=0.002, seed=13,
    )
    first = simulate_fbg_skin(**kwargs)
    second = simulate_fbg_skin(**kwargs)
    np.testing.assert_allclose(first["measured_shift_nm"], second["measured_shift_nm"])


def test_pressure_reconstruction_returns_nonnegative_dense_fields_and_metrics():
    result = simulate_pressure_reconstruction(
        scenario="双点接触", sparse_size=4, output_size=16,
        peak_pressure_kpa=80.0, bandwidth=0.16, noise_kpa=0.0, seed=17,
    )

    assert result["truth_kpa"].shape == (16, 16)
    assert result["reconstruction_kpa"].shape == (16, 16)
    assert result["error_kpa"].shape == (16, 16)
    assert np.all(result["truth_kpa"] >= 0.0)
    assert np.all(result["reconstruction_kpa"] >= 0.0)
    assert result["rmse_kpa"] >= 0.0
    assert 0.0 <= result["channel_saving_pct"] < 100.0
    assert np.isfinite(result["centroid_error_pct"])
    assert np.isfinite(result["total_force_error_pct"])


def test_pressure_reconstruction_validates_grid_sizes():
    with pytest.raises(ValueError, match="sparse_size"):
        simulate_pressure_reconstruction("单点接触", 5, 16)
    with pytest.raises(ValueError, match="output_size"):
        simulate_pressure_reconstruction("单点接触", 4, 20)


def test_dynamic_stable_and_imminent_slip_scenarios_have_explainable_alerts():
    stable = simulate_dynamic_skin_event(
        "稳定按压", sample_rate_hz=100, duration_s=4.0,
        normal_force_n=12.0, slip_threshold=0.35, temperature_c=25.0,
        noise_ratio=0.0, seed=3,
    )
    imminent = simulate_dynamic_skin_event(
        "即将滑移", sample_rate_hz=100, duration_s=4.0,
        normal_force_n=12.0, slip_threshold=0.35, temperature_c=25.0,
        noise_ratio=0.0, seed=3,
    )

    assert len(stable["time_s"]) == 400
    assert stable["alert"] is False
    assert stable["status"] == "稳定"
    assert imminent["alert"] is True
    assert imminent["status"] == "滑移风险"
    assert imminent["threshold_margin"] > 0.0
    assert imminent["peak_centroid_speed_mm_s"] > stable["peak_centroid_speed_mm_s"]


def test_dynamic_thermal_event_and_repeat_summary_are_reproducible():
    thermal = simulate_dynamic_skin_event(
        "热物体", sample_rate_hz=50, duration_s=3.0,
        normal_force_n=8.0, slip_threshold=0.35, temperature_c=25.0,
        noise_ratio=0.01, seed=5,
    )
    assert thermal["temperature_c"][-1] > thermal["temperature_c"][0] + 10.0

    first = repeat_dynamic_event(
        "即将滑移", repeats=20, sample_rate_hz=50, duration_s=3.0,
        normal_force_n=8.0, slip_threshold=0.35, temperature_c=25.0,
        noise_ratio=0.02, seed=5,
    )
    second = repeat_dynamic_event(
        "即将滑移", repeats=20, sample_rate_hz=50, duration_s=3.0,
        normal_force_n=8.0, slip_threshold=0.35, temperature_c=25.0,
        noise_ratio=0.02, seed=5,
    )
    assert first == second
    assert first["repeat_count"] == 20
    assert 0.0 <= first["alert_rate_pct"] <= 100.0
    assert first["peak_ratio_std"] >= 0.0
