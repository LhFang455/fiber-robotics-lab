import numpy as np
import pytest
import subprocess
import tempfile
from pathlib import Path
from streamlit.testing.v1 import AppTest

import fiber_robotics_sim.models as models
import fiber_robotics_sim.visuals as visuals


FBG_SIMPLUS_TUTORIAL_EXPORT = """% Model: COMSOL tutorial\n% x solid.elogxx solid.elogyy solid.elogzz solid.sx solid.sy solid.sz T\n0.0000 0.0020 0.0001 -0.0002 100.0 20.0 -10.0 293.15\n0.0010 0.0010 0.0002 -0.0001 80.0 15.0 -8.0 294.15\n"""
FBG_SIMPLUS_CSV_EXPORT = """position,exx,eyy,ezz,sxx,syy,szz,temperature\n0.0000,0.0020,0.0001,-0.0002,100.0,20.0,-10.0,293.15\n0.0010,0.0010,0.0002,-0.0001,80.0,15.0,-8.0,294.15\n"""


def test_fbg_simplus_parser_reads_the_public_tutorial_column_order():
    result = models.parse_fbg_simplus_comsol_export(FBG_SIMPLUS_TUTORIAL_EXPORT)

    assert np.array_equal(result["position_m"], np.array([0.0, .001]))
    assert np.array_equal(result["longitudinal_strain"], np.array([.002, .001]))
    assert np.array_equal(result["transverse_stress_pa"], np.array([[20.0, -10.0], [15.0, -8.0]]))
    assert np.array_equal(result["temperature_k"], np.array([293.15, 294.15]))
    assert result["sample_count"] == 2


def test_fbg_simplus_parser_rejects_invalid_columns_and_nonmonotonic_positions():
    with pytest.raises(ValueError, match="八列"):
        models.parse_fbg_simplus_comsol_export("0 1 2")
    with pytest.raises(ValueError, match="严格递增"):
        models.parse_fbg_simplus_comsol_export(
            "0 0 0 0 0 0 0 293.15\n0 0 0 0 0 0 0 293.15"
        )


def test_fbg_simplus_parser_accepts_csv_after_skipping_a_header_and_normalises_it():
    result = models.parse_fbg_simplus_comsol_export(
        FBG_SIMPLUS_CSV_EXPORT, delimiter="逗号（CSV）", skip_rows=1
    )

    assert result["source_delimiter"] == "逗号（CSV）"
    assert np.array_equal(result["position_m"], np.array([0.0, .001]))
    assert models.fbg_simplus_normalised_text(result).splitlines()[0].split() == [
        "0", "0.002", "0.0001", "-0.0002", "100", "20", "-10", "293.15"
    ]


def test_fbg_simplus_input_figure_exposes_strain_stress_and_temperature():
    parsed_export = {
        "position_m": np.array([0.0, .001]),
        "longitudinal_strain": np.array([.002, .001]),
        "transverse_stress_pa": np.array([[20.0, -10.0], [15.0, -8.0]]),
        "temperature_k": np.array([293.15, 294.15]),
    }

    figure = visuals.fbg_simplus_input_figure(parsed_export)

    assert [trace.name for trace in figure.data] == ["纵向应变 εxx", "横向应力 σy / σz", "温度"]
    assert np.array_equal(figure.data[0].x, np.array([0.0, 1.0]))


def test_app_exposes_the_fbg_simplus_compatibility_module_and_attribution():
    source = Path(__file__).resolve().parents[1].joinpath("app.py").read_text(encoding="utf-8")

    assert "FBG-SimPlus 兼容" in source
    assert "benfrey/FBG-SimPlus" in source
    assert "git clone https://github.com/benfrey/FBG-SimPlus.git" in source
    assert "GPL-3.0" in source
    assert "Frey, B., Snyder, P., Ziock, K., & Passian, A. (2021)" in source
    assert "python3.8 -m venv .venv" in source
    assert "python -m pip install PyQt5 scipy matplotlib sympy six numpy" in source
    assert "python run.py" in source
    assert "Windows（PowerShell）" in source
    assert "macOS / Linux（Terminal / Bash）" in source
    assert "自动识别" in source
    assert "逗号（CSV）" in source
    assert "跳过文件开头行数" in source
    assert "标准化八列文本" in source
    assert "原生模型文件" in source
    assert "Skip Rows" in source
    assert "Path Distance Input Units" in source
    assert "Generate" in source


def test_fbg_shift_combines_strain_and_temperature():
    assert hasattr(models, "fbg_wavelength_shift_nm")
    shift = models.fbg_wavelength_shift_nm(np.array([1e-3]), 10.0)
    expected = 1550.0 * ((1 - 0.22) * 1e-3 + 6.7e-6 * 10.0)
    assert shift == pytest.approx(np.array([expected]))


def test_noise_is_reproducible_for_a_seed():
    assert hasattr(models, "add_gaussian_noise")
    values = np.zeros(3)
    assert np.array_equal(
        models.add_gaussian_noise(values, 0.01, 9),
        models.add_gaussian_noise(values, 0.01, 9),
    )


def test_finger_angle_round_trip_without_noise():
    result = models.simulate_finger(
        45.0, 80.0, 1.0, np.array([20.0, 40.0, 60.0]), 0.0, 0.0, 1
    )
    estimate = models.estimate_finger_angle_deg(
        result["wavelength_shifts_nm"], 80.0, 1.0, 0.0
    )
    assert estimate == pytest.approx(45.0)


def test_zero_finger_angle_creates_a_straight_centerline():
    result = models.simulate_finger(
        0.0, 80.0, 1.0, np.array([20.0, 40.0, 60.0]), 0.0, 0.0, 1
    )
    assert np.allclose(result["centerline_xy_mm"][:, 1], 0.0)
    assert result["estimated_angle_deg"] == pytest.approx(0.0)


def test_contact_round_trip_without_noise():
    positions = np.array([15.0, 35.0, 55.0])
    result = models.simulate_contact(37.0, 4.0, positions, 12.0, 2e-4, 0.0, 0.0, 3)
    position_hat, force_hat = models.estimate_contact(
        result["wavelength_shifts_nm"], positions, 12.0, 2e-4, 0.0
    )
    assert position_hat == pytest.approx(37.0, abs=0.5)
    assert force_hat == pytest.approx(4.0, rel=0.03)


def test_zero_contact_force_is_estimated_as_zero():
    positions = np.array([15.0, 35.0, 55.0])
    result = models.simulate_contact(30.0, 0.0, positions, 12.0, 2e-4, 0.0, 0.0, 3)
    _, force_hat = models.estimate_contact(
        result["wavelength_shifts_nm"], positions, 12.0, 2e-4, 0.0
    )
    assert force_hat == pytest.approx(0.0)


def test_foot_fbg_round_trip_recovers_six_zone_loads_and_cop_without_noise():
    loads = np.array([18.0, 26.0, 34.0, 52.0, 44.0, 30.0])
    result = models.simulate_foot_fbg(loads, 12.0, 0.0, 5)
    estimate = models.estimate_foot_load_distribution(result["wavelength_shifts_nm"], 12.0)

    assert np.allclose(estimate["zone_loads_n"], loads)
    assert estimate["cop_region"] == pytest.approx(np.dot(np.arange(6), loads) / loads.sum())
    assert np.asarray(estimate["cop_xy"]).shape == (2,)


def test_foot_zone_loads_follow_terrain_and_phase_envelope():
    heel = models.simulate_foot_zone_loads(180.0, "平地", 0.0, "支撑期")
    toe = models.simulate_foot_zone_loads(180.0, "平地", 100.0, "支撑期")
    swing = models.simulate_foot_zone_loads(180.0, "平地", 55.0, "摆动期")

    assert heel.sum() == pytest.approx(180.0)
    assert toe.sum() == pytest.approx(180.0)
    assert heel[3:].sum() > heel[:3].sum()
    assert toe[:3].sum() > toe[3:].sum()
    assert swing.sum() == pytest.approx(180.0 * 0.03)


def test_replaceable_sole_normal_assembly_matches_the_temperature_compensated_baseline():
    result = models.simulate_replaceable_sole_assembly("正常装配", 18.0)

    assert result["assembly_prediction"] == "装配预测通过"
    assert np.allclose(result["baseline_residual_ue"], 0.0)
    assert result["left_right_difference_ue"] == pytest.approx(0.0)


def test_replaceable_sole_insufficient_insertion_is_rejected_by_baseline_residual():
    result = models.simulate_replaceable_sole_assembly("压入不足", 18.0)

    assert result["assembly_prediction"] == "压入不足：预测不通过"
    assert result["mean_baseline_residual_ue"] < -45.0


def test_replaceable_sole_single_side_misalignment_is_rejected_by_left_right_difference():
    result = models.simulate_replaceable_sole_assembly("单侧错位", 18.0)

    assert result["assembly_prediction"] == "单侧错位：预测不通过"
    assert result["left_right_difference_ue"] > 70.0
    assert result["transfer_centroid_x_mm"] > 0.0


def test_replaceable_sole_transfer_figure_is_a_renderable_two_dimensional_result_template():
    result = models.simulate_replaceable_sole_assembly("正常装配", 0.0)

    figure = visuals.replaceable_sole_transfer_figure(result)

    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"


def test_replaceable_sole_explainer_figure_connects_structure_check_and_gait_use():
    assembly = models.simulate_replaceable_sole_assembly("正常装配", 0.0)

    figure = visuals.replaceable_sole_explainer_figure(
        assembly, np.full(6, 30.0), "平地", 2.5, "支撑期"
    )

    assert len(figure.data) >= 2
    assert any("空载自检" in annotation.text for annotation in figure.layout.annotations)
    assert any("可更换" in annotation.text for annotation in figure.layout.annotations)


def test_sole_component_explorer_visually_separates_structure_seal_and_signal_paths():
    assembly = models.simulate_replaceable_sole_assembly("单侧错位", 0.0)

    figure = visuals.sole_component_explorer_figure(assembly, "周向密封圈")

    trace_names = {trace.name for trace in figure.data}
    assert {"受力路径", "密封路径", "光纤信号路径"} <= trace_names
    assert any("单侧错位" in annotation.text for annotation in figure.layout.annotations)
    assert len(figure.layout.shapes) >= 8


def test_replaceable_sole_tolerance_scan_is_reproducible_and_counts_every_case():
    first = models.simulate_replaceable_sole_tolerance_scan(40, 12.0, 0.002, 13)
    second = models.simulate_replaceable_sole_tolerance_scan(40, 12.0, 0.002, 13)

    assert first["confusion_matrix"].shape == (3, 3)
    assert np.array_equal(first["confusion_matrix"], second["confusion_matrix"])
    assert np.array_equal(first["confusion_matrix"].sum(axis=1), np.full(3, 40))


def test_reference_temperature_mismatch_creates_a_nonzero_baseline_bias():
    matched = models.simulate_reference_temperature_mismatch(15.0, 15.0)
    mismatched = models.simulate_reference_temperature_mismatch(15.0, 14.0)

    assert matched["baseline_bias_ue"] == pytest.approx(0.0)
    assert mismatched["baseline_bias_ue"] > 0.0


def test_seal_compression_screen_exposes_the_lowest_compression_location_without_ip_claim():
    centered = models.simulate_seal_compression_screen(0.20, 0.0)
    offset = models.simulate_seal_compression_screen(0.20, 0.80)

    assert centered["minimum_compression_ratio"] == pytest.approx(0.20)
    assert offset["minimum_compression_ratio"] < centered["minimum_compression_ratio"]
    assert offset["validation_boundary"] == "需要密封实物试验"


def test_preload_retention_sensitivity_is_monotonic_for_an_assumed_cycle_loss_rate():
    result = models.simulate_preload_retention_sensitivity(5000, 0.985)

    assert result["cycle_count"].shape == result["preload_ue"].shape
    assert result["preload_ue"][0] == pytest.approx(240.0)
    assert result["preload_ue"][-1] < result["preload_ue"][0]
    assert result["validation_boundary"] == "需要循环装拆与载荷试验"


def test_operational_load_is_separated_from_the_empty_load_assembly_check():
    result = models.simulate_assembly_operational_load_interference(180.0, 0.0)

    assert result["operational_signal_ue"].mean() > 0.0
    assert result["assembly_check_condition"] == "仅空载"
    assert result["validation_boundary"] == "需要步态载荷试验"


def test_foot_sensing_readiness_requires_an_empty_load_normal_assembly_candidate():
    normal_empty = models.assess_replaceable_sole_sensing_readiness("正常装配", 0.0)
    normal_loaded = models.assess_replaceable_sole_sensing_readiness("正常装配", 180.0)
    offset_empty = models.assess_replaceable_sole_sensing_readiness("单侧错位", 0.0)

    assert normal_empty["can_enter_foot_sensing_flow"] is True
    assert normal_loaded["can_enter_foot_sensing_flow"] is False
    assert offset_empty["can_enter_foot_sensing_flow"] is False


def test_discrete_sole_transfer_sensitivity_field_moves_with_the_assumed_lateral_offset():
    centered = models.solve_sole_transfer_sensitivity_field(1.0, 0.0)
    offset = models.solve_sole_transfer_sensitivity_field(1.0, 4.0)

    assert centered["relative_transfer"].ndim == 2
    assert centered["transfer_centroid_x_mm"] == pytest.approx(0.0, abs=0.3)
    assert offset["transfer_centroid_x_mm"] > 1.0
    assert centered["validation_boundary"] == "需要材料参数标定与有限元复核"


def test_tolerance_confusion_figure_is_renderable():
    scan = models.simulate_replaceable_sole_tolerance_scan(20, 0.0, 0.0, 5)

    figure = visuals.assembly_tolerance_confusion_figure(scan)

    assert len(figure.data) == 1
    assert figure.data[0].type == "heatmap"


def test_assembly_screening_line_figures_are_renderable():
    seal = models.simulate_seal_compression_screen(0.20, 0.80)
    retention = models.simulate_preload_retention_sensitivity(5000, 0.985)

    seal_figure = visuals.seal_compression_screen_figure(seal)
    retention_figure = visuals.preload_retention_sensitivity_figure(retention)

    assert len(seal_figure.data) == 1
    assert len(retention_figure.data) == 1


def test_arm_health_monitoring_localises_a_damage_peak_after_temperature_compensation():
    result = models.simulate_arm_health_fbg(80.0, 315.0, 0.70, 16.0, 0.0, 8)
    diagnosis = models.diagnose_arm_health(result["wavelength_shifts_nm"], 16.0)

    assert diagnosis["status"] == "需检查"
    assert diagnosis["suspected_location_mm"] == pytest.approx(320.0, abs=5.0)
    assert diagnosis["damage_index"] > 0.35
    assert diagnosis["location_uncertainty_mm"] == pytest.approx(60.0)


def test_arm_health_localisation_reports_uncertainty_and_improves_with_density():
    sparse = models.simulate_arm_health_fbg(80.0, 315.0, 0.70, 16.0, 0.0, 8)
    sparse_diagnosis = models.diagnose_arm_health(sparse["wavelength_shifts_nm"], 16.0)
    dense_positions = np.linspace(80.0, 440.0, 8)
    dense = models.simulate_arm_health_fbg(80.0, 315.0, 0.70, 16.0, 0.0, 8, sensor_positions_mm=dense_positions)
    dense_diagnosis = models.diagnose_arm_health(dense["wavelength_shifts_nm"], 16.0, sensor_positions_mm=dense_positions)

    assert sparse_diagnosis["location_uncertainty_mm"] > dense_diagnosis["location_uncertainty_mm"]
    assert abs(sparse_diagnosis["suspected_location_mm"] - 315.0) < 30.0


def test_redundant_fbg_diagnosis_excludes_a_drifting_channel_from_angle_estimate():
    result = models.simulate_redundant_finger_fbg(45.0, 80.0, 1.0, 10.0, "漂移", 2)
    diagnosis = models.diagnose_redundant_fbg(result["wavelength_shifts_nm"], 80.0, 1.0, 10.0)

    assert diagnosis["fault_channels"] == [2]
    assert diagnosis["estimated_angle_deg"] == pytest.approx(45.0)


def test_tactile_material_classifier_recovers_the_teaching_material_label():
    result = models.simulate_material_touch("海绵", 5.0, 20.0, 0.0)
    diagnosis = models.classify_tactile_material(result["finger_touch_n"], result["palm_touch_n"])

    assert diagnosis["material"] == "海绵"
    assert diagnosis["confidence"] >= 0.70
    assert diagnosis["probabilities"][diagnosis["material"]] == max(diagnosis["probabilities"].values())
    assert sum(diagnosis["probabilities"].values()) == pytest.approx(1.0)


def test_tactile_pattern_noise_can_misclassify_and_lowers_confidence():
    noisy = models.simulate_material_touch("海绵", 5.0, 20.0, 0.0, pattern_noise=0.60, noise_nm=0.0, seed=21)
    diagnosis = models.classify_tactile_material(noisy["finger_touch_n"], noisy["palm_touch_n"])

    assert diagnosis["material"] != "海绵"
    assert diagnosis["confidence"] < 0.95
    assert sum(diagnosis["probabilities"].values()) == pytest.approx(1.0)


def test_tactile_material_simulation_is_reproducible_for_a_seed():
    first = models.simulate_material_touch("硬块", 6.0, 30.0, 5.0, pattern_noise=0.15, noise_nm=0.01, seed=11)
    second = models.simulate_material_touch("硬块", 6.0, 30.0, 5.0, pattern_noise=0.15, noise_nm=0.01, seed=11)

    assert np.allclose(first["wavelength_shifts_nm"], second["wavelength_shifts_nm"])


def test_demodulation_chain_filters_and_temperature_compensates_a_control_signal():
    result = models.simulate_demodulation_chain(55.0, 15.0, 100, 0.010, 4)

    assert result["raw_wavelength_nm"].shape == result["filtered_wavelength_nm"].shape
    assert np.std(result["filtered_wavelength_nm"] - result["compensated_wavelength_nm"]) < 1e-12
    assert result["estimated_angle_deg"] == pytest.approx(55.0, abs=4.0)


def test_demodulation_figure_uses_time_as_a_real_plot_axis():
    result = models.simulate_demodulation_chain(55.0, 15.0, 100, 0.010, 4)
    figure = visuals.demodulation_figure(result)

    assert len(figure.data) == 3
    assert np.array_equal(figure.data[0].x, result["time_s"])


def test_sensor_frame_keeps_raw_compensated_and_quality_information_together():
    frame = models.build_sensor_frame("Rayleigh/OFDR", np.array([0.0, 10.0]), np.array([.01, .02]), np.array([.001, .002]), .92)

    assert frame["sensor_type"] == "Rayleigh/OFDR"
    assert frame["position_or_channel"].shape == (2,)
    assert frame["quality"] == pytest.approx(.92)


def test_distributed_models_return_spatial_and_temporal_measurements():
    rayleigh = models.simulate_rayleigh_ofdr(300.0, 140.0, 800.0, 1.0)
    das = models.simulate_das_event(300.0, 140.0, 60.0, 100)
    brillouin = models.simulate_brillouin_distribution(300.0, 40.0, 600.0)
    raman = models.simulate_raman_temperature(300.0, 140.0, 55.0)

    assert rayleigh["strain_ue"].ndim == 1
    assert das["amplitude"].ndim == 2
    assert brillouin["brillouin_frequency_ghz"].shape == brillouin["temperature_c"].shape
    assert raman["temperature_c"].max() > raman["temperature_c"].min()


def test_distributed_visuals_render_a_curve_and_a_time_distance_heatmap():
    rayleigh = models.simulate_rayleigh_ofdr(300.0, 140.0, 800.0, 1.0)
    das = models.simulate_das_event(300.0, 140.0, 60.0, 100)

    assert visuals.distributed_curve_figure(rayleigh, "Rayleigh").data[0].type == "scatter"
    assert visuals.das_event_figure(das).data[0].type == "heatmap"


def test_polarization_gyro_and_efpi_models_produce_their_distinct_optical_observables():
    polarization = models.simulate_polarization_sensing(120.0, 35.0, 8.0)
    gyro = models.simulate_sagnac_gyro(45.0, 120.0)
    efpi = models.simulate_efpi_pressure(0.4, 28.0)

    assert np.linalg.norm(polarization["stokes"]) == pytest.approx(1.0)
    assert gyro["phase_shift_rad"] > 0.0
    assert np.ptp(efpi["intensity"]) > 0.1


def test_multimodal_fusion_requires_quality_and_safety_to_report_ready():
    def quality(score: float) -> models.ModuleQuality:
        return models.ModuleQuality(score, 1.0, "test")

    ready = models.fuse_robot_sensing({
        "grasp": quality(1.0), "foot": quality(.85), "shape": quality(.90),
        "health": quality(1.0), "distributed": quality(.92),
    })
    warning = models.fuse_robot_sensing({
        "grasp": quality(1.0), "foot": quality(.85), "shape": quality(.90),
        "health": quality(.40), "distributed": quality(.92),
    })

    assert ready["status"] == "任务就绪"
    assert warning["status"] == "需人工复核"


def test_module_quality_assessors_are_bounded_and_center_aware():
    assert models.assess_foot_quality(2.5).score == pytest.approx(1.0)
    assert models.assess_foot_quality(5.0).score == pytest.approx(0.0)
    assert models.assess_shape_quality(8.0, 8.0).score == pytest.approx(1.0)
    assert models.assess_health_quality("需检查", 0.5).score < 0.75
    assert 0.0 <= models.assess_foot_quality(0.0).score <= 1.0


def test_simulate_distributed_mechanism_returns_result_and_quality_frame():
    for mode in ("Rayleigh/OFDR", "φ-OTDR / DAS", "Brillouin", "Raman"):
        result, frame = models.simulate_distributed_mechanism(mode, 300.0, 140.0, 600.0, sample_rate_hz=100)
        assert 0.0 < float(frame["quality"]) <= 1.0
        assert len(np.asarray(frame["position_or_channel"])) > 0
        assert len(result["position_mm"]) > 0


def test_decimate_distributed_result_reduces_spatial_points():
    fine = models.simulate_rayleigh_ofdr(300.0, 140.0, 600.0, 1.0)
    coarse = models.decimate_distributed_result(fine, 20.0)
    assert len(coarse["position_mm"]) < len(fine["position_mm"])
    assert coarse["strain_ue"].shape == coarse["position_mm"].shape
    das = models.simulate_das_event(300.0, 140.0, 60.0, 100)
    coarse_das = models.decimate_distributed_result(das, 20.0)
    assert coarse_das["amplitude"].shape[1] < das["amplitude"].shape[1]
    assert coarse_das["amplitude"].shape[0] == das["amplitude"].shape[0]


def test_distributed_finger_figure_highlights_contacts():
    result = models.simulate_distributed_sensing(np.full(5, 60.0), [0, 1, 2])
    figure = visuals.distributed_finger_figure(result, [0, 1, 2])
    assert len(figure.data) == 5
    assert any("（接触）" in trace.name for trace in figure.data)


def test_arm_distributed_vs_fbg_figure_overlays_discrete_and_continuous():
    distributed = models.simulate_rayleigh_ofdr(520.0, 320.0, 400.0, 1.0)
    positions = np.array([80.0, 200.0, 320.0, 440.0])
    figure = visuals.arm_distributed_vs_fbg_figure(distributed, positions, np.full(4, 100.0), 320.0, 60.0)
    assert len(figure.data) == 2
    assert "分布式" in figure.layout.title.text


def test_brillouin_raman_compensation_recovers_strain_after_temperature_decoupling():
    result = models.simulate_brillouin_raman_compensation(300.0, 140.0, 400.0, 5.0, 15.0, temperature_noise_c=0.0)
    noisy = models.simulate_brillouin_raman_compensation(300.0, 140.0, 400.0, 5.0, 15.0, temperature_noise_c=1.0)

    assert result["naive_peak_error_ue"] > 200.0
    assert result["compensated_peak_error_ue"] < 1.0
    assert 0.0 < noisy["compensated_peak_error_ue"] < noisy["naive_peak_error_ue"]


def test_brillouin_raman_compensation_figure_renders():
    result = models.simulate_brillouin_raman_compensation(300.0, 140.0, 400.0, 5.0, 15.0)
    figure = visuals.brillouin_raman_compensation_figure(result)
    assert len(figure.data) >= 4


def test_polarization_and_efpi_visuals_render_state_and_interference():
    polarization = models.simulate_polarization_sensing(120.0, 35.0, 8.0)
    efpi = models.simulate_efpi_pressure(0.4, 28.0)

    assert visuals.polarization_figure(polarization).data[0].type == "scatter3d"
    assert visuals.efpi_figure(efpi).data[0].type == "scatter"


def test_multicore_recovers_curvature_and_direction_without_noise():
    result = models.simulate_multicore_shape(8.0, 35.0, 0.0, 150.0, 125.0, 20.0, 0.0, 4)
    curvature, direction = models.estimate_multicore_curvature(
        result["wavelength_shifts_nm"], 125.0
    )
    assert curvature == pytest.approx(8.0)
    assert direction == pytest.approx(35.0)


def test_multicore_differential_estimate_rejects_common_temperature_shift():
    cold = models.simulate_multicore_shape(6.0, 70.0, 0.0, 100.0, 125.0, 0.0, 0.0, 5)
    hot = models.simulate_multicore_shape(6.0, 70.0, 0.0, 100.0, 125.0, 25.0, 0.0, 5)
    assert models.estimate_multicore_curvature(cold["wavelength_shifts_nm"], 125.0) == pytest.approx(
        models.estimate_multicore_curvature(hot["wavelength_shifts_nm"], 125.0)
    )


def test_multicore_temperature_gradient_biases_the_differential_estimate():
    clean = models.simulate_multicore_shape(8.0, 35.0, 0.0, 150.0, 125.0, 20.0, 0.0, 4, core_temperature_gradient_c=0.0)
    graded = models.simulate_multicore_shape(8.0, 35.0, 0.0, 150.0, 125.0, 20.0, 0.0, 4, core_temperature_gradient_c=10.0)

    clean_curvature, _ = models.estimate_multicore_curvature(clean["wavelength_shifts_nm"], 125.0)
    graded_curvature, _ = models.estimate_multicore_curvature(graded["wavelength_shifts_nm"], 125.0)
    assert clean_curvature == pytest.approx(8.0)
    assert abs(graded_curvature - 8.0) > 0.5
    assert graded["temperature_gradient_c"] == pytest.approx(10.0)
    assert np.allclose(graded["core_temperature_change_c"], [10.0, 20.0, 30.0])


def test_shape_distributed_link_returns_per_core_and_rayleigh_profiles():
    result = models.simulate_multicore_shape(8.0, 35.0, 0.0, 150.0, 125.0, 20.0, 0.0, 4)
    link = models.simulate_shape_distributed_link(150.0, result["strain"], 8.0)
    assert link["core_strain_ue"].shape == (3, len(link["position_mm"]))
    assert link["rayleigh_strain_ue"].shape == link["position_mm"].shape
    assert link["rayleigh_strain_ue"].max() > 0.0


def test_shape_distributed_link_figure_renders():
    link = models.simulate_shape_distributed_link(150.0, np.array([1e-3, 2e-3, 1.5e-3]), 8.0)
    figure = visuals.shape_distributed_link_figure(link)
    assert len(figure.data) == 4


def test_simulate_polarization_map_returns_grid():
    result = models.simulate_polarization_map(temperature_change_c=5.0, grid=11)
    assert result["azimuth_deg"].shape == (11, 11)
    assert result["ellipticity_deg"].shape == (11, 11)
    assert 0.0 <= result["azimuth_deg"].min() <= result["azimuth_deg"].max() < 180.0


def test_polarization_map_figure_renders():
    result = models.simulate_polarization_map(grid=11)
    figure = visuals.polarization_map_figure(result)
    assert sum(trace.type == "heatmap" for trace in figure.data) == 2


def test_sensing_chain_svg_contains_the_four_stages():
    markup = visuals.sensing_chain_svg()
    assert markup.startswith("<svg")
    assert "光纤传感" in markup and "控制与任务" in markup


def test_finger_figure_has_centerline_trace():
    result = models.simulate_finger(
        30.0, 80.0, 1.0, np.array([20.0, 40.0, 60.0]), 0.0, 0.0, 1
    )
    assert hasattr(visuals, "finger_figure")
    assert len(visuals.finger_figure(result).data) >= 2


def test_arm_figure_contains_arm_object_and_fiber_route():
    figure = visuals.arm_figure("抓取", "手指背侧", 45.0, 4.0)
    assert len(figure.data) >= 3


def test_arm_health_figure_marks_four_fbg_positions_and_the_suspected_location():
    result = models.simulate_arm_health_fbg(80.0, 315.0, 0.70, 0.0, 0.0, 1)
    diagnosis = models.diagnose_arm_health(result["wavelength_shifts_nm"], 0.0)
    figure = visuals.arm_health_figure(result, diagnosis)

    assert len(figure.data) == 2
    assert "可疑位置" in figure.layout.title.text


def test_arm_action_changes_planar_joint_coordinates():
    raised = visuals.arm_figure("抬臂", "手指背侧", 15.0, 0.0)
    grasping = visuals.arm_figure("抓取", "手指背侧", 55.0, 4.0)
    assert not np.allclose(raised.data[0].x, grasping.data[0].x)
    assert not np.allclose(raised.data[0].y, grasping.data[0].y)


def test_dexterous_hand_pose_has_five_fingers_and_grasp_closes_them():
    opened = models.dexterous_hand_pose("伸手")
    grasping = models.dexterous_hand_pose("抓取")
    assert len(opened["fingers"]) == 5
    assert np.linalg.norm(grasping["fingers"][1][-1] - grasping["palm_center"]) < np.linalg.norm(
        opened["fingers"][1][-1] - opened["palm_center"]
    )


def test_translated_hand_pose_moves_every_planar_feature_by_the_same_offset():
    base = models.dexterous_hand_pose("抓取")
    moved = models.dexterous_hand_pose("抓取", planar_translation=(2.5, -1.25))

    assert np.allclose(np.asarray(moved["arm_joints"]) - np.asarray(base["arm_joints"]), [2.5, -1.25])
    assert np.allclose(np.asarray(moved["palm_outline"]) - np.asarray(base["palm_outline"]), [2.5, -1.25])
    assert np.allclose(np.asarray(moved["fingers"][0]) - np.asarray(base["fingers"][0]), [2.5, -1.25])


def test_arm_figure_renders_five_fingers_and_five_fiber_branches():
    figure = visuals.arm_figure("抓取", "手指背侧", 55.0, 4.0)
    names = [trace.name for trace in figure.data]
    assert sum(name.startswith("手指") for name in names) == 5
    assert sum(name.startswith("FBG 支路") for name in names) == 5


def test_arm_3d_figure_contains_five_finger_traces():
    figure = visuals.arm_3d_figure("抓取", "手指背侧", 55.0)
    assert sum(trace.name.startswith("手指") for trace in figure.data) == 5


def test_can_grasp_requires_thumb_and_two_other_contacts():
    pose = models.dexterous_hand_pose("抓取")
    grasp = models.evaluate_can_grasp(pose, np.asarray(pose["target"]))
    opened = models.dexterous_hand_pose("伸手")
    open_grasp = models.evaluate_can_grasp(opened, np.asarray(opened["target"]))
    assert grasp["is_grasped"] is True
    assert 0 in grasp["contact_fingers"]
    assert len([index for index in grasp["contact_fingers"] if index != 0]) >= 2
    assert open_grasp["is_grasped"] is False


def test_planar_grasp_success_is_decided_from_temperature_compensated_fbg_contact_channels():
    sensed = models.simulate_planar_grasp_fbg((63.0, 84.0, 84.0, 84.0, 84.0), (1.0, 0.8, 0.6, 0.4, 0.3), 9.0)
    displaced = models.simulate_planar_grasp_fbg((63.0, 84.0, 84.0, 84.0, 84.0), np.zeros(5), 9.0)

    assert models.classify_planar_grasp_from_fbg(sensed, (63.0, 84.0, 84.0, 84.0, 84.0), 9.0)["is_grasped"] is True
    assert models.classify_planar_grasp_from_fbg(displaced, (63.0, 84.0, 84.0, 84.0, 84.0), 9.0)["is_grasped"] is False
    recovered = models.classify_planar_grasp_from_fbg(sensed, (63.0, 84.0, 84.0, 84.0, 84.0), 9.0)
    assert np.allclose(recovered["contact_force_n"], (1.0, 0.8, 0.6, 0.4, 0.3))


def test_planar_grasp_exposes_a_sixth_palm_fbg_without_changing_the_finger_grasp_rule():
    sensed = models.simulate_planar_grasp_fbg((63.0, 84.0, 84.0, 84.0, 84.0), (0.9, 0.6, 0.4, 0.0, 0.0), 0.0)
    decision = models.classify_planar_grasp_from_fbg(sensed, (63.0, 84.0, 84.0, 84.0, 84.0), 0.0)

    assert sensed["wavelength_shifts_nm"].shape == (6,)
    assert decision["is_grasped"] is True
    assert decision["palm_contact"] is True
    assert decision["palm_touch_n"] > 0.0


def test_grasp_calibration_changes_channel_sensitivity():
    curls = (63.0, 84.0, 84.0, 84.0, 84.0)
    forces = (1.0, 0.5, 0.3, 0.0, 0.0)
    default = models.simulate_planar_grasp_fbg(curls, forces, 0.0)
    sensitive = models.simulate_planar_grasp_fbg(
        curls, forces, 0.0,
        calibration=models.GraspCalibration(contact_strain_per_n=3.4e-4),
    )

    assert sensitive["contact_strain"][0] == pytest.approx(2.0 * default["contact_strain"][0])


def test_planar_animation_html_draws_the_palm_fbg_route():
    pose = models.dexterous_hand_pose("抓取")
    markup = visuals.planar_hand_animation_html(
        pose, pose, pose["target"], pose["target"], True, True,
        previous_contact_fingers=[0, 1, 2],
        current_contact_fingers=[0, 1, 2],
    )

    assert "__CONFIG__" not in markup
    assert "palmFibre" in markup
    assert '"contacts": [0, 1, 2]' in markup
    assert "#ffcc66" in markup and "#ffe06a" in markup
    assert "三维" not in markup


def test_planar_animation_frame_matches_the_shared_transition_contract():
    pose = models.dexterous_hand_pose("抓取")
    frame = visuals.planar_animation_frame(pose, pose["target"], True, [0, 1, 2])

    assert set(frame) == {"arm", "palm", "palmFibre", "fingers", "fibres", "can", "grasped", "contacts"}
    assert frame["contacts"] == [0, 1, 2]
    assert frame["grasped"] is True


def test_3d_grasp_sensing_is_independent_and_responds_to_depth_offset():
    grasped = models.evaluate_3d_grasp_sensing((72.0, 84.0, 84.0, 84.0, 84.0), (0.0, 0.0, 0.0))
    moved_away = models.evaluate_3d_grasp_sensing((72.0, 84.0, 84.0, 84.0, 84.0), (0.0, 0.0, 2.5))

    assert grasped["is_grasped"] is True
    assert 0 in grasped["contact_fingers"]
    assert len(grasped["contact_fingers"]) >= 3
    assert grasped["fbg_shifts_nm"].shape == (5,)
    assert moved_away["is_grasped"] is False
    assert moved_away["contact_force_n"].sum() < grasped["contact_force_n"].sum()


def test_3d_grasp_sensing_separates_arm_bend_from_palm_and_full_finger_touch_channels():
    result = models.evaluate_3d_grasp_sensing(
        (72.0, 84.0, 84.0, 84.0, 84.0), (0.0, 0.0, 0.0),
        arm_joint_angles_deg=(38.0, -58.0, 18.0),
        finger_joint_angles_deg=((90.0, 90.0), (72.0, 104.0, 74.0), (72.0, 104.0, 74.0), (72.0, 104.0, 74.0), (72.0, 104.0, 74.0)),
    )

    assert result["arm_bend_strain_ue"].shape == (3,)
    assert result["palm_touch_n"] > 0.0
    assert result["finger_segment_touch_n"].shape == (14,)
    assert result["tactile_fbg_shifts_nm"].shape == (15,)
    assert result["arm_fbg_shifts_nm"].shape == (3,)


def test_three_d_grasp_uses_a_palm_top_cylinder_and_clamps_fingers_at_collision():
    desired = ((90.0, 90.0), *((72.0, 104.0, 74.0),) * 4)
    result = models.evaluate_3d_grasp_sensing(
        (90.0, 83.3, 83.3, 83.3, 83.3),
        (0.0, 0.0, 0.0),
        finger_joint_angles_deg=desired,
    )

    assert np.allclose(result["can_center_xyz"], (0.30, -0.20, 0.76))
    assert result["palm_support_contact"] is True
    assert result["is_grasped"] is True
    assert np.all(result["collision_clearance"] >= -1e-3)
    assert all(
        np.all(np.asarray(actual) <= np.asarray(commanded) + 1e-9)
        for actual, commanded in zip(result["collision_limited_joint_angles_deg"], desired)
    )
    # 逐关节限位：近端关节可以完全屈曲，只有远端关节绕罐体折叠。
    assert result["collision_limited_joint_angles_deg"][1][1] == pytest.approx(104.0)
    assert result["collision_limited_joint_angles_deg"][1][2] < 74.0


def test_thumb_collision_capsules_bend_the_ip_around_the_renderer_z_axis():
    capsules = models._finger_capsules(0, (58.0, 74.0))
    tip = capsules[-1][1]
    lengths = models._HAND_FINGER_LENGTHS[0]
    transform = models._rotation_z(models._HAND_FINGER_SPREADS[0]) @ models._rotation_y(-np.deg2rad(58.0))
    point = models._HAND_FINGER_BASES[0].copy() + transform @ np.array((lengths[0], 0.0, 0.0))
    transform = transform @ models._rotation_z(np.deg2rad(74.0))
    point = point + transform @ np.array((lengths[1], 0.0, 0.0))

    assert np.allclose(tip, point)
    wrong_transform = models._rotation_z(models._HAND_FINGER_SPREADS[0]) @ models._rotation_y(-np.deg2rad(58.0))
    wrong_point = models._HAND_FINGER_BASES[0].copy() + wrong_transform @ np.array((lengths[0], 0.0, 0.0))
    wrong_transform = wrong_transform @ models._rotation_y(-np.deg2rad(74.0))
    wrong_point = wrong_point + wrong_transform @ np.array((lengths[1], 0.0, 0.0))
    assert np.linalg.norm(tip - wrong_point) > 0.5


def test_three_d_target_remains_fixed_while_the_hand_reaches_its_world_position():
    target = np.array((1.3, -0.8, 0.6))
    assert np.allclose(models.relative_3d_target_offset(target, target), (0.0, 0.0, 0.0))
    assert np.allclose(models.relative_3d_target_offset(target, (0.0, 0.0, 0.0)), target)


def test_three_d_grasp_success_is_classified_from_temperature_compensated_fbg_channels():
    desired = ((90.0, 90.0), *((72.0, 104.0, 74.0),) * 4)
    closed = models.evaluate_3d_grasp_sensing(
        (90.0, 83.3, 83.3, 83.3, 83.3), (0.0, 0.0, 0.0), 12.0,
        finger_joint_angles_deg=desired,
    )
    displaced = models.evaluate_3d_grasp_sensing(
        (90.0, 83.3, 83.3, 83.3, 83.3), (1.4, 0.0, 0.0), 12.0,
        finger_joint_angles_deg=desired,
    )

    assert models.classify_3d_grasp_from_fbg(closed, 12.0)["is_grasped"] is True
    assert models.classify_3d_grasp_from_fbg(displaced, 12.0)["is_grasped"] is False


def test_anthropomorphic_renderer_routes_fibre_across_arm_palm_and_every_finger_segment():
    markup = visuals.anthropomorphic_hand_html(
        "三维初始", (38.0, -58.0, 18.0), (0.0, 0.0, 0.0, 0.0, 0.0), False
    )

    assert "armFiber" in markup
    assert "palmFiberA" in markup
    assert "segmentFiber" in markup
    assert "segmentFiber.quaternion.copy(bone.quaternion)" in markup


def test_can_offset_from_target_uses_hand_forward_and_lateral_axes():
    pose = models.dexterous_hand_pose("抓取")
    joints = np.asarray(pose["arm_joints"])
    forward = joints[3] - joints[2]
    forward /= np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    can_center = np.asarray(pose["target"]) + 1.2 * forward - .4 * lateral

    assert np.allclose(models.can_offset_from_target(pose, can_center), [1.2, -.4])


def test_grasp_task_only_advances_to_transport_after_a_verified_closure():
    assert models.next_grasp_task_phase("寻找目标", False) == "对准目标"
    assert models.next_grasp_task_phase("对准目标", False) == "闭合抓取"
    assert models.next_grasp_task_phase("闭合抓取", False) == "抓取失败"
    assert models.next_grasp_task_phase("闭合抓取", True) == "搬运目标"
    assert models.next_grasp_task_phase("搬运目标", True) == "松开并放置"


def test_arm_3d_figure_has_can_mesh_trace():
    figure = visuals.arm_3d_figure("抓取", "手指背侧", 55.0)
    assert any(trace.name == "铝制饮料罐" for trace in figure.data)


def test_app_exposes_independent_thumb_control():
    app = AppTest.from_file("app.py").run()
    assert any(slider.label == "拇指屈曲 (°)" for slider in app.slider)


def test_app_exposes_can_position_controls():
    app = AppTest.from_file("app.py").run()
    controls = {slider.label: slider for slider in app.slider}
    assert {"饮料罐水平位置", "饮料罐垂直位置"} <= controls.keys()
    assert controls["饮料罐水平位置"].disabled is False


def test_app_exposes_the_stepwise_planar_find_grasp_and_place_actions():
    app = AppTest.from_file("app.py").run()
    labels = {button.label for button in app.button}
    assert {"开始二维寻找与抓取任务", "执行下一步"} <= labels


def test_app_keeps_the_planar_can_position_controls_for_world_targets():
    app = AppTest.from_file("app.py").run()
    labels = {slider.label for slider in app.slider}
    assert {"饮料罐水平位置", "饮料罐垂直位置"} <= labels


def test_app_exposes_shoulder_translation_controls():
    app = AppTest.from_file("app.py").run()
    labels = {slider.label for slider in app.slider}
    assert {"肩部水平位移", "肩部垂直位移"} <= labels


def test_app_keeps_automatically_aligned_shoulder_offsets_inside_the_control_range():
    source = Path("app.py").read_text(encoding="utf-8")

    assert 'st.slider("肩部水平位移", -12.0, 12.0' in source
    assert 'st.slider("肩部垂直位移", -12.0, 12.0' in source
    assert 'shoulder_translation_x = float(can_world[0] - target[0])' in source
    assert 'shoulder_translation_z = float(can_world[1] - target[1])' in source
    assert "can_depth_z" not in source


def test_distributed_sensing_returns_five_rayleigh_profiles_and_das_map():
    result = models.simulate_distributed_sensing(np.full(5, 60.0), [0, 1, 2])
    assert result["rayleigh_strain_ue"].shape[0] == 5
    assert result["das_amplitude"].ndim == 2


def test_distributed_figures_render_heatmaps():
    result = models.simulate_distributed_sensing(np.full(5, 60.0), [0, 1, 2])
    assert visuals.rayleigh_heatmap_figure(result).data[0].type == "heatmap"
    assert visuals.das_heatmap_figure(result).data[0].type == "heatmap"


def test_arm_3d_figure_contains_engineering_body_meshes():
    figure = visuals.arm_3d_figure("抓取", "手指背侧", 55.0)
    names = {trace.name for trace in figure.data}
    assert {"锥形前臂", "腕部关节", "一体化掌壳"} <= names


def test_arm_3d_figure_uses_solid_finger_meshes_attached_to_an_integrated_palm():
    figure = visuals.arm_3d_figure("抓取", "手指背侧", 55.0)
    names = {trace.name for trace in figure.data}

    assert "一体化掌壳" in names
    assert sum(name.startswith("实体手指") for name in names) == 5


def test_anthropomorphic_hand_renderer_has_arm_palm_thumb_and_five_articulated_fingers():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)

    assert "掌心轮廓" in markup
    assert "makeFingerFromData" in markup
    assert "makeArm" in markup
    assert "const links=[[6.3,.43" in markup
    assert "pivot.rotation.z=a[n]" in markup
    assert "link.rotation.z=-Math.PI/2" in markup
    assert '"fingerCapsules"' in markup
    assert '"previousFingerCapsules"' in markup


def test_anthropomorphic_hand_renderer_can_apply_a_page_specific_finger_curl_gain():
    markup = visuals.anthropomorphic_hand_html(
        "三维抓取", (38, -58, 18), (72, 84, 84, 84, 84), True, finger_curl_gain=1.65
    )

    assert '"curlGain": 1.65' in markup
    assert '"fingerJoints"' in markup
    assert '"fingerCapsules"' in markup
    assert "当前动作：" in markup
    assert "Orbit-like" in markup
    assert "can.visible=true" in markup
    assert "三维动作" not in markup
    assert "makeFingerFromData" in markup


def test_anthropomorphic_hand_renderer_supports_fourteen_independent_finger_joints():
    finger_joints = ((58, 74), (72, 104, 74), (72, 104, 74), (72, 104, 74), (72, 104, 74))
    markup = visuals.anthropomorphic_hand_html(
        "三维握拳", (38, -58, 18), (72, 84, 84, 84, 84), True, finger_joint_angles_deg=finger_joints
    )

    assert '"fingerJoints": [[58.0, 74.0], [72.0, 104.0, 74.0]' in markup
    assert '"fingerCapsules"' in markup
    assert '"previousFingerCapsules"' in markup


def test_anthropomorphic_hand_renderer_serialises_finger_capsules():
    markup = visuals.anthropomorphic_hand_html(
        "三维握拳", (38, -58, 18), (72, 84, 84, 84, 84), True,
        finger_joint_angles_deg=((58, 74), (72, 104, 74), (72, 104, 74), (72, 104, 74), (72, 104, 74)),
    )

    assert '"fingerCapsules"' in markup
    assert "makeFingerFromData" in markup
    assert "applyFingerData" in markup


def test_thumb_renderer_keeps_the_two_joint_chain_without_overlay():
    markup = visuals.anthropomorphic_hand_html(
        "三维握拳", (38, -58, 18), (72, 84, 84, 84, 84), True,
        finger_joint_angles_deg=((85, 74), (72, 104, 74), (72, 104, 74), (72, 104, 74), (72, 104, 74)),
    )

    assert '"fingerCapsules"' in markup
    assert '"previousFingerCapsules"' in markup
    assert "thumbOverlay" not in markup


def test_anthropomorphic_hand_emits_drag_rotation_without_wheel_zoom():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)

    assert "addEventListener('pointermove'" in markup
    assert "addEventListener('wheel'" not in markup
    assert "滚轮缩放" not in markup


def test_anthropomorphic_hand_inlines_local_three_runtime():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)

    assert '<script src="https://cdn.jsdelivr.net' not in markup
    assert "Copyright 2010-2023 Three.js Authors" in markup


def test_anthropomorphic_hand_animates_can_and_shoulder_translation():
    markup = visuals.anthropomorphic_hand_html(
        "抓取", (38, -58, 18), (60, 80, 80, 80, 80), True,
        can_offset=(1.2, -.4, 1.5), previous_can_offset=(0.0, 0.0, -.5),
        shoulder_offset=(1.0, 2.0, -1.0), previous_shoulder_offset=(0.0, 0.0, 0.0),
    )

    assert "previousCanOffset" in markup
    assert "shoulder.position.lerpVectors" in markup
    assert "graspFrame.getWorldPosition(canStart)" in markup
    assert "robot.add(can)" in markup
    assert "cfg.canOffset[2]" in markup
    assert "requestAnimationFrame(tick)" in markup


def test_anthropomorphic_hand_uses_fixed_axes_and_interpolates_only_the_hand_before_contact():
    markup = visuals.anthropomorphic_hand_html(
        "对准目标", (55, -35, 15), (72, 84, 84, 84, 84), False,
        shoulder_offset=(1.3, -.8, .6), previous_shoulder_offset=(0.0, 0.0, 0.0),
        previous_joint_angles_deg=(38, -58, 18),
        previous_finger_joint_angles_deg=((0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)),
    )

    assert 'new THREE.AxesHelper(2.0)' in markup
    assert 'camera.position.set(16,-18,14)' in markup
    assert 'armPivots.forEach((pivot,index)=>' in markup
    assert 'applyFingerData(finger,cfg.previousFingerCapsules[index],cfg.fingerCapsules[index],eased)' in markup
    assert 'if(!cfg.grasped)can.position.set(cfg.canOffset[0]+canStart.x,cfg.canOffset[1]+canStart.y,cfg.canOffset[2]+canStart.z)' in markup


def test_anthropomorphic_hand_renderer_embeds_shared_planar_geometry():
    pose = models.dexterous_hand_pose("抓取")
    markup = visuals.anthropomorphic_hand_html(
        "抓取", (38, -58, 18), (60, 80, 80, 80, 80), True, planar_pose=pose
    )

    assert '"sharedGeometry"' in markup
    assert '"armJoints"' in markup


def test_anthropomorphic_hand_renderer_keeps_the_solid_articulated_model_visible():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)

    assert "shoulder.visible=false" not in markup
    assert "sharedRobot=new THREE.Group" not in markup
    assert "viewDistance=viewRadius/Math.sin(THREE.MathUtils.degToRad(camera.fov/2))*1.2" in markup
    assert "graspFrame.position.set(.30,-.20,.76)" in markup
    assert "can.rotation.z=Math.PI/2" not in markup
    assert '"fingerCapsules"' in markup
    assert "new THREE.CylinderGeometry(.48,.48,1.72,32)" in markup
    assert "capsuleIntersectsCylinder" not in markup
    assert "firstCylinderContact" not in markup
    assert "new THREE.CapsuleGeometry(r,Math.max(.12,len-r*2),8,14),skin" in markup


def test_anthropomorphic_hand_renderer_javascript_parses():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)
    script = markup[markup.index("(() => {") : markup.index("})();</script>", markup.index("(() => {")) + 4]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".js") as source:
        source.write(script)
        source.flush()
        result = subprocess.run(["node", "--check", source.name], text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr


def test_planar_palm_uses_a_balanced_hand_proportion():
    pose = models.dexterous_hand_pose("抓取")
    joints = np.asarray(pose["arm_joints"])
    forward = joints[3] - joints[2]
    forward /= np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    outline = np.asarray(pose["palm_outline"])

    assert np.ptp(outline @ forward) == pytest.approx(1.18)
    assert np.ptp(outline @ lateral) == pytest.approx(.67)


def test_anthropomorphic_hand_frames_the_whole_robot_group():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)

    assert "new THREE.Box3().setFromObject(robot)" in markup
    assert "camera.lookAt(viewCenter)" in markup
    assert "MathUtils.degToRad(camera.fov/2))*1.2" in markup
    assert "can.visible=true" in markup


def test_thin_palm_shell_spans_finger_width_without_becoming_a_thick_ellipsoid():
    pose = models.dexterous_hand_pose("抓取")
    joints = np.asarray(pose["arm_joints"])
    forward = joints[3] - joints[2]
    forward /= np.linalg.norm(forward)
    vertices, *_ = visuals._arched_palm_mesh(np.asarray(pose["palm_center"]), forward)
    lateral = np.array([-forward[1], 0.0, forward[0]])

    # 五指横向排布在 3D y 轴，掌壳必须覆盖它们；厚度应只沿掌背法向保留很薄一层。
    assert np.ptp(vertices[:, 1]) >= 1.35
    assert np.ptp(vertices @ lateral) <= .36


def test_app_omits_the_redundant_planar_distributed_sensing_dashboard():
    app = AppTest.from_file("app.py").run()
    assert all(subheader.value != "多类型分布式光纤传感数据台" for subheader in app.subheader)


def test_planar_grasp_groups_commands_animation_and_sliders_in_order():
    source = Path("app.py").read_text(encoding="utf-8")

    hand_start = source.index("with hand_tab:")
    command_start = source.index('st.markdown("#### 指令")', hand_start)
    animation_start = source.index("visuals.planar_hand_animation_html(")
    sliders_start = source.index('st.markdown("#### 姿态与目标")', hand_start)

    assert command_start < sliders_start < animation_start
    assert '"二维抓取：五指与掌心六路 FBG 波长漂移"' in source


def test_standalone_finger_calibration_and_contact_inversion_live_in_the_calibration_tab():
    source = Path("app.py").read_text(encoding="utf-8")

    assert source.count("with hand_tab:") == 1
    calibration_start = source.index("with calibration_tab:")
    assert calibration_start < source.index('st.subheader("单根手指 FBG 弯曲标定")')
    assert calibration_start < source.index('st.subheader("指尖接触位置与法向力反演")')


def test_app_groups_the_relevant_modules_and_removes_the_standalone_pipeline_tab():
    source = Path("app.py").read_text(encoding="utf-8")

    assert '"② FBG 标定与诊断"' in source
    assert '"⑦ 连续体形状重建"' in source
    assert '"⑧ 机械臂健康监测"' in source
    assert "with pipeline_tab:" not in source


def test_app_exposes_basic_foot_and_arm_health_fbg_controls():
    app = AppTest.from_file("app.py").run()
    labels = {slider.label for slider in app.slider}

    assert {"机械臂载荷 (N)", "局部异常位置 (mm)", "异常程度"} <= labels


def test_app_exposes_the_complete_sensing_chain_pages_and_controls():
    source = Path("app.py").read_text(encoding="utf-8")
    app = AppTest.from_file("app.py").run()
    labels = {slider.label for slider in app.slider}

    assert '"⑤ 多材质触觉识别"' in source
    assert '"⑪ 解调器与实验任务"' in source
    assert {"故障通道", "握持力 (N)", "接触面积 (%)", "链路真实弯曲角 (°)"} <= labels


def test_overview_contains_a_clickable_module_directory_and_operational_summary():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "模块目录" in source
    assert "当前测量配置" in source
    assert "st.iframe" in source
    assert "推荐实验路径" in source
    assert "返回主页" in source


def test_app_exposes_demo_navigation_preset_and_auto_play():
    source = Path("app.py").read_text(encoding="utf-8")

    assert "演示预设" in source
    assert "平滑过渡动画" in source
    assert "tab_jump_button" in source
    assert "重置本页演示参数" in source
    assert "下一步 → ⑧ 机械臂健康监测" in source


def test_planar_task_commands_are_above_the_animation_and_sliders():
    source = Path("app.py").read_text(encoding="utf-8")
    hand_start = source.index("with hand_tab:")
    command_start = source.index('st.markdown("#### 指令")', hand_start)
    sliders_start = source.index('st.markdown("#### 姿态与目标")', hand_start)

    assert command_start < sliders_start
    assert sliders_start < source.index("visuals.planar_hand_animation_html(")


def test_planar_task_closure_resolves_to_a_clickable_next_phase():
    app = AppTest.from_file("app.py").run()
    two_d_start = next(button for button in app.button if button.key == "start_two_d_grasp_task")
    two_d_start.click().run()
    for _ in range(3):
        next(button for button in app.button if button.key == "advance_two_d_grasp_task").click().run()

    captions = [caption.value for caption in app.caption if "二维任务状态" in caption.value]
    next_button = next(button for button in app.button if button.key == "advance_two_d_grasp_task")
    assert any(phase in captions[0] for phase in ("搬运目标", "抓取失败"))
    assert next_button.disabled is False


def test_default_two_d_grab_preset_closes_on_the_initial_target_envelope():
    app = AppTest.from_file("app.py").run()
    next(button for button in app.button if button.key == "action_抓取").click().run()

    assert any("FBG 已抓稳" in alert.value for alert in app.success)


def test_planar_release_phase_reenables_manual_commands_before_final_acknowledgement():
    app = AppTest.from_file("app.py").run()
    get = lambda key: next(button for button in app.button if button.key == key)
    get("start_two_d_grasp_task").click().run()
    for _ in range(4):
        get("advance_two_d_grasp_task").click().run()

    assert "松开并放置" in next(caption.value for caption in app.caption if "二维任务状态" in caption.value)
    assert get("action_复位").disabled is False
    assert get("start_two_d_grasp_task").disabled is False


def test_planar_arm_figure_uses_a_slimmer_arm_and_can_silhouette():
    figure = visuals.arm_figure("抓取", "手指背侧", 55.0, 4.0)
    upper_arm, forearm, wrist, *rest = figure.data
    can = next(trace for trace in figure.data if trace.name == "铝制饮料罐")

    assert (upper_arm.line.width, forearm.line.width, wrist.line.width) == (20, 16, 10)
    assert max(can.x) - min(can.x) == pytest.approx(0.48)


def test_planar_hand_pose_uses_a_half_width_palm_without_changing_arm_geometry():
    opened = models.dexterous_hand_pose("伸手")

    assert np.linalg.norm(opened["palm_outline"][1] - opened["palm_outline"][2]) == pytest.approx(0.67)
    assert np.linalg.norm(opened["arm_joints"][1] - opened["arm_joints"][0]) == pytest.approx(3.5)


def test_planar_animation_html_serialises_previous_and_current_states():
    previous = models.dexterous_hand_pose("伸手")
    current = models.dexterous_hand_pose("抓取")

    markup = visuals.planar_hand_animation_html(previous, current, previous["target"], current["target"], False, True)

    assert "previous" in markup and "current" in markup
    assert "planar-grasp" in markup


def test_planar_search_uses_the_same_native_transition_view_as_every_other_task_step():
    source = Path("app.py").read_text(encoding="utf-8")
    hand_start = source.index("with hand_tab:")
    display_context = source[hand_start:source.index("st.session_state.two_d_previous_pose = display_pose", hand_start)]

    assert 'if st.session_state.two_d_task_phase == "寻找目标":' not in display_context
    assert "visuals.planar_hand_animation_html(" in display_context
    assert "st.iframe(" in display_context




def test_app_exposes_a_separate_3d_grasp_sensing_page():
    app = AppTest.from_file("app.py").run()
    assert any(subheader.value == "三维抓取传感：独立接触与 FBG 读数" for subheader in app.subheader)


def test_app_exposes_replaceable_sole_assembly_prediction_page():
    app = AppTest.from_file("app.py").run(timeout=10)
    headings = {subheader.value for subheader in app.subheader}
    assert "可更换式足底组件：二维装配状态预测" in headings
    assert "装配公差与阈值敏感性" in headings
    assert "下载装配验证参数摘要" in {button.label for button in app.get("download_button")}


def test_foot_page_keeps_reassembly_visuals_in_an_auxiliary_expander():
    app = AppTest.from_file("app.py").run(timeout=10)

    assert "复装与结构说明（辅助）" in {item.label for item in app.expander}


def test_app_exposes_a_3d_initial_pose_reset_button():
    app = AppTest.from_file("app.py").run()
    assert any(button.label == "恢复三维初始姿态" for button in app.button)


def test_app_exposes_a_stepwise_3d_search_grasp_and_release_flow():
    app = AppTest.from_file("app.py").run()
    labels = {button.label for button in app.button}
    assert {"开始三维寻找与抓取任务", "执行下一步"} <= labels


def test_anthropomorphic_hand_uses_a_wide_view_for_the_hand_and_target():
    markup = visuals.anthropomorphic_hand_html("抓取", (38, -58, 18), (60, 80, 80, 80, 80), True)
    assert "MathUtils.degToRad(camera.fov/2))*1.2" in markup


def test_arm_3d_and_foot_figures_are_renderable():
    assert len(visuals.arm_3d_figure("抓取", "手指背侧", 55.0).data) >= 2
    foot = visuals.foot_schematic_figure(np.full(6, 30.0), "平地")
    assert len(foot.data) == 1
    assert len(foot.layout.shapes) == 6
    with_cop = visuals.foot_schematic_figure(np.full(6, 30.0), "平地", np.array([1.5, 1.5]))
    assert len(with_cop.data) == 2


def test_foot_fbg_dashboard_shows_all_six_live_wavelength_channels_and_loads():
    figure = visuals.foot_fbg_dashboard_figure(
        np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006]),
        np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
    )

    assert {trace.name for trace in figure.data} == {"FBG 波长漂移", "区域载荷"}
    assert len(figure.data[0].x) == 6
