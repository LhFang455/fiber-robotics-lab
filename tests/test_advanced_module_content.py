import csv
import io

import numpy as np
import pytest

from fiber_robotics_sim import experiments, models, visuals


def test_shape_experiment_quantifies_centerline_and_tip_error():
    ideal = experiments.run_shape_experiment(experiments.SHAPE_PRESETS["理想恒曲率"])
    biased = experiments.run_shape_experiment(experiments.SHAPE_PRESETS["芯间温差对照"])

    assert ideal["results"]["centerline_rmse_mm"] == pytest.approx(0.0, abs=1e-9)
    assert ideal["results"]["tip_error_mm"] == pytest.approx(0.0, abs=1e-9)
    assert biased["results"]["centerline_rmse_mm"] > 0.0
    assert biased["results"]["tip_error_mm"] > 0.0
    assert len(biased["results"]["true_centerline_xyz_mm"]) == len(
        biased["results"]["estimated_centerline_xyz_mm"]
    )


def test_shape_export_preserves_true_and_reconstructed_centerlines():
    record = experiments.run_shape_experiment(experiments.SHAPE_PRESETS["芯间温差对照"])
    rows = list(csv.DictReader(io.StringIO(experiments.shape_csv_bytes(record).decode("utf-8-sig"))))
    report = experiments.shape_report(record)

    assert {"position_index", "true_x_mm", "estimated_x_mm", "point_error_mm"} <= set(rows[0])
    assert "中心线 RMSE" in report
    assert "末端误差" in report
    assert "不能替代" in report


def test_shape_marks_zero_curvature_direction_unidentifiable_and_twist_as_prior():
    record = experiments.run_shape_experiment({
        **experiments.SHAPE_PRESETS["扭转形状"],
        "curvature_per_m": 0.0,
        "direction_deg": 200.0,
    })

    assert record["results"]["direction_identifiable"] is False
    assert record["results"]["direction_error_deg"] is None
    assert record["results"]["twist_source"] == "known_prior"
    report = experiments.shape_report(record)
    assert "方向不适用" in report
    assert "已知重建先验" in report


def test_shape_twist_does_not_change_three_core_readings():
    params = experiments.SHAPE_PRESETS["理想恒曲率"]
    untwisted = experiments.run_shape_experiment({**params, "twist_per_m": 0.0})
    twisted = experiments.run_shape_experiment({**params, "twist_per_m": 4.0})

    assert twisted["results"]["wavelength_shifts_nm"] == pytest.approx(
        untwisted["results"]["wavelength_shifts_nm"]
    )
    assert "扭转率不是由当前三芯波长读数反演得到" in experiments.shape_report(twisted)


def test_health_experiment_distinguishes_detection_from_localisation():
    healthy = experiments.run_health_experiment(experiments.HEALTH_PRESETS["健康基线"])
    anomaly = experiments.run_health_experiment(experiments.HEALTH_PRESETS["高密度阵列对照"])

    assert healthy["results"]["detected"] is False
    assert healthy["results"]["localization_valid"] is False
    assert healthy["results"]["localization_error_mm"] is None
    assert anomaly["results"]["detected"] is True
    assert anomaly["results"]["detection_matches"] is True
    assert anomaly["results"]["localization_valid"] is True
    assert anomaly["results"]["localization_error_mm"] <= anomaly["results"]["location_uncertainty_mm"]


def test_healthy_baseline_suppresses_false_localisation_in_report_and_figure():
    record = experiments.run_health_experiment(experiments.HEALTH_PRESETS["健康基线"])
    results = record["results"]
    report = experiments.health_report(record)
    figure = visuals.arm_health_figure(
        {
            "sensor_positions_mm": results["sensor_positions_mm"],
            "strain": np.asarray(results["strain_ue"]) * 1e-6,
        },
        results,
    )

    assert "异常位置参数：未启用" in report
    assert "可疑位置：未形成有效异常定位" in report
    assert "未形成有效异常定位" in figure.layout.title.text
    assert not figure.layout.shapes


def test_health_export_states_uncertainty_and_safety_boundary():
    record = experiments.run_health_experiment(experiments.HEALTH_PRESETS["局部异常"])
    rows = list(csv.DictReader(io.StringIO(experiments.health_csv_bytes(record).decode("utf-8-sig"))))
    report = experiments.health_report(record)

    assert {"sensor_position_mm", "damage_profile", "strain_ue", "wavelength_shift_nm"} <= set(rows[0])
    assert "真实异常位置" in report
    assert "定位误差" in report
    assert "安全评估" in report


def test_missed_health_anomaly_preserves_the_configured_location_in_report():
    """捕获已设置但漏检的异常被报告成“参数未启用”的问题。"""
    record = experiments.run_health_experiment({
        "load_n": 120.0,
        "anomaly_position_mm": 0.0,
        "anomaly_severity": 0.7,
        "sensor_count": 4,
        "temperature_c": 0.0,
        "noise_nm": 0.0,
        "seed": 7,
    })

    report = experiments.health_report(record)

    assert record["results"]["expected_detection"] is True
    assert record["results"]["detected"] is False
    assert "真实异常位置：0.0 mm" in report
    assert "异常位置参数：未启用" not in report
    assert "未检出" in report


def test_redundant_report_contains_inputs_raw_channels_and_diagnosis():
    """捕获综合报告无法区分注入通道和诊断通道、不能复现实验的问题。"""
    report = experiments.redundant_fbg_report(
        true_angle_deg=45.0,
        temperature_c=10.0,
        fault_mode="漂移",
        injected_channel=2,
        wavelength_shifts_nm=[0.50, 0.57, 0.50, 0.50],
        diagnosed_channels=[2],
        estimated_angle_deg=44.8,
    )

    assert "真实弯曲角：45.0°" in report
    assert "注入故障通道：FBG 2" in report
    assert "诊断异常通道：FBG 2" in report
    assert "四路原始波长漂移" in report
    assert "0.570000" in report


def test_distributed_experiment_quantifies_location_and_exports_profile():
    """捕获分布式页只有图形、没有可追溯实验记录的问题。"""
    record = experiments.run_distributed_experiment(
        experiments.DISTRIBUTED_PRESETS["Rayleigh 局部应变"]
    )
    rows = list(csv.DictReader(io.StringIO(
        experiments.distributed_csv_bytes(record).decode("utf-8-sig")
    )))
    report = experiments.distributed_report(record)

    assert record["results"]["estimated_event_position_mm"] == pytest.approx(140.0, abs=2.0)
    assert record["results"]["location_error_mm"] <= 2.0
    assert len(rows) == record["results"]["sampled_points"]
    assert {"position_mm", "profile_value", "observable", "unit"} == set(rows[0])
    assert "事件定位误差" in report
    assert "空间采样间隔" in report
    assert "教学解析模型" in report


def test_grasp_report_preserves_conditions_and_model_boundary():
    report = experiments.grasp_report({
        "dimension": "三维",
        "task_phase": "搬运目标",
        "is_grasped": True,
        "contact_fingers": ["拇指", "食指", "中指"],
        "contact_force_n": [1.2, 1.0, 0.8, 0.0, 0.0],
        "palm_touch_n": 1.5,
        "wavelength_shifts_nm": [0.1, 0.2, 0.3, 0.0, 0.0, 0.4],
        "temperature_c": 5.0,
        "noise_nm": 0.002,
        "seed": 7,
        "target_position": [1.0, 2.0, 3.0],
    })

    assert "三维抓取实验记录" in report
    assert "任务阶段：搬运目标" in report
    assert "接触手指：拇指、食指、中指" in report
    assert "五指接触力" in report
    assert "六路 FBG 波长漂移" in report
    assert "不能替代" in report


def test_optical_experiment_exposes_stokes_cross_sensitivity_and_cavity_change():
    baseline = experiments.run_optical_experiment(experiments.OPTICAL_PRESETS["偏振基线"])
    crossed = experiments.run_optical_experiment(experiments.OPTICAL_PRESETS["温度交叉敏感"])

    assert np.linalg.norm(baseline["results"]["stokes"]) == pytest.approx(1.0)
    assert crossed["results"]["temperature_ellipticity_offset_deg"] > 0.0
    assert crossed["results"]["effective_cavity_um"] < crossed["parameters"]["cavity_um"]
    assert crossed["results"]["cavity_change_nm"] < 0.0
    report = experiments.optical_report(crossed)
    assert "Stokes" in report
    assert "Sagnac" in report
    assert "EFPI" in report
    assert "教学模型" in report


def test_sagnac_phase_preserves_rotation_direction_and_supports_inverse():
    positive = models.simulate_sagnac_gyro(90.0, 120.0)
    negative = models.simulate_sagnac_gyro(-90.0, 120.0)

    assert negative["phase_shift_rad"] == pytest.approx(-positive["phase_shift_rad"])
    assert positive["estimated_rate_deg_s"] == pytest.approx(90.0)
    assert negative["estimated_rate_deg_s"] == pytest.approx(-90.0)


def test_chain_report_contains_task_steps_current_record_and_shared_chain():
    report = experiments.sensing_chain_report(
        "连续体形状重建",
        "中心线 RMSE：1.25 mm",
        "采样率：100 Hz；融合状态：需人工复核",
    )

    assert "任务步骤" in report
    assert "中心线 RMSE：1.25 mm" in report
    assert "采样率：100 Hz" in report
    assert "不可替代真实系统" in report
