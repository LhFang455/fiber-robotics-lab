import json

import numpy as np
import pytest

from fiber_robotics_sim import experiments, models
from fiber_robotics_sim.experiments import (
    PRESETS,
    calibration_report,
    export_record,
    import_record,
    run_calibration,
)


def test_seeded_calibration_round_trip_is_deterministic():
    """捕获随机数未按 seed 固定或导入后沿用旧结果的问题。"""
    record = run_calibration({**PRESETS["噪声对照"], "seed": 29})

    restored, baseline = import_record(export_record(record))

    assert baseline is None
    assert restored == record


def test_baseline_is_exported_independently_from_current_record():
    """捕获导出时将 A、B 对照错误地共用同一参数的问题。"""
    baseline = run_calibration(PRESETS["理想标定"])
    current = run_calibration({**PRESETS["温漂对照"], "angle_deg": 30.0})

    restored, restored_baseline = import_record(export_record(current, baseline))

    assert restored["parameters"]["angle_deg"] == 30.0
    assert restored_baseline is not None
    assert restored_baseline["parameters"]["angle_deg"] == 45.0
    assert restored_baseline != restored


def test_temperature_correction_recovers_the_angle_while_uncompensated_does_not():
    """捕获温度补偿漏用、或未补偿结果被误当作补偿结果的问题。"""
    record = run_calibration(PRESETS["温漂对照"])

    assert record["results"]["estimated_angle_deg"] == pytest.approx(45.0)
    assert record["results"]["uncompensated_angle_deg"] != pytest.approx(45.0)
    assert record["results"]["error_deg"] == pytest.approx(0.0)


def test_hand_channel_failure_replaces_that_raw_reading_before_estimation():
    """捕获故障通道只改变展示读数、没有参与反演的问题。"""
    record = run_calibration({**PRESETS["理想标定"], "failed_channel": "手部 FBG 2", "drift_nm": 0.01})

    assert record["results"]["raw_shifts_nm"][1] == pytest.approx(0.01)
    assert record["results"]["estimated_angle_deg"] == pytest.approx(30.013, abs=0.001)


def test_sensor_positions_follow_the_current_finger_length(monkeypatch):
    """捕获实验层向正向模型传入固定传感器位置的问题。"""
    observed_positions = []
    original_simulator = experiments.simulate_finger

    def capture_positions(*args, **kwargs):
        observed_positions.append(args[3].tolist())
        return original_simulator(*args, **kwargs)

    monkeypatch.setattr(experiments, "simulate_finger", capture_positions)
    run_calibration({**PRESETS["理想标定"], "length_mm": 40.0})

    assert observed_positions == [[10.0, 20.0, 30.0]]


def test_zero_offset_is_explicitly_not_identifiable():
    """捕获零偏置被现有估计器静默报告为零角度的问题。"""
    record = run_calibration({**PRESETS["理想标定"], "fiber_offset_mm": 0.0})

    assert record["results"]["identifiable"] is False
    assert record["results"]["estimated_angle_deg"] is None
    assert record["results"]["uncompensated_angle_deg"] is None
    assert record["results"]["error_deg"] is None


@pytest.mark.parametrize(
    "params",
    [
        {**PRESETS["理想标定"], "angle_deg": 101.0},
        {**PRESETS["理想标定"], "noise_nm": True},
        {**PRESETS["理想标定"], "temperature_c": float("nan")},
        {key: value for key, value in PRESETS["理想标定"].items() if key != "seed"},
        {**PRESETS["理想标定"], "extra": 1},
    ],
)
def test_invalid_parameters_are_rejected(params):
    """捕获范围、布尔值、非有限值和参数集合校验缺失的问题。"""
    with pytest.raises(ValueError):
        run_calibration(params)


def test_import_rejects_invalid_payloads_and_regenerates_current_and_baseline_results():
    """捕获导入信任伪造结果或放过损坏格式、版本号、顶层字段的问题。"""
    record = run_calibration(PRESETS["理想标定"])
    baseline = run_calibration({**PRESETS["理想标定"], "angle_deg": 30.0})
    payload = json.loads(export_record(record, baseline).decode("utf-8"))
    payload["current"]["results"]["estimated_angle_deg"] = 999.0
    payload["baseline"]["results"]["estimated_angle_deg"] = -999.0

    restored, restored_baseline = import_record(json.dumps(payload).encode("utf-8"))
    assert restored == record
    assert restored_baseline == baseline
    with pytest.raises(ValueError):
        import_record(b"{")
    payload["schema_version"] = 2
    with pytest.raises(ValueError):
        import_record(json.dumps(payload).encode("utf-8"))
    payload["schema_version"] = True
    with pytest.raises(ValueError):
        import_record(json.dumps(payload).encode("utf-8"))
    payload["schema_version"] = 1
    payload["model_version"] = "other-model"
    with pytest.raises(ValueError):
        import_record(json.dumps(payload).encode("utf-8"))
    payload["model_version"] = "calibration-mean-strain-v1"
    payload["unknown"] = "field"
    with pytest.raises(ValueError):
        import_record(json.dumps(payload).encode("utf-8"))


def test_import_normalizes_bounded_size_numeric_and_deep_json_failures_to_value_error():
    """捕获上传文件的大小、超大整数或过深 JSON 让界面出现非 ValueError 异常的问题。"""
    record = run_calibration(PRESETS["理想标定"])
    payload = json.loads(export_record(record).decode("utf-8"))
    payload["current"]["parameters"]["angle_deg"] = 10**400

    for invalid_payload in (
        b" " * (128 * 1024 + 1),
        json.dumps(payload).encode("utf-8"),
        b"[" * 1100 + b"]" * 1100,
    ):
        with pytest.raises(ValueError):
            import_record(invalid_payload)


def test_report_includes_observed_values_parameters_and_model_limit():
    """捕获报告漏写 A/B 的实际参数、结果和模型限制的问题。"""
    baseline = run_calibration(PRESETS["理想标定"])
    current = run_calibration({**PRESETS["温漂对照"], "angle_deg": 30.0})

    report = calibration_report(current, baseline)

    assert "calibration-mean-strain-v1" in report
    assert "45.00° → 30.00°" in report
    assert "0.00°C → 20.00°C" in report
    assert "20.00°C" in report
    assert "seed：17" in report
    assert "补偿反演角（A → B）：45.00° → 30.00°" in report
    assert "原始读数（nm，A → B）" in report
    assert "已知温度" in report
    assert "不保证真实硬件" in report


def test_grasp_layout_comparison_distinguishes_planar_and_three_dimensional_rules():
    forces = [0.30, 0.20, 0.15, 0.0, 0.0]

    planar = experiments.compare_grasp_sensor_layouts(forces, 0.25, requires_palm=False)
    three_d = experiments.compare_grasp_sensor_layouts(forces, 0.25, requires_palm=True)

    assert planar[0]["受力覆盖率 (%)"] == pytest.approx(100.0)
    assert planar[1]["可完整执行当前判定"] == "是"
    assert three_d[1]["可完整执行当前判定"] == "否"
    assert three_d[2]["未观测通道"] == "中指、无名指、小指"
    zero_signal = experiments.compare_grasp_sensor_layouts([0.0] * 5, 0.0, requires_palm=False)
    assert [row["受力覆盖率 (%)"] for row in zero_signal] == [0.0, 0.0, 0.0]


def test_planar_grasp_noise_study_is_repeatable_and_uses_each_sample_for_classification():
    curls = (80.0, 80.0, 80.0, 0.0, 0.0)
    sensing = models.simulate_planar_grasp_fbg(curls, (0.30, 0.20, 0.15, 0.0, 0.0), 5.0)

    first = experiments.run_planar_grasp_noise_study(sensing, curls, 5.0, 0.002, 40, 23)
    repeated = experiments.run_planar_grasp_noise_study(sensing, curls, 5.0, 0.002, 40, 23)

    assert first == repeated
    assert first["sample_count"] == 40
    assert len(first["samples"]) == 40
    assert 0.0 <= first["decision_consistency_percent"] <= 100.0
    assert first["baseline_is_grasped"] is True
    assert first["total_force_std_n"] > 0.0


def test_three_d_grasp_noise_study_has_per_trial_force_and_zero_noise_is_stable():
    joint_angles = (
        (90.0, 90.0),
        *((72.0, 104.0, 74.0) for _ in range(4)),
    )
    curls = tuple(float(np.mean(angles)) for angles in joint_angles)
    sensing = models.evaluate_3d_grasp_sensing(
        curls, (0.0, 0.0, 0.0), 0.0, finger_joint_angles_deg=joint_angles
    )

    study = experiments.run_three_d_grasp_noise_study(sensing, 0.0, 0.0, 25, 7)

    assert study["baseline_is_grasped"] is True
    assert study["decision_consistency_percent"] == pytest.approx(100.0)
    assert study["decision_flip_count"] == 0
    assert study["samples"][0]["掌心反演力 (N)"] > 0.0
    assert study["samples"][0]["五指反演合力 (N)"] > 0.0


@pytest.mark.parametrize(
    "forces,palm",
    [([0.1, 0.1, 0.1, 0.1, float("-inf")], 0.2), ([0.1] * 5, float("-inf"))],
)
def test_grasp_layout_comparison_rejects_non_finite_force_inputs(forces, palm):
    with pytest.raises(ValueError):
        experiments.compare_grasp_sensor_layouts(forces, palm, requires_palm=True)


def test_grasp_noise_study_csv_and_report_include_repeatability_and_layout_limits():
    curls = (80.0, 80.0, 80.0, 0.0, 0.0)
    sensing = models.simulate_planar_grasp_fbg(curls, (0.30, 0.20, 0.15, 0.0, 0.0), 0.0)
    study = experiments.run_planar_grasp_noise_study(sensing, curls, 0.0, 0.001, 20, 11)
    layouts = experiments.compare_grasp_sensor_layouts(
        study["baseline_contact_force_n"], study["baseline_palm_touch_n"], requires_palm=False
    )

    csv_text = experiments.grasp_noise_study_csv(study).decode("utf-8-sig")
    report = experiments.grasp_robustness_report("二维", study, layouts)

    assert "样本序号,抓稳判定,接触手指数" in csv_text
    assert "20 次重复采样" in report
    assert "判定一致率" in report
    assert "三路：拇指＋食指＋掌心" in report
    assert "不能替代真实传感器选型" in report
