import csv
import io

import numpy as np
import pytest

from fiber_robotics_sim import experiments, models


def test_tactile_inverse_uses_temperature_compensated_wavelength_readings():
    """捕获触觉分类绕过 FBG 读数、直接使用正演接触力的问题。"""
    simulated = models.simulate_material_touch(
        "硬块", 6.0, 30.0, 18.0, pattern_noise=0.0, noise_nm=0.0, seed=7
    )

    estimated = models.estimate_tactile_touch(
        simulated["wavelength_shifts_nm"], 18.0
    )

    expected = np.r_[simulated["finger_touch_n"], simulated["palm_touch_n"]]
    assert estimated == pytest.approx(expected)


def test_tactile_experiment_reports_separation_pattern_error_and_repeatability():
    """捕获触觉页只能给单次标签、不能量化模板偏差和重复稳定性的问题。"""
    params = {**experiments.TACTILE_PRESETS["标准海绵"], "seed": 31}

    record = experiments.run_tactile_experiment(params, repeat_count=12)

    assert record["results"]["diagnosed_material"] == "海绵"
    assert record["results"]["runner_up_material"] in {"硬块", "圆柱", "薄板"}
    assert record["results"]["probability_margin"] > 0.0
    assert record["results"]["pattern_error_percent"] == pytest.approx(0.0, abs=1e-9)
    assert record["results"]["repeat_match_rate"] == pytest.approx(1.0)
    assert record["results"]["repeat_count"] == 12
    assert record == experiments.run_tactile_experiment(params, repeat_count=12)


def test_tactile_zero_contact_is_marked_invalid_instead_of_confident():
    """捕获无接触时仍显示材料识别结论的问题。"""
    record = experiments.run_tactile_experiment(
        {
            **experiments.TACTILE_PRESETS["标准海绵"],
            "grip_force_n": 0.0,
            "noise_nm": 0.01,
        },
        repeat_count=6,
    )

    assert record["results"]["valid_contact"] is False
    assert record["results"]["diagnosed_material"] == "未接触"
    assert record["results"]["probability_margin"] == pytest.approx(0.0)
    assert record["results"]["repeat_match_rate"] == pytest.approx(0.0)
    report = experiments.tactile_report(record)
    assert "有效接触：否" in report
    assert "模板偏差：不适用（接触不足）" in report


def test_foot_experiment_quantifies_load_and_cop_reconstruction():
    """捕获足底页只显示反演值、没有对照真实值和误差的问题。"""
    record = experiments.run_foot_experiment(experiments.FOOT_PRESETS["平地中期"])

    results = record["results"]
    assert results["true_total_load_n"] == pytest.approx(180.0)
    assert results["estimated_total_load_n"] == pytest.approx(180.0)
    assert results["zone_mae_n"] == pytest.approx(0.0, abs=1e-9)
    assert results["cop_error"] == pytest.approx(0.0, abs=1e-9)
    assert results["reliable_cop"] is True
    assert len(results["true_zone_loads_n"]) == 6
    assert len(results["estimated_zone_loads_n"]) == 6


def test_foot_swing_scenario_marks_cop_as_low_load_observation():
    """捕获摆动期低载荷仍被呈现为可靠压力中心的问题。"""
    record = experiments.run_foot_experiment(experiments.FOOT_PRESETS["摆动期低载荷"])

    assert record["results"]["true_total_load_n"] == pytest.approx(5.4)
    assert record["results"]["reliable_cop"] is False


def test_foot_noise_scenario_produces_reproducible_reconstruction_error():
    """捕获足底推荐场景全部为零误差、无法观察测量噪声影响的问题。"""
    first = experiments.run_foot_experiment(experiments.FOOT_PRESETS["波长噪声对照"])
    second = experiments.run_foot_experiment(experiments.FOOT_PRESETS["波长噪声对照"])

    assert first == second
    assert first["results"]["zone_mae_n"] > 0.0
    assert first["results"]["cop_error"] > 0.0


def test_foot_sensor_failure_preserves_true_load_and_changes_only_measurement():
    """捕获模拟失效时把真实区域载荷清零、从而掩盖传感误差的问题。"""
    record = experiments.run_foot_experiment({
        **experiments.FOOT_PRESETS["平地中期"], "failed_zone": 1, "drift_nm": 0.0,
    })

    assert record["results"]["true_zone_loads_n"][0] > 0.0
    assert record["results"]["estimated_zone_loads_n"][0] == pytest.approx(0.0)
    assert record["results"]["zone_mae_n"] > 0.0
    assert record["results"]["reliable_cop"] is False
    assert "CoP 仅供参考" in experiments.foot_report(record)


def test_module_exports_include_true_and_estimated_values():
    """捕获导出文件遗漏对照量或关键误差、无法复核页面结论的问题。"""
    tactile = experiments.run_tactile_experiment(
        experiments.TACTILE_PRESETS["硬块对照"], repeat_count=5
    )
    foot = experiments.run_foot_experiment(experiments.FOOT_PRESETS["脚跟着地"])

    tactile_rows = list(
        csv.DictReader(io.StringIO(experiments.tactile_csv_bytes(tactile).decode("utf-8-sig")))
    )
    foot_rows = list(
        csv.DictReader(io.StringIO(experiments.foot_csv_bytes(foot).decode("utf-8-sig")))
    )

    assert len(tactile_rows) == 6
    assert set(tactile_rows[0]) == {
        "channel", "ideal_touch_n", "estimated_touch_n", "wavelength_shift_nm"
    }
    assert len(foot_rows) == 6
    assert set(foot_rows[0]) == {
        "zone", "true_load_n", "estimated_load_n", "absolute_error_n",
        "wavelength_shift_nm",
    }


def test_module_reports_preserve_conditions_metrics_and_model_boundaries():
    """捕获报告只有结论、缺少条件、误差和教学模型边界的问题。"""
    tactile = experiments.run_tactile_experiment(
        experiments.TACTILE_PRESETS["高扰动薄板"], repeat_count=8
    )
    foot = experiments.run_foot_experiment(experiments.FOOT_PRESETS["摆动期低载荷"])

    tactile_report = experiments.tactile_report(tactile)
    foot_report = experiments.foot_report(foot)

    assert "薄板" in tactile_report
    assert "重复识别一致率" in tactile_report
    assert "类别间隔" in tactile_report
    assert "温度变化：0.00°C" in tactile_report
    assert "余弦相似度" in tactile_report
    assert "摆动期" in foot_report
    assert "区域平均绝对误差" in foot_report
    assert "CoP 仅供参考" in foot_report
    assert "动态冲击" in foot_report
