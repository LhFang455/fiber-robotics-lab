"""可复现实验记录的纯函数。"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .models import (
    classify_3d_grasp_from_fbg,
    classify_planar_grasp_from_fbg,
    classify_tactile_material,
    decimate_distributed_result,
    diagnose_arm_health,
    estimate_finger_angle_deg,
    estimate_foot_load_distribution,
    estimate_tactile_touch,
    fbg_wavelength_shift_nm,
    simulate_arm_health_fbg,
    simulate_efpi_pressure,
    simulate_finger,
    simulate_foot_fbg,
    simulate_foot_zone_loads,
    simulate_distributed_mechanism,
    simulate_material_touch,
    simulate_multicore_shape,
    simulate_polarization_sensing,
    simulate_sagnac_gyro,
)


ATTACHMENT_GAINS = {"嵌入式": 1.0, "粘接式": 0.78, "护套固定": 0.52}

_BASE_PARAMETERS = {
    "angle_deg": 45.0,
    "length_mm": 80.0,
    "fiber_offset_mm": 1.0,
    "attachment": "嵌入式",
    "temperature_c": 0.0,
    "noise_nm": 0.0,
    "drift_nm": 0.0,
    "failed_channel": "无",
    "seed": 17,
}
PRESETS = {
    "理想标定": dict(_BASE_PARAMETERS),
    "温漂对照": {**_BASE_PARAMETERS, "temperature_c": 20.0},
    "噪声对照": {**_BASE_PARAMETERS, "noise_nm": 0.02},
}

TACTILE_PRESETS = {
    "标准海绵": {
        "material": "海绵", "grip_force_n": 5.0, "contact_area_percent": 35.0,
        "pattern_noise": 0.0, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "硬块对照": {
        "material": "硬块", "grip_force_n": 5.0, "contact_area_percent": 35.0,
        "pattern_noise": 0.0, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "圆柱接触": {
        "material": "圆柱", "grip_force_n": 5.0, "contact_area_percent": 35.0,
        "pattern_noise": 0.0, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "标准薄板": {
        "material": "薄板", "grip_force_n": 5.0, "contact_area_percent": 35.0,
        "pattern_noise": 0.0, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "高扰动薄板": {
        "material": "薄板", "grip_force_n": 5.0, "contact_area_percent": 35.0,
        "pattern_noise": 0.35, "temperature_c": 0.0, "noise_nm": 0.01, "seed": 17,
    },
}
TACTILE_VALID_CONTACT_N = 0.5

FOOT_PRESETS = {
    "平地中期": {
        "terrain": "平地", "load_n": 180.0, "phase_percent": 55.0,
        "support": "支撑期", "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "脚跟着地": {
        "terrain": "平地", "load_n": 180.0, "phase_percent": 10.0,
        "support": "支撑期", "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "前掌蹬离": {
        "terrain": "平地", "load_n": 180.0, "phase_percent": 90.0,
        "support": "支撑期", "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "柔软地面对照": {
        "terrain": "柔软地面", "load_n": 180.0, "phase_percent": 55.0,
        "support": "支撑期", "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "波长噪声对照": {
        "terrain": "平地", "load_n": 180.0, "phase_percent": 55.0,
        "support": "支撑期", "temperature_c": 0.0, "noise_nm": 0.01, "seed": 17,
    },
    "摆动期低载荷": {
        "terrain": "平地", "load_n": 180.0, "phase_percent": 55.0,
        "support": "摆动期", "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
}

SHAPE_PRESETS = {
    "理想恒曲率": {
        "curvature_per_m": 8.0, "direction_deg": 35.0, "twist_per_m": 0.0,
        "length_mm": 150.0, "core_radius_um": 125.0,
        "temperature_c": 0.0, "noise_nm": 0.0,
        "core_temperature_gradient_c": 0.0, "seed": 17,
    },
    "扭转形状": {
        "curvature_per_m": 8.0, "direction_deg": 35.0, "twist_per_m": 12.0,
        "length_mm": 150.0, "core_radius_um": 125.0,
        "temperature_c": 0.0, "noise_nm": 0.0,
        "core_temperature_gradient_c": 0.0, "seed": 17,
    },
    "波长噪声对照": {
        "curvature_per_m": 8.0, "direction_deg": 35.0, "twist_per_m": 0.0,
        "length_mm": 150.0, "core_radius_um": 125.0,
        "temperature_c": 0.0, "noise_nm": 0.01,
        "core_temperature_gradient_c": 0.0, "seed": 17,
    },
    "芯间温差对照": {
        "curvature_per_m": 8.0, "direction_deg": 35.0, "twist_per_m": 0.0,
        "length_mm": 150.0, "core_radius_um": 125.0,
        "temperature_c": 0.0, "noise_nm": 0.0,
        "core_temperature_gradient_c": 8.0, "seed": 17,
    },
}

HEALTH_PRESETS = {
    "健康基线": {
        "load_n": 80.0, "anomaly_position_mm": 320.0, "anomaly_severity": 0.0,
        "sensor_count": 6, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "局部异常": {
        "load_n": 80.0, "anomaly_position_mm": 320.0, "anomaly_severity": 0.70,
        "sensor_count": 6, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "稀疏阵列对照": {
        "load_n": 80.0, "anomaly_position_mm": 320.0, "anomaly_severity": 0.70,
        "sensor_count": 4, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "高密度阵列对照": {
        "load_n": 80.0, "anomaly_position_mm": 320.0, "anomaly_severity": 0.70,
        "sensor_count": 8, "temperature_c": 0.0, "noise_nm": 0.0, "seed": 17,
    },
    "噪声定位对照": {
        "load_n": 80.0, "anomaly_position_mm": 320.0, "anomaly_severity": 0.70,
        "sensor_count": 8, "temperature_c": 0.0, "noise_nm": 0.01, "seed": 17,
    },
}

OPTICAL_PRESETS = {
    "偏振基线": {
        "stress_mpa": 0.0, "twist_deg": 0.0, "gyro_rate_deg_s": 0.0,
        "pressure_mpa": 0.0, "cavity_um": 28.0, "temperature_c": 0.0,
    },
    "横向应力": {
        "stress_mpa": 120.0, "twist_deg": 0.0, "gyro_rate_deg_s": 0.0,
        "pressure_mpa": 0.0, "cavity_um": 28.0, "temperature_c": 0.0,
    },
    "光纤扭转": {
        "stress_mpa": 0.0, "twist_deg": 35.0, "gyro_rate_deg_s": 0.0,
        "pressure_mpa": 0.0, "cavity_um": 28.0, "temperature_c": 0.0,
    },
    "温度交叉敏感": {
        "stress_mpa": 120.0, "twist_deg": 35.0, "gyro_rate_deg_s": 45.0,
        "pressure_mpa": 0.4, "cavity_um": 28.0, "temperature_c": 20.0,
    },
    "旋转与压力": {
        "stress_mpa": 0.0, "twist_deg": 0.0, "gyro_rate_deg_s": 90.0,
        "pressure_mpa": 0.8, "cavity_um": 28.0, "temperature_c": 0.0,
    },
}

DEMODULATION_PRESETS = {
    "快速响应基线": {
        "angle_deg": 30.0, "temperature_c": 0.0, "noise_nm": 0.002,
        "filter_window": 1, "control_threshold_deg": 35.0, "sample_rate_hz": 100,
    },
    "阈值附近噪声": {
        "angle_deg": 35.0, "temperature_c": 0.0, "noise_nm": 0.020,
        "filter_window": 1, "control_threshold_deg": 35.0, "sample_rate_hz": 100,
    },
    "强滤波对照": {
        "angle_deg": 35.0, "temperature_c": 0.0, "noise_nm": 0.020,
        "filter_window": 15, "control_threshold_deg": 35.0, "sample_rate_hz": 100,
    },
    "温漂补偿": {
        "angle_deg": 55.0, "temperature_c": 20.0, "noise_nm": 0.005,
        "filter_window": 5, "control_threshold_deg": 35.0, "sample_rate_hz": 100,
    },
}

DISTRIBUTED_PRESETS = {
    "Rayleigh 局部应变": {
        "mode": "Rayleigh/OFDR", "fiber_length_mm": 300.0,
        "event_position_mm": 140.0, "event_strength": 600.0,
        "spatial_spacing_mm": 2, "sample_rate_hz": 50,
    },
    "DAS 振动事件": {
        "mode": "φ-OTDR / DAS", "fiber_length_mm": 300.0,
        "event_position_mm": 140.0, "event_strength": 600.0,
        "spatial_spacing_mm": 10, "sample_rate_hz": 100,
    },
    "Brillouin 应变温度": {
        "mode": "Brillouin", "fiber_length_mm": 300.0,
        "event_position_mm": 140.0, "event_strength": 600.0,
        "spatial_spacing_mm": 10, "sample_rate_hz": 50,
    },
    "Raman 局部温升": {
        "mode": "Raman", "fiber_length_mm": 300.0,
        "event_position_mm": 140.0, "event_strength": 60.0,
        "spatial_spacing_mm": 10, "sample_rate_hz": 50,
    },
    "稀疏采样对照": {
        "mode": "Rayleigh/OFDR", "fiber_length_mm": 300.0,
        "event_position_mm": 140.0, "event_strength": 600.0,
        "spatial_spacing_mm": 40, "sample_rate_hz": 50,
    },
}

CHAIN_TASK_GUIDES = {
    "弯曲标定与温补": ("载入标准条件并保存基线", "改变温度或噪声", "比较补偿前后角度误差并导出记录"),
    "冗余故障诊断": ("建立无故障读数", "注入单通道漂移或断纤", "核对隔离通道与剩余通道反演"),
    "多材质触觉识别": ("载入标准材质并保存基线", "改变材质或接触扰动", "比较类别间隔、模板偏差与重复一致率"),
    "足底平衡": ("载入平地中期并保存基线", "改变步态、地形或通道状态", "比较六区载荷、区域 MAE 与 CoP 误差"),
    "连续体形状重建": ("载入理想恒曲率并保存基线", "改变已知扭转先验、噪声或芯间温差", "比较曲率、方向、中心线 RMSE 与末端误差"),
    "结构健康监测": ("载入健康基线", "载入局部异常并改变阵列密度", "核对检测状态、真实/可疑位置、定位误差与区间"),
    "分布式事件定位": ("载入 Rayleigh 局部应变基线", "改变机制或空间采样间隔", "比较估计位置、定位误差与数据质量"),
    "偏振与干涉传感": ("载入偏振基线", "分别施加应力、扭转、温度、旋转或压力", "比较 Stokes、Sagnac 相位与 EFPI 腔长变化"),
}

_NUMERIC_RANGES = {
    "angle_deg": (-100.0, 100.0),
    "length_mm": (40.0, 140.0),
    "fiber_offset_mm": (-2.0, 2.0),
    "temperature_c": (-20.0, 50.0),
    "noise_nm": (0.0, 0.02),
    "drift_nm": (0.0, 0.02),
}
_PARAMETER_KEYS = frozenset((*_NUMERIC_RANGES, "attachment", "failed_channel", "seed"))
_FAILED_CHANNELS = frozenset(("无", "手部 FBG 1", "手部 FBG 2", "手部 FBG 3", "足底区域 1"))
_RECORD_KEYS = frozenset(("parameters", "results"))
_TOP_LEVEL_KEYS = frozenset(("schema_version", "model_version", "current", "baseline"))
_MODEL_VERSION = "calibration-mean-strain-v1"
_MAX_PAYLOAD_BYTES = 128 * 1024


def _validation_error(message: str) -> ValueError:
    return ValueError(f"实验记录参数无效：{message}")


def _validated_parameters(params: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(params, dict):
        raise _validation_error("参数必须是字典")
    keys = set(params)
    if keys != _PARAMETER_KEYS:
        missing = _PARAMETER_KEYS - keys
        extra = keys - _PARAMETER_KEYS
        if missing:
            raise _validation_error(f"缺少参数：{', '.join(sorted(missing))}")
        raise _validation_error(f"不支持的参数：{', '.join(sorted(extra))}")

    validated: dict[str, Any] = {}
    for name, (minimum, maximum) in _NUMERIC_RANGES.items():
        value = params[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _validation_error(f"{name} 必须是数值")
        try:
            value = float(value)
        except OverflowError:
            raise _validation_error(f"{name} 超出可表示范围") from None
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise _validation_error(f"{name} 必须在 {minimum} 到 {maximum} 之间")
        validated[name] = value

    attachment = params["attachment"]
    if not isinstance(attachment, str) or attachment not in ATTACHMENT_GAINS:
        raise _validation_error("attachment 不是支持的光纤连接方式")
    validated["attachment"] = attachment

    failed_channel = params["failed_channel"]
    if not isinstance(failed_channel, str) or failed_channel not in _FAILED_CHANNELS:
        raise _validation_error("failed_channel 不是支持的故障通道")
    validated["failed_channel"] = failed_channel

    seed = params["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        raise _validation_error("seed 必须是 0 到 4294967295 的整数")
    validated["seed"] = seed
    return validated


def run_calibration(params: dict) -> dict:
    """按标定参数重算三路 FBG 教学实验结果。"""
    parameters = _validated_parameters(params)
    gain = ATTACHMENT_GAINS[parameters["attachment"]]
    effective_offset = parameters["fiber_offset_mm"] * gain
    simulated = simulate_finger(
        parameters["angle_deg"],
        parameters["length_mm"],
        effective_offset,
        np.asarray([0.25, 0.5, 0.75]) * parameters["length_mm"],
        parameters["temperature_c"],
        parameters["noise_nm"],
        parameters["seed"],
    )
    raw = np.asarray(simulated["wavelength_shifts_nm"], dtype=float).copy()
    failed_channel = parameters["failed_channel"]
    if failed_channel.startswith("手部 FBG "):
        raw[int(failed_channel[-1]) - 1] = parameters["drift_nm"]

    thermal_shift = float(fbg_wavelength_shift_nm(np.asarray([0.0]), parameters["temperature_c"])[0])
    compensated = raw - thermal_shift
    identifiable = abs(effective_offset) >= 1e-12
    if identifiable:
        estimated = float(
            estimate_finger_angle_deg(compensated, parameters["length_mm"], effective_offset, 0.0)
        )
        uncompensated = float(
            estimate_finger_angle_deg(raw, parameters["length_mm"], effective_offset, 0.0)
        )
        error = float(estimated - parameters["angle_deg"])
    else:
        estimated = None
        uncompensated = None
        error = None

    return {
        "parameters": parameters.copy(),
        "results": {
            "raw_shifts_nm": [float(value) for value in raw],
            "compensated_shifts_nm": [float(value) for value in compensated],
            "estimated_angle_deg": estimated,
            "uncompensated_angle_deg": uncompensated,
            "error_deg": error,
            "identifiable": identifiable,
            "gain": float(gain),
        },
    }


def _regenerate_record(record: dict[str, Any], label: str) -> dict:
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        raise _validation_error(f"{label} 记录格式不正确")
    if not isinstance(record["parameters"], dict) or not isinstance(record["results"], Mapping):
        raise _validation_error(f"{label} 记录字段类型不正确")
    return run_calibration(record["parameters"])


def export_record(current: dict, baseline: dict | None = None) -> bytes:
    """把重算后的当前记录与可选基线编码为可携带 JSON。"""
    regenerated_current = _regenerate_record(current, "当前")
    regenerated_baseline = None if baseline is None else _regenerate_record(baseline, "基线")
    payload = {
        "schema_version": 1,
        "model_version": _MODEL_VERSION,
        "current": regenerated_current,
        "baseline": regenerated_baseline,
    }
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def import_record(payload: bytes) -> tuple[dict, dict | None]:
    """验证并恢复导出记录，忽略其中保存的任何结果值。"""
    if not isinstance(payload, bytes):
        raise ValueError("实验记录导入失败：文件内容必须是 UTF-8 字节")
    if len(payload) > _MAX_PAYLOAD_BYTES:
        raise ValueError("实验记录导入失败：文件超过 128KiB")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (ValueError, RecursionError, OverflowError):
        raise ValueError("实验记录导入失败：不是有效的 UTF-8 JSON") from None
    if not isinstance(decoded, dict) or set(decoded) != _TOP_LEVEL_KEYS:
        raise ValueError("实验记录导入失败：顶层字段不正确")
    if isinstance(decoded["schema_version"], bool) or not isinstance(decoded["schema_version"], int) or decoded["schema_version"] != 1:
        raise ValueError("实验记录导入失败：不支持的 schema 版本")
    if not isinstance(decoded["model_version"], str) or decoded["model_version"] != _MODEL_VERSION:
        raise ValueError("实验记录导入失败：不支持的模型版本")
    try:
        current = _regenerate_record(decoded["current"], "当前")
        baseline_value = decoded["baseline"]
        baseline = None if baseline_value is None else _regenerate_record(baseline_value, "基线")
    except ValueError as error:
        raise ValueError(f"实验记录导入失败：{error}") from None
    return current, baseline


def _format_angle(value: float | None) -> str:
    return "不可反演" if value is None else f"{value:.2f}°"


def _format_parameter(name: str, value: Any) -> str:
    if name == "angle_deg":
        return f"{value:.2f}°"
    if name in {"length_mm", "fiber_offset_mm"}:
        return f"{value:.2f} mm"
    if name == "temperature_c":
        return f"{value:.2f}°C"
    if name in {"noise_nm", "drift_nm"}:
        return f"{value:.4f} nm"
    return str(value)


def calibration_report(current: dict, baseline: dict | None = None) -> str:
    """生成面向教学记录的中文文本报告。"""
    current = _regenerate_record(current, "当前")
    baseline = None if baseline is None else _regenerate_record(baseline, "基线")
    parameters = current["parameters"]
    results = current["results"]
    raw = ", ".join(f"{value:.6f}" for value in results["raw_shifts_nm"])
    compensated = ", ".join(f"{value:.6f}" for value in results["compensated_shifts_nm"])
    lines = [
        f"模型版本：{_MODEL_VERSION}",
        "实验参数：",
        f"- 弯曲角：{parameters['angle_deg']:.2f}°",
        f"- 手指长度：{parameters['length_mm']:.2f} mm",
        f"- 光纤偏置：{parameters['fiber_offset_mm']:.2f} mm",
        f"- 连接方式：{parameters['attachment']}（传力系数 {results['gain']:.2f}）",
        f"- 温度变化：{parameters['temperature_c']:.2f}°C",
        f"- 噪声：{parameters['noise_nm']:.4f} nm；漂移：{parameters['drift_nm']:.4f} nm",
        f"- 故障通道：{parameters['failed_channel']}；seed：{parameters['seed']}",
        f"原始读数（nm）：{raw}",
        f"温度补偿读数（nm）：{compensated}",
        f"补偿反演角：{_format_angle(results['estimated_angle_deg'])}",
        f"未补偿反演角：{_format_angle(results['uncompensated_angle_deg'])}",
        f"角度误差：{_format_angle(results['error_deg'])}",
    ]
    if baseline is not None:
        changed = [
            name for name, value in parameters.items() if baseline["parameters"][name] != value
        ]
        baseline_parameters = baseline["parameters"]
        baseline_results = baseline["results"]
        lines.extend((
            "对照 A（基线）与 B（当前）：",
            "参数变化（A → B）：",
        ))
        if changed:
            labels = {
                "angle_deg": "弯曲角",
                "length_mm": "手指长度",
                "fiber_offset_mm": "光纤偏置",
                "attachment": "连接方式",
                "temperature_c": "温度变化",
                "noise_nm": "噪声",
                "drift_nm": "漂移",
                "failed_channel": "故障通道",
                "seed": "seed",
            }
            lines.extend(
                f"- {labels[name]}：{_format_parameter(name, baseline_parameters[name])} → "
                f"{_format_parameter(name, parameters[name])}"
                for name in changed
            )
        else:
            lines.append("- 无")
        baseline_raw = ", ".join(f"{value:.6f}" for value in baseline_results["raw_shifts_nm"])
        baseline_compensated = ", ".join(
            f"{value:.6f}" for value in baseline_results["compensated_shifts_nm"]
        )
        lines.extend((
            "关键结果（A → B）：",
            f"- 原始读数（nm，A → B）：{baseline_raw} → {raw}",
            f"- 温度补偿读数（nm，A → B）：{baseline_compensated} → {compensated}",
            f"- 补偿反演角（A → B）：{_format_angle(baseline_results['estimated_angle_deg'])} → "
            f"{_format_angle(results['estimated_angle_deg'])}",
            f"- 未补偿反演角（A → B）：{_format_angle(baseline_results['uncompensated_angle_deg'])} → "
            f"{_format_angle(results['uncompensated_angle_deg'])}",
            f"- 角度误差（A → B）：{_format_angle(baseline_results['error_deg'])} → "
            f"{_format_angle(results['error_deg'])}",
            f"- 可反演（A → B）：{'是' if baseline_results['identifiable'] else '否'} → "
            f"{'是' if results['identifiable'] else '否'}",
        ))
    lines.append("说明：本记录采用已知温度补偿的教学模型，不保证真实硬件测量准确性。")
    return "\n".join(lines)


def _normalised_pattern(values: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(values))
    return np.zeros_like(values) if norm <= 1e-12 else values / norm


def run_tactile_experiment(params: dict, repeat_count: int = 24) -> dict:
    """运行一次触觉实验并给出模板偏差、类别间隔和重复识别率。"""
    parameters = dict(params)
    if repeat_count < 1:
        raise ValueError("重复次数必须为正整数")
    current = simulate_material_touch(
        parameters["material"], parameters["grip_force_n"],
        parameters["contact_area_percent"], parameters["temperature_c"],
        pattern_noise=parameters["pattern_noise"], noise_nm=parameters["noise_nm"],
        seed=parameters["seed"],
    )
    ideal = simulate_material_touch(
        parameters["material"], parameters["grip_force_n"],
        parameters["contact_area_percent"], parameters["temperature_c"],
        pattern_noise=0.0, noise_nm=0.0, seed=parameters["seed"],
    )
    estimated_touch = estimate_tactile_touch(
        current["wavelength_shifts_nm"], parameters["temperature_c"]
    )
    ideal_touch = np.r_[ideal["finger_touch_n"], ideal["palm_touch_n"]]
    valid_contact = (
        min(float(ideal_touch.sum()), float(estimated_touch.sum()))
        >= TACTILE_VALID_CONTACT_N
    )
    diagnosis = classify_tactile_material(
        estimated_touch[:5], float(estimated_touch[5])
    ) if valid_contact else classify_tactile_material(np.zeros(5), 0.0)
    ranked = sorted(
        diagnosis["probabilities"].items(), key=lambda item: item[1], reverse=True
    )
    if valid_contact:
        runner_up_material = ranked[1][0]
        probability_margin = float(ranked[0][1] - ranked[1][1])
        pattern_error_percent = float(
            np.linalg.norm(_normalised_pattern(estimated_touch) - _normalised_pattern(ideal_touch))
            * 100.0
        )
    else:
        runner_up_material = "无"
        probability_margin = 0.0
        pattern_error_percent = 0.0

    matches = 0
    for index in range(repeat_count):
        repeated = simulate_material_touch(
            parameters["material"], parameters["grip_force_n"],
            parameters["contact_area_percent"], parameters["temperature_c"],
            pattern_noise=parameters["pattern_noise"], noise_nm=parameters["noise_nm"],
            seed=parameters["seed"] + index,
        )
        repeated_touch = estimate_tactile_touch(
            repeated["wavelength_shifts_nm"], parameters["temperature_c"]
        )
        repeated_diagnosis = classify_tactile_material(
            repeated_touch[:5], float(repeated_touch[5])
        )
        matches += repeated_diagnosis["material"] == parameters["material"]

    return {
        "parameters": parameters,
        "results": {
            "diagnosed_material": diagnosis["material"],
            "confidence": float(diagnosis["confidence"]),
            "probabilities": dict(diagnosis["probabilities"]),
            "runner_up_material": runner_up_material,
            "probability_margin": probability_margin,
            "pattern_error_percent": pattern_error_percent,
            "repeat_match_rate": float(matches / repeat_count) if valid_contact else 0.0,
            "repeat_count": int(repeat_count),
            "valid_contact": valid_contact,
            "ideal_touch_n": ideal_touch.tolist(),
            "estimated_touch_n": estimated_touch.tolist(),
            "wavelength_shifts_nm": np.asarray(current["wavelength_shifts_nm"], dtype=float).tolist(),
        },
    }


def run_foot_experiment(params: dict) -> dict:
    """运行六区足底实验并量化载荷与压力中心重建误差。"""
    parameters = dict(params)
    true_loads = simulate_foot_zone_loads(
        parameters["load_n"], parameters["terrain"],
        parameters["phase_percent"], parameters["support"],
    )
    measured = simulate_foot_fbg(
        true_loads, parameters["temperature_c"], parameters["noise_nm"], parameters["seed"]
    )
    measured_shifts = np.asarray(measured["wavelength_shifts_nm"], dtype=float).copy()
    failed_zone = parameters.get("failed_zone")
    if failed_zone is not None:
        if failed_zone not in range(1, 7):
            raise ValueError("失效足底区域必须为 1 到 6")
        measured_shifts[failed_zone - 1] = float(parameters.get("drift_nm", 0.0))
    estimated = estimate_foot_load_distribution(
        measured_shifts, parameters["temperature_c"]
    )
    reference = estimate_foot_load_distribution(
        measured["clean_wavelength_shifts_nm"], parameters["temperature_c"]
    )
    estimated_loads = np.asarray(estimated["zone_loads_n"], dtype=float)
    true_total = float(true_loads.sum())
    estimated_total = float(estimated_loads.sum())
    cop_error = float(np.linalg.norm(np.asarray(estimated["cop_xy"]) - np.asarray(reference["cop_xy"])))
    return {
        "parameters": parameters,
        "results": {
            "true_zone_loads_n": true_loads.tolist(),
            "estimated_zone_loads_n": estimated_loads.tolist(),
            "wavelength_shifts_nm": measured_shifts.tolist(),
            "true_total_load_n": true_total,
            "estimated_total_load_n": estimated_total,
            "total_load_error_n": float(estimated_total - true_total),
            "zone_mae_n": float(np.mean(np.abs(estimated_loads - true_loads))),
            "true_cop_xy": np.asarray(reference["cop_xy"], dtype=float).tolist(),
            "estimated_cop_xy": np.asarray(estimated["cop_xy"], dtype=float).tolist(),
            "cop_error": cop_error,
            "reliable_cop": true_total >= 20.0 and estimated_total >= 20.0 and failed_zone is None,
        },
    }


def tactile_csv_bytes(record: dict) -> bytes:
    """导出六路触觉理想量、反演量与原始波长读数。"""
    results = record["results"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(("channel", "ideal_touch_n", "estimated_touch_n", "wavelength_shift_nm"))
    for index, values in enumerate(zip(
        results["ideal_touch_n"], results["estimated_touch_n"],
        results["wavelength_shifts_nm"], strict=True,
    ), start=1):
        writer.writerow((f"触觉通道 {index}", *values))
    return buffer.getvalue().encode("utf-8-sig")


def foot_csv_bytes(record: dict) -> bytes:
    """导出六区真实载荷、反演载荷及绝对误差。"""
    results = record["results"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow((
        "zone", "true_load_n", "estimated_load_n", "absolute_error_n",
        "wavelength_shift_nm",
    ))
    for index, (true_value, estimated_value, wavelength_shift) in enumerate(zip(
        results["true_zone_loads_n"], results["estimated_zone_loads_n"],
        results["wavelength_shifts_nm"], strict=True,
    ), start=1):
        writer.writerow((
            f"足底区域 {index}", true_value, estimated_value,
            abs(estimated_value - true_value), wavelength_shift,
        ))
    return buffer.getvalue().encode("utf-8-sig")


def tactile_report(record: dict) -> str:
    """生成包含实验条件、量化结果和适用边界的触觉报告。"""
    parameters = record["parameters"]
    results = record["results"]
    if results["valid_contact"]:
        separation_line = f"类别间隔：{results['probability_margin'] * 100:.2f} 个百分点；次高类别：{results['runner_up_material']}"
        pattern_line = f"模板偏差：{results['pattern_error_percent']:.2f}%"
        repeat_line = f"重复识别一致率：{results['repeat_match_rate'] * 100:.1f}%（{results['repeat_count']} 次）"
    else:
        separation_line = "类别间隔：不适用（接触不足）；次高类别：不适用"
        pattern_line = "模板偏差：不适用（接触不足）"
        repeat_line = f"重复识别一致率：不适用（接触不足；计划重复 {results['repeat_count']} 次）"
    return "\n".join((
        "多材质触觉识别实验记录",
        f"目标材质：{parameters['material']}；识别结果：{results['diagnosed_material']}；有效接触：{'是' if results['valid_contact'] else '否'}",
        f"握持力：{parameters['grip_force_n']:.2f} N；接触面积：{parameters['contact_area_percent']:.1f}%",
        f"接触模式扰动：{parameters['pattern_noise']:.2f}；温度变化：{parameters['temperature_c']:.2f}°C；波长噪声：{parameters['noise_nm']:.4f} nm；seed：{parameters['seed']}",
        separation_line,
        pattern_line,
        repeat_line,
        "说明：分类采用预设模板的余弦相似度，不是经过真实样本训练和验证的材料识别模型。",
    ))


def foot_report(record: dict) -> str:
    """生成包含真实/反演对照、误差和适用边界的足底报告。"""
    parameters = record["parameters"]
    results = record["results"]
    reliability = (
        "载荷充足，可结合误差观察"
        if results["reliable_cop"]
        else "低载荷或模拟通道失效，CoP 仅供参考"
    )
    return "\n".join((
        "足底平衡与步态实验记录",
        f"地形：{parameters['terrain']}；状态：{parameters['support']}；步态相位：{parameters['phase_percent']:.1f}%",
        f"温度变化：{parameters['temperature_c']:.2f}°C；波长噪声：{parameters['noise_nm']:.4f} nm；seed：{parameters['seed']}",
        f"模拟失效区域：{parameters.get('failed_zone') or '无'}；替代漂移值：{parameters.get('drift_nm', 0.0):.4f} nm",
        f"输入载荷：{parameters['load_n']:.2f} N；真实支撑力：{results['true_total_load_n']:.2f} N；反演支撑力：{results['estimated_total_load_n']:.2f} N",
        f"总载荷误差：{results['total_load_error_n']:.3f} N；区域平均绝对误差：{results['zone_mae_n']:.3f} N",
        f"CoP 位置误差：{results['cop_error']:.4f}；可靠性提示：{reliability}",
        "说明：本结果来自独立线性区域标定教学模型，未包含动态冲击、足部姿态、材料迟滞和真实封装标定。",
    ))


def _wrapped_direction_error_deg(estimated_deg: float, true_deg: float) -> float:
    return float((estimated_deg - true_deg + 180.0) % 360.0 - 180.0)


def run_shape_experiment(params: dict) -> dict:
    """运行三芯形状实验，并量化整条中心线与末端重建误差。"""
    parameters = dict(params)
    simulation = simulate_multicore_shape(
        parameters["curvature_per_m"], parameters["direction_deg"],
        parameters["twist_per_m"], parameters["length_mm"],
        parameters["core_radius_um"], parameters["temperature_c"],
        parameters["noise_nm"], parameters["seed"],
        core_temperature_gradient_c=parameters["core_temperature_gradient_c"],
    )
    truth = np.asarray(simulation["centerline_xyz_mm"], dtype=float)
    estimate = np.asarray(simulation["estimated_centerline_xyz_mm"], dtype=float)
    point_error = np.linalg.norm(estimate - truth, axis=1)
    estimated_curvature = float(simulation["estimated_curvature_per_m"])
    estimated_direction = float(simulation["estimated_direction_deg"])
    direction_identifiable = abs(float(parameters["curvature_per_m"])) > 1e-9
    return {
        "parameters": parameters,
        "results": {
            "estimated_curvature_per_m": estimated_curvature,
            "curvature_error_per_m": estimated_curvature - float(parameters["curvature_per_m"]),
            "estimated_direction_deg": estimated_direction,
            "direction_identifiable": direction_identifiable,
            "direction_error_deg": (
                _wrapped_direction_error_deg(estimated_direction, float(parameters["direction_deg"]))
                if direction_identifiable else None
            ),
            "twist_source": "known_prior",
            "centerline_rmse_mm": float(np.sqrt(np.mean(point_error ** 2))),
            "tip_error_mm": float(point_error[-1]),
            "point_error_mm": point_error.tolist(),
            "true_centerline_xyz_mm": truth.tolist(),
            "estimated_centerline_xyz_mm": estimate.tolist(),
            "core_angles_deg": np.asarray(simulation["core_angles_deg"], dtype=float).tolist(),
            "strain": np.asarray(simulation["strain"], dtype=float).tolist(),
            "wavelength_shifts_nm": np.asarray(simulation["wavelength_shifts_nm"], dtype=float).tolist(),
        },
    }


def shape_csv_bytes(record: dict) -> bytes:
    """导出真实/反演中心线和逐点空间误差。"""
    results = record["results"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow((
        "position_index", "true_x_mm", "true_y_mm", "true_z_mm",
        "estimated_x_mm", "estimated_y_mm", "estimated_z_mm", "point_error_mm",
    ))
    for index, (truth, estimate, error) in enumerate(zip(
        results["true_centerline_xyz_mm"], results["estimated_centerline_xyz_mm"],
        results["point_error_mm"], strict=True,
    )):
        writer.writerow((index, *truth, *estimate, error))
    return buffer.getvalue().encode("utf-8-sig")


def shape_report(record: dict) -> str:
    """生成带实验条件、几何误差和模型边界的形状重建报告。"""
    parameters = record["parameters"]
    results = record["results"]
    direction_line = (
        f"真实方向/反演方向：{parameters['direction_deg']:.1f}° / {results['estimated_direction_deg']:.1f}°；方向误差：{results['direction_error_deg']:+.1f}°"
        if results["direction_identifiable"]
        else "真实方向/反演方向/方向误差：方向不适用（真实曲率为 0）"
    )
    return "\n".join((
        "连续体形状重建实验记录",
        f"真实曲率/反演曲率：{parameters['curvature_per_m']:.3f} / {results['estimated_curvature_per_m']:.3f} 1/m",
        f"曲率误差：{results['curvature_error_per_m']:+.3f} 1/m",
        direction_line,
        f"长度：{parameters['length_mm']:.1f} mm；扭转率（已知重建先验）：{parameters['twist_per_m']:.2f} 1/m；纤芯半径：{parameters['core_radius_um']:.1f} μm",
        f"温度变化：{parameters['temperature_c']:.2f}°C；芯间温度梯度：{parameters['core_temperature_gradient_c']:+.2f}°C/芯；波长噪声：{parameters['noise_nm']:.4f} nm；seed：{parameters['seed']}",
        f"中心线 RMSE：{results['centerline_rmse_mm']:.3f} mm；末端误差：{results['tip_error_mm']:.3f} mm",
        "说明：当前扭转率不是由当前三芯波长读数反演得到，而是作为已知重建先验；本结果不能替代真实连续体机器人的空间标定、动态重建或误差认证。",
    ))


def redundant_fbg_report(
    true_angle_deg: float,
    temperature_c: float,
    fault_mode: str,
    injected_channel: int,
    wavelength_shifts_nm: list[float] | np.ndarray,
    diagnosed_channels: list[int],
    estimated_angle_deg: float,
) -> str:
    """生成包含注入条件、原始四路读数和诊断结果的冗余实验记录。"""
    raw_values = ", ".join(
        f"FBG {index}={float(value):.6f} nm"
        for index, value in enumerate(wavelength_shifts_nm, start=1)
    )
    diagnosed = "、".join(f"FBG {channel}" for channel in diagnosed_channels) or "无"
    injected = "无" if fault_mode == "无" else f"FBG {injected_channel}"
    return "\n".join((
        "冗余 FBG 故障诊断实验记录",
        f"真实弯曲角：{true_angle_deg:.1f}°；温度变化：{temperature_c:.1f}°C",
        f"故障设定：{fault_mode}；注入故障通道：{injected}",
        f"四路原始波长漂移：{raw_values}",
        f"诊断异常通道：{diagnosed}",
        f"剔除异常通道后的反演角：{estimated_angle_deg:.2f}°",
        "说明：冗余中值与阈值仅用于教学，不等同于真实故障诊断认证。",
    ))



def compare_grasp_sensor_layouts(
    contact_force_n: list[float] | np.ndarray,
    palm_touch_n: float,
    *,
    requires_palm: bool,
) -> list[dict[str, str | float]]:
    """Compare current-force coverage without claiming unmodelled hardware accuracy."""
    raw_forces = np.asarray(contact_force_n, dtype=float)
    raw_palm = float(palm_touch_n)
    if raw_forces.shape != (5,) or not np.all(np.isfinite(raw_forces)):
        raise ValueError("抓取布置对照需要五根手指的有限接触力")
    if not np.isfinite(raw_palm):
        raise ValueError("掌心接触力必须为有限数值")
    forces = np.maximum(raw_forces, 0.0)
    palm = max(0.0, raw_palm)

    names = ("拇指", "食指", "中指", "无名指", "小指", "掌心")
    values = np.r_[forces, palm]
    total = float(values.sum())
    layouts = (
        ("六路：五指＋掌心", (0, 1, 2, 3, 4, 5)),
        ("五路：仅五指", (0, 1, 2, 3, 4)),
        ("三路：拇指＋食指＋掌心", (0, 1, 5)),
    )
    required = {0, 1, 2, 3, 4}
    if requires_palm:
        required.add(5)
    rows: list[dict[str, str | float]] = []
    for label, channels in layouts:
        observed = float(values[list(channels)].sum())
        missing = [names[index] for index in range(6) if index not in channels]
        rows.append({
            "传感器布置": label,
            "当前观测受力 (N)": observed,
            "受力覆盖率 (%)": 0.0 if total == 0.0 else observed / total * 100.0,
            "可完整执行当前判定": "是" if required.issubset(channels) else "否",
            "未观测通道": "无" if not missing else "、".join(missing),
        })
    return rows


def _validate_grasp_noise_inputs(noise_std_nm: float, sample_count: int, seed: int) -> None:
    if isinstance(sample_count, bool) or not isinstance(sample_count, (int, np.integer)) or sample_count < 1:
        raise ValueError("重复采样次数必须为正整数")
    if not np.isfinite(noise_std_nm) or noise_std_nm < 0.0:
        raise ValueError("波长噪声标准差必须是非负有限数值")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= int(seed) <= 2**32 - 1:
        raise ValueError("随机种子超出范围")


def _summarise_grasp_noise_samples(
    samples: list[dict[str, int | float | str]],
    *,
    baseline_is_grasped: bool,
    baseline_contact_force_n: np.ndarray,
    baseline_palm_touch_n: float,
    noise_std_nm: float,
    seed: int,
) -> dict[str, Any]:
    decisions = np.asarray([row["抓稳判定"] == "是" for row in samples], dtype=bool)
    total_force = np.asarray([row["五指反演合力 (N)"] for row in samples], dtype=float)
    palm_force = np.asarray([row["掌心反演力 (N)"] for row in samples], dtype=float)
    consistent = decisions == baseline_is_grasped
    return {
        "sample_count": len(samples),
        "noise_std_nm": float(noise_std_nm),
        "seed": int(seed),
        "baseline_is_grasped": bool(baseline_is_grasped),
        "baseline_contact_force_n": np.asarray(baseline_contact_force_n, dtype=float).tolist(),
        "baseline_palm_touch_n": float(baseline_palm_touch_n),
        "grasped_rate_percent": float(decisions.mean() * 100.0),
        "decision_consistency_percent": float(consistent.mean() * 100.0),
        "decision_flip_count": int(np.count_nonzero(~consistent)),
        "total_force_mean_n": float(total_force.mean()),
        "total_force_std_n": float(total_force.std()),
        "palm_force_mean_n": float(palm_force.mean()),
        "palm_force_std_n": float(palm_force.std()),
        "samples": samples,
    }


def run_planar_grasp_noise_study(
    sensing: Mapping[str, Any],
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    temperature_change_c: float,
    noise_std_nm: float,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    """Repeat the planar FBG decision with seeded wavelength noise."""
    _validate_grasp_noise_inputs(noise_std_nm, sample_count, seed)
    clean_shifts = np.asarray(sensing["wavelength_shifts_nm"], dtype=float)
    baseline = classify_planar_grasp_from_fbg(dict(sensing), finger_curls_deg, temperature_change_c)
    rng = np.random.default_rng(int(seed))
    samples: list[dict[str, int | float | str]] = []
    for trial in range(1, int(sample_count) + 1):
        noisy = dict(sensing)
        noisy["wavelength_shifts_nm"] = clean_shifts + rng.normal(0.0, noise_std_nm, clean_shifts.shape)
        decision = classify_planar_grasp_from_fbg(noisy, finger_curls_deg, temperature_change_c)
        samples.append({
            "样本序号": trial,
            "抓稳判定": "是" if decision["is_grasped"] else "否",
            "接触手指数": len(decision["contact_fingers"]),
            "五指反演合力 (N)": float(np.asarray(decision["contact_force_n"]).sum()),
            "掌心反演力 (N)": float(decision["palm_touch_n"]),
        })
    return _summarise_grasp_noise_samples(
        samples,
        baseline_is_grasped=bool(baseline["is_grasped"]),
        baseline_contact_force_n=np.asarray(baseline["contact_force_n"]),
        baseline_palm_touch_n=float(baseline["palm_touch_n"]),
        noise_std_nm=noise_std_nm,
        seed=seed,
    )


def run_three_d_grasp_noise_study(
    sensing: Mapping[str, Any],
    temperature_change_c: float,
    noise_std_nm: float,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    """Repeat the 3D tactile-channel decision with seeded wavelength noise."""
    _validate_grasp_noise_inputs(noise_std_nm, sample_count, seed)
    clean_tactile = np.asarray(sensing["tactile_fbg_shifts_nm"], dtype=float)
    baseline = classify_3d_grasp_from_fbg(dict(sensing), temperature_change_c)
    rng = np.random.default_rng(int(seed))
    samples: list[dict[str, int | float | str]] = []
    for trial in range(1, int(sample_count) + 1):
        noisy = dict(sensing)
        noisy["tactile_fbg_shifts_nm"] = clean_tactile + rng.normal(0.0, noise_std_nm, clean_tactile.shape)
        decision = classify_3d_grasp_from_fbg(noisy, temperature_change_c)
        samples.append({
            "样本序号": trial,
            "抓稳判定": "是" if decision["is_grasped"] else "否",
            "接触手指数": len(decision["contact_fingers"]),
            "五指反演合力 (N)": float(np.asarray(decision["contact_force_n"]).sum()),
            "掌心反演力 (N)": float(decision["palm_touch_n"]),
        })
    return _summarise_grasp_noise_samples(
        samples,
        baseline_is_grasped=bool(baseline["is_grasped"]),
        baseline_contact_force_n=np.asarray(baseline["contact_force_n"]),
        baseline_palm_touch_n=float(baseline["palm_touch_n"]),
        noise_std_nm=noise_std_nm,
        seed=seed,
    )


def grasp_noise_study_csv(study: Mapping[str, Any]) -> bytes:
    """Export every repeat, rather than only the aggregate metrics."""
    buffer = io.StringIO()
    fieldnames = ["样本序号", "抓稳判定", "接触手指数", "五指反演合力 (N)", "掌心反演力 (N)"]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(study["samples"])
    return buffer.getvalue().encode("utf-8-sig")


def grasp_robustness_report(
    dimension: str,
    study: Mapping[str, Any],
    layouts: list[Mapping[str, Any]],
) -> str:
    """Summarise repeatability and the observability limit of three layouts."""
    layout_lines = [
        f"- {row['传感器布置']}：受力覆盖率 {float(row['受力覆盖率 (%)']):.1f}%；"
        f"完整判定 {'可' if row['可完整执行当前判定'] == '是' else '不可'}；"
        f"未观测 {row['未观测通道']}"
        for row in layouts
    ]
    return "\n".join((
        f"{dimension}抓取稳健性实验记录",
        f"{study['sample_count']} 次重复采样；波长噪声 σ={float(study['noise_std_nm']):.4f} nm；seed：{study['seed']}",
        f"无噪声基准判定：{'抓稳' if study['baseline_is_grasped'] else '未抓稳'}；"
        f"判定一致率：{float(study['decision_consistency_percent']):.1f}%；"
        f"翻转次数：{study['decision_flip_count']}",
        f"抓稳率：{float(study['grasped_rate_percent']):.1f}%；"
        f"五指反演合力：{float(study['total_force_mean_n']):.3f} ± {float(study['total_force_std_n']):.3f} N；"
        f"掌心反演力：{float(study['palm_force_mean_n']):.3f} ± {float(study['palm_force_std_n']):.3f} N",
        "传感器布置对照（基于当前工况的可观测性）：",
        *layout_lines,
        "说明：重复采样仅扰动教学模型中的波长读数；布置对照只反映当前通道覆盖与规则可观测性，不能替代真实传感器选型、标定或安全验证。",
    ))

def grasp_report(record: dict) -> str:
    """生成二维或三维抓取的可复现实验条件与传感结果摘要。"""
    contact_fingers = "、".join(str(item) for item in record["contact_fingers"]) or "无"
    contact_forces = ", ".join(f"{float(value):.3f}" for value in record["contact_force_n"])
    wavelength_shifts = ", ".join(
        f"{float(value):.6f}" for value in record["wavelength_shifts_nm"]
    )
    target_position = ", ".join(f"{float(value):.2f}" for value in record["target_position"])
    return "\n".join((
        f"{record['dimension']}抓取实验记录",
        f"任务阶段：{record['task_phase']}；抓稳判定：{'是' if record['is_grasped'] else '否'}",
        f"接触手指：{contact_fingers}；掌心接触力：{float(record['palm_touch_n']):.3f} N",
        f"五指接触力 (N)：{contact_forces}",
        f"六路 FBG 波长漂移 (nm)：{wavelength_shifts}",
        f"目标位置：({target_position})；温度变化：{float(record['temperature_c']):.2f}°C；波长噪声：{float(record['noise_nm']):.4f} nm；seed：{record['seed']}",
        "说明：抓稳判定来自简化接触与 FBG 教学模型，不能替代真实机械手的碰撞、摩擦、力控或安全验证。",
    ))


def _distributed_profile(result: dict, mode: str) -> tuple[np.ndarray, np.ndarray, str, str]:
    positions = np.asarray(result["position_mm"], dtype=float)
    if mode == "Rayleigh/OFDR":
        return positions, np.asarray(result["strain_ue"], dtype=float), "应变", "με"
    if mode == "φ-OTDR / DAS":
        amplitude = np.asarray(result["amplitude"], dtype=float)
        return positions, np.max(np.abs(amplitude), axis=0), "最大振动幅值", "a.u."
    if mode == "Brillouin":
        values = np.asarray(result["brillouin_frequency_ghz"], dtype=float)
        return positions, values, "Brillouin 频移", "GHz"
    values = np.asarray(result["temperature_c"], dtype=float)
    return positions, values, "温度", "°C"


def run_distributed_experiment(params: dict) -> dict:
    """运行分布式教学实验并量化采样后的事件定位误差。"""
    parameters = dict(params)
    result, frame = simulate_distributed_mechanism(
        parameters["mode"], parameters["fiber_length_mm"],
        parameters["event_position_mm"], parameters["event_strength"],
        int(parameters["sample_rate_hz"]),
    )
    result = decimate_distributed_result(result, int(parameters["spatial_spacing_mm"]))
    positions, profile, observable, unit = _distributed_profile(result, parameters["mode"])
    centered_profile = np.abs(profile - np.median(profile))
    estimated_position = float(positions[int(np.argmax(centered_profile))])
    return {
        "parameters": parameters,
        "results": {
            "sensor_type": str(frame["sensor_type"]),
            "quality": float(frame["quality"]),
            "sampled_points": int(len(positions)),
            "estimated_event_position_mm": estimated_position,
            "location_error_mm": abs(estimated_position - float(parameters["event_position_mm"])),
            "observable": observable,
            "unit": unit,
            "profile_position_mm": positions.tolist(),
            "profile_value": profile.tolist(),
        },
    }


def distributed_csv_bytes(record: dict) -> bytes:
    """导出分布式实验的降采样空间剖面。"""
    results = record["results"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(("position_mm", "profile_value", "observable", "unit"))
    for position, value in zip(
        results["profile_position_mm"], results["profile_value"], strict=True
    ):
        writer.writerow((position, value, results["observable"], results["unit"]))
    return buffer.getvalue().encode("utf-8-sig")


def distributed_report(record: dict) -> str:
    """生成包含机制、定位误差、采样条件和模型边界的分布式报告。"""
    parameters = record["parameters"]
    results = record["results"]
    return "\n".join((
        "分布式光纤感知实验记录",
        f"机制：{parameters['mode']}；光纤长度：{parameters['fiber_length_mm']:.1f} mm",
        f"真实/估计事件位置：{parameters['event_position_mm']:.1f} / {results['estimated_event_position_mm']:.1f} mm；事件定位误差：{results['location_error_mm']:.2f} mm",
        f"事件幅值：{parameters['event_strength']:.1f}；空间采样间隔：{parameters['spatial_spacing_mm']} mm；采样率：{parameters['sample_rate_hz']} Hz",
        f"显示采样点：{results['sampled_points']}；数据质量：{results['quality'] * 100:.1f}%；导出观测量：{results['observable']} ({results['unit']})",
        "说明：定位值来自降采样剖面的峰值，只用于比较机制与采样影响；这是教学解析模型，不代表商用解调设备性能。",
    ))


def run_health_experiment(params: dict) -> dict:
    """运行点式 FBG 健康监测实验，分开报告检测与定位。"""
    parameters = dict(params)
    positions = np.linspace(80.0, 440.0, int(parameters["sensor_count"]))
    simulation = simulate_arm_health_fbg(
        parameters["load_n"], parameters["anomaly_position_mm"],
        parameters["anomaly_severity"], parameters["temperature_c"],
        parameters["noise_nm"], parameters["seed"], sensor_positions_mm=positions,
    )
    diagnosis = diagnose_arm_health(
        simulation["wavelength_shifts_nm"], parameters["temperature_c"],
        sensor_positions_mm=positions,
    )
    detected = diagnosis["status"] == "需检查"
    expected_detection = float(parameters["anomaly_severity"]) >= 0.35
    localization_valid = bool(detected and expected_detection)
    localization_error = (
        abs(float(diagnosis["suspected_location_mm"]) - float(parameters["anomaly_position_mm"]))
        if localization_valid else None
    )
    return {
        "parameters": parameters,
        "results": {
            "status": str(diagnosis["status"]),
            "detected": detected,
            "expected_detection": expected_detection,
            "detection_matches": detected == expected_detection,
            "localization_valid": localization_valid,
            "suspected_location_mm": float(diagnosis["suspected_location_mm"]),
            "localization_error_mm": localization_error,
            "location_uncertainty_mm": float(diagnosis["location_uncertainty_mm"]),
            "sensor_spacing_mm": float(diagnosis["sensor_spacing_mm"]),
            "damage_index": float(diagnosis["damage_index"]),
            "sensor_positions_mm": positions.tolist(),
            "damage_profile": np.asarray(simulation["damage_profile"], dtype=float).tolist(),
            "strain_ue": (np.asarray(simulation["strain"], dtype=float) * 1e6).tolist(),
            "wavelength_shifts_nm": np.asarray(simulation["wavelength_shifts_nm"], dtype=float).tolist(),
        },
    }


def health_csv_bytes(record: dict) -> bytes:
    """导出点式阵列位置、异常剖面、应变与波长读数。"""
    results = record["results"]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(("sensor_position_mm", "damage_profile", "strain_ue", "wavelength_shift_nm"))
    for row in zip(
        results["sensor_positions_mm"], results["damage_profile"],
        results["strain_ue"], results["wavelength_shifts_nm"], strict=True,
    ):
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")


def health_report(record: dict) -> str:
    """生成区分检测、定位和安全结论边界的健康监测报告。"""
    parameters = record["parameters"]
    results = record["results"]
    has_configured_anomaly = float(parameters["anomaly_severity"]) > 0.0
    if has_configured_anomaly:
        anomaly_line = f"真实异常位置：{parameters['anomaly_position_mm']:.1f} mm；阵列数量：{parameters['sensor_count']}；传感器间距：{results['sensor_spacing_mm']:.1f} mm"
    else:
        anomaly_line = f"异常位置参数：未启用（当前为健康场景）；阵列数量：{parameters['sensor_count']}；传感器间距：{results['sensor_spacing_mm']:.1f} mm"
    if results["localization_valid"]:
        location_line = f"可疑位置：{results['suspected_location_mm']:.1f} ± {results['location_uncertainty_mm']:.1f} mm；定位误差：{results['localization_error_mm']:.2f} mm"
    elif has_configured_anomaly and not results["detected"]:
        location_line = "检测结果：已设置异常但未检出；可疑位置：未形成有效异常定位；定位误差：不适用"
    else:
        location_line = "可疑位置：未形成有效异常定位；定位误差：不适用"
    return "\n".join((
        "机械臂结构健康监测实验记录",
        f"载荷：{parameters['load_n']:.1f} N；异常程度：{parameters['anomaly_severity']:.2f}",
        anomaly_line,
        f"温度变化：{parameters['temperature_c']:.2f}°C；波长噪声：{parameters['noise_nm']:.4f} nm；seed：{parameters['seed']}",
        f"检测状态：{results['status']}；局部异常指数：{results['damage_index']:.3f}；检测是否符合设定：{'是' if results['detection_matches'] else '否'}",
        location_line,
        "说明：‘需检查’只是教学阈值触发，不能替代真实结构的安全评估、无损检测或维护决策。",
    ))


def run_optical_experiment(params: dict) -> dict:
    """汇总偏振、Sagnac 与 EFPI 三种光学观测量。"""
    parameters = dict(params)
    polarization = simulate_polarization_sensing(
        parameters["stress_mpa"], parameters["twist_deg"], parameters["temperature_c"]
    )
    zero_temperature = simulate_polarization_sensing(
        parameters["stress_mpa"], parameters["twist_deg"], 0.0
    )
    gyro = simulate_sagnac_gyro(parameters["gyro_rate_deg_s"], 120.0)
    efpi = simulate_efpi_pressure(parameters["pressure_mpa"], parameters["cavity_um"])
    effective_cavity = float(efpi["effective_cavity_um"])
    return {
        "parameters": parameters,
        "results": {
            "stokes": np.asarray(polarization["stokes"], dtype=float).tolist(),
            "azimuth_deg": float(polarization["azimuth_deg"]),
            "ellipticity_deg": float(polarization["ellipticity_deg"]),
            "temperature_ellipticity_offset_deg": float(
                polarization["ellipticity_deg"] - zero_temperature["ellipticity_deg"]
            ),
            "sagnac_phase_shift_rad": float(gyro["phase_shift_rad"]),
            "estimated_rate_deg_s": float(gyro["estimated_rate_deg_s"]),
            "effective_cavity_um": effective_cavity,
            "cavity_change_nm": (effective_cavity - float(parameters["cavity_um"])) * 1000.0,
        },
    }


def optical_csv_bytes(record: dict) -> bytes:
    """按观测量导出偏振、Sagnac 与 EFPI 数值。"""
    results = record["results"]
    rows = (
        ("偏振", "Stokes S1", results["stokes"][0], "1"),
        ("偏振", "Stokes S2", results["stokes"][1], "1"),
        ("偏振", "Stokes S3", results["stokes"][2], "1"),
        ("偏振", "方位角", results["azimuth_deg"], "deg"),
        ("偏振", "椭圆率角", results["ellipticity_deg"], "deg"),
        ("Sagnac", "相位差", results["sagnac_phase_shift_rad"], "rad"),
        ("EFPI", "有效腔长", results["effective_cavity_um"], "um"),
        ("EFPI", "腔长变化", results["cavity_change_nm"], "nm"),
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(("mechanism", "observable", "value", "unit"))
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def optical_report(record: dict) -> str:
    """生成三类光学机制的参数、结果和交叉敏感性说明。"""
    parameters = record["parameters"]
    results = record["results"]
    stokes = ", ".join(f"{value:+.4f}" for value in results["stokes"])
    return "\n".join((
        "偏振与干涉传感实验记录",
        f"横向应力：{parameters['stress_mpa']:.1f} MPa；扭转：{parameters['twist_deg']:.1f}°；温度变化：{parameters['temperature_c']:.1f}°C",
        f"Stokes (S1, S2, S3)：({stokes})；方位角：{results['azimuth_deg']:.2f}°；椭圆率角：{results['ellipticity_deg']:.2f}°",
        f"温度引起的椭圆率偏移：{results['temperature_ellipticity_offset_deg']:+.2f}°",
        f"Sagnac 输入/反演角速度：{parameters['gyro_rate_deg_s']:.1f} / {results['estimated_rate_deg_s']:.1f} °/s；相位差：{results['sagnac_phase_shift_rad']:.6e} rad",
        f"EFPI 压力：{parameters['pressure_mpa']:.3f} MPa；初始/有效腔长：{parameters['cavity_um']:.3f} / {results['effective_cavity_um']:.3f} μm；腔长变化：{results['cavity_change_nm']:+.2f} nm",
        "说明：三部分均为彼此独立的解析教学模型，只用于理解观测量，不代表真实偏振解调、惯导或压力传感器精度。",
    ))


def sensing_chain_report(task_name: str, task_record: str, chain_summary: str) -> str:
    """把选中任务的操作步骤、当前记录与共享数据链摘要合成报告。"""
    if task_name not in CHAIN_TASK_GUIDES:
        raise ValueError("不支持的实验任务")
    steps = CHAIN_TASK_GUIDES[task_name]
    return "\n".join((
        "光纤机器人感知链实验报告（教学仿真）",
        f"任务：{task_name}",
        "任务步骤：",
        *(f"{index}. {step}" for index, step in enumerate(steps, start=1)),
        "当前任务记录：",
        task_record,
        "共享感知链摘要：",
        chain_summary,
        "说明：本报告不可替代真实系统的标定、风险评估或安全决策。",
    ))
