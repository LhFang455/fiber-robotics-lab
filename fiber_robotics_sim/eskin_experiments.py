"""Presets and export helpers for electronic-skin teaching experiments."""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping

import numpy as np


ESKIN_PRESETS = {
    "三轴温漂校正": {
        "mode": "三轴单元",
        "description": "在曲率、应变和升温共同存在时比较主动信号与参考校正。",
        "state": {
            "eskin_fx_n": 2.0, "eskin_fy_n": -1.5, "eskin_fz_n": 8.0,
            "eskin_curvature": 3.0, "eskin_strain_milli": 1.0,
            "eskin_taxel_temperature": 38.0, "eskin_noise_pf": 0.01,
            "eskin_reference_match": 0.98,
        },
    },
    "双点光学触觉": {
        "mode": "光学皮肤",
        "description": "用八个 FBG 的重叠感受野观察双点载荷的合力与压力质心。",
        "state": {
            "eskin_fbg_sensor_count": 8, "eskin_contact_mode": "双点",
            "eskin_touch1_x": 24.0, "eskin_touch1_y": 30.0, "eskin_touch1_force": 6.0,
            "eskin_touch2_x": 58.0, "eskin_touch2_y": 22.0, "eskin_touch2_force": 4.0,
            "eskin_skin_width": 80.0, "eskin_skin_height": 60.0,
            "eskin_receptive_width": 18.0, "eskin_fbg_temperature": 36.0,
            "eskin_noise_nm": 0.001,
        },
    },
    "稀疏双点重建": {
        "mode": "压力重建",
        "description": "比较 4×4 稀疏采样与 16×16 压力场的双点重建误差。",
        "state": {
            "eskin_pressure_scenario": "双点接触", "eskin_sparse_size": 4,
            "eskin_output_size": 16, "eskin_peak_pressure": 80.0,
            "eskin_kernel_bandwidth": 0.16, "eskin_pressure_noise": 0.5,
        },
    },
    "即将滑移": {
        "mode": "动态判别",
        "description": "让剪切比和压力质心速度同时越过教学阈值，观察风险告警。",
        "state": {
            "eskin_dynamic_event": "即将滑移", "eskin_dynamic_sample_rate": 100,
            "eskin_dynamic_duration": 4.0, "eskin_dynamic_force": 12.0,
            "eskin_slip_threshold": 0.35, "eskin_dynamic_temperature": 25.0,
            "eskin_dynamic_noise": 0.01, "eskin_repeat_count": 50,
        },
    },
    "热物体": {
        "mode": "动态判别",
        "description": "在稳定接触中加入升温过程，区分热事件与滑移风险。",
        "state": {
            "eskin_dynamic_event": "热物体", "eskin_dynamic_sample_rate": 50,
            "eskin_dynamic_duration": 4.0, "eskin_dynamic_force": 8.0,
            "eskin_slip_threshold": 0.35, "eskin_dynamic_temperature": 25.0,
            "eskin_dynamic_noise": 0.01, "eskin_repeat_count": 50,
        },
    },
}


def eskin_result_records(section: str, result: Mapping) -> list[dict]:
    """Convert one model result to tidy row records for CSV export."""
    if section == "三轴单元":
        return [
            {
                "channel": index + 1,
                "active_pf": float(result["active_pf"][index]),
                "reference_pf": float(result["reference_pf"][index]),
                "corrected_pf": float(result["corrected_pf"][index]),
            }
            for index in range(len(result["active_pf"]))
        ]
    if section == "光学皮肤":
        return [
            {
                "sensor": result["sensor_labels"][index],
                "x_mm": float(result["sensor_positions_mm"][index, 0]),
                "y_mm": float(result["sensor_positions_mm"][index, 1]),
                "measured_shift_nm": float(result["measured_shift_nm"][index]),
                "compensated_shift_nm": float(result["compensated_shift_nm"][index]),
            }
            for index in range(len(result["sensor_labels"]))
        ]
    if section == "压力重建":
        truth = np.asarray(result["truth_kpa"])
        reconstruction = np.asarray(result["reconstruction_kpa"])
        error = np.asarray(result["error_kpa"])
        return [
            {
                "row": row,
                "column": column,
                "truth_kpa": float(truth[row, column]),
                "reconstruction_kpa": float(reconstruction[row, column]),
                "error_kpa": float(error[row, column]),
            }
            for row in range(truth.shape[0])
            for column in range(truth.shape[1])
        ]
    if section == "动态判别":
        return [
            {
                "time_s": float(result["time_s"][index]),
                "normal_force_n": float(result["normal_force_n"][index]),
                "shear_force_n": float(result["shear_force_n"][index]),
                "shear_ratio": float(result["shear_ratio"][index]),
                "centroid_x_mm": float(result["centroid_x_mm"][index]),
                "temperature_c": float(result["temperature_c"][index]),
            }
            for index in range(len(result["time_s"]))
        ]
    raise ValueError(f"unknown electronic-skin section: {section}")


def eskin_csv_bytes(section: str, result: Mapping) -> bytes:
    """Encode one result as an Excel-friendly UTF-8 CSV."""
    records = eskin_result_records(section, result)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
    return buffer.getvalue().encode("utf-8-sig")


def eskin_report_bytes(
    title: str,
    parameters: Mapping[str, object],
    metrics: Mapping[str, object],
    boundary: str,
) -> bytes:
    """Build a compact Markdown report that states the evidence boundary."""
    lines = [
        "# 电子皮肤教学实验报告",
        "",
        f"实验：{title}",
        "",
        "## 输入参数",
        "",
    ]
    lines.extend(f"- {name}：{value}" for name, value in parameters.items())
    lines.extend(("", "## 结果指标", ""))
    lines.extend(f"- {name}：{value}" for name, value in metrics.items())
    lines.extend((
        "", "## 解释边界", "",
        f"本报告来自可复现的教学仿真。{boundary}",
        "结果用于理解传感与算法关系，不能替代器件标定、实物测试或安全认证。",
    ))
    return "\n".join(lines).encode("utf-8-sig")
