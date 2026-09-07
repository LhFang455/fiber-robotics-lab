"""Reproducible presets and evidence for the short visitor showcase."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from . import eskin, experiments


CHAPTERS = (
    {
        "id": "foot",
        "label": "01 · 足底重心",
        "title": "脚底如何感知重心？",
        "question": "脚跟到前掌的受力变化，怎样变成可以读取的位置？",
        "action": "点击播放，观察六区载荷由后向前迁移，再对照白点与橙环。",
        "explanation": "六路 FBG 先反演区域载荷，再计算压力中心；两个标记接近时，说明当前理想条件下的反演跟随了真实重心。",
        "boundary": "这里扫描支撑相位，不模拟完整步行动力学；低载荷或通道失效时 CoP 不能作为可靠定位。",
        "kind": "foot",
        "target": (3, "foot_navigation", "平衡与步态"),
    },
    {
        "id": "fbg",
        "label": "02 · 温度补偿",
        "title": "波长变了，手指一定弯了吗？",
        "question": "机械弯曲保持不变时，升温为什么仍会改变光栅波长？",
        "action": "播放前半段的机械加载和后半段的升温，比较未温补角与温补反演角。",
        "explanation": "原始波长同时包含应变与温度项；使用已知温差扣除共模漂移后，反演角更接近设定弯曲角。",
        "boundary": "温补使用已知温差，未模拟温度参考器件、温度梯度或真实封装误差。",
        "kind": "fbg",
        "target": (1, "foundation_navigation", "弯曲标定与诊断"),
    },
    {
        "id": "pressure",
        "label": "03 · 压力重建",
        "title": "少量传感点能还原整张压力图吗？",
        "question": "采样点从 4×4 增加到 8×8，哪些误差会改变？",
        "action": "切换采样密度并播放，使用同一真实压力场、输出网格与色标比较采样和重建。",
        "explanation": "稀疏采样可以通过高斯核插值形成连续图，但通道数量、峰值、质心和总载荷误差之间存在具体取舍。",
        "boundary": "位置为归一化坐标，重建采用高斯核插值；通道节省率不等于实物成本节省。",
        "kind": "pressure",
        "target": (6, "eskin_navigation", "稀疏压力重建"),
    },
)


def chapter(index: int) -> dict:
    """Return one immutable-by-convention chapter definition."""
    return CHAPTERS[index]


def demo_parameters(chapter_id: str, pressure_size: int = 4) -> dict:
    """Return fixed parameters that never depend on Streamlit session state."""
    if chapter_id == "foot":
        return deepcopy(experiments.FOOT_PRESETS["平地中期"])
    if chapter_id == "fbg":
        return deepcopy(experiments.PRESETS["温漂对照"])
    if chapter_id == "pressure":
        if pressure_size not in (4, 8):
            raise ValueError("showcase pressure grid must be 4 or 8")
        return {
            "scenario": "双点接触",
            "sparse_size": pressure_size,
            "output_size": 32,
            "peak_pressure_kpa": 80.0,
            "bandwidth": 0.16,
            "noise_kpa": 0.0,
            "seed": 17,
        }
    raise ValueError(f"unknown showcase chapter: {chapter_id}")


def chapter_metrics(chapter_id: str, pressure_size: int = 4) -> list[tuple[str, str, str | None]]:
    """Calculate concise evidence cards from the same models used by each demo."""
    if chapter_id == "foot":
        params = demo_parameters("foot")
        heel = experiments.run_foot_experiment({**params, "phase_percent": 10.0})["results"]
        toe = experiments.run_foot_experiment({**params, "phase_percent": 90.0})["results"]
        movement = np.asarray(toe["estimated_cop_xy"]) - np.asarray(heel["estimated_cop_xy"])
        return [
            ("支撑载荷", f"{toe['estimated_total_load_n']:.1f} N", None),
            ("反演 CoP 迁移", f"{np.linalg.norm(movement):.2f}", "归一化距离"),
            ("末端区域 MAE", f"{toe['zone_mae_n']:.2f} N", None),
        ]
    if chapter_id == "fbg":
        params = demo_parameters("fbg")
        result = experiments.run_calibration(params)["results"]
        raw_error = abs(result["uncompensated_angle_deg"] - params["angle_deg"])
        corrected_error = abs(result["estimated_angle_deg"] - params["angle_deg"])
        return [
            ("设定弯曲角", f"{params['angle_deg']:.1f}°", None),
            ("未温补误差", f"{raw_error:.1f}°", None),
            ("温补后误差", f"{corrected_error:.1f}°", None),
        ]
    result = eskin.simulate_pressure_reconstruction(**demo_parameters("pressure", pressure_size))
    return [
        ("采样通道", str(pressure_size**2), f"{pressure_size}×{pressure_size}"),
        ("压力 RMSE", f"{result['rmse_kpa']:.2f} kPa", None),
        ("峰值误差", f"{result['peak_error_pct']:.1f}%", None),
        ("质心误差", f"{result['centroid_error_pct']:.1f}%", None),
    ]


def pressure_comparison() -> list[dict]:
    """Return a fair 4x4/8x8 comparison with every other parameter fixed."""
    rows = []
    for size in (4, 8):
        result = eskin.simulate_pressure_reconstruction(**demo_parameters("pressure", size))
        rows.append({
            "采样网格": f"{size}×{size}",
            "通道数": size**2,
            "压力 RMSE (kPa)": result["rmse_kpa"],
            "峰值误差 (%)": result["peak_error_pct"],
            "质心误差 (%)": result["centroid_error_pct"],
            "总载荷误差 (%)": result["total_force_error_pct"],
        })
    return rows
