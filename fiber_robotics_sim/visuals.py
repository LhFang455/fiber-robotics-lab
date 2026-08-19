"""Plotly visualisations for the learning lab."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fiber_robotics_sim import models


COLORS = {"truth": "#17a2b8", "estimate": "#ff7f0e", "sensor": "#6f42c1"}


def fbg_simplus_input_figure(result: dict) -> go.Figure:
    """Preview FEM fields in the public FBG-SimPlus tutorial input order."""
    position_mm = np.asarray(result["position_m"], dtype=float) * 1000.0
    transverse_stress = np.asarray(result["transverse_stress_pa"], dtype=float)
    stress_magnitude_mpa = np.linalg.norm(transverse_stress, axis=1) / 1e6
    figure = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=.08,
        subplot_titles=("纵向应变", "横向应力合量", "温度"),
    )
    figure.add_scatter(x=position_mm, y=result["longitudinal_strain"], mode="lines", name="纵向应变 εxx", line={"color": COLORS["truth"], "width": 3}, row=1, col=1)
    figure.add_scatter(x=position_mm, y=stress_magnitude_mpa, mode="lines", name="横向应力 σy / σz", line={"color": COLORS["sensor"], "width": 3}, row=2, col=1)
    figure.add_scatter(x=position_mm, y=result["temperature_k"], mode="lines", name="温度", line={"color": COLORS["estimate"], "width": 3}, row=3, col=1)
    figure.update_layout(title="FBG-SimPlus 兼容输入预览（FEM 导出数据）", template="plotly_white", height=620, legend={"orientation": "h", "y": 1.08})
    figure.update_xaxes(title_text="位置 (mm)", row=3, col=1)
    figure.update_yaxes(title_text="应变", row=1, col=1)
    figure.update_yaxes(title_text="MPa", row=2, col=1)
    figure.update_yaxes(title_text="K", row=3, col=1)
    return figure


def _base_layout(title: str, xaxis: str, yaxis: str) -> dict:
    return {"title": title, "xaxis_title": xaxis, "yaxis_title": yaxis, "template": "plotly_white", "height": 410}


def finger_figure(result: dict) -> go.Figure:
    line = np.asarray(result["centerline_xy_mm"])
    sensors = np.asarray(result["sensor_positions_mm"])
    sensor_y = np.interp(sensors, line[:, 0], line[:, 1])
    figure = go.Figure()
    figure.add_scatter(x=line[:, 0], y=line[:, 1], mode="lines", name="柔性手指", line={"color": COLORS["truth"], "width": 6})
    figure.add_scatter(x=sensors, y=sensor_y, mode="markers+text", text=["FBG 1", "FBG 2", "FBG 3"], textposition="top center", name="FBG", marker={"size": 11, "color": COLORS["sensor"]})
    figure.update_layout(**_base_layout("常曲率柔性手指", "长度方向 (mm)", "横向位移 (mm)"))
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def sensor_bar_figure(positions: np.ndarray, shifts: np.ndarray, title: str) -> go.Figure:
    figure = go.Figure(go.Bar(x=[f"FBG {index + 1}" for index in range(len(positions))], y=shifts, marker_color=COLORS["sensor"], text=[f"{value:.4f}" for value in shifts], textposition="outside"))
    figure.update_layout(**_base_layout(title, "传感器", "波长漂移 Δλ (nm)"))
    return figure


def material_probability_figure(probabilities: dict[str, float]) -> go.Figure:
    """Show the four candidate materials and their recognition probabilities."""
    names = list(probabilities.keys())
    values = [float(probabilities[name]) for name in names]
    figure = go.Figure(go.Bar(
        x=names,
        y=values,
        marker_color=COLORS["sensor"],
        text=[f"{value * 100:.0f}%" for value in values],
        textposition="outside",
    ))
    figure.update_layout(**_base_layout("触觉识别概率分布", "候选材质", "概率"))
    return figure


def sensing_chain_svg() -> str:
    """Return an inline dark-theme SVG of the fibre-sensing chain."""
    return """<svg viewBox="0 0 1000 220" width="100%" height="100%" xmlns="http://www.w3.org/2000/svg" font-family="sans-serif">
  <defs><marker id="chain-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#44b8d5"/></marker></defs>
  <rect x="20" y="55" width="190" height="110" rx="12" fill="#132a3b" stroke="#315064"/>
  <text x="115" y="90" text-anchor="middle" fill="#8fd8ea" font-size="15" font-weight="700">光纤传感</text>
  <text x="115" y="113" text-anchor="middle" fill="#a9c0cf" font-size="12">抓取 · 触觉 · 足底</text>
  <text x="115" y="133" text-anchor="middle" fill="#a9c0cf" font-size="12">连续体 · 分布式</text>
  <line x1="210" y1="110" x2="272" y2="110" stroke="#44b8d5" stroke-width="4" marker-end="url(#chain-arrow)"/>
  <rect x="278" y="55" width="190" height="110" rx="12" fill="#132a3b" stroke="#315064"/>
  <text x="373" y="90" text-anchor="middle" fill="#8fd8ea" font-size="15" font-weight="700">光纤解调</text>
  <text x="373" y="113" text-anchor="middle" fill="#a9c0cf" font-size="12">波长读取 · 滤波温补</text>
  <text x="373" y="133" text-anchor="middle" fill="#a9c0cf" font-size="12">冗余故障隔离</text>
  <line x1="468" y1="110" x2="530" y2="110" stroke="#44b8d5" stroke-width="4" marker-end="url(#chain-arrow)"/>
  <rect x="536" y="55" width="190" height="110" rx="12" fill="#132a3b" stroke="#315064"/>
  <text x="631" y="90" text-anchor="middle" fill="#8fd8ea" font-size="15" font-weight="700">状态估计</text>
  <text x="631" y="113" text-anchor="middle" fill="#a9c0cf" font-size="12">CoP · 曲率 · 异常位置</text>
  <text x="631" y="133" text-anchor="middle" fill="#a9c0cf" font-size="12">分布式定位 · 偏振态</text>
  <line x1="726" y1="110" x2="788" y2="110" stroke="#44b8d5" stroke-width="4" marker-end="url(#chain-arrow)"/>
  <rect x="794" y="55" width="190" height="110" rx="12" fill="#132a3b" stroke="#315064"/>
  <text x="889" y="90" text-anchor="middle" fill="#8fd8ea" font-size="15" font-weight="700">控制与任务</text>
  <text x="889" y="113" text-anchor="middle" fill="#a9c0cf" font-size="12">张开 / 闭合命令</text>
  <text x="889" y="133" text-anchor="middle" fill="#a9c0cf" font-size="12">多模态任务报告</text>
</svg>"""


def replaceable_sole_transfer_figure(result: dict) -> go.Figure:
    """Render the two-dimensional transmission field for an assembly prediction."""
    figure = go.Figure(
        go.Heatmap(
            x=result["transfer_x_mm"],
            y=result["transfer_y_mm"],
            z=result["transfer_index"],
            colorscale="Viridis",
            colorbar={"title": "相对传力"},
            hovertemplate="横向 %{x:.1f} mm<br>纵向 %{y:.1f} mm<br>相对传力 %{z:.3f}<extra></extra>",
        )
    )
    figure.add_shape(type="rect", x0=-18, x1=18, y0=-34, y1=34, line={"color": "#ffffff", "width": 2})
    figure.add_shape(type="rect", x0=-9, x1=9, y0=-22, y1=22, line={"color": COLORS["sensor"], "width": 2, "dash": "dot"})
    figure.add_annotation(x=0, y=39, text="可更换外底 / 分区传力模块边界", showarrow=False)
    figure.add_annotation(x=0, y=-39, text="固定感知芯与布线通道（示意）", showarrow=False, font={"color": COLORS["sensor"]})
    figure.update_layout(**_base_layout("二维传力场结果模板（空载装配预测）", "横向位置 (mm)", "纵向位置 (mm)"))
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def replaceable_sole_explainer_figure(
    assembly: dict,
    zones: np.ndarray,
    terrain: str,
    cop_region: float,
    support: str,
) -> go.Figure:
    """Connect replaceable-sole structure, empty-load screening and sole use in one view.

    This is an explanatory interface diagram.  Its values come from the same
    simplified teaching models as the detailed charts and are not a product
    qualification result.
    """
    zones = np.asarray(zones, dtype=float)
    if zones.shape != (6,):
        raise ValueError("足底总览需要六个分区载荷")
    prediction = str(assembly["assembly_prediction"])
    passed = prediction == "装配预测通过"
    status_color = "#1f9d8a" if passed else "#d66a34"
    residual = float(assembly["mean_baseline_residual_ue"])
    difference = float(assembly["left_right_difference_ue"])
    maximum = max(float(zones.max()), 1.0)
    figure = go.Figure()

    # Three large reading panels: hardware, unloaded check, and operation.
    for x0, x1, title, color in (
        (1, 31, "① 换装结构", "#d9eef3"),
        (35, 65, "② 空载自检", "#efe6fa"),
        (69, 99, "③ 足底使用", "#fff0dd"),
    ):
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=21, y1=94,
            fillcolor=color, line={"color": "#7d98a7", "width": 1}, layer="below",
        )
        figure.add_annotation(x=(x0 + x1) / 2, y=90, text=f"<b>{title}</b>", showarrow=False,
                              font={"size": 17, "color": "#173445"})

    # Exploded, sectional representation: the sensing core remains in place.
    layers = (
        (77, 84, "可更换耐磨外底 / 分区传力模块", "#e9a85b"),
        (65, 72, "柔性隔离膜：受力与密封分离", "#f3d7b4"),
        (52, 60, "固定光纤感知芯：FBG 与布线通道", "#5fb6c5"),
        (39, 47, "基板 / 轴向限位台阶", "#7895a3"),
    )
    for y0, y1, label, color in layers:
        figure.add_shape(type="rect", x0=5, x1=27, y0=y0, y1=y1,
                         fillcolor=color, line={"color": "#315064", "width": 1.4})
        figure.add_annotation(x=16, y=(y0 + y1) / 2, text=label, showarrow=False,
                              font={"size": 11, "color": "#102a38"})
    for x in (8, 24):
        figure.add_shape(type="line", x0=x, x1=x, y0=41, y1=81,
                         line={"color": "#345a6a", "width": 3})
    figure.add_annotation(x=29, y=61, text="定位柱 + 锁止件\n控制复装位置", showarrow=True,
                          ax=50, ay=0, arrowhead=2, font={"size": 11, "color": "#173445"})
    figure.add_annotation(x=16, y=85, text="复装方向", showarrow=True, ax=0, ay=-30,
                          arrowhead=3, arrowwidth=2, arrowcolor="#c75d2c", font={"size": 11})
    figure.add_annotation(x=16, y=31, text="感知芯不随外底更换", showarrow=False,
                          font={"size": 12, "color": "#176f7d"})

    # Unloaded working/reference grating comparison and decision.
    for x, label, color in ((40, "W1", COLORS["sensor"]), (48, "W2", COLORS["sensor"]), (56, "R", "#6c757d")):
        figure.add_shape(type="circle", x0=x - 3.4, x1=x + 3.4, y0=69, y1=76,
                         fillcolor=color, line={"color": "#ffffff", "width": 1})
        figure.add_annotation(x=x, y=72.5, text=f"<b>{label}</b>", showarrow=False,
                              font={"color": "white", "size": 13})
    figure.add_annotation(x=48, y=62, text="温度补偿后比较空载基线", showarrow=True, ax=0, ay=-28,
                          arrowhead=2, font={"size": 12, "color": "#173445"})
    figure.add_shape(type="rect", x0=39, x1=57, y0=42, y1=53, fillcolor="#ffffff",
                     line={"color": "#9474b6", "width": 1.5})
    figure.add_annotation(x=48, y=47.5,
                          text=f"平均残差 {residual:.1f} με<br>|W1−W2| {difference:.1f} με",
                          showarrow=False, font={"size": 12, "color": "#35234a"})
    figure.add_shape(type="rect", x0=39, x1=57, y0=28, y1=37, fillcolor=status_color,
                     line={"color": status_color})
    figure.add_annotation(x=48, y=32.5, text=f"<b>{prediction}</b>", showarrow=False,
                          font={"size": 12, "color": "white"})

    # Six coloured zones show the current operational load separately from the check.
    zone_positions = ((74, 70), (82, 70), (90, 70), (74, 48), (82, 48), (90, 48))
    for index, ((x, y), value) in enumerate(zip(zone_positions, zones, strict=True), start=1):
        alpha = .22 + .70 * float(value) / maximum
        figure.add_shape(type="rect", x0=x, x1=x + 6.2, y0=y, y1=y + 17,
                         fillcolor=f"rgba(31,157,138,{alpha:.2f})",
                         line={"color": "#315064", "width": 1.2})
        figure.add_annotation(x=x + 3.1, y=y + 8.5, text=f"区 {index}<br>{value:.0f} N",
                              showarrow=False, font={"size": 11, "color": "#102a38"})
    figure.add_scatter(
        x=[75, 78, 84, 90, 93, 90, 84, 78, 75],
        y=[78, 75, 77, 74, 68, 55, 53, 55, 50],
        mode="lines+markers", name="固定光纤走线",
        line={"color": "#176f7d", "width": 3}, marker={"size": 5, "color": "#176f7d"},
    )
    cop_x = 77.1 + 3.1 * float(np.clip(cop_region, 0.0, 5.0))
    figure.add_scatter(x=[cop_x], y=[45], mode="markers+text", name="压力中心 CoP",
                       text=["CoP"], textposition="bottom center",
                       marker={"size": 14, "color": "#d14646", "symbol": "x"})
    figure.add_annotation(x=84, y=31, text=f"{terrain} · {support}<br>六区载荷 → CoP / 步态教学",
                          showarrow=False, font={"size": 12, "color": "#173445"})

    # Bottom flow makes the required order explicit to a first-time reader.
    flow = ((4, 26, "模块复装", "#315064"), (29, 51, "空载自检\n（必须卸载）", "#6f42c1"),
            (54, 76, "装配候选", status_color), (79, 97, "载荷 / 步态读取", "#17a2b8"))
    for x0, x1, label, color in flow:
        figure.add_shape(type="rect", x0=x0, x1=x1, y0=6, y1=16, fillcolor=color,
                         line={"color": color}, layer="below")
        figure.add_annotation(x=(x0 + x1) / 2, y=11, text=f"<b>{label}</b>", showarrow=False,
                              font={"size": 11, "color": "white"})
    for x in (27.5, 52.5, 77.5):
        figure.add_annotation(x=x, y=11, text="", showarrow=True, ax=-18, ay=0,
                              arrowhead=3, arrowcolor="#5b7280")
    figure.add_annotation(x=50, y=98,
                          text="教学总览：显示当前简化模型的结构关系、筛查逻辑与载荷示意；不等同于实物装配、密封或耐久结论。",
                          showarrow=False, font={"size": 11, "color": "#526c79"})
    figure.update_layout(
        title="从复装到步态：可更换足底组件一图读懂",
        template="plotly_white", height=610, showlegend=False,
        xaxis={"visible": False, "range": [0, 100], "fixedrange": True},
        yaxis={"visible": False, "range": [0, 102], "fixedrange": True},
        margin={"l": 8, "r": 8, "t": 55, "b": 8},
        hovermode=False,
    )
    return figure


def sole_component_explorer_figure(assembly: dict, selected_component: str) -> go.Figure:
    """Render an annotated 2D sectional digital mock-up for first-time readers.

    Each part is drawn as an independent layer.  The selected part receives a
    heavier outline, while force, sealing and optical paths remain visible at
    the same time so their different roles are not conflated.
    """
    case = str(assembly["assembly_case"])
    if selected_component not in _SOLE_COMPONENT_COPY:
        raise ValueError("未知足底点读部件")
    insufficient = case == "压入不足"
    offset = 5 if case == "单侧错位" else 0
    figure = go.Figure()
    components = (
        ("可更换耐磨外底", 11 + offset, 49 + offset, 78, 89, "#d8893f"),
        ("分区传力模块", 15 + offset, 45 + offset, 65, 75, "#eab76f"),
        ("柔性隔离膜", 13 + offset, 47 + offset, 59, 63, "#efd4b0"),
        ("周向密封圈", 9, 51, 54, 58, "#45a58c"),
        ("定位柱与锁止件", 18, 23, 34, 54, "#6a8798"),
        ("固定光纤感知芯", 14, 46, 42, 51, "#45aeca"),
        ("基板与限位台阶", 10, 50, 27, 40, "#7895a3"),
    )
    for name, x0, x1, y0, y1, color in components:
        selected = name == selected_component
        figure.add_shape(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=color,
            line={"color": "#d14646" if selected else "#29495a", "width": 3 if selected else 1.3},
        )
        figure.add_annotation(x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=name,
                              showarrow=False, font={"size": 11, "color": "#132a38"})

    # A small bottom view lets readers see the six independently replaceable force regions.
    for index in range(6):
        column, row = index % 3, index // 3
        x0, y0 = 66 + column * 10, 57 - row * 18
        figure.add_shape(type="rect", x0=x0, x1=x0 + 8, y0=y0, y1=y0 + 14,
                         fillcolor="#eab76f", line={"color": "#29495a", "width": 1.2})
        figure.add_annotation(x=x0 + 4, y=y0 + 7, text=f"分区 {index + 1}", showarrow=False,
                              font={"size": 10, "color": "#132a38"})
    figure.add_shape(type="rect", x0=63, x1=98, y0=18, y1=77, line={"color": "#45a58c", "width": 3, "dash": "dash"})
    figure.add_annotation(x=80.5, y=13, text="底视：六区传力模块与周向密封边界", showarrow=False,
                          font={"size": 11, "color": "#29495a"})

    # These three traces are deliberately visually distinct, not alternative measurements.
    figure.add_scatter(x=[30 + offset, 30 + offset, 30], y=[90, 62, 42], mode="lines+markers",
                       name="受力路径", line={"color": "#cf6c2b", "width": 5},
                       marker={"size": 7, "symbol": "triangle-down"}, hoverinfo="skip")
    figure.add_scatter(x=[10, 50, 50, 10, 10], y=[56, 56, 29, 29, 56], mode="lines",
                       name="密封路径", line={"color": "#16866f", "width": 3, "dash": "dash"}, hoverinfo="skip")
    figure.add_scatter(x=[16, 24, 32, 40, 46], y=[47, 45, 48, 45, 47], mode="lines+markers",
                       name="光纤信号路径", line={"color": "#176f9a", "width": 4},
                       marker={"size": 6}, hoverinfo="skip")

    if insufficient:
        figure.add_shape(type="rect", x0=11, x1=49, y0=75, y1=78, fillcolor="rgba(209,70,70,.18)",
                         line={"color": "#d14646", "width": 1.5, "dash": "dot"})
        figure.add_annotation(x=52, y=76.5, text="未到位间隙", showarrow=True, ax=55, ay=0,
                              arrowhead=2, font={"color": "#b83a3a", "size": 12})
    elif offset:
        figure.add_annotation(x=52, y=75, text="上部模块横向偏移", showarrow=True, ax=55, ay=-25,
                              arrowhead=2, font={"color": "#b83a3a", "size": 12})
    else:
        figure.add_annotation(x=53, y=66, text="定位与轴向限位到位（模型设定）", showarrow=True, ax=50, ay=-5,
                              arrowhead=2, font={"color": "#176f7d", "size": 12})

    title = _SOLE_COMPONENT_COPY[selected_component]["title"]
    body = _SOLE_COMPONENT_COPY[selected_component]["body"]
    figure.add_annotation(x=30, y=8, text=f"<b>点读：{title}</b><br>{body}", showarrow=False,
                          align="left", font={"size": 12, "color": "#173445"})
    figure.add_annotation(x=50, y=98, text=f"<b>装配状态：{case}</b>　红框为当前点读部件；橙=受力，绿虚线=密封，蓝=光纤信号。",
                          showarrow=False, font={"size": 13, "color": "#173445"})
    figure.update_layout(
        title="足底组件数字样机：结构、受力、密封与信号分层可读",
        template="plotly_white", height=570, hovermode=False,
        xaxis={"visible": False, "range": [0, 102], "fixedrange": True},
        yaxis={"visible": False, "range": [0, 102], "fixedrange": True},
        legend={"orientation": "h", "x": .25, "y": .2, "font": {"size": 12}},
        margin={"l": 8, "r": 8, "t": 55, "b": 8},
    )
    return figure


_SOLE_COMPONENT_COPY = {
    "可更换耐磨外底": {"title": "可更换耐磨外底", "body": "接触地面的磨耗件；更换时不移动固定感知芯。"},
    "分区传力模块": {"title": "分区传力模块", "body": "把接触载荷分配至不同区域，供六区载荷教学模型读取。"},
    "柔性隔离膜": {"title": "柔性隔离膜", "body": "使主要受力传递与周向密封功能分离；其实际材料行为仍需实物标定。"},
    "周向密封圈": {"title": "周向密封圈", "body": "围绕接口形成密封路径；本界面不表示 IP 等级或实物密封结论。"},
    "定位柱与锁止件": {"title": "定位柱与锁止件", "body": "控制周向位置并协助锁定；与轴向限位共同约束复装深度。"},
    "固定光纤感知芯": {"title": "固定光纤感知芯", "body": "容纳工作/参考 FBG 与布线通道，作为复装后空载基线比较的固定对象。"},
    "基板与限位台阶": {"title": "基板与限位台阶", "body": "提供安装基准与轴向止挡，避免仅凭外观判断是否压入到位。"},
}


def assembly_tolerance_confusion_figure(result: dict) -> go.Figure:
    """Render a count matrix for the assumed tolerance-screening simulation."""
    labels = list(result["labels"])
    figure = go.Figure(
        go.Heatmap(
            x=["预测正常", "预测压入不足", "预测单侧错位"],
            y=[f"设定{label}" for label in labels],
            z=result["confusion_matrix"],
            colorscale="Blues",
            texttemplate="%{z}",
            colorbar={"title": "样本数"},
        )
    )
    figure.update_layout(**_base_layout("装配公差扫描：设定工况与预测标签", "模型预测", "设定工况"))
    return figure


def seal_compression_screen_figure(result: dict) -> go.Figure:
    """Render relative seal compression around the assumed circumference."""
    figure = go.Figure(
        go.Scatter(
            x=result["angle_deg"],
            y=np.asarray(result["compression_ratio"]) * 100.0,
            mode="lines",
            line={"color": COLORS["estimate"], "width": 3},
            name="相对压缩率",
        )
    )
    figure.add_hline(y=0.0, line_dash="dot", line_color="#6c757d")
    figure.update_layout(**_base_layout("周向密封压缩敏感性（非密封等级结论）", "周向角度 (°)", "相对压缩率 (%)"))
    return figure


def preload_retention_sensitivity_figure(result: dict) -> go.Figure:
    """Render an assumed preload-retention sensitivity curve for test planning."""
    figure = go.Figure(
        go.Scatter(
            x=result["cycle_count"],
            y=result["preload_ue"],
            mode="lines",
            line={"color": COLORS["sensor"], "width": 3},
            name="假设预应变",
        )
    )
    figure.update_layout(**_base_layout("循环后预应变保持敏感性（非寿命预测）", "循环次数", "预应变 (με)"))
    return figure


def demodulation_figure(result: dict) -> go.Figure:
    """Render raw, filtered and temperature-compensated wavelength signals on time."""
    time_s = np.asarray(result["time_s"], dtype=float)
    figure = go.Figure()
    for name, key, color in (
        ("原始波长漂移", "raw_wavelength_nm", "#8ca0b3"),
        ("滤波后波长漂移", "filtered_wavelength_nm", "#ff9f43"),
        ("温补后机械信号", "compensated_wavelength_nm", "#17a2b8"),
    ):
        figure.add_scatter(x=time_s, y=result[key], mode="lines", name=name, line={"color": color, "width": 2})
    figure.update_layout(**_base_layout("FBG 解调与温度补偿数据流", "时间 (s)", "波长变化 (nm)"))
    return figure


def distributed_curve_figure(result: dict, mechanism: str) -> go.Figure:
    """Render a spatial distributed-sensing curve using the primary available quantity."""
    x = np.asarray(result["position_mm"], dtype=float)
    if mechanism == "Brillouin":
        y, label = result["brillouin_frequency_ghz"], "Brillouin 频移 (GHz)"
    elif mechanism == "Raman":
        y, label = result["temperature_c"], "温度 (°C)"
    else:
        y, label = result["strain_ue"], "应变 (με)"
    figure = go.Figure(go.Scatter(x=x, y=y, mode="lines", line={"color": COLORS["truth"], "width": 3}, name=mechanism))
    figure.update_layout(**_base_layout(f"{mechanism} 分布式测量", "光纤位置 (mm)", label))
    return figure


def das_event_figure(result: dict) -> go.Figure:
    """Render a phi-OTDR/DAS time-distance event map."""
    x = result.get("position_mm", result.get("das_distance_mm"))
    y = result.get("time_s", result.get("das_time_ms"))
    z = result.get("amplitude", result.get("das_amplitude"))
    time_label = "时间 (ms)" if "das_time_ms" in result else "时间 (s)"
    figure = go.Figure(go.Heatmap(x=x, y=y, z=z, colorscale="Magma", colorbar={"title": "振动幅值"}))
    figure.update_layout(title="φ-OTDR / DAS：时间—距离振动事件", template="plotly_dark", height=390, xaxis_title="光纤位置 (mm)", yaxis_title=time_label)
    return figure


def distributed_finger_figure(result: dict, contact_fingers: list[int] | None = None) -> go.Figure:
    """Plot five-finger Rayleigh strain profiles, highlighting contacting fingers."""
    distance = np.asarray(result["distance_mm"], dtype=float)
    rayleigh = np.asarray(result["rayleigh_strain_ue"], dtype=float)
    contacts = set(int(index) for index in (contact_fingers or []))
    palette = ["#29c4d7", "#ffcc66", "#f0a58c", "#9be39b", "#c39bf3"]
    figure = go.Figure()
    for index in range(rayleigh.shape[0]):
        active = index in contacts
        figure.add_scatter(
            x=distance,
            y=rayleigh[index],
            mode="lines",
            name=f"手指 {index + 1}{'（接触）' if active else ''}",
            line={"width": 4 if active else 2, "color": "#ff8a4c" if active else palette[index]},
        )
    figure.update_layout(**_base_layout("五指分布式 Rayleigh 应变（接触段出现峰）", "光纤距离 (mm)", "应变 (με)"))
    return figure


def arm_distributed_vs_fbg_figure(
    distributed_result: dict,
    fbg_positions_mm: np.ndarray,
    fbg_strain_ue: np.ndarray,
    suspected_mm: float,
    uncertainty_mm: float,
) -> go.Figure:
    """Overlay a continuous Rayleigh strain profile on the discrete FBG array."""
    position = np.asarray(distributed_result["position_mm"], dtype=float)
    strain = np.asarray(distributed_result["strain_ue"], dtype=float)
    figure = go.Figure()
    figure.add_scatter(x=position, y=strain, mode="lines", name="分布式 Rayleigh 应变", line={"color": "#29c4d7", "width": 3})
    figure.add_scatter(
        x=np.asarray(fbg_positions_mm, dtype=float),
        y=np.asarray(fbg_strain_ue, dtype=float),
        mode="markers",
        name="点式 FBG",
        marker={"size": 12, "color": "#ff8a4c", "line": {"color": "#ffffff", "width": 1}},
    )
    figure.add_vrect(
        x0=float(suspected_mm) - float(uncertainty_mm),
        x1=float(suspected_mm) + float(uncertainty_mm),
        fillcolor="rgba(255,77,79,.08)",
        line_width=0,
    )
    figure.add_vline(x=float(suspected_mm), line_dash="dash", line_color="#ff4d4f")
    figure.update_layout(**_base_layout("分布式 vs 点式 FBG：同一损伤的定位对比", "构件长度 (mm)", "应变 (με)"))
    return figure


def brillouin_raman_compensation_figure(result: dict) -> go.Figure:
    """Show true/apparent/compensated strain with the Raman temperature profile."""
    position = np.asarray(result["position_mm"], dtype=float)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_scatter(x=position, y=result["true_strain_ue"], mode="lines", name="真实应变", line={"color": COLORS["truth"], "width": 3}, secondary_y=False)
    figure.add_scatter(x=position, y=result["naive_strain_ue"], mode="lines", name="未温补（表观应变）", line={"color": "#ff8a4c", "width": 2, "dash": "dash"}, secondary_y=False)
    figure.add_scatter(x=position, y=result["compensated_strain_ue"], mode="lines", name="温补后应变", line={"color": "#29c4d7", "width": 3}, secondary_y=False)
    figure.add_scatter(x=position, y=result["temperature_change_c"], mode="lines", name="温度变化", line={"color": "#c39bf3", "width": 2}, secondary_y=True)
    figure.update_layout(**_base_layout("Brillouin × Raman 温补解耦", "光纤位置 (mm)", "应变 (με)"))
    figure.update_yaxes(title_text="温度变化 (°C)", secondary_y=True)
    return figure


def polarization_figure(result: dict) -> go.Figure:
    """Render a normalised Stokes vector on a compact Poincare-sphere view."""
    stokes = np.asarray(result["stokes"], dtype=float)
    figure = go.Figure(go.Scatter3d(x=[0.0, stokes[0]], y=[0.0, stokes[1]], z=[0.0, stokes[2]], mode="lines+markers", line={"color": "#17a2b8", "width": 7}, marker={"size": 5}, name="Stokes 向量"))
    figure.update_layout(title="偏振态：Stokes 向量 / 庞加莱球坐标", template="plotly_white", height=420, scene={"xaxis_title": "S1", "yaxis_title": "S2", "zaxis_title": "S3", "xaxis": {"range": [-1, 1]}, "yaxis": {"range": [-1, 1]}, "zaxis": {"range": [-1, 1]}, "aspectmode": "cube"})
    return figure


def efpi_figure(result: dict) -> go.Figure:
    """Render the simulated EFPI interference spectrum."""
    figure = go.Figure(go.Scatter(x=result["wavelength_nm"], y=result["intensity"], mode="lines", line={"color": "#6f42c1", "width": 2}, name="干涉强度"))
    figure.update_layout(**_base_layout("EFPI：腔长变化引起的干涉谱", "波长 (nm)", "归一化反射强度"))
    return figure


def polarization_map_figure(result: dict) -> go.Figure:
    """Show azimuth and ellipticity over a stress-twist grid."""
    figure = make_subplots(
        rows=1, cols=2,
        subplot_titles=("偏振方位角 (°)", "椭圆率角 (°)"),
        horizontal_spacing=0.12,
    )
    figure.add_heatmap(x=result["twist_deg"], y=result["stress_mpa"], z=result["azimuth_deg"], colorscale="Viridis", row=1, col=1)
    figure.add_heatmap(x=result["twist_deg"], y=result["stress_mpa"], z=result["ellipticity_deg"], colorscale="Magma", row=1, col=2)
    figure.update_layout(title="双折射—扭转—温度联合视图（网格扫描）", template="plotly_white", height=430, xaxis_title="光纤扭转 (°)", yaxis_title="横向应力 (MPa)")
    figure.update_xaxes(title_text="光纤扭转 (°)", row=1, col=2)
    return figure


def rayleigh_heatmap_figure(result: dict) -> go.Figure:
    """Plot continuous Rayleigh/OFDR strain over the five finger branches."""
    figure = go.Figure(go.Heatmap(
        x=result["distance_mm"],
        y=["拇指", "食指", "中指", "无名指", "小指"],
        z=result["rayleigh_strain_ue"],
        colorscale="Viridis",
        colorbar={"title": "应变 (με)"},
        hovertemplate="%{y}<br>位置 %{x:.1f} mm<br>应变 %{z:.1f} με<extra>Rayleigh/OFDR</extra>",
    ))
    figure.update_layout(title="Rayleigh/OFDR：五指连续应变分布（教学仿真）", template="plotly_dark", height=360, xaxis_title="每根手指上的光纤位置 (mm)", yaxis_title="光纤支路", margin={"l": 20, "r": 20, "t": 55, "b": 45})
    return figure


def das_heatmap_figure(result: dict) -> go.Figure:
    """Plot a DAS teaching waterfall for transient touch and slip events."""
    figure = go.Figure(go.Heatmap(
        x=result["das_distance_mm"],
        y=result["das_time_ms"],
        z=result["das_amplitude"],
        colorscale="Magma",
        colorbar={"title": "相对振动"},
        hovertemplate="时间 %{y:.0f} ms<br>光纤位置 %{x:.1f} mm<br>相对振动 %{z:.3f}<extra>DAS/φ-OTDR</extra>",
    ))
    figure.update_layout(title="DAS/φ-OTDR：抓取接触瞬态（教学仿真）", template="plotly_dark", height=360, xaxis_title="五条光纤支路串联位置 (mm)", yaxis_title="相对事件时间 (ms)", margin={"l": 20, "r": 20, "t": 55, "b": 45})
    return figure


def contact_figure(result: dict, estimated_position_mm: float) -> go.Figure:
    positions = np.asarray(result["sensor_positions_mm"])
    figure = go.Figure()
    figure.add_scatter(x=positions, y=result["strain"] * 1e6, mode="lines+markers", name="各 FBG 应变", line={"color": COLORS["sensor"]})
    figure.add_vline(x=result["contact_position_mm"], line_dash="dash", line_color=COLORS["truth"], annotation_text="真实接触点")
    figure.add_vline(x=estimated_position_mm, line_dash="dot", line_color=COLORS["estimate"], annotation_text="反演接触点")
    figure.update_layout(**_base_layout("局部接触引起的应变分布", "指尖位置 (mm)", "应变 (με)"))
    return figure


def multicore_figure(result: dict) -> go.Figure:
    truth = np.asarray(result["centerline_xyz_mm"])
    estimate = np.asarray(result["estimated_centerline_xyz_mm"])
    figure = go.Figure()
    figure.add_scatter3d(x=truth[:, 0], y=truth[:, 1], z=truth[:, 2], mode="lines", name="真实形状", line={"color": COLORS["truth"], "width": 7})
    figure.add_scatter3d(x=estimate[:, 0], y=estimate[:, 1], z=estimate[:, 2], mode="lines", name="反演形状", line={"color": COLORS["estimate"], "width": 4, "dash": "dash"})
    figure.update_layout(title="三芯光纤 3D 中心线", template="plotly_white", height=480, scene={"xaxis_title": "X (mm)", "yaxis_title": "Y (mm)", "zaxis_title": "Z (mm)", "aspectmode": "data"})
    return figure


def shape_distributed_link_figure(result: dict) -> go.Figure:
    """Show per-core strain along the fibre next to a distributed Rayleigh peak."""
    position = np.asarray(result["position_mm"], dtype=float)
    core_profiles = np.asarray(result["core_strain_ue"], dtype=float)
    palette = ["#29c4d7", "#ffcc66", "#c39bf3"]
    figure = go.Figure()
    for index in range(core_profiles.shape[0]):
        figure.add_scatter(
            x=position,
            y=core_profiles[index],
            mode="lines",
            name=f"纤芯 {index + 1} 应变",
            line={"width": 2.5, "color": palette[index]},
        )
    figure.add_scatter(
        x=position,
        y=result["rayleigh_strain_ue"],
        mode="lines",
        name="分布式 Rayleigh 局部峰（对比）",
        line={"width": 3, "color": "#ff8a4c", "dash": "dash"},
    )
    figure.update_layout(**_base_layout("应变沿长度分布：形状重建与分布式传感的同一主线", "光纤位置 (mm)", "应变 (με)"))
    return figure


def planar_animation_frame(
    pose: dict,
    can_center: np.ndarray,
    grasped: bool,
    contact_fingers: list[int] | None = None,
) -> dict:
    """Build one animation frame of the shared previous/current transition contract.

    Both the 2D SVG renderer and the 3D renderer animate from a previous frame
    to a current frame; the 2D side serialises a full frame here, while the 3D
    side keeps the same previous/current convention with its capsule geometry.
    """
    contacts = [int(index) for index in (contact_fingers if contact_fingers is not None else (range(5) if grasped else []))]
    return {
        "arm": np.asarray(pose["arm_joints"], dtype=float).tolist(),
        "palm": np.asarray(pose["palm_outline"], dtype=float).tolist(),
        "palmFibre": np.asarray(pose["palm_fiber_route"], dtype=float).tolist(),
        "fingers": [np.asarray(finger, dtype=float).tolist() for finger in pose["fingers"]],
        "fibres": [np.asarray(route, dtype=float).tolist() for route in pose["fiber_routes"]],
        "can": np.asarray(can_center, dtype=float).tolist(),
        "grasped": bool(grasped),
        "contacts": contacts,
    }


def planar_hand_animation_html(
    previous_pose: dict,
    current_pose: dict,
    previous_can_center: np.ndarray,
    current_can_center: np.ndarray,
    previous_grasped: bool,
    current_grasped: bool,
    previous_contact_fingers: list[int] | None = None,
    current_contact_fingers: list[int] | None = None,
    animate: bool = True,
) -> str:
    """Render one continuous SVG transition between two planar grasp states."""
    config = json.dumps({
        "previous": planar_animation_frame(previous_pose, previous_can_center, previous_grasped, previous_contact_fingers),
        "current": planar_animation_frame(current_pose, current_can_center, current_grasped, current_contact_fingers),
        "animate": bool(animate),
    }, ensure_ascii=False)
    html = r'''<div style="height:620px;border-radius:18px;overflow:hidden;background:linear-gradient(135deg,#0b1119,#152938)">
<svg id="planar-grasp" viewBox="0 0 1000 620" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"></svg></div>
<script>
(() => {
 const cfg=__CONFIG__, svg=document.getElementById('planar-grasp'), ns='http://www.w3.org/2000/svg';
 const flat=s=>[...s.arm,...s.palm,...s.palmFibre,...s.fingers.flat(),...s.fibres.flat(),s.can];
 const points=[...flat(cfg.previous),...flat(cfg.current)], xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
 const centerX=(Math.min(...xs)+Math.max(...xs))/2,centerY=(Math.min(...ys)+Math.max(...ys))/2;
 const viewWidth=12,viewHeight=9,scale=Math.min(900/viewWidth,500/viewHeight), ox=500-scale*centerX, oy=325+scale*centerY;
 const xy=p=>`${ox+scale*p[0]},${oy-scale*p[1]}`;
 const mix=(a,b,t)=>Array.isArray(a)?a.map((v,i)=>mix(v,b[i],t)):a+(b-a)*t;
 const el=(tag,attrs)=>{const node=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));svg.appendChild(node);return node};
 const arm=[['#4e7189',20],['#6f98ae',16],['#385a6d',10]].map(([stroke,width])=>el('polyline',{fill:'none',stroke,'stroke-width':width,'stroke-linecap':'round'}));
 const palm=el('polygon',{fill:'rgba(112,146,164,.84)',stroke:'#c2d9e5','stroke-width':3});
 const palmFibre=el('polyline',{fill:'none',stroke:'#29c4d7','stroke-width':5,'stroke-linecap':'round'});
 const fingers=Array.from({length:5},()=>el('polyline',{fill:'none',stroke:'#d8e7ef','stroke-width':8,'stroke-linecap':'round','stroke-linejoin':'round'}));
 const fibres=Array.from({length:5},()=>el('polyline',{fill:'none',stroke:'#29c4d7','stroke-width':5,'stroke-linecap':'round'}));
 const can=el('rect',{fill:'#c33237',stroke:'#e8eff4','stroke-width':3,rx:12});
 const ring=el('line',{stroke:'#f6d365','stroke-width':4,'stroke-linecap':'round'});
 const text=el('text',{x:500,y:46,fill:'#eff8ff','font-size':20,'font-weight':'700','text-anchor':'middle'});
 function draw(t){const s={arm:mix(cfg.previous.arm,cfg.current.arm,t),palm:mix(cfg.previous.palm,cfg.current.palm,t),palmFibre:mix(cfg.previous.palmFibre,cfg.current.palmFibre,t),fingers:mix(cfg.previous.fingers,cfg.current.fingers,t),fibres:mix(cfg.previous.fibres,cfg.current.fibres,t),can:mix(cfg.previous.can,cfg.current.can,t),grasped:t<.5?cfg.previous.grasped:cfg.current.grasped,contacts:t<.5?cfg.previous.contacts:cfg.current.contacts};
  arm.forEach((node,i)=>node.setAttribute('points',[s.arm[i],s.arm[i+1]].map(xy).join(' ')));
  palm.setAttribute('points',s.palm.map(xy).join(' ')); palmFibre.setAttribute('points',s.palmFibre.map(xy).join(' '));
  fingers.forEach((node,i)=>{node.setAttribute('points',s.fingers[i].map(xy).join(' '));node.setAttribute('stroke',s.contacts.includes(i)?'#ffcc66':'#d8e7ef');});
  fibres.forEach((node,i)=>{node.setAttribute('points',s.fibres[i].map(xy).join(' '));node.setAttribute('stroke',s.contacts.includes(i)?'#ffe06a':'#29c4d7');});
  const w=.48*scale,h=.74*scale,x=ox+scale*s.can[0]-w/2,y=oy-scale*s.can[1]-h/2;can.setAttribute('x',x);can.setAttribute('y',y);can.setAttribute('width',w);can.setAttribute('height',h);ring.setAttribute('x1',x+w*.32);ring.setAttribute('x2',x+w*.68);ring.setAttribute('y1',y+12);ring.setAttribute('y2',y+12);text.textContent=s.grasped?'二维 FBG 已抓稳 · 正在搬运':'二维寻找与对准 · 物体保持原位';}
 if(cfg.animate){const started=performance.now(),duration=520; function animate(now){const t=Math.min(1,(now-started)/duration),e=t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;draw(e);if(t<1)requestAnimationFrame(animate)} draw(0);requestAnimationFrame(animate);}else{draw(1);}
})();
</script>'''
    return html.replace("__CONFIG__", config)


def _can_mesh(can_center: np.ndarray, radius: float = .52, height: float = 1.75, sides: int = 24) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Create a simple closed cylinder mesh for an aluminium beverage can."""
    can_center = np.asarray(can_center, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    lower = np.column_stack([can_center[0] + radius * np.cos(theta), .42 + radius * np.sin(theta), np.full(sides, can_center[1] - height / 2)])
    upper = lower.copy()
    upper[:, 2] = can_center[1] + height / 2
    vertices = np.vstack([lower, upper, [[can_center[0], .42, can_center[1] - height / 2], [can_center[0], .42, can_center[1] + height / 2]]])
    lower_center, upper_center = 2 * sides, 2 * sides + 1
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        i.extend([index, index, lower_center, upper_center])
        j.extend([nxt, sides + index, nxt, sides + nxt])
        k.extend([sides + index, sides + nxt, index, sides + index])
    return vertices, i, j, k


def _frustum_mesh(start: np.ndarray, end: np.ndarray, start_radius: float, end_radius: float, sides: int = 12) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Create a tapered engineering shell between two 3D points."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    axis /= np.linalg.norm(axis)
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < .9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    angle = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circle = np.outer(np.cos(angle), basis_u) + np.outer(np.sin(angle), basis_v)
    vertices = np.vstack([start + start_radius * circle, end + end_radius * circle, start, end])
    start_center, end_center = 2 * sides, 2 * sides + 1
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        i.extend([index, index, start_center, end_center])
        j.extend([nxt, sides + index, nxt, sides + nxt])
        k.extend([sides + index, sides + nxt, index, sides + index])
    return vertices, i, j, k


def _finger_tube_mesh(points: np.ndarray, radius: float = .115) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Join all three phalanges into one solid mesh, with a slight overlap at each knuckle."""
    points = np.asarray(points, dtype=float)
    vertices: list[np.ndarray] = []
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    offset = 0
    for start, end in zip(points[:-1], points[1:]):
        segment_vertices, segment_i, segment_j, segment_k = _frustum_mesh(start, end, radius, radius * .88, sides=10)
        vertices.append(segment_vertices)
        i.extend([value + offset for value in segment_i])
        j.extend([value + offset for value in segment_j])
        k.extend([value + offset for value in segment_k])
        offset += len(segment_vertices)
    return np.vstack(vertices), i, j, k


def _arched_palm_mesh(center: np.ndarray, forward_2d: np.ndarray, rings: int = 8, sides: int = 14) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Create a thin rounded palm shell whose width follows the five-finger spread."""
    center_3d = np.array([center[0], 0.0, center[1]])
    forward_3d = np.array([forward_2d[0], 0.0, forward_2d[1]])
    normal_3d = np.array([-forward_2d[1], 0.0, forward_2d[0]])
    # 局部坐标为（掌长、五指横向）。宽度沿 y 轴展开，五指根部才会真正落在掌面内。
    outline = np.asarray([
        (-.52, -.46), (-.45, -.67), (-.23, -.79), (.24, -.79),
        (.48, -.66), (.55, -.45), (.55, .45), (.48, .66),
        (.24, .79), (-.23, .79), (-.45, .67), (-.52, .46),
    ])
    shell_thickness = .13
    front = np.asarray([
        center_3d + along * forward_3d + width * np.array([0.0, 1.0, 0.0]) + shell_thickness * normal_3d
        for along, width in outline
    ])
    back = np.asarray([
        center_3d + along * forward_3d + width * np.array([0.0, 1.0, 0.0]) - shell_thickness * normal_3d
        for along, width in outline
    ])
    vertices = np.vstack([front, back, center_3d + shell_thickness * normal_3d, center_3d - shell_thickness * normal_3d])
    count = len(outline)
    front_center, back_center = 2 * count, 2 * count + 1
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(count):
        nxt = (index + 1) % count
        i.extend([index, index, front_center, back_center])
        j.extend([nxt, count + index, index, count + nxt])
        k.extend([count + index, count + nxt, nxt, count + index])
    return vertices, i, j, k


def arm_figure(action: str, route: str, finger_angle_deg: float, contact_force_n: float, joint_angles_deg: tuple[float, float, float] | None = None, finger_curls_deg: tuple[float, float, float, float, float] | None = None, wrist_rotation_deg: float = 0.0, can_center: np.ndarray | None = None) -> go.Figure:
    """Draw a five-finger teaching hand whose parts follow the selected action."""
    pose = models.dexterous_hand_pose(action, joint_angles_deg, finger_curls_deg, wrist_rotation_deg)
    joints = np.asarray(pose["arm_joints"])
    palm_outline = np.asarray(pose["palm_outline"])
    target = np.asarray(pose["target"] if can_center is None else can_center)
    fingers = pose["fingers"]
    fiber_routes = pose["fiber_routes"]
    grasp = models.evaluate_can_grasp(pose, target)
    figure = go.Figure()
    # 以不同长度和厚度绘制上臂、前臂与腕部，避免一条等粗折线破坏手臂比例。
    for name, start, end, width, color in (
        ("上臂", joints[0], joints[1], 20, "#4e7189"),
        ("前臂", joints[1], joints[2], 16, "#6f98ae"),
        ("腕部", joints[2], joints[3], 10, "#385a6d"),
    ):
        figure.add_scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode="lines+markers", name=name, line={"width": width, "color": color}, marker={"size": max(9, width - 6), "color": "#d8e7ef"}, hovertemplate=f"{name}<extra></extra>")
    figure.add_scatter(x=palm_outline[:, 0], y=palm_outline[:, 1], mode="lines", fill="toself", name="掌壳", line={"width": 3, "color": "#9db7c7"}, fillcolor="rgba(112, 146, 164, .82)")
    figure.add_scatter(x=pose["palm_fiber_route"][:, 0], y=pose["palm_fiber_route"][:, 1], mode="lines+markers", name="掌心 FBG", line={"width": 5, "color": "#29c4d7"}, marker={"size": 6, "color": "#75eef5"}, hovertemplate="掌心 FBG：第六触觉通道<extra></extra>")
    for index, (finger, fiber) in enumerate(zip(fingers, fiber_routes), 1):
        contact = index - 1 in grasp["contact_fingers"]
        figure.add_scatter(x=finger[:, 0], y=finger[:, 1], mode="lines+markers", name=f"手指 {index}", line={"width": 8, "color": "#ffcc66" if contact else "#d8e7ef"}, marker={"size": 7, "color": "#ff8a4c" if contact else "#506d80"}, hovertemplate=f"手指 {index}<extra></extra>")
        fiber_width = 5 if route in {"手指背侧", "指尖"} else 3
        figure.add_scatter(x=fiber[:, 0], y=fiber[:, 1], mode="lines+markers", name=f"FBG 支路 {index}", line={"width": fiber_width, "color": "#ffe06a" if contact else "#29c4d7"}, marker={"size": 5, "color": "#fff2aa" if contact else "#75eef5"}, hovertemplate=f"FBG 支路 {index}<extra></extra>")
    can_radius, can_height = .24, .74
    figure.add_scatter(x=[target[0] - can_radius, target[0] + can_radius, target[0] + can_radius, target[0] - can_radius, target[0] - can_radius], y=[target[1] - can_height / 2, target[1] - can_height / 2, target[1] + can_height / 2, target[1] + can_height / 2, target[1] - can_height / 2], mode="lines", fill="toself", name="铝制饮料罐", line={"color":"#d7e0e8","width":3}, fillcolor="rgba(195, 50, 55, .92)")
    figure.add_scatter(x=[target[0] - .20, target[0] + .20], y=[target[1] + can_height / 2 - .10, target[1] + can_height / 2 - .10], mode="lines", name="拉环", line={"color":"#f6d365","width":4}, hovertemplate="饮料罐拉环<extra></extra>")
    annotation_y = max(joints[:, 1].max(), target[1], palm_outline[:, 1].max()) + .65
    grasp_text = "抓取成功：罐体已绑定" if grasp["is_grasped"] else "尚未抓稳：继续调整五指"
    figure.add_annotation(x=joints[:, 0].mean(), y=annotation_y, text=f"动作：{action}<br>手指弯曲：{finger_angle_deg:.0f}°<br>{grasp_text}", showarrow=False, bgcolor="#172a3a", font={"color": "white"})
    all_points = np.vstack([joints, palm_outline, target, *fingers, *fiber_routes])
    figure.update_layout(title=f"五指灵巧手实体模拟与 FBG 走线：{route}", template="plotly_dark", height=560, showlegend=True, legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "font":{"size":10}}, xaxis={"visible": False, "range": [all_points[:, 0].min() - .8, all_points[:, 0].max() + 1.0]}, yaxis={"visible": False, "range": [all_points[:, 1].min() - .8, annotation_y + .5]}, margin={"l": 20, "r": 20, "t": 72, "b": 18})
    return figure


def arm_3d_figure(action: str, route: str, finger_angle_deg: float, joint_angles_deg: tuple[float, float, float] | None = None, finger_curls_deg: tuple[float, float, float, float, float] | None = None, wrist_rotation_deg: float = 0.0, can_center: np.ndarray | None = None) -> go.Figure:
    """Render a rotatable 3D teaching view of the current arm action."""
    pose = models.dexterous_hand_pose(action, joint_angles_deg, finger_curls_deg, wrist_rotation_deg)
    planar_joints = np.asarray(pose["arm_joints"])
    joints = np.column_stack([planar_joints[:, 0], [0.0, .12, .26, .38], planar_joints[:, 1]])
    fingers = pose["fingers"]
    fiber_routes = pose["fiber_routes"]
    target = np.asarray(pose["target"] if can_center is None else can_center)
    grasp = models.evaluate_can_grasp(pose, target)
    forward_2d = planar_joints[3] - planar_joints[2]
    forward_2d /= np.linalg.norm(forward_2d)
    palm_vertices, palm_i, palm_j, palm_k = _arched_palm_mesh(np.asarray(pose["palm_center"]), forward_2d)
    wrist_rotation = np.deg2rad(wrist_rotation_deg)
    palm_z = float(np.asarray(pose["palm_center"])[1])
    palm_y_local = palm_vertices[:, 1].copy()
    palm_z_local = palm_vertices[:, 2] - palm_z
    palm_vertices[:, 1] = palm_y_local * np.cos(wrist_rotation) - palm_z_local * np.sin(wrist_rotation)
    palm_vertices[:, 2] = palm_z + palm_y_local * np.sin(wrist_rotation) + palm_z_local * np.cos(wrist_rotation)
    figure = go.Figure()
    upper_vertices, upper_i, upper_j, upper_k = _frustum_mesh(joints[0], joints[1], .34, .29)
    forearm_vertices, forearm_i, forearm_j, forearm_k = _frustum_mesh(joints[1], joints[2], .29, .20)
    wrist_vertices, wrist_i, wrist_j, wrist_k = _frustum_mesh(joints[2], joints[3], .20, .15)
    figure.add_mesh3d(x=upper_vertices[:, 0], y=upper_vertices[:, 1], z=upper_vertices[:, 2], i=upper_i, j=upper_j, k=upper_k, name="上臂外壳", color="#54758b", opacity=1.0, flatshading=False)
    figure.add_mesh3d(x=forearm_vertices[:, 0], y=forearm_vertices[:, 1], z=forearm_vertices[:, 2], i=forearm_i, j=forearm_j, k=forearm_k, name="锥形前臂", color="#6f95aa", opacity=1.0, flatshading=False)
    figure.add_mesh3d(x=wrist_vertices[:, 0], y=wrist_vertices[:, 1], z=wrist_vertices[:, 2], i=wrist_i, j=wrist_j, k=wrist_k, name="腕部连接", color="#3f5b6d", opacity=1.0, flatshading=False)
    figure.add_scatter3d(x=[joints[1, 0], joints[2, 0]], y=[joints[1, 1], joints[2, 1]], z=[joints[1, 2], joints[2, 2]], mode="markers", name="腕部关节", marker={"size": [7, 6], "color": "#d9e5ec", "line":{"color":"#314b5d","width":2}})
    figure.add_mesh3d(x=palm_vertices[:, 0], y=palm_vertices[:, 1], z=palm_vertices[:, 2], i=palm_i, j=palm_j, k=palm_k, name="一体化掌壳", color="#7f9faf", opacity=1.0, flatshading=False, hovertemplate="一体化掌壳<extra></extra>")
    finger_depths = (-.28, -.13, .03, .18, .31)
    for index, (finger, fiber, depth) in enumerate(zip(fingers, fiber_routes, finger_depths), 1):
        contact = index - 1 in grasp["contact_fingers"]
        finger_y = np.full(len(finger), depth)
        finger_z_local = finger[:, 1] - palm_z
        finger_y_rotated = finger_y * np.cos(wrist_rotation) - finger_z_local * np.sin(wrist_rotation)
        finger_z_rotated = palm_z + finger_y * np.sin(wrist_rotation) + finger_z_local * np.cos(wrist_rotation)
        finger_points_3d = np.column_stack([finger[:, 0], finger_y_rotated, finger_z_rotated])
        finger_vertices, finger_i, finger_j, finger_k = _finger_tube_mesh(finger_points_3d)
        figure.add_mesh3d(x=finger_vertices[:, 0], y=finger_vertices[:, 1], z=finger_vertices[:, 2], i=finger_i, j=finger_j, k=finger_k, name=f"实体手指 {index}", color="#f2c978" if contact else "#d4e0e6", opacity=1.0, flatshading=False, hovertemplate=f"实体手指 {index}<extra></extra>")
        figure.add_scatter3d(x=finger[:, 0], y=finger_y_rotated, z=finger_z_rotated, mode="lines+markers", name=f"手指 {index}", line={"width": 11, "color": "#ffcc66" if contact else "#d8e7ef"}, marker={"size": 4, "color": "#ff8a4c" if contact else "#506d80"}, hovertemplate=f"手指 {index}<extra></extra>")
        fiber_width = 8 if route in {"手指背侧", "指尖"} else 4
        fiber_y = np.full(len(fiber), depth + .07)
        fiber_z_local = fiber[:, 1] - palm_z
        fiber_y_rotated = fiber_y * np.cos(wrist_rotation) - fiber_z_local * np.sin(wrist_rotation)
        fiber_z_rotated = palm_z + fiber_y * np.sin(wrist_rotation) + fiber_z_local * np.cos(wrist_rotation)
        figure.add_scatter3d(x=fiber[:, 0], y=fiber_y_rotated, z=fiber_z_rotated, mode="lines+markers", name=f"FBG 支路 {index}", line={"width": fiber_width, "color": "#ffe06a" if contact else "#29c4d7"}, marker={"size": 3, "color": "#fff2aa" if contact else "#75eef5"}, hovertemplate=f"FBG 支路 {index}<extra></extra>")
    can_vertices, can_i, can_j, can_k = _can_mesh(target)
    figure.add_mesh3d(x=can_vertices[:, 0], y=can_vertices[:, 1], z=can_vertices[:, 2], i=can_i, j=can_j, k=can_k, name="铝制饮料罐", color="#c83f45", opacity=.96, hovertemplate="铝制饮料罐<extra></extra>")
    figure.add_scatter3d(x=[target[0] - .20, target[0] + .20], y=[.42, .42], z=[target[1] + .80, target[1] + .80], mode="lines", name="拉环", line={"color":"#f6d365","width":7}, hovertemplate="饮料罐拉环<extra></extra>")
    figure.update_layout(title=f"三维五指灵巧手与饮料罐：{action}（{'已抓稳' if grasp['is_grasped'] else '接触调整中'}）", template="plotly_dark", height=600, scene={"xaxis":{"visible":False},"yaxis":{"visible":False},"zaxis":{"visible":False},"aspectmode":"manual","aspectratio":{"x":1.35,"y":.70,"z":1.0},"camera":{"eye":{"x":1.75,"y":-2.2,"z":1.25}}}, margin={"l":0,"r":0,"t":50,"b":0}, legend={"font":{"size":10}})
    return figure


def anthropomorphic_hand_html(action: str, joint_angles_deg: tuple[float, float, float], finger_curls_deg: tuple[float, float, float, float, float], grasped: bool, can_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), previous_can_offset: tuple[float, float, float] | None = None, shoulder_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), previous_shoulder_offset: tuple[float, float, float] | None = None, planar_pose: dict | None = None, can_center: np.ndarray | None = None, previous_can_center: np.ndarray | None = None, can_depth: float = 0.0, previous_can_depth: float | None = None, finger_curl_gain: float = 1.0, finger_joint_angles_deg: tuple[tuple[float, ...], ...] | None = None, previous_joint_angles_deg: tuple[float, float, float] | None = None, previous_finger_joint_angles_deg: tuple[tuple[float, ...], ...] | None = None, animate: bool = True) -> str:
    """Return a self-contained Three.js teaching renderer for a rounded five-finger robot hand."""
    def three_axis_offset(offset: tuple[float, ...]) -> tuple[float, float, float]:
        values = tuple(float(value) for value in offset)
        return (values + (0.0, 0.0, 0.0))[:3]

    can_offset = three_axis_offset(can_offset)
    previous_can_offset = can_offset if previous_can_offset is None else three_axis_offset(previous_can_offset)
    previous_shoulder_offset = shoulder_offset if previous_shoulder_offset is None else previous_shoulder_offset
    planar_pose = planar_pose or models.dexterous_hand_pose(action, joint_angles_deg, finger_curls_deg)
    shared_geometry = {
        "armJoints": np.asarray(planar_pose["arm_joints"], dtype=float).tolist(),
        "palmOutline": np.asarray(planar_pose["palm_outline"], dtype=float).tolist(),
        "fingers": [np.asarray(finger, dtype=float).tolist() for finger in planar_pose["fingers"]],
        "fiberRoutes": [np.asarray(route, dtype=float).tolist() for route in planar_pose["fiber_routes"]],
    }
    can_center = np.asarray(planar_pose["target"] if can_center is None else can_center, dtype=float)
    previous_can_center = can_center if previous_can_center is None else np.asarray(previous_can_center, dtype=float)
    previous_can_depth = can_depth if previous_can_depth is None else previous_can_depth
    if finger_joint_angles_deg is None:
        curls = np.asarray(finger_curls_deg, dtype=float) * float(finger_curl_gain)
        finger_joint_angles_deg = (
            (0.0, curls[0] * .47),
            *((0.0, curl * .47, curl * .35) for curl in curls[1:]),
        )
    joint_counts = (2, 3, 3, 3, 3)
    if len(finger_joint_angles_deg) != 5 or any(len(angles) != count for angles, count in zip(finger_joint_angles_deg, joint_counts)):
        raise ValueError("finger_joint_angles_deg 必须包含拇指 2 个和其余四指各 3 个关节角")
    previous_joint_angles_deg = joint_angles_deg if previous_joint_angles_deg is None else previous_joint_angles_deg
    previous_finger_joint_angles_deg = finger_joint_angles_deg if previous_finger_joint_angles_deg is None else previous_finger_joint_angles_deg
    if len(previous_joint_angles_deg) != 3 or len(previous_finger_joint_angles_deg) != 5 or any(len(angles) != count for angles, count in zip(previous_finger_joint_angles_deg, joint_counts)):
        raise ValueError("上一帧必须包含三个手臂关节和 14 个手指关节")
    finger_joints = [list(map(float, angles)) for angles in finger_joint_angles_deg]

    def serialise_capsules(spec: list[tuple[np.ndarray, np.ndarray, float]]) -> list[list[object]]:
        return [
            [start.tolist(), end.tolist(), float(radius)]
            for start, end, radius in spec
        ]

    capsules = models.three_d_finger_capsules(finger_joint_angles_deg)
    previous_capsules = models.three_d_finger_capsules(previous_finger_joint_angles_deg)
    config = json.dumps({
        "action": action,
        "joints": list(joint_angles_deg),
        "previousJoints": list(previous_joint_angles_deg),
        "curls": list(finger_curls_deg),
        "curlGain": float(finger_curl_gain),
        "fingerJoints": finger_joints,
        "previousFingerJoints": [list(map(float, angles)) for angles in previous_finger_joint_angles_deg],
        "fingerCapsules": [serialise_capsules(finger) for finger in capsules],
        "previousFingerCapsules": [serialise_capsules(finger) for finger in previous_capsules],
        "grasped": grasped,
        "canOffset": list(can_offset),
        "previousCanOffset": list(previous_can_offset),
        "shoulderOffset": list(shoulder_offset),
        "previousShoulderOffset": list(previous_shoulder_offset),
        "sharedGeometry": shared_geometry,
        "canPosition": can_center.tolist(),
        "previousCanPosition": previous_can_center.tolist(),
        "canDepth": float(can_depth),
        "previousCanDepth": float(previous_can_depth),
        "animate": bool(animate),
    }, ensure_ascii=False)
    three_runtime = (Path(__file__).with_name("vendor") / "three.min.js").read_text(encoding="utf-8")
    html = r'''<div id="bio-hand" style="height:640px;border-radius:18px;overflow:hidden;background:linear-gradient(135deg,#0b1119,#152938);position:relative"></div>
<script>__THREE_RUNTIME__</script>
<script>
(() => {
 const cfg=__CONFIG__, host=document.getElementById('bio-hand');
 const viewW=Math.max(host.clientWidth,300),viewH=Math.max(host.clientHeight,300);
 const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(36,viewW/viewH,.1,100);
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true}); renderer.setSize(viewW,viewH); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); host.appendChild(renderer.domElement);
 scene.add(new THREE.HemisphereLight(0xe8f5ff,0x15222e,2.3)); const key=new THREE.DirectionalLight(0xffffff,2.4); key.position.set(4,-3,7); scene.add(key);
 const robot=new THREE.Group(); scene.add(robot); scene.add(new THREE.AxesHelper(2.0)); const grid=new THREE.GridHelper(16,16,0x315366,0x193140); scene.add(grid); const hand=new THREE.Group();
 const shoulderStart=new THREE.Vector3(-6.7+cfg.previousShoulderOffset[0],-1.8+cfg.previousShoulderOffset[1],cfg.previousShoulderOffset[2]),shoulderTarget=new THREE.Vector3(-6.7+cfg.shoulderOffset[0],-1.8+cfg.shoulderOffset[1],cfg.shoulderOffset[2]);
 const metal=new THREE.MeshStandardMaterial({color:0x6d9bb5,metalness:.75,roughness:.28}); const joint=new THREE.MeshStandardMaterial({color:0x2e5268,metalness:.7,roughness:.25}); const skin=new THREE.MeshStandardMaterial({color:0xa5c8d9,metalness:.35,roughness:.48}); const cyan=new THREE.MeshStandardMaterial({color:0x23d5e8,emissive:0x087482,emissiveIntensity:.65});
 // 掌心轮廓：圆润的虎口、收窄腕部和拱起的掌背，不使用球体或平板。
 const outline=new THREE.Shape(); outline.moveTo(-1.12,-.43); outline.quadraticCurveTo(-.94,-.93,-.46,-.98); outline.lineTo(.58,-.93); outline.quadraticCurveTo(.96,-.82,1.04,-.48); outline.lineTo(1.04,.52); outline.quadraticCurveTo(.90,.91,.52,1.00); outline.lineTo(-.42,.96); outline.quadraticCurveTo(-.94,.87,-1.12,.43); outline.closePath();
 const palm=new THREE.Mesh(new THREE.ExtrudeGeometry(outline,{depth:.48,bevelEnabled:true,bevelThickness:.15,bevelSize:.14,bevelSegments:4,curveSegments:16}),metal); palm.position.set(0,0,-.24); hand.add(palm);
 const palmFiberA=new THREE.Mesh(new THREE.CapsuleGeometry(.026,1.76,5,8),cyan),palmFiberB=new THREE.Mesh(new THREE.CapsuleGeometry(.026,1.48,5,8),cyan); palmFiberA.rotation.z=-Math.PI/2; palmFiberA.position.set(.02,-.36,.28); palmFiberB.rotation.z=-Math.PI/2; palmFiberB.position.set(.02,.38,.28); hand.add(palmFiberA,palmFiberB);
 const cuff=new THREE.Mesh(new THREE.CapsuleGeometry(.38,.50,8,16),joint); cuff.rotation.y=Math.PI/2; cuff.position.set(-1.35,0,0); hand.add(cuff);
 const hub=new THREE.Mesh(new THREE.TorusGeometry(.34,.10,10,22),joint); hub.rotation.y=Math.PI/2; hub.position.set(-1.03,0,0); hand.add(hub);
 // 三段机器人手臂与二维图共用同一运动学：每一段按肩、肘、腕角依次累积旋转。
 const armPivots=[];
 function makeArm(){ const a=cfg.joints.map(v=>v*Math.PI/180), shoulder=new THREE.Group(); shoulder.position.copy(shoulderStart); robot.add(shoulder); let pivot=shoulder; const links=[[6.3,.43,0x4f7690],[5.4,.37,0x638ba2],[2.25,.30,0x36586c]]; links.forEach((item,n)=>{ armPivots.push(pivot); pivot.rotation.z=a[n]; const link=new THREE.Mesh(new THREE.CapsuleGeometry(item[1],item[0]-item[1]*2,8,16),new THREE.MeshStandardMaterial({color:item[2],metalness:.78,roughness:.26})); link.rotation.z=-Math.PI/2; link.position.x=item[0]/2; pivot.add(link); const armFiber=new THREE.Mesh(new THREE.CapsuleGeometry(.026,Math.max(.1,item[0]-.12),5,8),cyan); armFiber.rotation.z=-Math.PI/2; armFiber.position.set(item[0]/2,0,item[1]*.94); pivot.add(armFiber); const next=new THREE.Group(); next.position.x=item[0]; pivot.add(next); if(n<2){ const jointDisc=new THREE.Mesh(new THREE.TorusGeometry(item[1]*1.05,.075,8,18),joint); jointDisc.rotation.y=Math.PI/2; jointDisc.position.x=item[0]; pivot.add(jointDisc); } pivot=next; }); pivot.add(hand); return shoulder; } const shoulder=makeArm(); armPivots.forEach((pivot,index)=>pivot.rotation.z=cfg.previousJoints[index]*Math.PI/180);
 // 手指几何由 Python 端按同一套运动学计算并序列化，JS 只负责渲染与插值。
 const fingerMeshes=[];
 function makeFingerFromData(caps){const knuckle=new THREE.Mesh(new THREE.SphereGeometry(caps[0][2]*1.18,12,12),joint);knuckle.position.set(...caps[0][0]);hand.add(knuckle);const bones=[],fibres=[],rings=[];caps.forEach(cap=>{const r=cap[2],p1=new THREE.Vector3(...cap[0]),p2=new THREE.Vector3(...cap[1]);const len=p1.distanceTo(p2),mid=p1.clone().add(p2).multiplyScalar(.5),dir=p2.clone().sub(p1).normalize();const bone=new THREE.Mesh(new THREE.CapsuleGeometry(r,Math.max(.12,len-r*2),8,14),skin);bone.position.copy(mid);bone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);hand.add(bone);bones.push(bone);const segmentFiber=new THREE.Mesh(new THREE.CapsuleGeometry(.025,Math.max(.1,len-.08),5,8),cyan);segmentFiber.position.copy(mid);segmentFiber.position.z+=r*.94;segmentFiber.quaternion.copy(bone.quaternion);hand.add(segmentFiber);fibres.push(segmentFiber);const ring=new THREE.Mesh(new THREE.TorusGeometry(r*1.04,.035,8,16),joint);ring.quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),dir);ring.position.copy(p2);hand.add(ring);rings.push(ring);});fingerMeshes.push({bones,fibres,rings,knuckle});}
 function applyFingerData(finger,prevCaps,currCaps,t){finger.bones.forEach((bone,si)=>{const p1=new THREE.Vector3().lerpVectors(new THREE.Vector3(...prevCaps[si][0]),new THREE.Vector3(...currCaps[si][0]),t);const p2=new THREE.Vector3().lerpVectors(new THREE.Vector3(...prevCaps[si][1]),new THREE.Vector3(...currCaps[si][1]),t);const len=p1.distanceTo(p2),mid=p1.clone().add(p2).multiplyScalar(.5),dir=p2.clone().sub(p1).normalize();bone.position.copy(mid);bone.quaternion.setFromUnitVectors(new THREE.Vector3(0,1,0),dir);finger.fibres[si].position.copy(mid);finger.fibres[si].position.z+=prevCaps[si][2]*.94;finger.fibres[si].quaternion.copy(bone.quaternion);finger.rings[si].position.copy(p2);finger.rings[si].quaternion.setFromUnitVectors(new THREE.Vector3(0,0,1),dir);});finger.knuckle.position.copy(new THREE.Vector3().lerpVectors(new THREE.Vector3(...prevCaps[0][0]),new THREE.Vector3(...currCaps[0][0]),t));}
 cfg.fingerCapsules.forEach(caps=>makeFingerFromData(caps)); fingerMeshes.forEach((finger,index)=>applyFingerData(finger,cfg.previousFingerCapsules[index],cfg.fingerCapsules[index],0));
 // 把机器人整体平移，使默认抓取位对准坐标系原点：罐子放在滑杆对应的 (X,Y,Z) 上，
 // 手部通过 reach 从原点出发去找它，抓取前罐子保持世界位置不动。
 const graspFrame=new THREE.Group(); graspFrame.position.set(.30,-.20,.76); hand.add(graspFrame); const originalShoulder=shoulder.position.clone(); shoulder.position.set(-6.7,-1.8,0); scene.updateMatrixWorld(true); const canStart=new THREE.Vector3(); graspFrame.getWorldPosition(canStart); shoulder.position.copy(originalShoulder); robot.position.sub(canStart); scene.updateMatrixWorld(true); const can=new THREE.Mesh(new THREE.CylinderGeometry(.48,.48,1.72,32),new THREE.MeshStandardMaterial({color:0xc94d4f,metalness:.65,roughness:.25})); can.visible=true; if(cfg.grasped){can.position.set(.30,-.20,.76);hand.add(can);}else{can.position.set(cfg.canOffset[0]+canStart.x,cfg.canOffset[1]+canStart.y,cfg.canOffset[2]+canStart.z);robot.add(can);}
 // 相机按机器人包围盒自动取景：距离随整体尺寸缩放，交互只保留拖动旋转。
 camera.position.set(16,-18,14); const frameBox=new THREE.Box3().setFromObject(robot),viewCenter=frameBox.getCenter(new THREE.Vector3()),viewRadius=Math.max(frameBox.getSize(new THREE.Vector3()).length()/2,1e-3),viewDistance=viewRadius/Math.sin(THREE.MathUtils.degToRad(camera.fov/2))*1.2; camera.position.copy(new THREE.Vector3(16,-18,14).normalize().multiplyScalar(viewDistance).add(viewCenter)); camera.lookAt(viewCenter);
 // 相机固定取景整个工作空间（覆盖物块 ±3 与手部活动范围），镜头不跟随手部。
 function frameCamera(){const b=new THREE.Box3(new THREE.Vector3(-16,-5,-4),new THREE.Vector3(4,3,4)),c=b.getCenter(new THREE.Vector3()),r=Math.max(b.getSize(new THREE.Vector3()).length()/2,1e-3),d=r/Math.sin(THREE.MathUtils.degToRad(camera.fov/2))*1.2;camera.position.copy(new THREE.Vector3(16,-18,14).normalize().multiplyScalar(d).add(c));camera.lookAt(c);}
 const label=document.createElement('div'); label.textContent='当前动作：'+cfg.action+' · 仿生五指机器人手 · 拖动旋转'; label.style.cssText='position:absolute;left:18px;bottom:14px;color:#d8f2ff;font:600 14px sans-serif;background:#132a3baa;padding:8px 11px;border-radius:9px'; host.appendChild(label);
 // Orbit-like interaction without external controls.
 let drag=false,last; host.addEventListener('pointerdown',e=>{drag=true;last=e;}); addEventListener('pointerup',()=>drag=false); addEventListener('pointermove',e=>{if(!drag)return;robot.rotation.y+=(e.clientX-last.clientX)*.01;robot.rotation.x+=(e.clientY-last.clientY)*.01;last=e;});
 new ResizeObserver(()=>{camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();renderer.setSize(host.clientWidth,host.clientHeight)}).observe(host);
 if(cfg.animate){const motionStart=performance.now(),motionDuration=520; function tick(now){const progress=Math.min((now-motionStart)/motionDuration,1),eased=1-Math.pow(1-progress,3); shoulder.position.lerpVectors(shoulderStart,shoulderTarget,eased); armPivots.forEach((pivot,index)=>pivot.rotation.z=THREE.MathUtils.lerp(cfg.previousJoints[index],cfg.joints[index],eased)*Math.PI/180); fingerMeshes.forEach((finger,index)=>applyFingerData(finger,cfg.previousFingerCapsules[index],cfg.fingerCapsules[index],eased)); if(!cfg.grasped)can.position.set(cfg.canOffset[0]+canStart.x,cfg.canOffset[1]+canStart.y,cfg.canOffset[2]+canStart.z); frameCamera(); renderer.render(scene,camera);requestAnimationFrame(tick);} requestAnimationFrame(tick);}else{shoulder.position.copy(shoulderTarget);armPivots.forEach((pivot,index)=>pivot.rotation.z=cfg.joints[index]*Math.PI/180);fingerMeshes.forEach((finger,index)=>applyFingerData(finger,cfg.fingerCapsules[index],cfg.fingerCapsules[index],1));if(!cfg.grasped)can.position.set(cfg.canOffset[0]+canStart.x,cfg.canOffset[1]+canStart.y,cfg.canOffset[2]+canStart.z);frameCamera();renderer.render(scene,camera);}
})();</script>'''
    return html.replace("__THREE_RUNTIME__", three_runtime).replace("__CONFIG__", config)


def foot_schematic_figure(zones: np.ndarray, terrain: str, cop_xy: np.ndarray | None = None) -> go.Figure:
    """Render a six-zone sole and cyan FBG routing for teaching."""
    zones = np.asarray(zones, dtype=float)
    figure = go.Figure()
    for index, value in enumerate(zones):
        x, y = index % 3, 1 - index // 3
        figure.add_shape(type="rect", x0=x, x1=x + .9, y0=y, y1=y + .85, fillcolor=f"rgba(255,150,50,{min(.9,.18+value/80):.2f})", line={"color":"#dcebf3"})
        figure.add_annotation(x=x+.45, y=y+.42, text=f"区 {index+1}<br>{value:.0f} N", showarrow=False, font={"color":"white"})
    figure.add_scatter(x=[.1,.8,1.5,2.2,2.7], y=[1.65,1.4,1.55,1.3,.25], mode="lines+markers", name="足底 FBG 走线", line={"color":"#29c4d7","width":6})
    if cop_xy is not None:
        figure.add_scatter(x=[float(cop_xy[0])], y=[float(cop_xy[1])], mode="markers", name="压力中心 CoP", marker={"size": 15, "symbol": "x", "color": "#ffffff", "line": {"color": "#ff4d4f", "width": 3}})
    figure.update_layout(title=f"足底六区接触与光纤阵列：{terrain}", template="plotly_dark", height=390, xaxis={"visible":False,"range":[-.1,3.1]}, yaxis={"visible":False,"range":[-.1,2.1],"scaleanchor":"x"}, margin={"l":10,"r":10,"t":50,"b":10})
    return figure


def foot_fbg_dashboard_figure(wavelength_shifts_nm: np.ndarray, zone_loads_n: np.ndarray) -> go.Figure:
    """Show the six live FBG channels and their matching teaching loads together."""
    shifts = np.asarray(wavelength_shifts_nm, dtype=float)
    loads = np.asarray(zone_loads_n, dtype=float)
    if shifts.shape != (6,) or loads.shape != (6,):
        raise ValueError("足底 FBG 数据看板需要六路波长漂移与六个区域载荷")
    labels = [f"FBG {index}" for index in range(1, 7)]
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_bar(
        x=labels, y=shifts, name="FBG 波长漂移", marker_color=COLORS["sensor"],
        text=[f"{value:.4f}" for value in shifts], textposition="outside", secondary_y=False,
    )
    figure.add_scatter(
        x=labels, y=loads, mode="lines+markers+text", name="区域载荷",
        line={"color": COLORS["estimate"], "width": 3}, marker={"size": 9},
        text=[f"{value:.0f} N" for value in loads], textposition="top center", secondary_y=True,
    )
    figure.update_layout(
        title="实时足底结果：六路 FBG 波长漂移与六区载荷",
        template="plotly_white", height=360, legend={"orientation": "h", "y": 1.14},
        margin={"l": 20, "r": 20, "t": 65, "b": 25},
    )
    figure.update_yaxes(title_text="波长漂移 Δλ (nm)", secondary_y=False)
    figure.update_yaxes(title_text="区域载荷 (N)", secondary_y=True)
    return figure


def arm_health_figure(result: dict, diagnosis: dict) -> go.Figure:
    """Show an arm-link FBG array and its localised structural-health indication."""
    positions = np.asarray(result["sensor_positions_mm"], dtype=float)
    strain = np.asarray(result["strain"], dtype=float) * 1e6
    suspected = float(diagnosis["suspected_location_mm"])
    uncertainty = float(diagnosis.get("location_uncertainty_mm", 60.0))
    figure = go.Figure()
    figure.add_scatter(
        x=[0.0, 520.0], y=[0.0, 0.0], mode="lines", name="机械臂构件",
        line={"color": "#315064", "width": 24},
    )
    figure.add_scatter(
        x=positions, y=np.zeros_like(positions), mode="markers+text", name="FBG 阵列",
        text=[f"FBG {index + 1}<br>{value:.0f} με" for index, value in enumerate(strain)],
        textposition="top center", marker={"size": 18, "color": strain, "colorscale": "YlOrRd", "showscale": True, "colorbar": {"title": "应变 (με)"}},
    )
    figure.add_vrect(
        x0=suspected - uncertainty, x1=suspected + uncertainty,
        fillcolor="rgba(255,77,79,.10)", line_width=0,
    )
    figure.add_vline(x=suspected, line_dash="dash", line_color="#ff4d4f", annotation_text=f"可疑位置 {suspected:.0f} ± {uncertainty:.0f} mm")
    figure.update_layout(
        title=f"机械臂结构健康：可疑位置 {suspected:.0f} ± {uncertainty:.0f} mm",
        template="plotly_dark", height=360, xaxis_title="构件长度 (mm)",
        yaxis={"visible": False, "range": [-1.0, 1.8]}, margin={"l": 20, "r": 70, "t": 55, "b": 35},
    )
    return figure
