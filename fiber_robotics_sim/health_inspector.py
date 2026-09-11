"""Inspect the existing one-dimensional health array without inventing geometry."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from . import models
from .foot_inspector import selection_value


def channels(record):
    r = record["results"]
    positions = r["sensor_positions_mm"]
    shifts = np.asarray(r["wavelength_shifts_nm"], dtype=float)
    strain = (
        models._strain_from_shift(shifts, record["parameters"]["temperature_c"]) * 1e6
    )
    valid = np.isfinite(strain)
    excess = (
        np.maximum(0.0, strain - np.median(strain))
        if valid.all()
        else np.full(len(strain), np.nan)
    )
    return [
        dict(
            id=f"ARM-N{len(positions):02d}-{i + 1:02d}",
            channel=i + 1,
            position=float(x),
            wavelength=float(shifts[i]) if np.isfinite(shifts[i]) else None,
            strain=float(strain[i]) if valid[i] else None,
            excess=float(excess[i]) if np.isfinite(excess[i]) else None,
        )
        for i, x in enumerate(positions)
    ]


def figures(record, selected, show_reference):
    rows = channels(record)
    r = record["results"]
    p = record["parameters"]
    ids = [c["id"] for c in rows]
    positions = [c["position"] for c in rows]
    spatial = go.Figure()
    spatial.add_shape(
        type="rect",
        x0=0,
        x1=520,
        y0=-0.17,
        y1=0.17,
        fillcolor="#20384a",
        line_color="#608196",
        layer="below",
    )
    spatial.add_scatter(
        x=positions,
        y=[0] * len(rows),
        customdata=list(range(len(rows))),
        mode="markers+text",
        text=[f"FBG {c['channel']}" for c in rows],
        textposition="top center",
        marker=dict(
            size=22,
            color=["#ffcf69" if i == selected else "#66cec5" for i in range(len(rows))],
        ),
        unselected=dict(marker=dict(opacity=1)),
        hovertext=ids,
        hovertemplate="%{hovertext}<br>x=%{x:.1f} mm<extra></extra>",
    )
    profile = go.Figure(
        go.Scatter(
            x=positions,
            y=[c["excess"] for c in rows],
            customdata=list(range(len(rows))),
            mode="lines+markers",
            marker=dict(
                size=13,
                color=[
                    "#ffcf69" if i == selected else "#66cec5" for i in range(len(rows))
                ],
            ),
            line=dict(color="#66cec5"),
            connectgaps=False,
            hovertemplate="x=%{x:.1f} mm<br>局部超额=%{y:.2f} με<extra></extra>",
        )
    )
    profile.add_hline(
        y=280,
        line_dash="dot",
        line_color="#ef9686",
        annotation_text="教学触发阈值 280 με",
    )
    profile.add_vline(x=positions[selected], line_color="#ffcf69", line_dash="dash")
    usable = all(c["excess"] is not None for c in rows)
    for fig in (spatial, profile):
        if r["localization_valid"] and usable:
            x = r["suspected_location_mm"]
            half = r["location_uncertainty_mm"]
            fig.add_vrect(
                x0=max(0, x - half),
                x1=min(520, x + half),
                fillcolor="#ee9861",
                opacity=0.2,
                line_width=0,
                layer="below",
            )
            fig.add_vline(x=x, line_color="#ee9861", line_dash="dash")
        if show_reference and p["anomaly_severity"] > 0:
            fig.add_vline(
                x=p["anomaly_position_mm"], line_color="#be9afa", line_dash="dot"
            )
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            clickmode="event+select",
            xaxis=dict(title="梁段轴向位置 (mm)", range=[0, 520]),
            margin=dict(l=45, r=15, t=45, b=40),
        )
    spatial.update_layout(
        title="梁段位置示意 · 点击测点",
        height=255,
        yaxis=dict(visible=False, range=[-0.6, 0.6]),
    )
    profile.update_layout(
        title="局部超额应变 · 点击通道", height=330, yaxis_title="超额应变 (με)"
    )
    return spatial, profile


def render(record):
    st.markdown("#### 结构测点与定位观察器")
    st.caption(
        "模拟数据 · 梁段仅表示原模型 0～520 mm 轴向范围，厚度为示意；测点沿轴向测量应变，不代表已确认的实物截面或封装。"
    )
    rows = channels(record)
    st.session_state.setdefault("health_inspect_channel", 0)
    if st.session_state.health_inspect_channel >= len(rows):
        st.session_state.health_inspect_channel = len(rows) - 1

    def pick(key):
        value = selection_value(st.session_state.get(key, {}))
        if isinstance(value, int) and 0 <= value < len(rows):
            st.session_state.health_inspect_channel = value

    selected = st.selectbox(
        "查看结构测点",
        range(len(rows)),
        key="health_inspect_channel",
        format_func=lambda i: f"{rows[i]['id']} · {rows[i]['position']:.1f} mm",
    )
    c = rows[selected]
    a, b, d = st.columns(3)

    def display(v, unit):
        return "缺测" if v is None else f"{v:.3f} {unit}"

    a.metric("测点波长 Δλ", display(c["wavelength"], "nm"))
    b.metric("温补反演应变", display(c["strain"], "με"))
    d.metric("相对阵列中位数的超额", display(c["excess"], "με"))
    st.caption(
        f"{c['id']} → FBG {c['channel']}，梁段局部坐标 ({c['position']:.1f}, 0, 0) mm，测量方向 +X。编号包含阵列数量；切换密度后，不应把同序号当作原物理测点。"
    )
    reference = st.checkbox(
        "显示异常位置设定（模拟参考）", value=True, key="health_inspect_reference"
    )
    space, profile = figures(record, selected, reference)
    st.plotly_chart(
        space,
        key="health_space_pick",
        on_select=lambda: pick("health_space_pick"),
        selection_mode="points",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        profile,
        key="health_profile_pick",
        on_select=lambda: pick("health_profile_pick"),
        selection_mode="points",
        config={"displayModeBar": False},
    )
    st.caption(
        "金色表示所选通道；橙色虚线与阴影表示有效定位及教学区间；紫色点线是模拟异常设定。连线只连接离散测点，不代表连续实测或真实裂纹形状。超额应变使用温补应变减去本阵列中位数并截断负值，沿用原诊断算法，不等同于独立健康基线。"
    )
    if not record["results"]["localization_valid"]:
        st.info(
            "本观察器不显示定位区间：当前未满足原模型的有效定位条件。阈值报警不等同于裂纹确认；无报警也不构成安全保证。"
        )
    st.caption(
        "阈值 280 με = 0.35 × 800 με，复用原教学诊断尺度；橙色区间由传感器间距估计，并非统计置信区间，边界处显示范围裁剪至梁段。真实判断仍需基线、载荷与温度工况以及独立检测。"
    )
