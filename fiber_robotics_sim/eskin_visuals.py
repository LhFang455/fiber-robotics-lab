"""Plotly figures for electronic-skin teaching models."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


_COLORS = {"navy": "#16324F", "cyan": "#3AAFA9", "orange": "#F2994A", "red": "#D1495B"}


def _layout(figure: go.Figure, title: str) -> go.Figure:
    figure.update_layout(
        title=title,
        template="plotly_white",
        margin=dict(l=35, r=25, t=60, b=35),
        legend=dict(orientation="h", y=1.08, x=0),
        hovermode="x unified",
    )
    return figure


def taxel_channel_figure(result: dict) -> go.Figure:
    channels = [f"C{index + 1}" for index in range(len(result["active_pf"]))]
    figure = go.Figure()
    figure.add_bar(name="主动信号", x=channels, y=result["active_pf"], marker_color=_COLORS["navy"])
    figure.add_bar(name="参考信号", x=channels, y=result["reference_pf"], marker_color=_COLORS["orange"])
    figure.add_bar(name="校正信号", x=channels, y=result["corrected_pf"], marker_color=_COLORS["cyan"])
    figure.update_layout(barmode="group", xaxis_title="电容通道", yaxis_title="电容变化 (pF)")
    return _layout(figure, "五通道主动/参考/校正响应")


def fbg_skin_figure(result: dict, skin_width_mm: float, skin_height_mm: float) -> go.Figure:
    positions = np.asarray(result["sensor_positions_mm"])
    response = np.asarray(result["compensated_shift_nm"])
    figure = go.Figure()
    figure.add_scatter(
        x=positions[:, 0], y=positions[:, 1], mode="markers+text",
        text=result["sensor_labels"], textposition="top center", name="FBG 传感点",
        marker=dict(size=18, color=response, colorscale="Turbo", showscale=True,
                    colorbar=dict(title="温补后 Δλ (nm)")),
    )
    figure.add_scatter(
        x=[result["estimated_centroid_mm"][0]], y=[result["estimated_centroid_mm"][1]],
        mode="markers", name="估计压力质心",
        marker=dict(symbol="x", size=16, color=_COLORS["red"], line=dict(width=3)),
    )
    figure.update_xaxes(range=[0, skin_width_mm], title="x (mm)")
    figure.update_yaxes(range=[0, skin_height_mm], title="y (mm)", scaleanchor="x", scaleratio=1)
    return _layout(figure, "FBG 光学皮肤感受野响应")


def pressure_reconstruction_figure(result: dict) -> go.Figure:
    figure = make_subplots(rows=1, cols=3, subplot_titles=("目标压力场", "稀疏重建", "重建误差"))
    figure.add_trace(go.Heatmap(z=result["truth_kpa"], colorscale="Turbo", colorbar=dict(title="kPa", x=0.29)), row=1, col=1)
    figure.add_trace(go.Heatmap(z=result["reconstruction_kpa"], colorscale="Turbo", colorbar=dict(title="kPa", x=0.64)), row=1, col=2)
    max_error = max(float(np.max(np.abs(result["error_kpa"]))), 1e-9)
    figure.add_trace(
        go.Heatmap(z=result["error_kpa"], colorscale="RdBu", zmin=-max_error, zmax=max_error,
                   colorbar=dict(title="误差 kPa", x=1.0)),
        row=1, col=3,
    )
    figure.update_xaxes(title_text="列")
    figure.update_yaxes(title_text="行", autorange="reversed")
    return _layout(figure, "稀疏采样到致密压力场")


def dynamic_event_figure(result: dict) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    time_s = result["time_s"]
    figure.add_scatter(x=time_s, y=result["normal_force_n"], name="法向力 (N)", line=dict(color=_COLORS["navy"]), secondary_y=False)
    figure.add_scatter(x=time_s, y=result["shear_force_n"], name="剪切力 (N)", line=dict(color=_COLORS["orange"]), secondary_y=False)
    figure.add_scatter(x=time_s, y=result["shear_ratio"], name="剪切比", line=dict(color=_COLORS["red"]), secondary_y=True)
    figure.add_scatter(x=time_s, y=result["centroid_x_mm"], name="压力质心 x (mm)", line=dict(color=_COLORS["cyan"], dash="dot"), secondary_y=True)
    figure.add_scatter(x=time_s, y=result["temperature_c"], name="温度 (°C)", line=dict(color="#7A5195", dash="dash"), secondary_y=True)
    figure.update_xaxes(title_text="时间 (s)")
    figure.update_yaxes(title_text="力 (N)", secondary_y=False)
    figure.update_yaxes(title_text="剪切比 / 质心 / 温度", secondary_y=True)
    return _layout(figure, f"动态多模态事件：{result['event']}")
