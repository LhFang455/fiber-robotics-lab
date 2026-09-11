"""Time-linked inspection of the existing offline dynamic skin rule."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .foot_inspector import selection_value

CHANNELS = {
    "剪切比": ("shear_ratio", "无量纲"),
    "质心速度绝对值": ("centroid_velocity_mm_s", "mm/s"),
    "法向力": ("normal_force_n", "N"),
    "剪切力": ("shear_force_n", "N"),
    "温度": ("temperature_c", "°C"),
    "质心位置": ("centroid_x_mm", "mm"),
}


def series(result, name, force):
    key, _ = CHANNELS[name]
    values = np.asarray(result[key], dtype=float).copy()
    if name == "质心速度绝对值":
        values = np.abs(values)
    if name == "剪切比":
        values[np.asarray(result["normal_force_n"]) <= max(0.05 * force, 1e-9)] = np.nan
    return values


def peak_indices(result, duration):
    window = np.flatnonzero(np.asarray(result["time_s"]) / duration >= 0.2)
    return tuple(
        int(window[np.argmax(np.abs(np.asarray(result[key])[window]))])
        for key in ("shear_ratio", "centroid_velocity_mm_s")
    )


def timeline(result, name, index, duration, force, threshold):
    time = np.asarray(result["time_s"])
    values = series(result, name, force)
    fig = go.Figure(
        go.Scatter(
            x=time,
            y=[float(v) if np.isfinite(v) else None for v in values],
            mode="lines",
            hoverinfo="skip",
            connectgaps=False,
            line=dict(color="#65d0c8"),
            showlegend=False,
        )
    )
    points = sorted(
        set(np.linspace(0, len(time) - 1, min(45, len(time)), dtype=int))
        | {index}
        | set(peak_indices(result, duration))
    )
    fig.add_scatter(
        x=time[points],
        y=[float(values[i]) if np.isfinite(values[i]) else None for i in points],
        customdata=points,
        mode="markers",
        marker=dict(
            size=9, color=["#ffcf69" if i == index else "#65d0c8" for i in points]
        ),
        unselected=dict(marker=dict(opacity=1)),
        showlegend=False,
        hovertemplate="t=%{x:.3f} s<br>%{y:.4f}<extra>"
        + CHANNELS[name][1]
        + "</extra>",
    )
    fig.add_vrect(
        x0=0.2 * duration,
        x1=time[-1],
        fillcolor="#25445c",
        opacity=0.25,
        line_width=0,
        layer="below",
    )
    if name in ("剪切比", "质心速度绝对值"):
        fig.add_hline(
            y=threshold if name == "剪切比" else 2.0,
            line_dash="dash",
            line_color="#ed91bd",
        )
    fig.add_vline(x=time[index], line_color="#ffcf69")
    fig.update_layout(
        title=f"{name} · 点击圆点选时刻",
        template="plotly_dark",
        height=350,
        clickmode="event+select",
        xaxis_title="时间 (s)",
        yaxis_title=CHANNELS[name][1],
        margin=dict(l=45, r=15, t=45, b=45),
    )
    return fig


def render(result, duration, force, threshold):
    st.markdown("#### 动态事件时刻观察器")
    st.caption(
        "模拟记录 · 蓝色背景是记录 20% 后的判定窗口；粉色虚线为当前信号阈值；金色为选定时刻。完整曲线保留全部采样，圆点只用于稀疏选取，不是降采样分析。"
    )
    time = np.asarray(result["time_s"])
    key = "dynamic_inspect_index"
    st.session_state[key] = min(
        st.session_state.get(key, len(time) // 2), len(time) - 1
    )
    peaks = peak_indices(result, duration)

    def jump(i):
        st.session_state[key] = i

    a, b = st.columns(2)
    a.button(
        "定位剪切比峰值", on_click=jump, args=(peaks[0],), key="dynamic_jump_ratio"
    )
    b.button(
        "定位质心速度峰值", on_click=jump, args=(peaks[1],), key="dynamic_jump_speed"
    )
    index = st.slider("检查采样序号", 0, len(time) - 1, key=key)
    name = st.selectbox("查看动态信号", list(CHANNELS), key="dynamic_inspect_channel")
    st.caption(
        f"FRAME-{index:04d} · t={time[index]:.3f} s · {'判定窗口内' if time[index] / duration >= 0.2 else '初始接触阶段，不参与峰值判定'}。编号从 0 开始，切换事件或采样率后对应新记录。"
    )
    rows = []
    for label, (_, unit) in CHANNELS.items():
        v = series(result, label, force)[index]
        rows.append(
            {
                "信号": label,
                "当前值": "未定义／缺测" if not np.isfinite(v) else f"{v:.4f}",
                "单位": unit,
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")

    def pick():
        value = selection_value(st.session_state.get("dynamic_time_pick", {}))
        if isinstance(value, int) and 0 <= value < len(time):
            st.session_state[key] = value

    st.plotly_chart(
        timeline(result, name, index, duration, force, threshold),
        key="dynamic_time_pick",
        on_select=pick,
        selection_mode="points",
        config={"displayModeBar": False},
    )
    x = np.asarray(result["centroid_x_mm"])
    ruler = go.Figure()
    ruler.add_scatter(
        x=[float(x[index]) if np.isfinite(x[index]) else None],
        y=[0],
        mode="markers",
        marker=dict(size=22, color="#ffcf69"),
        hovertemplate="质心 x=%{x:.3f} mm<extra></extra>",
    )
    valid = x[np.isfinite(x)]
    ruler.update_layout(
        title="所选时刻压力质心 · 一维位置示意",
        template="plotly_dark",
        height=190,
        xaxis=dict(
            title="模型局部 x (mm)",
            range=[
                min(-1.0, float(valid.min()) - 1.0) if len(valid) else -1.0,
                max(1.0, float(valid.max()) + 1.0) if len(valid) else 1.0,
            ],
        ),
        yaxis=dict(visible=False, range=[-1, 1]),
        margin=dict(l=25, r=20, t=45, b=50),
    )
    st.plotly_chart(ruler, key="dynamic_position", config={"displayModeBar": False})
    st.caption(
        "仅显示模型已有的 x 方向质心；不虚构皮肤尺寸、接触轮廓或 y 坐标。低法向力时剪切比未定义：原算法使用零占位，这里不将其显示为有效测量。"
    )
    st.info(
        f"整段记录结论：{result['status']}。剪切比峰值在 {time[peaks[0]]:.3f} s，速度绝对值峰值在 {time[peaks[1]]:.3f} s。"
        + result["rule"]
        + " 当前时刻不单独产生告警；温度不是该规则的判定条件。此结论使用整段记录，不是实时告警时刻。"
    )
