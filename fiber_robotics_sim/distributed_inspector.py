"""Spatial sample inspector for existing distributed models; no interpolation."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .foot_inspector import selection_value


def sample(record, index):
    r = record["results"]
    x = float(r["profile_position_mm"][index])
    v = r["profile_value"][index]
    return dict(
        id=f"D-{x:.3f}mm",
        position=x,
        value=float(v) if np.isfinite(v) else None,
        unit=r["unit"],
    )


def figures(record, index):
    r = record["results"]
    x = r["profile_position_mm"]
    values = r["profile_value"]
    spatial = go.Figure(
        go.Scatter(
            x=x,
            y=[0] * len(x),
            customdata=list(range(len(x))),
            mode="markers",
            marker=dict(
                size=10,
                color=["#ffcf69" if i == index else "#64ccc5" for i in range(len(x))],
            ),
            unselected=dict(marker=dict(opacity=1)),
            hovertemplate="采样位置 %{x:.2f} mm<extra></extra>",
        )
    )
    spatial.add_shape(
        type="line",
        x0=0,
        x1=record["parameters"]["fiber_length_mm"],
        y0=0,
        y1=0,
        line=dict(color="#59788d", width=6),
        layer="below",
    )
    profile = go.Figure(
        go.Scatter(
            x=x,
            y=[v if np.isfinite(v) else None for v in values],
            customdata=list(range(len(x))),
            mode="lines+markers",
            connectgaps=False,
            marker=dict(
                size=8,
                color=["#ffcf69" if i == index else "#64ccc5" for i in range(len(x))],
            ),
            line=dict(color="#64ccc5"),
            hovertemplate="位置 %{x:.2f} mm<br>值 %{y:.4f} "
            + r["unit"]
            + "<extra></extra>",
        )
    )
    for fig in (spatial, profile):
        fig.add_vline(x=x[index], line_color="#ffcf69", line_dash="dash")
        if r["localization_valid"]:
            fig.add_vline(
                x=r["estimated_event_position_mm"],
                line_color="#e99a66",
                line_dash="dot",
            )
        fig.update_layout(
            template="plotly_dark",
            showlegend=False,
            clickmode="event+select",
            xaxis=dict(
                title="光纤轴向位置 (mm)",
                range=[0, record["parameters"]["fiber_length_mm"]],
            ),
            margin=dict(l=50, r=15, t=45, b=40),
        )
    spatial.update_layout(
        title="光纤采样位置 · 点击选取",
        height=200,
        yaxis=dict(visible=False, range=[-0.4, 0.4]),
    )
    profile.update_layout(
        title="空间剖面 · 点击采样点",
        height=320,
        yaxis_title=r["observable"] + " (" + r["unit"] + ")",
    )
    return spatial, profile


def waveform(result, index):
    x = np.asarray(result["position_mm"])
    time = np.asarray(result["time_s"])
    amplitude = np.asarray(result["amplitude"])
    if amplitude.shape != (len(time), len(x)):
        raise ValueError("DAS 时间与距离维度不匹配")
    y = amplitude[:, index]
    fig = go.Figure(
        go.Scatter(
            x=time,
            y=[float(v) if np.isfinite(v) else None for v in y],
            mode="lines+markers",
            connectgaps=False,
            line=dict(color="#ffcf69"),
            marker=dict(size=4),
        )
    )
    fig.update_layout(
        title=f"DAS：{x[index]:.2f} mm 处的时间波形",
        template="plotly_dark",
        height=300,
        xaxis_title="模型相对时间 (s)",
        yaxis_title="振动幅值 (a.u.)",
        margin=dict(l=50, r=15, t=45, b=40),
    )
    return fig


def render(record, result):
    st.markdown("#### 分布式位置与信号观察器")
    st.caption(
        "模拟数据 · 光纤按轴向距离展开，不代表真实铺设路径。D 编号使用实际采样位置；圆点为原模型保留下来的空间样本，连线不代表新增测量或插值结果。"
    )
    r = record["results"]
    count = len(r["profile_position_mm"])
    st.session_state.setdefault("distributed_inspect_index", 0)
    if st.session_state.distributed_inspect_index >= count:
        st.session_state.distributed_inspect_index = count - 1

    def pick(key):
        v = selection_value(st.session_state.get(key, {}))
        if isinstance(v, int) and 0 <= v < count:
            st.session_state.distributed_inspect_index = v

    index = st.select_slider(
        "查看空间采样点",
        options=list(range(count)),
        key="distributed_inspect_index",
        format_func=lambda i: f"{r['profile_position_mm'][i]:.2f} mm",
    )
    c = sample(record, index)
    a, b = st.columns(2)
    a.metric("选中位置", f"{c['position']:.2f} mm")
    b.metric(
        r["observable"] + " · 选中值",
        "缺测" if c["value"] is None else f"{c['value']:.4f} {c['unit']}",
    )
    spacing = np.diff(r["profile_position_mm"])
    st.caption(
        f"{c['id']} · 第 {index + 1}/{count} 个样本。实际相邻距离 {spacing.min():.3f}～{spacing.max():.3f} mm；金色线标记当前选择，橙色点线仅在原模型存在有效定位时显示。采样间隔不等同于仪器空间分辨率。"
    )
    spatial, profile = figures(record, index)
    st.plotly_chart(
        spatial,
        key="distributed_space_pick",
        on_select=lambda: pick("distributed_space_pick"),
        selection_mode="points",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        profile,
        key="distributed_profile_pick",
        on_select=lambda: pick("distributed_profile_pick"),
        selection_mode="points",
        config={"displayModeBar": False},
    )
    if record["parameters"]["mode"] == "φ-OTDR / DAS":
        np.testing.assert_allclose(result["position_mm"], r["profile_position_mm"])
        st.plotly_chart(
            waveform(result, index),
            key="distributed_time_view",
            config={"displayModeBar": False},
        )
        dt = float(np.diff(result["time_s"])[0])
        actual_rate = 1 / dt
        st.caption(
            f"空间剖面为各位置在当前时间窗内的最大绝对振幅；时间波形显示所选位置的原始有符号数据。时间间隔 {dt:.5f} s，对应生成采样率 {actual_rate:.1f} Hz；侧栏输入为 {record['parameters']['sample_rate_hz']} Hz，原生成器在固定 0.5 s 时间窗内使用至少 40 个样本，两者可能不同。"
        )
        if actual_rate <= 120:
            st.warning(
                "模型振动频率固定为 60 Hz；当前实际生成采样率未超过 120 Hz，不能据此可靠辨认原振动频率，需警惕混叠。"
            )
    st.caption(
        r["model_note"]
        + " 本观察器只读取已有模拟结果，不接入上传数据，也不改变原定位算法。"
    )
