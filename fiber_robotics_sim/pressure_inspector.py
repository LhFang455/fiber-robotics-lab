"""Inspect existing Gaussian reconstruction without inventing extra sensors."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .foot_inspector import selection_value


def contribution(result, row, col, bandwidth):
    samples = np.asarray(result["sparse_samples_kpa"])
    n = samples.shape[0]
    m = result["reconstruction_kpa"].shape[0]
    x, y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    weights = np.exp(
        -((x - col / (m - 1)) ** 2 + (y - row / (m - 1)) ** 2) / (2 * bandwidth**2)
    )
    weights /= max(weights.sum(), np.finfo(float).eps)
    return weights, weights * samples


def figures(result, row, col, bandwidth):
    field = np.asarray(result["reconstruction_kpa"])
    truth = np.asarray(result["truth_kpa"])
    samples = np.asarray(result["sparse_samples_kpa"])
    m, n = len(field), len(samples)
    axis = np.linspace(0, 1, m)
    sx, sy = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    finite = np.concatenate([a[np.isfinite(a)] for a in (field, truth, samples)])
    maximum = max(1.0, float(finite.max())) if len(finite) else 1.0
    space = go.Figure(
        go.Heatmap(
            x=axis,
            y=axis,
            z=field,
            zmin=0,
            zmax=maximum,
            colorscale="Viridis",
            colorbar=dict(title="kPa", thickness=12),
            hoverinfo="skip",
            showscale=True,
        )
    )
    labels = [
        f"SAMPLE-N{n:02d}-R{i // n + 1:02d}-C{i % n + 1:02d} · 通道 {i + 1}"
        for i in range(n * n)
    ]
    space.add_scatter(
        x=sx.ravel(),
        y=sy.ravel(),
        mode="markers",
        name="模拟采样位置",
        marker=dict(symbol="square-open", color="white", size=9),
        text=labels,
        hovertemplate="%{text}<br>归一化位置 (%{x:.3f}, %{y:.3f})<extra></extra>",
    )
    space.add_scatter(
        x=axis,
        y=[axis[row]] * m,
        customdata=list(range(m)),
        mode="markers",
        name="所选行重建节点（可点击）",
        marker=dict(
            size=10, color=["#ffcf69" if i == col else "#ed91bd" for i in range(m)]
        ),
        hovertemplate="重建节点 x=%{x:.3f}<extra></extra>",
        unselected=dict(marker=dict(opacity=1)),
    )
    space.update_layout(
        title="重建场与采样位置",
        template="plotly_dark",
        height=420,
        clickmode="event+select",
        xaxis=dict(title="归一化 x", range=[-0.05, 1.05], constrain="domain"),
        yaxis=dict(
            title="归一化 y", range=[-0.05, 1.05], scaleanchor="x", constrain="domain"
        ),
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=40, r=20, t=45, b=90),
    )
    profile = go.Figure()
    for a, name, color in [
        (truth, "模拟参考场", "#65d0c8"),
        (field, "插值重建场", "#c19bef"),
    ]:
        profile.add_scatter(
            x=axis,
            y=[float(v) if np.isfinite(v) else None for v in a[row]],
            customdata=list(range(m)),
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            connectgaps=False,
            hovertemplate="x=%{x:.3f}<br>%{y:.3f} kPa<extra>" + name + "</extra>",
        )
    profile.add_vline(x=axis[col], line_color="#ffcf69", line_dash="dash")
    profile.update_layout(
        title=f"第 {row + 1} 行压力剖面 · 点击节点",
        template="plotly_dark",
        height=330,
        xaxis_title="归一化 x",
        yaxis_title="压力 (kPa)",
        clickmode="event+select",
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=45, r=15, t=45, b=85),
    )
    weights, values = contribution(result, row, col, bandwidth)
    bars = go.Figure(
        go.Bar(
            x=list(range(1, n * n + 1)),
            y=[float(v) if np.isfinite(v) else None for v in values.ravel()],
            customdata=weights.ravel(),
            hovertext=labels,
            hovertemplate="%{hovertext}<br>贡献 %{y:.4f} kPa<br>权重 %{customdata:.4f}<extra></extra>",
        )
    )
    bars.update_layout(
        title="各模拟采样通道对所选节点的贡献",
        template="plotly_dark",
        height=300,
        xaxis_title="采样通道编号（按行展开）",
        yaxis_title="权重 × 采样压力 (kPa)",
        margin=dict(l=45, r=15, t=45, b=45),
    )
    return space, profile, bars


def render(result, bandwidth):
    st.markdown("#### 压力重建位置观察器")
    st.caption(
        "模拟采样 → 高斯核插值 → 重建网格。白色空心方框是模拟传感通道；粉色圆点是所选行的计算节点，金色为当前节点。网格加密不增加传感器，也不保证提高空间分辨率。"
    )
    m = len(result["reconstruction_kpa"])
    n = len(result["sparse_samples_kpa"])
    for key in ("pressure_inspect_row", "pressure_inspect_col"):
        st.session_state[key] = min(st.session_state.get(key, m // 2), m - 1)
    a, b = st.columns(2)
    row = a.selectbox(
        "检查重建网格行",
        range(m),
        format_func=lambda i: f"第 {i + 1} 行",
        key="pressure_inspect_row",
    )
    col = b.selectbox(
        "检查重建网格列",
        range(m),
        format_func=lambda i: f"第 {i + 1} 列",
        key="pressure_inspect_col",
    )

    def pick(key):
        value = selection_value(st.session_state.get(key, {}))
        if isinstance(value, int) and 0 <= value < m:
            st.session_state.pressure_inspect_col = value

    st.caption(
        f"GRID-N{m:02d}-R{row + 1:02d}-C{col + 1:02d} · 归一化位置 ({col / (m - 1):.3f}, {row / (m - 1):.3f})；不是独立测点编号。当前只有 {n * n} 个模拟采样通道。"
    )
    cols = st.columns(3)
    for container, key, label in zip(
        cols,
        ("truth_kpa", "reconstruction_kpa", "error_kpa"),
        ("模型参考压力", "插值压力", "有符号误差（重建−参考）"),
    ):
        v = result[key][row, col]
        container.metric(label, f"{v:+.3f} kPa" if np.isfinite(v) else "缺测")
    space, profile, bars = figures(result, row, col, bandwidth)
    for fig, key in (
        (space, "pressure_space_pick"),
        (profile, "pressure_profile_pick"),
    ):
        st.plotly_chart(
            fig,
            key=key,
            on_select=lambda key=key: pick(key),
            selection_mode="points",
            config={"displayModeBar": False},
        )
    st.plotly_chart(
        bars, key="pressure_contributions", config={"displayModeBar": False}
    )
    st.caption(
        "贡献值相加等于该节点的插值压力；权重由归一化距离与当前带宽决定，不是置信度。高斯核平滑通常不严格经过采样值。缺失值保留为空，不按零压力解释。"
    )
    st.info(
        "当前模型没有实物面积，不能把压力之和直接当作 N。页面总载荷误差是在相同归一化网格上的平均压力相对误差；实际合力还需要面积与积分约定。"
    )
