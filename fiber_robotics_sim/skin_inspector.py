"""Inspect existing FBG skin channels and normalized contact contributions."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .foot_inspector import selection_value


def channel(result, index):
    position = np.asarray(result["sensor_positions_mm"])[index]

    def finite(v):
        return float(v) if np.isfinite(v) else None

    return dict(
        id=f"SKIN-N{len(result['sensor_labels']):02d}-{index + 1:02d}",
        position=position.tolist(),
        raw=finite(result["measured_shift_nm"][index]),
        compensated=finite(result["compensated_shift_nm"][index]),
    )


def figures(result, index, width, height, show_reference):
    positions = np.asarray(result["sensor_positions_mm"])
    n = len(positions)
    color = ["#ffcf69" if i == index else "#65d0c8" for i in range(n)]
    spatial = go.Figure(
        go.Scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            customdata=list(range(n)),
            mode="markers+text",
            text=result["sensor_labels"],
            textposition="top center",
            marker=dict(size=17, color=color),
            unselected=dict(marker=dict(opacity=1)),
            hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<extra></extra>",
        )
    )
    spatial.add_shape(
        type="rect",
        x0=0,
        x1=width,
        y0=0,
        y1=height,
        fillcolor="#173044",
        line_color="#557489",
        layer="below",
    )
    for key, name, c in [
        ("estimated_centroid_mm", "估计载荷质心", "#f3a270"),
        ("true_centroid_mm", "模型参考质心", "#c19bef"),
    ]:
        xy = np.asarray(result[key])
        if (key != "true_centroid_mm" or show_reference) and np.all(np.isfinite(xy)):
            spatial.add_scatter(
                x=[xy[0]],
                y=[xy[1]],
                mode="markers",
                marker=dict(size=17, symbol="x", color=c),
                name=name,
            )
    spatial.update_layout(
        title="光学皮肤阵列 · 点击测点",
        height=420,
        template="plotly_dark",
        clickmode="event+select",
        xaxis=dict(title="局部 x (mm)", range=[-5, width + 5], constrain="domain"),
        yaxis=dict(
            title="局部 y (mm)",
            range=[-5, height + 5],
            scaleanchor="x",
            constrain="domain",
        ),
        legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=15, t=45, b=60),
    )
    spatial.data[0].showlegend = False
    profile = go.Figure()
    for key, name, c in [
        ("measured_shift_nm", "原始波长", "#c19bef"),
        ("compensated_shift_nm", "温补波长", "#65d0c8"),
    ]:
        values = [float(v) if np.isfinite(v) else None for v in result[key]]
        profile.add_scatter(
            x=list(range(1, n + 1)),
            y=values,
            customdata=list(range(n)),
            mode="lines+markers",
            name=name,
            line=dict(color=c),
            marker=dict(size=10),
            connectgaps=False,
            hovertemplate="FBG %{x}<br>Δλ=%{y:.5f} nm<extra>" + name + "</extra>",
        )
    profile.add_vline(x=index + 1, line_color="#ffcf69", line_dash="dash")
    profile.update_layout(
        title="原始与温补通道 · 点击圆点",
        height=340,
        template="plotly_dark",
        clickmode="event+select",
        xaxis=dict(title="FBG 通道编号", dtick=1),
        yaxis_title="波长漂移 Δλ (nm)",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=45, r=15, t=45, b=70),
    )
    return spatial, profile


def render(result, width, height, touches):
    st.markdown("#### 光学皮肤测点与质心观察器")
    st.caption(
        "模拟数据 · 沿用当前阵列位置与归一化高斯感受野。SKIN 编号包含阵列数量；尺寸或数量改变后，应以显示的局部坐标核对测点，不代表实物封装编号。"
    )
    n = len(result["sensor_labels"])
    st.session_state.setdefault("skin_inspect_channel", 0)
    if st.session_state.skin_inspect_channel >= n:
        st.session_state.skin_inspect_channel = n - 1

    def pick(key):
        v = selection_value(st.session_state.get(key, {}))
        if isinstance(v, int) and 0 <= v < n:
            st.session_state.skin_inspect_channel = v

    index = st.selectbox(
        "查看光学皮肤测点",
        range(n),
        key="skin_inspect_channel",
        format_func=lambda i: result["sensor_labels"][i],
    )
    c = channel(result, index)
    a, b, d = st.columns(3)
    a.metric("所选原始波长", "缺测" if c["raw"] is None else f"{c['raw']:.5f} nm")
    b.metric(
        "所选温补波长",
        "缺测" if c["compensated"] is None else f"{c['compensated']:+.5f} nm",
    )
    d.metric("已知温度共模", f"{result['temperature_shift_nm']:+.5f} nm")
    st.caption(
        f"{c['id']} → FBG {index + 1}，局部位置 ({c['position'][0]:.2f}, {c['position'][1]:.2f}, 0) mm。感受野将表面法向载荷编码为波长，未提供真实纤芯朝向或三轴力标定。"
    )
    show_reference = st.checkbox(
        "显示模拟参考质心", value=True, key="skin_inspect_reference"
    )
    spatial, profile = figures(result, index, width, height, show_reference)
    st.plotly_chart(
        spatial,
        key="skin_space_pick",
        on_select=lambda: pick("skin_space_pick"),
        selection_mode="points",
        config={"displayModeBar": False},
    )
    st.plotly_chart(
        profile,
        key="skin_profile_pick",
        on_select=lambda: pick("skin_profile_pick"),
        selection_mode="points",
        config={"displayModeBar": False},
    )
    st.caption(
        "金色圆点／竖线为同一选中通道；叉号为估计与模型参考质心，不是额外测点。通道曲线只连接离散读数，不表示沿空间连续测量。负温补波长保留显示；原估计器求质心时会将负响应截断为零。"
    )
    st.dataframe(
        [
            dict(
                接触=f"接触 {j + 1}",
                位置_mm=f"({x:.1f}, {y:.1f})",
                模型载荷_N=float(force),
                分配到所选测点的权重=float(result["receptive_fields"][j, index]),
            )
            for j, (x, y, force) in enumerate(touches)
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "权重为该接触在全部 FBG 间归一化后的模型分配系数，不是接触概率、识别置信度或独立实测力；单个测点可同时响应两个接触。"
    )
    if not np.all(np.isfinite(result["true_centroid_mm"])):
        st.info(
            "当前模型总载荷为零：参考质心未定义。若噪声仍产生估计质心，不应认定为真实接触。"
        )
    st.info(
        result["interpretation"]
        + " 合力与质心来自全部有效模型响应，不是所选单通道独立识别的结果。"
    )
