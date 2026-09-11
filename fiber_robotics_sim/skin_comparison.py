"""Same-load, same-centroid contact hypotheses; no contact classifier."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from . import eskin


@st.cache_data(show_spinner=False, max_entries=24)
def compare(width, height, receptive, force, separation):
    center = (width / 2, height / 2)
    single = [(*center, force)]
    double = [
        (center[0] - separation / 2, center[1], force / 2),
        (center[0] + separation / 2, center[1], force / 2),
    ]
    cases = []
    for n in (4, 8, 16):
        p = dict(
            sensor_count=n,
            skin_width_mm=width,
            skin_height_mm=height,
            receptive_width_mm=receptive,
            temperature_c=25.0,
            noise_nm=0.0,
        )
        a = eskin.simulate_fbg_skin(touch_points=single, **p)
        b = eskin.simulate_fbg_skin(touch_points=double, **p)
        delta = b["compensated_shift_nm"] - a["compensated_shift_nm"]
        cases.append(
            dict(
                n=n,
                single=a,
                double=b,
                rms=float(np.sqrt(np.mean(delta**2))),
                maximum=float(np.max(np.abs(delta))),
            )
        )
    return cases, single, double


def figures(case, single, double, width, height):
    fig = go.Figure()
    xy = case["single"]["sensor_positions_mm"]
    fig.add_scatter(
        x=xy[:, 0],
        y=xy[:, 1],
        mode="markers",
        name="模拟 FBG 位置",
        marker=dict(color="#65d0c8", symbol="square-open", size=10),
    )
    for contacts, name, color, symbol in (
        (single, "假设 A：中心单点", "#ffcf69", "cross"),
        (double, "假设 B：对称双点", "#ed91bd", "circle"),
    ):
        fig.add_scatter(
            x=[p[0] for p in contacts],
            y=[p[1] for p in contacts],
            mode="markers",
            name=name,
            marker=dict(color=color, symbol=symbol, size=16),
            text=[f"{p[2]:.2f} N" for p in contacts],
            hovertemplate="(%{x:.2f}, %{y:.2f}) mm<br>%{text}<extra></extra>",
        )
    fig.update_layout(
        title="两个接触假设 · 不是识别输出",
        template="plotly_dark",
        height=380,
        xaxis=dict(title="x (mm)", range=[-5, width + 5], constrain="domain"),
        yaxis=dict(
            title="y (mm)", range=[-5, height + 5], scaleanchor="x", constrain="domain"
        ),
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=40, r=15, t=45, b=100),
    )
    signals = go.Figure()
    for key, name, color in (
        ("single", "单点响应", "#ffcf69"),
        ("double", "双点响应", "#ed91bd"),
    ):
        signals.add_scatter(
            x=list(range(1, case["n"] + 1)),
            y=case[key]["compensated_shift_nm"],
            mode="lines+markers",
            name=name,
            line=dict(color=color),
        )
    signals.update_layout(
        title="无噪声温补响应对照",
        template="plotly_dark",
        height=320,
        xaxis=dict(title="FBG 通道编号", dtick=1),
        yaxis=dict(
            title="波长漂移 (nm)",
            range=[
                0,
                max(
                    1e-6,
                    max(
                        float(case[k]["compensated_shift_nm"].max())
                        for k in ("single", "double")
                    )
                    * 1.1,
                ),
            ],
            tickformat=".4f",
        ),
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=45, r=15, t=45, b=85),
    )
    return fig, signals


def render(width, height, receptive, force, noise):
    st.markdown("#### 单点与双点可辨识性对照")
    if not st.checkbox("展开同合力同质心对照", key="skin_compare_enabled"):
        st.caption("比较中心单点与对称双点，检查通道响应是否保留接触分布信息。")
        return
    fraction = st.slider(
        "对照双点间距 / 皮肤宽度", 0.0, 0.8, 0.4, 0.05, key="skin_compare_separation"
    )
    n = st.selectbox("查看对照阵列", (4, 8, 16), index=1, key="skin_compare_count")
    if force <= 0:
        st.info("当前总载荷为零，载荷质心未定义；请提高上方有效接触载荷后进行对照。")
        return
    cases, single, double = compare(width, height, receptive, force, fraction * width)
    case = next(c for c in cases if c["n"] == n)
    st.caption(
        f"模拟假设 · 两组总载荷均为 {force:.2f} N，真实质心均为 ({width / 2:.2f}, {height / 2:.2f}) mm；双点间距 {fraction * width:.2f} mm，各承受 {force / 2:.2f} N。沿用当前尺寸、感受野宽度 {receptive:.1f} mm 与有效接触总载荷；接触位置在本对照中重新规定，不使用上方位置。"
    )
    st.caption(
        "两组均固定 25°C、无噪声，隔离接触分布的影响；上方温度不用于此对照，噪声仅作尺度参考。不修改主实验设置。"
    )
    st.markdown(
        "**先预测：** 同合力、同真实质心，是否意味着所有通道读数也相同？切换四测点阵列后再判断。"
    )
    for fig, key in zip(
        figures(case, single, double, width, height),
        ("skin_compare_layout", "skin_compare_response"),
    ):
        st.plotly_chart(fig, key=key, config={"displayModeBar": False}, width="stretch")
    st.dataframe(
        [
            {
                "通道数": c["n"],
                "响应差 RMS (nm)": c["rms"],
                "最大通道差 (nm)": c["maximum"],
                "估计质心间距 (mm)": float(
                    np.linalg.norm(
                        c["single"]["estimated_centroid_mm"]
                        - c["double"]["estimated_centroid_mm"]
                    )
                ),
            }
            for c in cases
        ],
        hide_index=True,
        column_config={
            name: st.column_config.NumberColumn(format="%.3e")
            for name in ("响应差 RMS (nm)", "最大通道差 (nm)", "估计质心间距 (mm)")
        },
        width="stretch",
    )
    if fraction == 0:
        st.info(
            "间距为零：两个载荷位置重合，已退化为同一个接触假设，响应相同不能作为两个分离接触不可辨识的证据。"
        )
    elif case["maximum"] < 1e-12:
        st.info(
            "本配置下，两种不同接触假设的通道响应在数值容差内相同。当前数据无法区分这两个假设；四角对称阵列与等载荷对称接触会出现这种情况。"
        )
    else:
        st.info(
            "本配置下，通道向量保留了两种假设的差异，尽管它们的合力和质心相同。这只区分了本次两个预设，不证明任意双点都可唯一恢复；当前算法仍只估计合力与质心。"
        )
    st.caption(
        f"当前主实验单通道噪声 σ={noise:.5f} nm；本对照响应差 RMS={case['rms']:.5f} nm。若两次测量噪声独立且同方差，差值噪声标准差为 √2σ={np.sqrt(2) * noise:.5f} nm。这里只提供量纲一致的尺度比较，不设置检测阈值、不计算识别率或置信度。"
    )
