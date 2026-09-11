"""Rejudge fixed synthetic records; decision thresholds never regenerate events."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from . import eskin

REPEATS = 20


@st.cache_data(show_spinner=False, max_entries=16)
def records(parameters):
    groups = []
    for noisy in (False, True):
        samples = []
        for i in range(REPEATS):
            r = eskin.simulate_dynamic_skin_event(
                **{
                    **parameters,
                    "noise_ratio": parameters["noise_ratio"] if noisy else 0.0,
                    "slip_threshold": 0.35,
                    "seed": parameters["seed"] + i,
                }
            )
            samples.append((r["peak_shear_ratio"], r["peak_centroid_speed_mm_s"]))
        groups.append(np.asarray(samples))
    return groups


def decisions(peaks, threshold):
    return (peaks[:, 0] > threshold) & (peaks[:, 1] > 2.0)


def figure(groups, selected):
    thresholds = sorted(set([0.15, 0.25, 0.35, 0.45, 0.55, 0.7, selected]))
    fig = go.Figure()
    for peaks, name, color in zip(
        groups, ("无噪声", "当前噪声"), ("#65d0c8", "#ed91bd")
    ):
        fig.add_scatter(
            x=thresholds,
            y=[100 * decisions(peaks, t).mean() for t in thresholds],
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            hovertemplate="阈值 %{x:.2f}<br>告警率 %{y:.1f}%<extra>"
            + name
            + "</extra>",
        )
    fig.add_vline(x=selected, line_color="#ffcf69", line_dash="dash")
    fig.update_layout(
        title="固定记录的阈值扫描",
        template="plotly_dark",
        height=340,
        xaxis_title="判定剪切比阈值",
        yaxis=dict(title="模拟记录告警率 (%)", range=[-5, 105]),
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=45, r=15, t=45, b=85),
    )
    return fig


def render(parameters):
    st.markdown("#### 固定记录的噪声与阈值对照")
    if not st.checkbox("展开动态判定对照", key="dynamic_compare_enabled"):
        st.caption("冻结事件生成条件，再独立扫描判定阈值，区分输入变化与规则变化。")
        return
    threshold = st.slider(
        "对照判定阈值（不重新生成事件）",
        0.15,
        0.7,
        0.35,
        0.01,
        key="dynamic_compare_threshold",
    )
    groups = records(parameters)
    baseline, current = [decisions(p, threshold) for p in groups]
    st.caption(
        f"模拟对照 · {parameters['event']}；{parameters['sample_rate_hz']} Hz；{parameters['duration_s']:.1f} s；目标法向力 {parameters['normal_force_n']:.1f} N；初始温度 {parameters['temperature_c']:.1f} °C。每组 {REPEATS} 条记录，种子 {parameters['seed']}～{parameters['seed'] + REPEATS - 1}；噪声分别为 0 与 {parameters['noise_ratio']:.3f}。"
    )
    st.caption(
        "生成端阈值固定 0.35，上方滑移阈值与重复次数不用于本对照。只改变本对照判定阈值时复用记录；更改事件、采样率、时长、力、温度、噪声或种子会生成新记录。主实验设置保持不变。"
    )
    st.markdown(
        "**先预测：** 对同一批记录提高阈值，告警数能否增加？速度条件没有越阈时，仅降低剪切比阈值有用吗？"
    )
    st.plotly_chart(
        figure(groups, threshold),
        key="dynamic_compare_chart",
        config={"displayModeBar": False},
        width="stretch",
    )
    st.dataframe(
        [
            {
                "组别": name,
                "剪切比峰值范围": f"{p[:, 0].min():.4f}～{p[:, 0].max():.4f}",
                "速度峰值范围 (mm/s)": f"{p[:, 1].min():.3f}～{p[:, 1].max():.3f}",
                "速度条件满足数": int((p[:, 1] > 2).sum()),
                "告警数": int(decisions(p, threshold).sum()),
                "总记录数": REPEATS,
            }
            for p, name in zip(groups, ("无噪声", "当前噪声"))
        ],
        hide_index=True,
        width="stretch",
    )
    added = int((~baseline & current).sum())
    removed = int((baseline & ~current).sum())
    st.info(
        f"当前阈值 {threshold:.2f}：无噪声告警 {baseline.sum()}/{REPEATS}，当前噪声告警 {current.sum()}/{REPEATS}；按同种子配对，噪声组新增告警 {added} 条、取消告警 {removed} 条。这是相对无噪声规则结果的变化，不是相对真实滑移标签的误报或漏报。"
    )
    st.caption(
        "判据沿用原模型：记录后 80% 窗口的剪切比峰值 > 阈值，且速度绝对值峰值 > 2 mm/s；两个峰值不必同时出现。扫描曲线只连接所评估阈值，不表示连续概率模型。无噪声组的记录相同，不是 20 次独立实物验证。"
    )
    st.markdown(
        "**再试：** 比较稳定按压与横向滑动，再提高噪声或采样率。速度由质心差分计算，噪声与采样间隔会影响速度峰值；有限重复告警率不是识别准确率，也不用于推荐真实设备阈值。"
    )
