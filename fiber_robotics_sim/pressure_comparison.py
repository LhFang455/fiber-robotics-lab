"""Controlled, noiseless comparisons using the existing pressure model."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from . import eskin


@st.cache_data(show_spinner=False, max_entries=16)
def compare(scenario, peak, output):
    return [
        dict(
            size=n,
            bandwidth=b,
            result=eskin.simulate_pressure_reconstruction(
                scenario,
                n,
                output,
                peak_pressure_kpa=peak,
                bandwidth=b,
                noise_kpa=0,
                seed=7,
            ),
        )
        for n in (4, 8)
        for b in (0.08, 0.24)
    ]


def label(case):
    return f"{case['size']}×{case['size']} · 带宽 {case['bandwidth']:.2f}"


def rows(cases):
    return [
        {
            "方案": label(c),
            "通道数": c["size"] ** 2,
            "采样峰值 (kPa)": float(c["result"]["sparse_samples_kpa"].max()),
            "重建峰值 (kPa)": float(c["result"]["reconstruction_kpa"].max()),
            "峰值绝对误差 (%)": c["result"]["peak_error_pct"],
            "全场 RMSE (kPa)": c["result"]["rmse_kpa"],
            "质心误差 (%)": c["result"]["centroid_error_pct"],
        }
        for c in cases
    ]


def profile(cases):
    truth = cases[0]["result"]["truth_kpa"]
    row, col = np.unravel_index(np.argmax(truth), truth.shape)
    axis = np.linspace(0, 1, len(truth))
    fig = go.Figure(
        go.Scatter(
            x=axis, y=truth[row], name="模拟参考场", line=dict(color="white", width=3)
        )
    )
    for c, color in zip(cases, ("#65d0c8", "#ed91bd", "#ffcf69", "#a798ef")):
        fig.add_scatter(
            x=axis,
            y=c["result"]["reconstruction_kpa"][row],
            name=label(c),
            line=dict(color=color),
            mode="lines+markers",
        )
    fig.update_layout(
        title=f"参考峰值所在行 · y={axis[row]:.3f}",
        template="plotly_dark",
        height=460,
        xaxis_title="归一化 x",
        yaxis_title="压力 (kPa)",
        legend=dict(orientation="h", y=-0.25),
        margin=dict(l=45, r=15, t=45, b=130),
    )
    return fig, int(row), int(col)


def render(scenario, peak, output):
    st.markdown("#### 压力重建对照实验")
    if not st.checkbox("展开采样密度与带宽对照", key="pressure_compare_enabled"):
        st.caption("在相同参考场下比较四种方案，不修改上方参数。")
        return
    st.caption(
        f"模拟对照 · 场景：{scenario}；峰值参数 {peak:.1f} kPa；共同输出网格 {output}×{output}；各组噪声固定为 0 kPa。上方噪声、采样密度和带宽不用于本对照；场景、峰值和输出网格变化会重新计算。"
    )
    st.markdown(
        "**先预测：** 增加通道是否能捕获原来遗漏的峰值？同一批采样扩大核带宽后，峰值和全场误差是否一起改善？"
    )
    cases = compare(scenario, peak, output)
    fig, row, col = profile(cases)
    truth = cases[0]["result"]["truth_kpa"]
    st.caption(
        f"共同参考网格峰值：{truth.max():.3f} kPa。峰值参数是生成函数的幅度设置，不一定等于离散网格最大值；下图剖面使用参考峰值选行，仅用于评估，不向重建算法提供真值。"
    )
    st.plotly_chart(
        fig,
        key="pressure_compare_profile",
        width="stretch",
        config={"displayModeBar": False},
    )
    st.dataframe(rows(cases), hide_index=True, width="stretch")
    best = min(cases, key=lambda c: c["result"]["rmse_kpa"])
    st.markdown(
        f"**本次结果：** 四组中全场 RMSE 最小的是 {label(best)}（{best['result']['rmse_kpa']:.3f} kPa）。这只说明本工况下的误差，不是通用最优参数。"
    )
    for start in (0, 2):
        narrow, wide = cases[start : start + 2]
        a, b = narrow["result"], wide["result"]
        st.write(
            f"• {narrow['size']}×{narrow['size']}：采样峰值 {a['sparse_samples_kpa'].max():.3f} kPa；带宽 0.08 → 0.24 时，重建峰值 {a['reconstruction_kpa'].max():.3f} → {b['reconstruction_kpa'].max():.3f} kPa，全场 RMSE {a['rmse_kpa']:.3f} → {b['rmse_kpa']:.3f} kPa。"
        )
    st.info(
        "解释：当前重建是非负、归一化的加权平均，不能生成高于采样最大值的压力峰。若稀疏采样没覆盖窄峰，仅加密输出网格无法恢复它。此处没有接触检测阈值或漏检标签，因此评价峰值损失，不将其称为实测漏检率。"
    )
    st.caption(
        "再试：保持场景不变比较两个密度；然后切换边缘／双点场景，看排序是否改变。4×4 与 8×8 不是嵌套采样网格，位置也发生变化；增加通道的收益不能只归因于通道数量。"
    )
