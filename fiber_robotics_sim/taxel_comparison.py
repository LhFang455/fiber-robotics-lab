"""Paired reference-matching comparisons of the existing teaching model."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from . import eskin


@st.cache_data(show_spinner=False, max_entries=16)
def compare(parameters):
    cases = []
    for noisy in (False, True):
        for match in (0.8, 0.9, 1.0):
            p = {
                **parameters,
                "reference_match": match,
                "noise_pf": parameters["noise_pf"] if noisy else 0.0,
            }
            cases.append(
                dict(
                    noisy=noisy, match=match, result=eskin.simulate_triaxial_taxel(**p)
                )
            )
    return cases


def table(cases):
    return [
        {
            "噪声组": "当前噪声" if c["noisy"] else "无噪声",
            "参考匹配度": f"{c['match']:.0%}",
            "理论失配残差 (pF)": (1 - c["match"]) * c["result"]["common_mode_pf"],
            "未校正 MAE (N)": c["result"]["raw_mae_n"],
            "校正 MAE (N)": c["result"]["corrected_mae_n"],
            "Fx 误差 (N)": float(c["result"]["corrected_error_n"][0]),
            "Fy 误差 (N)": float(c["result"]["corrected_error_n"][1]),
            "Fz 误差 (N)": float(c["result"]["corrected_error_n"][2]),
        }
        for c in cases
    ]


def figure(cases):
    fig = go.Figure()
    for start, name, color in ((0, "无噪声", "#65d0c8"), (3, "当前噪声", "#ed91bd")):
        group = cases[start : start + 3]
        fig.add_scatter(
            x=[80, 90, 100],
            y=[c["result"]["corrected_mae_n"] for c in group],
            name=name + " · 校正后",
            mode="lines+markers",
            line=dict(color=color),
        )
        fig.add_scatter(
            x=[80, 90, 100],
            y=[c["result"]["raw_mae_n"] for c in group],
            name=name + " · 未校正",
            mode="lines",
            line=dict(color=color, dash="dash"),
        )
    fig.update_layout(
        title="参考匹配度与三轴平均绝对误差",
        template="plotly_dark",
        height=390,
        xaxis=dict(title="参考匹配度 (%)", tickvals=[80, 90, 100]),
        yaxis_title="力 MAE (N)",
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=45, r=15, t=45, b=110),
    )
    return fig


def render(parameters):
    st.markdown("#### 参考失配与校正对照实验")
    if not st.checkbox("展开参考匹配度对照", key="taxel_compare_enabled"):
        st.caption("固定受力、温度、曲率和应变，比较三种匹配度与两种噪声条件。")
        return
    cases = compare(parameters)
    r = cases[0]["result"]
    st.caption(
        f"模拟对照 · 力 {tuple(parameters[k] for k in ('fx_n', 'fy_n', 'fz_n'))} N；温度 {parameters['temperature_c']:.1f} °C；曲率 {parameters['curvature_per_m']:.2f} 1/m；应变 {parameters['strain_fraction'] * 1000:.2f} ‰。共模 {r['common_mode_pf']:+.4f} pF。"
    )
    st.caption(
        f"匹配度固定比较 80%、90%、100%，不使用上方匹配度设置；噪声比较 0 与当前 σ={parameters['noise_pf']:.4f} pF，种子 {parameters['seed']}。各匹配度复用同一组噪声样本，不是独立重复试验，也不计算成功率。不会改写上方参数。"
    )
    st.markdown(
        "**先预测：** 完全匹配是否意味着零误差？当共模很小时，参考相减是否仍然有益？"
    )
    st.plotly_chart(
        figure(cases),
        key="taxel_compare_chart",
        config={"displayModeBar": False},
        width="stretch",
    )
    st.dataframe(table(cases), hide_index=True, width="stretch")
    for c in cases[3:]:
        r = c["result"]
        difference = r["raw_mae_n"] - r["corrected_mae_n"]
        outcome = (
            "基本不变"
            if np.isclose(difference, 0, atol=1e-10)
            else ("降低" if difference > 0 else "增加")
        )
        st.write(
            f"• 当前噪声、匹配度 {c['match']:.0%}：校正后 MAE {r['corrected_mae_n']:.5f} N，相对未校正{outcome}"
            + ("。" if outcome == "基本不变" else f" {abs(difference):.5f} N。")
        )
    st.info(
        "解释：校正残差 = (1−匹配度)×共模 + 主动噪声 − 参考噪声。无噪声且完全匹配时，模型可消除共模；有噪声时，参考相减还会引入参考噪声。因此本次最小误差不等于通用最优匹配度，偶然抵消也不是标定改善。"
    )
    st.caption(
        "再试：把温度设为 25°C、曲率和应变设为 0，保持非零噪声，观察参考相减是否仍然降低误差；再恢复干扰条件进行比较。本实验没有真实器件标定或统计置信区间。"
    )
