"""Channel-linked inspection of the existing triaxial teaching matrix."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from .foot_inspector import selection_value


def contributions(result):
    matrix = np.asarray(result["sensitivity_matrix"])
    forward = matrix * np.asarray(result["forces_true_n"])[None, :]
    inverse = np.linalg.pinv(matrix) * np.asarray(result["corrected_pf"])[None, :]
    return forward, inverse


def figures(result, selected):
    labels = [f"TAXEL-C{i + 1}" for i in range(5)]
    layout = go.Figure()
    profile = go.Figure()
    for stage, (key, name, color) in enumerate(
        (
            ("active_pf", "主动信号", "#c19bef"),
            ("reference_pf", "参考信号", "#ed91bd"),
            ("corrected_pf", "校正信号", "#65d0c8"),
        )
    ):
        values = [float(v) if np.isfinite(v) else None for v in result[key]]
        layout.add_scatter(
            x=[stage] * 5,
            y=list(range(1, 6)),
            customdata=list(range(5)),
            mode="markers+text",
            text=labels,
            textposition="top center",
            name=name,
            marker=dict(
                size=16, color=["#ffcf69" if i == selected else color for i in range(5)]
            ),
            hovertext=["缺测" if v is None else f"{v:+.4f} pF" for v in values],
            hovertemplate="%{text}<br>%{hovertext}<extra>" + name + "</extra>",
            unselected=dict(marker=dict(opacity=1)),
        )
        profile.add_scatter(
            x=list(range(1, 6)),
            y=values,
            customdata=list(range(5)),
            mode="lines+markers",
            name=name,
            line=dict(color=color),
            connectgaps=False,
            hovertemplate="C%{x}<br>%{y:.4f} pF<extra>" + name + "</extra>",
        )
    layout.update_layout(
        title="同一触觉单元的五通道关系图",
        template="plotly_dark",
        height=420,
        showlegend=False,
        clickmode="event+select",
        xaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["主动", "参考", "主动 − 参考"],
            range=[-0.5, 2.5],
            fixedrange=True,
        ),
        yaxis=dict(visible=False, range=[0.4, 5.6], fixedrange=True),
        margin=dict(l=10, r=10, t=45, b=40),
    )
    profile.add_vline(x=selected + 1, line_color="#ffcf69", line_dash="dash")
    profile.update_layout(
        title="电容变化 · 点击通道圆点",
        template="plotly_dark",
        height=320,
        clickmode="event+select",
        xaxis=dict(title="通道编号", dtick=1),
        yaxis_title="电容变化 ΔC (pF)",
        legend=dict(orientation="h", y=-0.3),
        margin=dict(l=45, r=15, t=45, b=80),
    )
    return layout, profile


def render(result):
    st.markdown("#### 三轴触觉通道观察器")
    st.caption(
        "模拟数据 · 下图按通道编号排布，不表示电极实物位置、间距或内部结构。三列是主动、参考与相减结果，不是三个独立器件。"
    )
    st.session_state.setdefault("taxel_inspect_channel", 0)
    selected = st.selectbox(
        "查看三轴触觉通道",
        range(5),
        format_func=lambda i: f"TAXEL-C{i + 1}",
        key="taxel_inspect_channel",
    )

    def pick(key):
        value = selection_value(st.session_state.get(key, {}))
        if isinstance(value, int) and 0 <= value < 5:
            st.session_state.taxel_inspect_channel = value

    cols = st.columns(3)
    for col, key, label in zip(
        cols,
        ("active_pf", "reference_pf", "corrected_pf"),
        ("主动 ΔC", "参考 ΔC", "校正 ΔC"),
    ):
        v = result[key][selected]
        col.metric(label, f"{v:+.4f} pF" if np.isfinite(v) else "缺测")
    for fig, key in zip(
        figures(result, selected), ("taxel_space_pick", "taxel_profile_pick")
    ):
        st.plotly_chart(
            fig,
            key=key,
            on_select=lambda key=key: pick(key),
            selection_mode="points",
            config={"displayModeBar": False},
        )
    forward, inverse = contributions(result)

    def formatted(v):
        return f"{v:+.5f}" if np.isfinite(v) else "缺测"

    st.markdown(f"**TAXEL-C{selected + 1}：从受力到电容，再到力反演**")
    st.dataframe(
        [
            {
                "分量": axis,
                "灵敏度 (pF/N)": formatted(result["sensitivity_matrix"][selected, i]),
                "该力产生的机械信号 (pF)": formatted(forward[selected, i]),
                "该通道对校正反演力的贡献 (N)": formatted(inverse[i, selected]),
            }
            for i, axis in enumerate(("Fx", "Fy", "Fz"))
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "前三轴机械信号相加后，再叠加共模与主动噪声得到主动信号；参考相减不一定能消除全部干扰。反演贡献使用原矩阵的伪逆，五个通道的贡献相加才得到某一轴估计力，单通道不能独立确定三轴力。负号表示响应方向或代数贡献，不是负的绝对电容。"
    )
    residual = result["corrected_pf"][selected] - forward[selected].sum()
    st.info(
        f"已知模型共模：{result['common_mode_pf']:+.4f} pF；该通道校正后相对理想机械响应的残差：{formatted(residual)} pF。残差包含参考失配与两路噪声，不能仅凭它认定真实器件故障。灵敏度与通道关系均为教学设置，尚无实物位置和标定依据。"
    )
