"""Native, selectable six-zone inspection; original model coordinates and data."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from . import experiments


ZONES = tuple(dict(id=f"SOLE-{i+1:02d}", channel=i+1,
                   part="前掌" if i < 3 else "脚跟",
                   local=(i % 3 + .45, 1 - i // 3 + .42, 0.)) for i in range(6))


def observation(record, zone):
    result = record["results"]
    failed = record["parameters"].get("failed_zone") == zone + 1
    value = result["estimated_zone_loads_n"][zone]
    return None if failed or not np.isfinite(value) else float(value)


@st.cache_data(show_spinner=False, max_entries=12)
def phase_scan(parameters):
    # Same seeded noise at each phase: controlled comparison, not independent sampling.
    phases = sorted(set(range(0, 101, 5)) | {int(parameters["phase_percent"])})
    return [experiments.run_foot_experiment(dict(parameters, phase_percent=p)) for p in phases]


def spatial_figure(record, selected):
    values = [observation(record, i) for i in range(6)]
    fig = go.Figure()
    for i, zone in enumerate(ZONES):
        x, y, _ = zone["local"]
        fig.add_shape(type="rect", layer="below", x0=x-.44, x1=x+.44, y0=y-.4, y1=y+.4,
                      fillcolor="#172c3d", line_color="#ffcf69" if i == selected else "#456075",
                      line_width=3 if i == selected else 1)
    valid = [i for i, v in enumerate(values) if v is not None]
    fig.add_scatter(x=[ZONES[i]["local"][0] for i in valid], y=[ZONES[i]["local"][1] for i in valid],
        customdata=valid, mode="markers+text", text=[ZONES[i]["id"] for i in valid],
        unselected=dict(marker=dict(opacity=1), textfont=dict(color="#e0eef4")),
        textposition="top center", marker=dict(size=35, color=[values[i] for i in valid],
        colorscale="Tealrose", cmin=0, cmax=max(1., record["parameters"]["load_n"], *[v for v in values if v is not None]),
        colorbar=dict(title="载荷 N")), hovertext=[f"FBG {i+1} · {v:.2f} N" if v is not None else f"FBG {i+1} · 无有效观测" for i,v in enumerate(values) if i in valid],
        hovertemplate="%{hovertext}<extra></extra>", name="区域")
    invalid = [i for i,v in enumerate(values) if v is None]
    if invalid:
        fig.add_scatter(x=[ZONES[i]["local"][0] for i in invalid], y=[ZONES[i]["local"][1] for i in invalid],
            customdata=invalid, mode="markers+text",text=[ZONES[i]["id"] for i in invalid],textposition="top center", marker=dict(size=35,color="#88939d",symbol="x"),
            name="无有效观测", hovertemplate="已知失效或缺测<extra></extra>")
    if record["results"]["reliable_cop"] and not invalid:
        xy = record["results"]["estimated_cop_xy"]
        fig.add_scatter(x=[xy[0]], y=[xy[1]],mode="markers",marker=dict(symbol="cross",color="white",size=17),name="CoP（反演）")
    fig.update_layout(title="六区空间位置 · 点击区域选择", height=370, template="plotly_dark",
        clickmode="event+select", showlegend=False, margin=dict(l=15,r=15,t=45,b=30),
        xaxis=dict(range=[-.15,3.25],visible=False), yaxis=dict(range=[-.25,2.2],scaleanchor="x",
        tickvals=[.42,1.42],ticktext=["脚跟","前掌"]), uirevision="sole-layout")
    return fig


def phase_figure(records, selected, phase):
    fig = go.Figure()
    x = [r["parameters"]["phase_percent"] for r in records]
    fig.add_scatter(x=x,y=[r["results"]["true_zone_loads_n"][selected] for r in records],
        customdata=x,mode="lines+markers",name="模型参考载荷",line=dict(color="#75d9ca"))
    fig.add_scatter(x=x,y=[observation(r,selected) for r in records],customdata=x,
        mode="lines+markers",name="FBG 反演（有效观测）",connectgaps=False,line=dict(color="#ffcf69",dash="dot"))
    fig.add_vline(x=phase,line_dash="dash",line_color="white")
    fig.update_layout(title=f"{ZONES[selected]['id']} · 点击采样点切换相位",height=370,
        template="plotly_dark",xaxis_title="规定步态相位 (%)",yaxis_title="区域载荷 (N)",
        clickmode="event+select",legend=dict(orientation="h",y=-.25),margin=dict(l=40,r=15,t=45,b=70),
        uirevision=f"sole-phase-{selected}")
    return fig


def selection_value(event):
    points = event.get("selection", {}).get("points", [])
    return points[-1].get("customdata") if points else None


def render(record):
    st.markdown("#### 足底区域与通道观察器")
    st.caption("模拟数据 · SOLE-01～03 为前掌，04～06 为脚跟；位置沿用原模型归一化坐标，不代表毫米尺寸或实物封装。")
    st.session_state.setdefault("foot_inspector_zone", 0)

    def select_zone():
        value = selection_value(st.session_state.get("foot_space_selection", {}))
        if isinstance(value, int) and 0 <= value < 6:
            st.session_state.foot_inspector_zone = value

    def select_phase():
        value = selection_value(st.session_state.get("foot_phase_selection", {}))
        if isinstance(value, (int, float)) and 0 <= value <= 100:
            st.session_state.foot_phase = int(value)

    selected = st.selectbox("查看足底区域", range(6),key="foot_inspector_zone",
        format_func=lambda i:f"{ZONES[i]['id']} · {ZONES[i]['part']} · FBG {i+1}")
    zone = ZONES[selected]
    value = observation(record, selected)
    result = record["results"]
    a,b,c = st.columns(3)
    a.metric("模型参考载荷",f"{result['true_zone_loads_n'][selected]:.2f} N")
    b.metric("有效反演载荷", "无有效观测" if value is None else f"{value:.2f} N")
    shift = result['wavelength_shifts_nm'][selected]
    c.metric("通道波长 Δλ",f"{shift:.4f} nm" if np.isfinite(shift) else "缺测")
    if value is None:
        st.warning("所选区域已知失效或缺测：灰色叉号与曲线空缺表示无有效观测。已知失效时波长栏保留模型注入的替代读数，不能据此认定该区域载荷为零。")
    st.caption(f"{zone['id']} → FBG {zone['channel']}；局部位置 {zone['local']}，受力方向为足底法向。当前相位 {record['parameters']['phase_percent']:g}%。金色边框为选中区域，白色十字为可靠条件下的反演 CoP。")
    left,right = st.columns(2)
    with left:
        st.plotly_chart(spatial_figure(record,selected),key="foot_space_selection",on_select=select_zone,
            selection_mode="points",config={"displayModeBar":False})
    with right:
        st.plotly_chart(phase_figure(phase_scan(record['parameters']),selected,record['parameters']['phase_percent']),
            key="foot_phase_selection",on_select=select_phase,selection_mode="points",config={"displayModeBar":False})
    st.caption("相位扫描固定载荷、地形、温度、噪声与随机种子，每 5% 计算一次并包含当前相位；连线仅连接计算点，不是连续测量或真实时间记录。点击曲线采样点会更新本页步态相位；区域下拉框也可用键盘选择。已知失效不显示有效反演曲线，低载荷或失效时不绘制可靠 CoP。")
