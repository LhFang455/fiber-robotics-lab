"""Link three core channels to a section diagram and computed centreline nodes."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from . import models
from .foot_inspector import selection_value


def core_registry(record):
    radius = record["parameters"]["core_radius_um"]
    return [dict(id=f"CORE-{i+1:02d}", channel=i+1, angle=float(a),
                 local_um=(float(radius*np.cos(np.deg2rad(a))), float(radius*np.sin(np.deg2rad(a)))))
            for i,a in enumerate(record["results"]["core_angles_deg"])]


def differential_strain(record):
    shifts = np.asarray(record["results"]["wavelength_shifts_nm"],dtype=float)
    if not np.all(np.isfinite(shifts)):
        return [None]*3
    strain = models._strain_from_shift(shifts,0.)
    return ((strain-strain.mean())*1e6).tolist()


def section_figure(record, selected):
    cores = core_registry(record)
    radius = record["parameters"]["core_radius_um"]
    fig = go.Figure()
    fig.add_shape(type="circle",x0=-radius,y0=-radius,x1=radius,y1=radius,
                  line=dict(color="#617e93",dash="dot"),layer="below")
    fig.add_scatter(x=[c["local_um"][0] for c in cores],y=[c["local_um"][1] for c in cores],
        customdata=list(range(3)),mode="markers+text",text=[c["id"] for c in cores],
        textposition=["top left","top right","bottom right"],
        marker=dict(size=22,color=["#ffcf69" if i==selected else "#65d0c8" for i in range(3)]),
        unselected=dict(marker=dict(opacity=1)),hovertemplate="x=%{x:.1f} μm<br>y=%{y:.1f} μm<extra></extra>")
    fig.update_layout(title="三芯横截面 · 点击纤芯选取",template="plotly_dark",height=340,
        showlegend=False,clickmode="event+select",margin=dict(l=40,r=20,t=45,b=40),
        xaxis=dict(title="局部 x (μm)",range=[-radius*1.55,radius*1.55]),
        yaxis=dict(title="局部 y (μm)",range=[-radius*1.55,radius*1.55],scaleanchor="x"))
    return fig


def position_figure(record, index, reference=True):
    r=record["results"]
    fig=go.Figure()
    series=[("反演中心线",r["estimated_centerline_xyz_mm"],"#ffcf69")]
    if reference:
        series.insert(0,("模型参考中心线",r["true_centerline_xyz_mm"],"#66d2c7"))
    for name,points,color in series:
        p=np.asarray(points)
        fig.add_scatter3d(x=p[:,0],y=p[:,1],z=p[:,2],mode="lines",name=name,line=dict(color=color,width=5))
        fig.add_scatter3d(x=[p[index,0]],y=[p[index,1]],z=[p[index,2]],mode="markers",
            name=f"{name} · 节点 {index:03d}",marker=dict(color=color,size=6),showlegend=False,
            hovertemplate="x=%{x:.2f} mm<br>y=%{y:.2f} mm<br>z=%{z:.2f} mm<extra></extra>")
    fig.update_layout(title=f"三维计算节点 S-{index:03d}",template="plotly_dark",height=390,
        scene=dict(aspectmode="data",xaxis_title="x (mm)",yaxis_title="y (mm)",zaxis_title="z (mm)"),
        legend=dict(orientation="h"),margin=dict(l=0,r=0,t=40,b=0),uirevision="shape-inspector-camera")
    return fig


def error_figure(record,index):
    error=record["results"]["point_error_mm"]
    distance=np.linspace(0,record["parameters"]["length_mm"],len(error))
    fig=go.Figure(go.Scatter(x=distance,y=error,customdata=list(range(len(error))),mode="lines",
        marker=dict(size=8),line=dict(color="#80c9f2"),connectgaps=False,hoverinfo="skip"))
    picks=sorted(set(range(0,len(error),10)) | {index,len(error)-1})
    fig.add_scatter(x=distance[picks],y=[error[i] for i in picks],customdata=picks,
        mode="markers",marker=dict(size=10,color="#80c9f2"),showlegend=False,
        hovertemplate="节点 S-%{customdata}<br>s=%{x:.2f} mm<br>误差=%{y:.4f} mm<extra></extra>")
    fig.add_vline(x=distance[index],line_color="#ffcf69",line_dash="dash")
    fig.update_layout(title="沿程空间误差 · 点击计算节点",showlegend=False,template="plotly_dark",height=340,
        xaxis_title="模型弧长 s (mm)",yaxis_title="逐点空间误差 (mm)",yaxis=dict(tickformat=".2e"),clickmode="event+select",
        margin=dict(l=50,r=20,t=45,b=40))
    return fig


def render(record):
    st.markdown("#### 三芯与沿程形状观察器")
    st.caption("模拟数据 · 三芯是输入通道，S-000～S-160 是求解器计算节点，不能当作 161 组实测 FBG。截面采用当前模型的芯半径与 0°/120°/240° 布置；虚圆仅表示芯中心所在半径，不是包层边界。")
    st.session_state.setdefault("shape_inspect_core",0)
    st.session_state.setdefault("shape_inspect_node",80)

    def pick_core():
        v=selection_value(st.session_state.get("shape_section_pick",{}))
        if isinstance(v,int) and 0<=v<3:
            st.session_state.shape_inspect_core=v

    def pick_node():
        v=selection_value(st.session_state.get("shape_error_pick",{}))
        if isinstance(v,int) and 0<=v<len(record["results"]["point_error_mm"]):
            st.session_state.shape_inspect_node=v

    selected=st.selectbox("查看纤芯",range(3),format_func=lambda i:f"CORE-{i+1:02d} · 通道 {i+1}",key="shape_inspect_core")
    core=core_registry(record)[selected]
    shifts=record["results"]["wavelength_shifts_nm"]
    differential=differential_strain(record)[selected]
    a,b=st.columns(2)
    a.metric("所选芯波长 Δλ",f"{shifts[selected]:.4f} nm" if np.isfinite(shifts[selected]) else "缺测")
    b.metric("去共模差分应变",f"{differential:+.2f} με" if differential is not None else "无法求解")
    st.caption(f"{core['id']} → 通道 {core['channel']}，截面角 {core['angle']:.0f}°，芯中心 ({core['local_um'][0]:.1f}, {core['local_um'][1]:.1f}) μm。差分应变由三芯波长共同计算，包含噪声及芯间温差影响，不能当成独立实测机械应变。")
    left,right=st.columns(2)
    with left:
        st.plotly_chart(section_figure(record,selected),key="shape_section_pick",on_select=pick_core,
            selection_mode="points",config={"displayModeBar":False})
    with right:
        index=st.slider("沿程计算节点",0,len(record["results"]["point_error_mm"])-1,key="shape_inspect_node")
        st.plotly_chart(error_figure(record,index),key="shape_error_pick",on_select=pick_node,
            selection_mode="points",config={"displayModeBar":False})
    s=record['parameters']['length_mm']*index/(len(record['results']['point_error_mm'])-1)
    error=record['results']['point_error_mm'][index]
    if max(record["results"]["point_error_mm"]) < 1e-9:
        st.caption("当前误差处于浮点舍入量级（小于 1e-9 mm），不应解读为实物测量精度；可调高芯间温度梯度观察可见偏差。")
    st.caption(f"选中 S-{index:03d} · 模型弧长 {s:.2f} mm · 空间误差 {error:.4f} mm。金色竖线与三维亮点对应同一计算节点；曲线保留全部计算值，圆点每 10 个节点显示一次并包含当前节点，滑杆可逐节点选择。")
    reference=st.checkbox("显示模型参考中心线",value=True,key="shape_inspect_reference")
    st.plotly_chart(position_figure(record,index,reference),key="shape_inspector_3d",config={"scrollZoom":False})
    r=record['results']
    st.caption(f"重建输入曲率：{r['estimated_curvature_per_m']:.3f} 1/m（沿用恒定参数）；扭转率 {record['parameters']['twist_per_m']:.2f} 1/m 是已知先验，并非三芯读数测得。此处沿用原中心线积分器，不提供经过验证的局部曲率场或力学形变。三维图可拖动旋转、双击复位视角，滚轮缩放关闭。")
