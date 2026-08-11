"""Plotly visualisations for the learning lab."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


COLORS = {"truth": "#17a2b8", "estimate": "#ff7f0e", "sensor": "#6f42c1"}


def _base_layout(title: str, xaxis: str, yaxis: str) -> dict:
    return {"title": title, "xaxis_title": xaxis, "yaxis_title": yaxis, "template": "plotly_white", "height": 410}


def finger_figure(result: dict) -> go.Figure:
    line = np.asarray(result["centerline_xy_mm"])
    sensors = np.asarray(result["sensor_positions_mm"])
    sensor_y = np.interp(sensors, line[:, 0], line[:, 1])
    figure = go.Figure()
    figure.add_scatter(x=line[:, 0], y=line[:, 1], mode="lines", name="柔性手指", line={"color": COLORS["truth"], "width": 6})
    figure.add_scatter(x=sensors, y=sensor_y, mode="markers+text", text=["FBG 1", "FBG 2", "FBG 3"], textposition="top center", name="FBG", marker={"size": 11, "color": COLORS["sensor"]})
    figure.update_layout(**_base_layout("常曲率柔性手指", "长度方向 (mm)", "横向位移 (mm)"))
    figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def sensor_bar_figure(positions: np.ndarray, shifts: np.ndarray, title: str) -> go.Figure:
    figure = go.Figure(go.Bar(x=[f"FBG {index + 1}" for index in range(len(positions))], y=shifts, marker_color=COLORS["sensor"], text=[f"{value:.4f}" for value in shifts], textposition="outside"))
    figure.update_layout(**_base_layout(title, "传感器", "波长漂移 Δλ (nm)"))
    return figure


def demodulation_figure(result: dict) -> go.Figure:
    """Render raw, filtered and temperature-compensated wavelength signals on time."""
    time_s = np.asarray(result["time_s"], dtype=float)
    figure = go.Figure()
    for name, key, color in (
        ("原始波长漂移", "raw_wavelength_nm", "#8ca0b3"),
        ("滤波后波长漂移", "filtered_wavelength_nm", "#ff9f43"),
        ("温补后机械信号", "compensated_wavelength_nm", "#17a2b8"),
    ):
        figure.add_scatter(x=time_s, y=result[key], mode="lines", name=name, line={"color": color, "width": 2})
    figure.update_layout(**_base_layout("FBG 解调与温度补偿数据流", "时间 (s)", "波长变化 (nm)"))
    return figure


def distributed_curve_figure(result: dict, mechanism: str) -> go.Figure:
    """Render a spatial distributed-sensing curve using the primary available quantity."""
    x = np.asarray(result["position_mm"], dtype=float)
    if mechanism == "Brillouin":
        y, label = result["brillouin_frequency_ghz"], "Brillouin 频移 (GHz)"
    elif mechanism == "Raman":
        y, label = result["temperature_c"], "温度 (°C)"
    else:
        y, label = result["strain_ue"], "应变 (με)"
    figure = go.Figure(go.Scatter(x=x, y=y, mode="lines", line={"color": COLORS["truth"], "width": 3}, name=mechanism))
    figure.update_layout(**_base_layout(f"{mechanism} 分布式测量", "光纤位置 (mm)", label))
    return figure


def das_event_figure(result: dict) -> go.Figure:
    """Render a phi-OTDR/DAS time-distance event map."""
    figure = go.Figure(go.Heatmap(x=result["position_mm"], y=result["time_s"], z=result["amplitude"], colorscale="Magma", colorbar={"title": "振动幅值"}))
    figure.update_layout(title="φ-OTDR / DAS：时间—距离振动事件", template="plotly_dark", height=390, xaxis_title="光纤位置 (mm)", yaxis_title="时间 (s)")
    return figure


def polarization_figure(result: dict) -> go.Figure:
    """Render a normalised Stokes vector on a compact Poincare-sphere view."""
    stokes = np.asarray(result["stokes"], dtype=float)
    figure = go.Figure(go.Scatter3d(x=[0.0, stokes[0]], y=[0.0, stokes[1]], z=[0.0, stokes[2]], mode="lines+markers", line={"color": "#17a2b8", "width": 7}, marker={"size": 5}, name="Stokes 向量"))
    figure.update_layout(title="偏振态：Stokes 向量 / 庞加莱球坐标", template="plotly_white", height=420, scene={"xaxis_title": "S1", "yaxis_title": "S2", "zaxis_title": "S3", "xaxis": {"range": [-1, 1]}, "yaxis": {"range": [-1, 1]}, "zaxis": {"range": [-1, 1]}, "aspectmode": "cube"})
    return figure


def efpi_figure(result: dict) -> go.Figure:
    """Render the simulated EFPI interference spectrum."""
    figure = go.Figure(go.Scatter(x=result["wavelength_nm"], y=result["intensity"], mode="lines", line={"color": "#6f42c1", "width": 2}, name="干涉强度"))
    figure.update_layout(**_base_layout("EFPI：腔长变化引起的干涉谱", "波长 (nm)", "归一化反射强度"))
    return figure


def rayleigh_heatmap_figure(result: dict) -> go.Figure:
    """Plot continuous Rayleigh/OFDR strain over the five finger branches."""
    figure = go.Figure(go.Heatmap(
        x=result["distance_mm"],
        y=["拇指", "食指", "中指", "无名指", "小指"],
        z=result["rayleigh_strain_ue"],
        colorscale="Viridis",
        colorbar={"title": "应变 (με)"},
        hovertemplate="%{y}<br>位置 %{x:.1f} mm<br>应变 %{z:.1f} με<extra>Rayleigh/OFDR</extra>",
    ))
    figure.update_layout(title="Rayleigh/OFDR：五指连续应变分布（教学仿真）", template="plotly_dark", height=360, xaxis_title="每根手指上的光纤位置 (mm)", yaxis_title="光纤支路", margin={"l": 20, "r": 20, "t": 55, "b": 45})
    return figure


def das_heatmap_figure(result: dict) -> go.Figure:
    """Plot a DAS teaching waterfall for transient touch and slip events."""
    figure = go.Figure(go.Heatmap(
        x=result["das_distance_mm"],
        y=result["das_time_ms"],
        z=result["das_amplitude"],
        colorscale="Magma",
        colorbar={"title": "相对振动"},
        hovertemplate="时间 %{y:.0f} ms<br>光纤位置 %{x:.1f} mm<br>相对振动 %{z:.3f}<extra>DAS/φ-OTDR</extra>",
    ))
    figure.update_layout(title="DAS/φ-OTDR：抓取接触瞬态（教学仿真）", template="plotly_dark", height=360, xaxis_title="五条光纤支路串联位置 (mm)", yaxis_title="相对事件时间 (ms)", margin={"l": 20, "r": 20, "t": 55, "b": 45})
    return figure


def contact_figure(result: dict, estimated_position_mm: float) -> go.Figure:
    positions = np.asarray(result["sensor_positions_mm"])
    figure = go.Figure()
    figure.add_scatter(x=positions, y=result["strain"] * 1e6, mode="lines+markers", name="各 FBG 应变", line={"color": COLORS["sensor"]})
    figure.add_vline(x=result["contact_position_mm"], line_dash="dash", line_color=COLORS["truth"], annotation_text="真实接触点")
    figure.add_vline(x=estimated_position_mm, line_dash="dot", line_color=COLORS["estimate"], annotation_text="反演接触点")
    figure.update_layout(**_base_layout("局部接触引起的应变分布", "指尖位置 (mm)", "应变 (με)"))
    return figure


def multicore_figure(result: dict) -> go.Figure:
    truth = np.asarray(result["centerline_xyz_mm"])
    estimate = np.asarray(result["estimated_centerline_xyz_mm"])
    figure = go.Figure()
    figure.add_scatter3d(x=truth[:, 0], y=truth[:, 1], z=truth[:, 2], mode="lines", name="真实形状", line={"color": COLORS["truth"], "width": 7})
    figure.add_scatter3d(x=estimate[:, 0], y=estimate[:, 1], z=estimate[:, 2], mode="lines", name="反演形状", line={"color": COLORS["estimate"], "width": 4, "dash": "dash"})
    figure.update_layout(title="三芯光纤 3D 中心线", template="plotly_white", height=480, scene={"xaxis_title": "X (mm)", "yaxis_title": "Y (mm)", "zaxis_title": "Z (mm)", "aspectmode": "data"})
    return figure


def _arm_joint_coordinates(action: str) -> np.ndarray:
    """Return base, elbow, wrist and hand locations for a named teaching action."""
    poses_deg = {
        "抬臂": (72.0, -50.0, -12.0),
        "伸手": (18.0, 2.0, 0.0),
        "抓取": (38.0, -58.0, 18.0),
        "按压": (20.0, -68.0, -18.0),
        "松开": (32.0, -35.0, 10.0),
        "复位": (45.0, -60.0, 15.0),
    }
    angles = np.deg2rad(poses_deg.get(action, poses_deg["复位"]))
    lengths = (3.5, 3.0, 1.25)
    points = [np.array([0.0, 0.0])]
    direction = 0.0
    for length, angle in zip(lengths, angles):
        direction += angle
        points.append(points[-1] + length * np.array([np.cos(direction), np.sin(direction)]))
    return np.asarray(points)


def _arm_fiber_route(joints: np.ndarray, route: str) -> np.ndarray:
    """Place the selected FBG route relative to the current arm pose."""
    elbow, wrist, hand = joints[1], joints[2], joints[3]
    hand_direction = (hand - wrist) / np.linalg.norm(hand - wrist)
    routes = {
        "手指背侧": np.vstack([wrist + .45 * (hand - wrist), hand, hand + .45 * hand_direction]),
        "指尖": np.vstack([hand - .10 * hand_direction, hand + .25 * hand_direction, hand + .55 * hand_direction]),
        "手掌": np.vstack([wrist + .18 * (hand - wrist), wrist + .72 * (hand - wrist), hand + .15 * hand_direction]),
        "前臂": np.vstack([elbow + .15 * (wrist - elbow), elbow + .55 * (wrist - elbow), wrist]),
    }
    return routes[route]


def dexterous_hand_pose(
    action: str,
    joint_angles_deg: tuple[float, float, float] | None = None,
    finger_curls_deg: tuple[float, float, float, float, float] | None = None,
    wrist_rotation_deg: float = 0.0,
    planar_translation: tuple[float, float] = (0.0, 0.0),
) -> dict[str, np.ndarray | list[np.ndarray] | float]:
    """Return a planar five-finger hand pose tied to the selected arm action."""
    arm_joints = _arm_joint_coordinates(action) if joint_angles_deg is None else _arm_joint_coordinates_from_angles(joint_angles_deg)
    arm_joints = arm_joints + np.asarray(planar_translation, dtype=float)
    wrist, hand = arm_joints[2], arm_joints[3]
    forward = (hand - wrist) / np.linalg.norm(hand - wrist)
    lateral = np.array([-forward[1], forward[0]])
    palm_center = hand + .35 * forward
    palm_length, palm_width = 1.18, .67
    palm_outline = np.vstack([
        palm_center - .50 * palm_length * forward - .50 * palm_width * lateral,
        palm_center + .50 * palm_length * forward - .50 * palm_width * lateral,
        palm_center + .50 * palm_length * forward + .50 * palm_width * lateral,
        palm_center - .50 * palm_length * forward + .50 * palm_width * lateral,
        palm_center - .50 * palm_length * forward - .50 * palm_width * lateral,
    ])
    curl_by_action = {"抬臂": 8.0, "伸手": 4.0, "抓取": 84.0, "按压": 62.0, "松开": 18.0, "复位": 12.0}
    curl = curl_by_action.get(action, 12.0)
    curls = finger_curls_deg if finger_curls_deg is not None else (curl * .75, curl, curl, curl, curl)
    finger_offsets = (-.275, -.10, .10, .275)
    finger_lengths = ((.62, .48, .36), (.70, .52, .40), (.66, .50, .38), (.58, .43, .32))
    fingers: list[np.ndarray] = []
    fiber_routes: list[np.ndarray] = []
    for index, (offset, lengths) in enumerate(zip(finger_offsets, finger_lengths)):
        base = palm_center + .50 * palm_length * forward + offset * lateral
        splay = np.deg2rad(offset * 18.0)
        local_curl = 88.0 if action == "按压" and index == 1 and finger_curls_deg is None else curls[index + 1]
        direction = np.arctan2(forward[1], forward[0]) + splay
        points = [base]
        for length, joint_angle in zip(lengths, (0.0, local_curl, local_curl)):
            direction += np.deg2rad(joint_angle)
            points.append(points[-1] + length * np.array([np.cos(direction), np.sin(direction)]))
        finger = np.asarray(points)
        fingers.append(finger)
        fiber_routes.append(finger + .055 * lateral)
    thumb_base = palm_center + .10 * palm_length * forward - .60 * palm_width * lateral
    thumb_direction = np.arctan2(forward[1], forward[0]) + np.deg2rad(50.0)
    thumb_points = [thumb_base]
    thumb_curl = curls[0]
    for length, joint_angle in zip((.55, .42, .30), (0.0, thumb_curl, thumb_curl)):
        thumb_direction += np.deg2rad(joint_angle)
        thumb_points.append(thumb_points[-1] + length * np.array([np.cos(thumb_direction), np.sin(thumb_direction)]))
    thumb = np.asarray(thumb_points)
    fingers.insert(0, thumb)
    fiber_routes.insert(0, thumb - .055 * lateral)
    # 目标位于掌心偏拇指侧，闭合时由拇指与多根手指形成包络。
    target = palm_center + .10 * forward + .46 * lateral
    return {
        "arm_joints": arm_joints,
        "palm_center": palm_center,
        "palm_outline": palm_outline,
        "fingers": fingers,
        "fiber_routes": fiber_routes,
        "target": target,
        "finger_curls_deg": np.asarray(curls, dtype=float),
        "wrist_rotation_deg": float(wrist_rotation_deg),
    }


def planar_hand_animation_html(
    previous_pose: dict,
    current_pose: dict,
    previous_can_center: np.ndarray,
    current_can_center: np.ndarray,
    previous_grasped: bool,
    current_grasped: bool,
) -> str:
    """Render one continuous SVG transition between two planar grasp states."""
    def serialise(pose: dict, can_center: np.ndarray, grasped: bool) -> dict:
        return {
            "arm": np.asarray(pose["arm_joints"], dtype=float).tolist(),
            "palm": np.asarray(pose["palm_outline"], dtype=float).tolist(),
            "fingers": [np.asarray(finger, dtype=float).tolist() for finger in pose["fingers"]],
            "fibres": [np.asarray(route, dtype=float).tolist() for route in pose["fiber_routes"]],
            "can": np.asarray(can_center, dtype=float).tolist(),
            "grasped": bool(grasped),
        }

    config = json.dumps({
        "previous": serialise(previous_pose, previous_can_center, previous_grasped),
        "current": serialise(current_pose, current_can_center, current_grasped),
    }, ensure_ascii=False)
    html = r'''<div style="height:620px;border-radius:18px;overflow:hidden;background:linear-gradient(135deg,#0b1119,#152938)">
<svg id="planar-grasp" viewBox="0 0 1000 620" width="100%" height="100%" preserveAspectRatio="xMidYMid meet"></svg></div>
<script>
(() => {
 const cfg=__CONFIG__, svg=document.getElementById('planar-grasp'), ns='http://www.w3.org/2000/svg';
 const flat=s=>[...s.arm,...s.palm,...s.fingers.flat(),...s.fibres.flat(),s.can];
 const points=[...flat(cfg.previous),...flat(cfg.current)], xs=points.map(p=>p[0]), ys=points.map(p=>p[1]);
 const centerX=(Math.min(...xs)+Math.max(...xs))/2,centerY=(Math.min(...ys)+Math.max(...ys))/2;
 const viewWidth=12,viewHeight=9,scale=Math.min(900/viewWidth,500/viewHeight), ox=500-scale*centerX, oy=325+scale*centerY;
 const xy=p=>`${ox+scale*p[0]},${oy-scale*p[1]}`;
 const mix=(a,b,t)=>Array.isArray(a)?a.map((v,i)=>mix(v,b[i],t)):a+(b-a)*t;
 const el=(tag,attrs)=>{const node=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,v));svg.appendChild(node);return node};
 const arm=[['#4e7189',20],['#6f98ae',16],['#385a6d',10]].map(([stroke,width])=>el('polyline',{fill:'none',stroke,'stroke-width':width,'stroke-linecap':'round'}));
 const palm=el('polygon',{fill:'rgba(112,146,164,.84)',stroke:'#c2d9e5','stroke-width':3});
 const fingers=Array.from({length:5},()=>el('polyline',{fill:'none',stroke:'#d8e7ef','stroke-width':8,'stroke-linecap':'round','stroke-linejoin':'round'}));
 const fibres=Array.from({length:5},()=>el('polyline',{fill:'none',stroke:'#29c4d7','stroke-width':5,'stroke-linecap':'round'}));
 const can=el('rect',{fill:'#c33237',stroke:'#e8eff4','stroke-width':3,rx:12});
 const ring=el('line',{stroke:'#f6d365','stroke-width':4,'stroke-linecap':'round'});
 const text=el('text',{x:500,y:46,fill:'#eff8ff','font-size':20,'font-weight':'700','text-anchor':'middle'});
 function draw(t){const s={arm:mix(cfg.previous.arm,cfg.current.arm,t),palm:mix(cfg.previous.palm,cfg.current.palm,t),fingers:mix(cfg.previous.fingers,cfg.current.fingers,t),fibres:mix(cfg.previous.fibres,cfg.current.fibres,t),can:mix(cfg.previous.can,cfg.current.can,t),grasped:t<.5?cfg.previous.grasped:cfg.current.grasped};
  arm.forEach((node,i)=>node.setAttribute('points',[s.arm[i],s.arm[i+1]].map(xy).join(' ')));
  palm.setAttribute('points',s.palm.map(xy).join(' ')); fingers.forEach((node,i)=>node.setAttribute('points',s.fingers[i].map(xy).join(' '))); fibres.forEach((node,i)=>node.setAttribute('points',s.fibres[i].map(xy).join(' ')));
  const w=.42*scale,h=.74*scale,x=ox+scale*s.can[0]-w/2,y=oy-scale*s.can[1]-h/2;can.setAttribute('x',x);can.setAttribute('y',y);can.setAttribute('width',w);can.setAttribute('height',h);ring.setAttribute('x1',x+w*.32);ring.setAttribute('x2',x+w*.68);ring.setAttribute('y1',y+12);ring.setAttribute('y2',y+12);text.textContent=s.grasped?'二维 FBG 已抓稳 · 正在搬运':'二维寻找与对准 · 物体保持原位';}
 const started=performance.now(),duration=900; function animate(now){const t=Math.min(1,(now-started)/duration),e=t<.5?2*t*t:1-Math.pow(-2*t+2,2)/2;draw(e);if(t<1)requestAnimationFrame(animate)} draw(0);requestAnimationFrame(animate);
})();
</script>'''
    return html.replace("__CONFIG__", config)


def _arm_joint_coordinates_from_angles(angles_deg: tuple[float, float, float]) -> np.ndarray:
    """Return planar arm coordinates from independently controlled joint angles."""
    angles = np.deg2rad(angles_deg)
    lengths = (3.5, 3.0, 1.25)
    points = [np.array([0.0, 0.0])]
    direction = 0.0
    for length, angle in zip(lengths, angles):
        direction += angle
        points.append(points[-1] + length * np.array([np.cos(direction), np.sin(direction)]))
    return np.asarray(points)


def evaluate_can_grasp(pose: dict, can_center: np.ndarray, can_radius: float = .52, contact_margin: float = .42) -> dict:
    """Evaluate teaching-level fingertip contacts around a cylindrical beverage can."""
    can_center = np.asarray(can_center, dtype=float)
    fingers = pose["fingers"]
    distances = np.asarray([np.linalg.norm(np.asarray(finger)[-1] - can_center) for finger in fingers])
    contact_fingers = [index for index, distance in enumerate(distances) if distance <= can_radius + contact_margin]
    curl_mean = float(np.mean(np.asarray(pose["finger_curls_deg"])))
    non_thumb_contacts = [index for index in contact_fingers if index != 0]
    is_grasped = 0 in contact_fingers and len(non_thumb_contacts) >= 2 and curl_mean >= 35.0
    stability = min(1.0, .16 * len(contact_fingers) + .004 * curl_mean)
    return {"contact_fingers": contact_fingers, "tip_distances": distances, "stability": stability, "is_grasped": is_grasped}


def can_offset_from_target(pose: dict, can_center: np.ndarray) -> np.ndarray:
    """Express the can displacement from the default target in hand-local axes."""
    joints = np.asarray(pose["arm_joints"], dtype=float)
    forward = joints[3] - joints[2]
    forward /= np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    displacement = np.asarray(can_center, dtype=float) - np.asarray(pose["target"], dtype=float)
    return np.array([np.dot(displacement, forward), np.dot(displacement, lateral)])


def _can_mesh(can_center: np.ndarray, radius: float = .52, height: float = 1.75, sides: int = 24) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Create a simple closed cylinder mesh for an aluminium beverage can."""
    can_center = np.asarray(can_center, dtype=float)
    theta = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    lower = np.column_stack([can_center[0] + radius * np.cos(theta), .42 + radius * np.sin(theta), np.full(sides, can_center[1] - height / 2)])
    upper = lower.copy()
    upper[:, 2] = can_center[1] + height / 2
    vertices = np.vstack([lower, upper, [[can_center[0], .42, can_center[1] - height / 2], [can_center[0], .42, can_center[1] + height / 2]]])
    lower_center, upper_center = 2 * sides, 2 * sides + 1
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        i.extend([index, index, lower_center, upper_center])
        j.extend([nxt, sides + index, nxt, sides + nxt])
        k.extend([sides + index, sides + nxt, index, sides + index])
    return vertices, i, j, k


def _frustum_mesh(start: np.ndarray, end: np.ndarray, start_radius: float, end_radius: float, sides: int = 12) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Create a tapered engineering shell between two 3D points."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    axis /= np.linalg.norm(axis)
    reference = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < .9 else np.array([0.0, 1.0, 0.0])
    basis_u = np.cross(axis, reference)
    basis_u /= np.linalg.norm(basis_u)
    basis_v = np.cross(axis, basis_u)
    angle = np.linspace(0.0, 2.0 * np.pi, sides, endpoint=False)
    circle = np.outer(np.cos(angle), basis_u) + np.outer(np.sin(angle), basis_v)
    vertices = np.vstack([start + start_radius * circle, end + end_radius * circle, start, end])
    start_center, end_center = 2 * sides, 2 * sides + 1
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        i.extend([index, index, start_center, end_center])
        j.extend([nxt, sides + index, nxt, sides + nxt])
        k.extend([sides + index, sides + nxt, index, sides + index])
    return vertices, i, j, k


def _finger_tube_mesh(points: np.ndarray, radius: float = .115) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Join all three phalanges into one solid mesh, with a slight overlap at each knuckle."""
    points = np.asarray(points, dtype=float)
    vertices: list[np.ndarray] = []
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    offset = 0
    for start, end in zip(points[:-1], points[1:]):
        segment_vertices, segment_i, segment_j, segment_k = _frustum_mesh(start, end, radius, radius * .88, sides=10)
        vertices.append(segment_vertices)
        i.extend([value + offset for value in segment_i])
        j.extend([value + offset for value in segment_j])
        k.extend([value + offset for value in segment_k])
        offset += len(segment_vertices)
    return np.vstack(vertices), i, j, k


def _arched_palm_mesh(center: np.ndarray, forward_2d: np.ndarray, rings: int = 8, sides: int = 14) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    """Create a thin rounded palm shell whose width follows the five-finger spread."""
    center_3d = np.array([center[0], 0.0, center[1]])
    forward_3d = np.array([forward_2d[0], 0.0, forward_2d[1]])
    normal_3d = np.array([-forward_2d[1], 0.0, forward_2d[0]])
    # 局部坐标为（掌长、五指横向）。宽度沿 y 轴展开，五指根部才会真正落在掌面内。
    outline = np.asarray([
        (-.52, -.46), (-.45, -.67), (-.23, -.79), (.24, -.79),
        (.48, -.66), (.55, -.45), (.55, .45), (.48, .66),
        (.24, .79), (-.23, .79), (-.45, .67), (-.52, .46),
    ])
    shell_thickness = .13
    front = np.asarray([
        center_3d + along * forward_3d + width * np.array([0.0, 1.0, 0.0]) + shell_thickness * normal_3d
        for along, width in outline
    ])
    back = np.asarray([
        center_3d + along * forward_3d + width * np.array([0.0, 1.0, 0.0]) - shell_thickness * normal_3d
        for along, width in outline
    ])
    vertices = np.vstack([front, back, center_3d + shell_thickness * normal_3d, center_3d - shell_thickness * normal_3d])
    count = len(outline)
    front_center, back_center = 2 * count, 2 * count + 1
    i: list[int] = []
    j: list[int] = []
    k: list[int] = []
    for index in range(count):
        nxt = (index + 1) % count
        i.extend([index, index, front_center, back_center])
        j.extend([nxt, count + index, index, count + nxt])
        k.extend([count + index, count + nxt, nxt, count + index])
    return vertices, i, j, k


def arm_figure(action: str, route: str, finger_angle_deg: float, contact_force_n: float, joint_angles_deg: tuple[float, float, float] | None = None, finger_curls_deg: tuple[float, float, float, float, float] | None = None, wrist_rotation_deg: float = 0.0, can_center: np.ndarray | None = None) -> go.Figure:
    """Draw a five-finger teaching hand whose parts follow the selected action."""
    pose = dexterous_hand_pose(action, joint_angles_deg, finger_curls_deg, wrist_rotation_deg)
    joints = np.asarray(pose["arm_joints"])
    palm_outline = np.asarray(pose["palm_outline"])
    target = np.asarray(pose["target"] if can_center is None else can_center)
    fingers = pose["fingers"]
    fiber_routes = pose["fiber_routes"]
    grasp = evaluate_can_grasp(pose, target)
    figure = go.Figure()
    # 以不同长度和厚度绘制上臂、前臂与腕部，避免一条等粗折线破坏手臂比例。
    for name, start, end, width, color in (
        ("上臂", joints[0], joints[1], 20, "#4e7189"),
        ("前臂", joints[1], joints[2], 16, "#6f98ae"),
        ("腕部", joints[2], joints[3], 10, "#385a6d"),
    ):
        figure.add_scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode="lines+markers", name=name, line={"width": width, "color": color}, marker={"size": max(9, width - 6), "color": "#d8e7ef"}, hovertemplate=f"{name}<extra></extra>")
    figure.add_scatter(x=palm_outline[:, 0], y=palm_outline[:, 1], mode="lines", fill="toself", name="掌壳", line={"width": 3, "color": "#9db7c7"}, fillcolor="rgba(112, 146, 164, .82)")
    for index, (finger, fiber) in enumerate(zip(fingers, fiber_routes), 1):
        contact = index - 1 in grasp["contact_fingers"]
        figure.add_scatter(x=finger[:, 0], y=finger[:, 1], mode="lines+markers", name=f"手指 {index}", line={"width": 8, "color": "#ffcc66" if contact else "#d8e7ef"}, marker={"size": 7, "color": "#ff8a4c" if contact else "#506d80"}, hovertemplate=f"手指 {index}<extra></extra>")
        fiber_width = 5 if route in {"手指背侧", "指尖"} else 3
        figure.add_scatter(x=fiber[:, 0], y=fiber[:, 1], mode="lines+markers", name=f"FBG 支路 {index}", line={"width": fiber_width, "color": "#ffe06a" if contact else "#29c4d7"}, marker={"size": 5, "color": "#fff2aa" if contact else "#75eef5"}, hovertemplate=f"FBG 支路 {index}<extra></extra>")
    can_radius, can_height = .21, .74
    figure.add_scatter(x=[target[0] - can_radius, target[0] + can_radius, target[0] + can_radius, target[0] - can_radius, target[0] - can_radius], y=[target[1] - can_height / 2, target[1] - can_height / 2, target[1] + can_height / 2, target[1] + can_height / 2, target[1] - can_height / 2], mode="lines", fill="toself", name="铝制饮料罐", line={"color":"#d7e0e8","width":3}, fillcolor="rgba(195, 50, 55, .92)")
    figure.add_scatter(x=[target[0] - .20, target[0] + .20], y=[target[1] + can_height / 2 - .10, target[1] + can_height / 2 - .10], mode="lines", name="拉环", line={"color":"#f6d365","width":4}, hovertemplate="饮料罐拉环<extra></extra>")
    annotation_y = max(joints[:, 1].max(), target[1], palm_outline[:, 1].max()) + .65
    grasp_text = "抓取成功：罐体已绑定" if grasp["is_grasped"] else "尚未抓稳：继续调整五指"
    figure.add_annotation(x=joints[:, 0].mean(), y=annotation_y, text=f"动作：{action}<br>手指弯曲：{finger_angle_deg:.0f}°<br>{grasp_text}", showarrow=False, bgcolor="#172a3a", font={"color": "white"})
    all_points = np.vstack([joints, palm_outline, target, *fingers, *fiber_routes])
    figure.update_layout(title=f"五指灵巧手实体模拟与 FBG 走线：{route}", template="plotly_dark", height=560, showlegend=True, legend={"orientation": "h", "y": 1.02, "x": 1, "xanchor": "right", "font":{"size":10}}, xaxis={"visible": False, "range": [all_points[:, 0].min() - .8, all_points[:, 0].max() + 1.0]}, yaxis={"visible": False, "range": [all_points[:, 1].min() - .8, annotation_y + .5]}, margin={"l": 20, "r": 20, "t": 72, "b": 18})
    return figure


def planar_hand_transition_figure(
    previous_pose: dict,
    current_pose: dict,
    previous_can_center: np.ndarray,
    current_can_center: np.ndarray,
    previous_grasped: bool,
    current_grasped: bool,
) -> go.Figure:
    """Create a native Plotly animation, avoiding a remounted HTML document."""
    def blend(start: np.ndarray, end: np.ndarray, amount: float) -> np.ndarray:
        return (1.0 - amount) * np.asarray(start, dtype=float) + amount * np.asarray(end, dtype=float)

    def traces(amount: float) -> list[go.Scatter]:
        arm = blend(previous_pose["arm_joints"], current_pose["arm_joints"], amount)
        palm = blend(previous_pose["palm_outline"], current_pose["palm_outline"], amount)
        fingers = [blend(before, after, amount) for before, after in zip(previous_pose["fingers"], current_pose["fingers"])]
        fibres = [blend(before, after, amount) for before, after in zip(previous_pose["fiber_routes"], current_pose["fiber_routes"])]
        can = blend(previous_can_center, current_can_center, amount)
        grasped = current_grasped if amount >= .5 else previous_grasped
        scene: list[go.Scatter] = []
        for name, start, end, width, color in (
            ("上臂", arm[0], arm[1], 20, "#4e7189"),
            ("前臂", arm[1], arm[2], 16, "#6f98ae"),
            ("腕部", arm[2], arm[3], 10, "#385a6d"),
        ):
            scene.append(go.Scatter(x=[start[0], end[0]], y=[start[1], end[1]], mode="lines+markers", name=name, line={"width": width, "color": color}, marker={"size": max(9, width - 6), "color": "#d8e7ef"}))
        scene.append(go.Scatter(x=palm[:, 0], y=palm[:, 1], mode="lines", fill="toself", name="掌壳", line={"width": 3, "color": "#9db7c7"}, fillcolor="rgba(112,146,164,.82)"))
        for index, (finger, fibre) in enumerate(zip(fingers, fibres), 1):
            active = grasped and index in (1, 2, 3, 4)
            scene.append(go.Scatter(x=finger[:, 0], y=finger[:, 1], mode="lines+markers", name=f"手指 {index}", line={"width": 8, "color": "#ffcc66" if active else "#d8e7ef"}, marker={"size": 7, "color": "#ff8a4c" if active else "#506d80"}))
            scene.append(go.Scatter(x=fibre[:, 0], y=fibre[:, 1], mode="lines", name=f"FBG 支路 {index}", line={"width": 5, "color": "#ffe06a" if active else "#29c4d7"}))
        radius, height = .21, .74
        scene.append(go.Scatter(x=[can[0] - radius, can[0] + radius, can[0] + radius, can[0] - radius, can[0] - radius], y=[can[1] - height / 2, can[1] - height / 2, can[1] + height / 2, can[1] + height / 2, can[1] - height / 2], mode="lines", fill="toself", name="铝制饮料罐", line={"color": "#e8eff4", "width": 3}, fillcolor="rgba(195,50,55,.92)"))
        return scene

    all_points = np.vstack([
        np.asarray(previous_pose["arm_joints"]), np.asarray(current_pose["arm_joints"]),
        np.asarray(previous_pose["palm_outline"]), np.asarray(current_pose["palm_outline"]),
        np.asarray(previous_can_center), np.asarray(current_can_center),
    ])
    center = (all_points.min(axis=0) + all_points.max(axis=0)) / 2
    figure = go.Figure(data=traces(0.0), frames=[go.Frame(data=traces(amount), name=str(index)) for index, amount in enumerate(np.linspace(0.0, 1.0, 13))])
    figure.update_layout(
        template="plotly_dark", height=620, showlegend=False,
        xaxis={"visible": False, "range": [center[0] - 6.0, center[0] + 6.0]},
        yaxis={"visible": False, "range": [center[1] - 4.5, center[1] + 4.5], "scaleanchor": "x", "scaleratio": 1},
        margin={"l": 8, "r": 8, "t": 42, "b": 8},
        title="二维抓取连续动画（点击播放）",
        updatemenus=[{"type": "buttons", "showactive": False, "x": .5, "xanchor": "center", "y": 1.08, "yanchor": "top", "buttons": [{"label": "播放本步骤动画", "method": "animate", "args": [None, {"frame": {"duration": 75, "redraw": True}, "transition": {"duration": 0}, "fromcurrent": True}]}]}],
    )
    return figure


def planar_hand_snapshot_svg(pose: dict, can_center: np.ndarray, grasped: bool) -> str:
    """Return a script-free SVG frame for the non-moving planar search step."""
    joints = np.asarray(pose["arm_joints"], dtype=float)
    palm = np.asarray(pose["palm_outline"], dtype=float)
    can = np.asarray(can_center, dtype=float)
    points = np.vstack([joints, palm, *pose["fingers"], can])
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    scale, offset_x, offset_y = min(900 / 12, 500 / 9), 500 - 900 / 12 * center[0], 325 + 500 / 9 * center[1]

    def xy(point: np.ndarray) -> str:
        return f"{offset_x + scale * point[0]:.2f},{offset_y - scale * point[1]:.2f}"

    def polyline(points: np.ndarray) -> str:
        return " ".join(xy(point) for point in np.asarray(points, dtype=float))

    arm = "".join(
        f'<polyline points="{polyline([start, end])}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round"/>'
        for start, end, width, color in ((joints[0], joints[1], 20, "#4e7189"), (joints[1], joints[2], 16, "#6f98ae"), (joints[2], joints[3], 10, "#385a6d"))
    )
    fingers = "".join(
        f'<polyline points="{polyline(finger)}" fill="none" stroke="{"#ffcc66" if grasped else "#d8e7ef"}" stroke-width="8" stroke-linecap="round" stroke-linejoin="round"/><polyline points="{polyline(fibre)}" fill="none" stroke="{"#ffe06a" if grasped else "#29c4d7"}" stroke-width="5" stroke-linecap="round"/>'
        for finger, fibre in zip(pose["fingers"], pose["fiber_routes"])
    )
    can_x, can_y = offset_x + scale * (can[0] - .21), offset_y - scale * (can[1] + .37)
    return f'<div style="height:620px;border-radius:18px;overflow:hidden;background:linear-gradient(135deg,#0b1119,#152938)"><svg viewBox="0 0 1000 620" width="100%" height="620" aria-label="二维寻找目标姿态"><text x="500" y="42" fill="#eff8ff" font-size="20" font-weight="700" text-anchor="middle">二维寻找目标：物体保持原位</text>{arm}<polygon points="{polyline(palm)}" fill="#7092a4" fill-opacity=".85" stroke="#c2d9e5" stroke-width="3"/>{fingers}<rect x="{can_x:.2f}" y="{can_y:.2f}" width="{.42 * scale:.2f}" height="{.74 * scale:.2f}" rx="12" fill="#c33237" stroke="#e8eff4" stroke-width="3"/></svg></div>'


def arm_3d_figure(action: str, route: str, finger_angle_deg: float, joint_angles_deg: tuple[float, float, float] | None = None, finger_curls_deg: tuple[float, float, float, float, float] | None = None, wrist_rotation_deg: float = 0.0, can_center: np.ndarray | None = None) -> go.Figure:
    """Render a rotatable 3D teaching view of the current arm action."""
    pose = dexterous_hand_pose(action, joint_angles_deg, finger_curls_deg, wrist_rotation_deg)
    planar_joints = np.asarray(pose["arm_joints"])
    joints = np.column_stack([planar_joints[:, 0], [0.0, .12, .26, .38], planar_joints[:, 1]])
    fingers = pose["fingers"]
    fiber_routes = pose["fiber_routes"]
    target = np.asarray(pose["target"] if can_center is None else can_center)
    grasp = evaluate_can_grasp(pose, target)
    forward_2d = planar_joints[3] - planar_joints[2]
    forward_2d /= np.linalg.norm(forward_2d)
    palm_vertices, palm_i, palm_j, palm_k = _arched_palm_mesh(np.asarray(pose["palm_center"]), forward_2d)
    wrist_rotation = np.deg2rad(wrist_rotation_deg)
    palm_z = float(np.asarray(pose["palm_center"])[1])
    palm_y_local = palm_vertices[:, 1].copy()
    palm_z_local = palm_vertices[:, 2] - palm_z
    palm_vertices[:, 1] = palm_y_local * np.cos(wrist_rotation) - palm_z_local * np.sin(wrist_rotation)
    palm_vertices[:, 2] = palm_z + palm_y_local * np.sin(wrist_rotation) + palm_z_local * np.cos(wrist_rotation)
    figure = go.Figure()
    upper_vertices, upper_i, upper_j, upper_k = _frustum_mesh(joints[0], joints[1], .34, .29)
    forearm_vertices, forearm_i, forearm_j, forearm_k = _frustum_mesh(joints[1], joints[2], .29, .20)
    wrist_vertices, wrist_i, wrist_j, wrist_k = _frustum_mesh(joints[2], joints[3], .20, .15)
    figure.add_mesh3d(x=upper_vertices[:, 0], y=upper_vertices[:, 1], z=upper_vertices[:, 2], i=upper_i, j=upper_j, k=upper_k, name="上臂外壳", color="#54758b", opacity=1.0, flatshading=False)
    figure.add_mesh3d(x=forearm_vertices[:, 0], y=forearm_vertices[:, 1], z=forearm_vertices[:, 2], i=forearm_i, j=forearm_j, k=forearm_k, name="锥形前臂", color="#6f95aa", opacity=1.0, flatshading=False)
    figure.add_mesh3d(x=wrist_vertices[:, 0], y=wrist_vertices[:, 1], z=wrist_vertices[:, 2], i=wrist_i, j=wrist_j, k=wrist_k, name="腕部连接", color="#3f5b6d", opacity=1.0, flatshading=False)
    figure.add_scatter3d(x=[joints[1, 0], joints[2, 0]], y=[joints[1, 1], joints[2, 1]], z=[joints[1, 2], joints[2, 2]], mode="markers", name="腕部关节", marker={"size": [7, 6], "color": "#d9e5ec", "line":{"color":"#314b5d","width":2}})
    figure.add_mesh3d(x=palm_vertices[:, 0], y=palm_vertices[:, 1], z=palm_vertices[:, 2], i=palm_i, j=palm_j, k=palm_k, name="一体化掌壳", color="#7f9faf", opacity=1.0, flatshading=False, hovertemplate="一体化掌壳<extra></extra>")
    finger_depths = (-.28, -.13, .03, .18, .31)
    for index, (finger, fiber, depth) in enumerate(zip(fingers, fiber_routes, finger_depths), 1):
        contact = index - 1 in grasp["contact_fingers"]
        finger_y = np.full(len(finger), depth)
        finger_z_local = finger[:, 1] - palm_z
        finger_y_rotated = finger_y * np.cos(wrist_rotation) - finger_z_local * np.sin(wrist_rotation)
        finger_z_rotated = palm_z + finger_y * np.sin(wrist_rotation) + finger_z_local * np.cos(wrist_rotation)
        finger_points_3d = np.column_stack([finger[:, 0], finger_y_rotated, finger_z_rotated])
        finger_vertices, finger_i, finger_j, finger_k = _finger_tube_mesh(finger_points_3d)
        figure.add_mesh3d(x=finger_vertices[:, 0], y=finger_vertices[:, 1], z=finger_vertices[:, 2], i=finger_i, j=finger_j, k=finger_k, name=f"实体手指 {index}", color="#f2c978" if contact else "#d4e0e6", opacity=1.0, flatshading=False, hovertemplate=f"实体手指 {index}<extra></extra>")
        figure.add_scatter3d(x=finger[:, 0], y=finger_y_rotated, z=finger_z_rotated, mode="lines+markers", name=f"手指 {index}", line={"width": 11, "color": "#ffcc66" if contact else "#d8e7ef"}, marker={"size": 4, "color": "#ff8a4c" if contact else "#506d80"}, hovertemplate=f"手指 {index}<extra></extra>")
        fiber_width = 8 if route in {"手指背侧", "指尖"} else 4
        fiber_y = np.full(len(fiber), depth + .07)
        fiber_z_local = fiber[:, 1] - palm_z
        fiber_y_rotated = fiber_y * np.cos(wrist_rotation) - fiber_z_local * np.sin(wrist_rotation)
        fiber_z_rotated = palm_z + fiber_y * np.sin(wrist_rotation) + fiber_z_local * np.cos(wrist_rotation)
        figure.add_scatter3d(x=fiber[:, 0], y=fiber_y_rotated, z=fiber_z_rotated, mode="lines+markers", name=f"FBG 支路 {index}", line={"width": fiber_width, "color": "#ffe06a" if contact else "#29c4d7"}, marker={"size": 3, "color": "#fff2aa" if contact else "#75eef5"}, hovertemplate=f"FBG 支路 {index}<extra></extra>")
    can_vertices, can_i, can_j, can_k = _can_mesh(target)
    figure.add_mesh3d(x=can_vertices[:, 0], y=can_vertices[:, 1], z=can_vertices[:, 2], i=can_i, j=can_j, k=can_k, name="铝制饮料罐", color="#c83f45", opacity=.96, hovertemplate="铝制饮料罐<extra></extra>")
    figure.add_scatter3d(x=[target[0] - .20, target[0] + .20], y=[.42, .42], z=[target[1] + .80, target[1] + .80], mode="lines", name="拉环", line={"color":"#f6d365","width":7}, hovertemplate="饮料罐拉环<extra></extra>")
    figure.update_layout(title=f"三维五指灵巧手与饮料罐：{action}（{'已抓稳' if grasp['is_grasped'] else '接触调整中'}）", template="plotly_dark", height=600, scene={"xaxis":{"visible":False},"yaxis":{"visible":False},"zaxis":{"visible":False},"aspectmode":"manual","aspectratio":{"x":1.35,"y":.70,"z":1.0},"camera":{"eye":{"x":1.75,"y":-2.2,"z":1.25}}}, margin={"l":0,"r":0,"t":50,"b":0}, legend={"font":{"size":10}})
    return figure


def anthropomorphic_hand_html(action: str, joint_angles_deg: tuple[float, float, float], finger_curls_deg: tuple[float, float, float, float, float], grasped: bool, can_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), previous_can_offset: tuple[float, float, float] | None = None, shoulder_offset: tuple[float, float, float] = (0.0, 0.0, 0.0), previous_shoulder_offset: tuple[float, float, float] | None = None, planar_pose: dict | None = None, can_center: np.ndarray | None = None, previous_can_center: np.ndarray | None = None, can_depth: float = 0.0, previous_can_depth: float | None = None, finger_curl_gain: float = 1.0, finger_joint_angles_deg: tuple[tuple[float, ...], ...] | None = None, previous_joint_angles_deg: tuple[float, float, float] | None = None, previous_finger_joint_angles_deg: tuple[tuple[float, ...], ...] | None = None) -> str:
    """Return a self-contained Three.js teaching renderer for a rounded five-finger robot hand."""
    def three_axis_offset(offset: tuple[float, ...]) -> tuple[float, float, float]:
        values = tuple(float(value) for value in offset)
        return (values + (0.0, 0.0, 0.0))[:3]

    can_offset = three_axis_offset(can_offset)
    previous_can_offset = can_offset if previous_can_offset is None else three_axis_offset(previous_can_offset)
    previous_shoulder_offset = shoulder_offset if previous_shoulder_offset is None else previous_shoulder_offset
    planar_pose = planar_pose or dexterous_hand_pose(action, joint_angles_deg, finger_curls_deg)
    shared_geometry = {
        "armJoints": np.asarray(planar_pose["arm_joints"], dtype=float).tolist(),
        "palmOutline": np.asarray(planar_pose["palm_outline"], dtype=float).tolist(),
        "fingers": [np.asarray(finger, dtype=float).tolist() for finger in planar_pose["fingers"]],
        "fiberRoutes": [np.asarray(route, dtype=float).tolist() for route in planar_pose["fiber_routes"]],
    }
    can_center = np.asarray(planar_pose["target"] if can_center is None else can_center, dtype=float)
    previous_can_center = can_center if previous_can_center is None else np.asarray(previous_can_center, dtype=float)
    previous_can_depth = can_depth if previous_can_depth is None else previous_can_depth
    if finger_joint_angles_deg is None:
        curls = np.asarray(finger_curls_deg, dtype=float) * float(finger_curl_gain)
        finger_joint_angles_deg = (
            (0.0, curls[0] * .47),
            *((0.0, curl * .47, curl * .35) for curl in curls[1:]),
        )
    joint_counts = (2, 3, 3, 3, 3)
    if len(finger_joint_angles_deg) != 5 or any(len(angles) != count for angles, count in zip(finger_joint_angles_deg, joint_counts)):
        raise ValueError("finger_joint_angles_deg 必须包含拇指 2 个和其余四指各 3 个关节角")
    previous_joint_angles_deg = joint_angles_deg if previous_joint_angles_deg is None else previous_joint_angles_deg
    previous_finger_joint_angles_deg = finger_joint_angles_deg if previous_finger_joint_angles_deg is None else previous_finger_joint_angles_deg
    if len(previous_joint_angles_deg) != 3 or len(previous_finger_joint_angles_deg) != 5 or any(len(angles) != count for angles, count in zip(previous_finger_joint_angles_deg, joint_counts)):
        raise ValueError("上一帧必须包含三个手臂关节和 14 个手指关节")
    finger_joints = [list(map(float, angles)) for angles in finger_joint_angles_deg]
    config = json.dumps({"action": action, "joints": list(joint_angles_deg), "previousJoints": list(previous_joint_angles_deg), "curls": list(finger_curls_deg), "curlGain": float(finger_curl_gain), "fingerJoints": finger_joints, "previousFingerJoints": [list(map(float, angles)) for angles in previous_finger_joint_angles_deg], "grasped": grasped, "canOffset": list(can_offset), "previousCanOffset": list(previous_can_offset), "shoulderOffset": list(shoulder_offset), "previousShoulderOffset": list(previous_shoulder_offset), "sharedGeometry": shared_geometry, "canPosition": can_center.tolist(), "previousCanPosition": previous_can_center.tolist(), "canDepth": float(can_depth), "previousCanDepth": float(previous_can_depth)}, ensure_ascii=False)
    three_runtime = (Path(__file__).with_name("vendor") / "three.min.js").read_text(encoding="utf-8")
    html = r'''<div id="bio-hand" style="height:640px;border-radius:18px;overflow:hidden;background:linear-gradient(135deg,#0b1119,#152938);position:relative"></div>
<script>__THREE_RUNTIME__</script>
<script>
(() => {
 const cfg=__CONFIG__, host=document.getElementById('bio-hand');
 const scene=new THREE.Scene(), camera=new THREE.PerspectiveCamera(36,host.clientWidth/host.clientHeight,.1,100);
 const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true}); renderer.setSize(host.clientWidth,host.clientHeight); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); host.appendChild(renderer.domElement);
 scene.add(new THREE.HemisphereLight(0xe8f5ff,0x15222e,2.3)); const key=new THREE.DirectionalLight(0xffffff,2.4); key.position.set(4,-3,7); scene.add(key);
 const robot=new THREE.Group(); scene.add(robot); scene.add(new THREE.AxesHelper(2.0)); const grid=new THREE.GridHelper(24,24,0x315366,0x193140); grid.rotation.x=Math.PI/2; scene.add(grid); const hand=new THREE.Group();
 const shoulderStart=new THREE.Vector3(-6.7+cfg.previousShoulderOffset[0],-1.8+cfg.previousShoulderOffset[1],cfg.previousShoulderOffset[2]),shoulderTarget=new THREE.Vector3(-6.7+cfg.shoulderOffset[0],-1.8+cfg.shoulderOffset[1],cfg.shoulderOffset[2]);
 const metal=new THREE.MeshStandardMaterial({color:0x6d9bb5,metalness:.75,roughness:.28}); const joint=new THREE.MeshStandardMaterial({color:0x2e5268,metalness:.7,roughness:.25}); const skin=new THREE.MeshStandardMaterial({color:0xa5c8d9,metalness:.35,roughness:.48}); const cyan=new THREE.MeshStandardMaterial({color:0x23d5e8,emissive:0x087482,emissiveIntensity:.65});
 // 掌心轮廓：圆润的虎口、收窄腕部和拱起的掌背，不使用球体或平板。
 const outline=new THREE.Shape(); outline.moveTo(-1.12,-.43); outline.quadraticCurveTo(-.94,-.93,-.46,-.98); outline.lineTo(.58,-.93); outline.quadraticCurveTo(.96,-.82,1.04,-.48); outline.lineTo(1.04,.52); outline.quadraticCurveTo(.90,.91,.52,1.00); outline.lineTo(-.42,.96); outline.quadraticCurveTo(-.94,.87,-1.12,.43); outline.closePath();
 const palm=new THREE.Mesh(new THREE.ExtrudeGeometry(outline,{depth:.48,bevelEnabled:true,bevelThickness:.15,bevelSize:.14,bevelSegments:4,curveSegments:16}),metal); palm.position.set(0,0,-.24); hand.add(palm);
 const palmFiberA=new THREE.Mesh(new THREE.CapsuleGeometry(.026,1.76,5,8),cyan),palmFiberB=new THREE.Mesh(new THREE.CapsuleGeometry(.026,1.48,5,8),cyan); palmFiberA.rotation.z=-Math.PI/2; palmFiberA.position.set(.02,-.36,.28); palmFiberB.rotation.z=-Math.PI/2; palmFiberB.position.set(.02,.38,.28); hand.add(palmFiberA,palmFiberB);
 const cuff=new THREE.Mesh(new THREE.CapsuleGeometry(.38,.50,8,16),joint); cuff.rotation.y=Math.PI/2; cuff.position.set(-1.35,0,0); hand.add(cuff);
 const hub=new THREE.Mesh(new THREE.TorusGeometry(.34,.10,10,22),joint); hub.rotation.y=Math.PI/2; hub.position.set(-1.03,0,0); hand.add(hub);
 // 三段机器人手臂与二维图共用同一运动学：每一段按肩、肘、腕角依次累积旋转。
 const armPivots=[],fingerPivots=[];
 function makeArm(){ const a=cfg.joints.map(v=>v*Math.PI/180), shoulder=new THREE.Group(); shoulder.position.copy(shoulderStart); robot.add(shoulder); let pivot=shoulder; const links=[[6.3,.43,0x4f7690],[5.4,.37,0x638ba2],[2.25,.30,0x36586c]]; links.forEach((item,n)=>{ armPivots.push(pivot); pivot.rotation.z=a[n]; const link=new THREE.Mesh(new THREE.CapsuleGeometry(item[1],item[0]-item[1]*2,8,16),new THREE.MeshStandardMaterial({color:item[2],metalness:.78,roughness:.26})); link.rotation.z=-Math.PI/2; link.position.x=item[0]/2; pivot.add(link); const armFiber=new THREE.Mesh(new THREE.CapsuleGeometry(.026,Math.max(.1,item[0]-.12),5,8),cyan); armFiber.rotation.z=-Math.PI/2; armFiber.position.set(item[0]/2,0,item[1]*.94); pivot.add(armFiber); const next=new THREE.Group(); next.position.x=item[0]; pivot.add(next); if(n<2){ const jointDisc=new THREE.Mesh(new THREE.TorusGeometry(item[1]*1.05,.075,8,18),joint); jointDisc.rotation.y=Math.PI/2; jointDisc.position.x=item[0]; pivot.add(jointDisc); } pivot=next; }); pivot.add(hand); return shoulder; } const shoulder=makeArm(); armPivots.forEach((pivot,index)=>pivot.rotation.z=cfg.previousJoints[index]*Math.PI/180);
 function makeFinger(base,lengths,jointAngles,spread,thumb){const root=new THREE.Group();root.position.set(...base);root.rotation.z=spread;if(thumb){root.rotateY(-jointAngles[0]*Math.PI/180);}else{root.rotateY(-jointAngles[0]*Math.PI/180);}hand.add(root);let pivot=root;const bends=[root],radii=thumb?[.19,.16,.13]:[.17,.145,.12],knuckle=new THREE.Mesh(new THREE.SphereGeometry(radii[0]*1.18,12,12),joint);knuckle.position.set(...base);hand.add(knuckle);lengths.forEach((len,n)=>{if(n){bends.push(pivot);pivot.rotation.y=-jointAngles[n]*Math.PI/180;}const bone=new THREE.Mesh(new THREE.CapsuleGeometry(radii[n],Math.max(.12,len-radii[n]*2),8,14),skin);bone.rotation.z=-Math.PI/2;bone.position.x=len/2;pivot.add(bone);const segmentFiber=new THREE.Mesh(new THREE.CapsuleGeometry(.025,Math.max(.1,len-.08),5,8),cyan);segmentFiber.rotation.z=-Math.PI/2;segmentFiber.position.set(len/2,0,radii[n]*.94);pivot.add(segmentFiber);const ring=new THREE.Mesh(new THREE.TorusGeometry(radii[n]*1.04,.035,8,16),joint);ring.rotation.y=Math.PI/2;ring.position.x=len;pivot.add(ring);const next=new THREE.Group();next.position.x=len;pivot.add(next);pivot=next;});fingerPivots.push(bends);}
 const fj=cfg.fingerJoints; makeFinger([.18,-.98,.04],[1.31,.85],fj[0],-.83,true); makeFinger([.92,-.56,.08],[1.384,1.0,.72],fj[1],-.09,false); makeFinger([1.04,-.18,.10],[1.512,1.072,.768],fj[2],-.03,false); makeFinger([.98,.23,.08],[1.36,.976,.688],fj[3],.05,false); makeFinger([.84,.59,.02],[1.104,.768,.56],fj[4],.15,false); fingerPivots.forEach((finger,index)=>finger.forEach((pivot,jointIndex)=>pivot.rotation.y=-cfg.previousFingerJoints[index][jointIndex]*Math.PI/180));
 const graspFrame=new THREE.Group(); graspFrame.position.set(.30,-.20,.76); hand.add(graspFrame); const originalShoulder=shoulder.position.clone(); shoulder.position.set(-6.7,-1.8,0); scene.updateMatrixWorld(true); const canStart=new THREE.Vector3(); graspFrame.getWorldPosition(canStart); canStart.add(new THREE.Vector3(cfg.canOffset[0],cfg.canOffset[1],cfg.canOffset[2])); shoulder.position.copy(originalShoulder); const can=new THREE.Mesh(new THREE.CylinderGeometry(.48,.48,1.72,32),new THREE.MeshStandardMaterial({color:0xc94d4f,metalness:.65,roughness:.25})); can.visible=true; if(cfg.grasped){can.position.set(.30,-.20,.76);hand.add(can);}else{can.position.copy(canStart);robot.add(can);}
 camera.position.set(16,-18,14); camera.lookAt(new THREE.Vector3(1,0,0));
 const floor=new THREE.Mesh(new THREE.CircleGeometry(9,48),new THREE.MeshStandardMaterial({color:0x102331,roughness:.82})); floor.rotation.x=-Math.PI/2; floor.position.y=-1.65; scene.add(floor);
 const label=document.createElement('div'); label.textContent='当前动作：'+cfg.action+' · 仿生五指机器人手 · 拖动旋转'; label.style.cssText='position:absolute;left:18px;bottom:14px;color:#d8f2ff;font:600 14px sans-serif;background:#132a3baa;padding:8px 11px;border-radius:9px'; host.appendChild(label);
 // Orbit-like interaction without external controls.
 let drag=false,last; host.addEventListener('pointerdown',e=>{drag=true;last=e;}); addEventListener('pointerup',()=>drag=false); addEventListener('pointermove',e=>{if(!drag)return;robot.rotation.y+=(e.clientX-last.clientX)*.01;robot.rotation.x+=(e.clientY-last.clientY)*.01;last=e;});
 new ResizeObserver(()=>{camera.aspect=host.clientWidth/host.clientHeight;camera.updateProjectionMatrix();renderer.setSize(host.clientWidth,host.clientHeight)}).observe(host); const motionStart=performance.now(),motionDuration=900; function tick(now){const progress=Math.min((now-motionStart)/motionDuration,1),eased=1-Math.pow(1-progress,3); shoulder.position.lerpVectors(shoulderStart,shoulderTarget,eased); armPivots.forEach((pivot,index)=>pivot.rotation.z=THREE.MathUtils.lerp(cfg.previousJoints[index],cfg.joints[index],eased)*Math.PI/180); fingerPivots.forEach((finger,index)=>finger.forEach((pivot,jointIndex)=>pivot.rotation.y=-THREE.MathUtils.lerp(cfg.previousFingerJoints[index][jointIndex],cfg.fingerJoints[index][jointIndex],eased)*Math.PI/180)); if(!cfg.grasped)can.position.copy(canStart); renderer.render(scene,camera);requestAnimationFrame(tick);} requestAnimationFrame(tick);
})();</script>'''
    return html.replace("__THREE_RUNTIME__", three_runtime).replace("__CONFIG__", config)


def foot_schematic_figure(zones: np.ndarray, terrain: str) -> go.Figure:
    """Render a six-zone sole and cyan FBG routing for teaching."""
    zones = np.asarray(zones, dtype=float)
    figure = go.Figure()
    for index, value in enumerate(zones):
        x, y = index % 3, 1 - index // 3
        figure.add_shape(type="rect", x0=x, x1=x + .9, y0=y, y1=y + .85, fillcolor=f"rgba(255,150,50,{min(.9,.18+value/80):.2f})", line={"color":"#dcebf3"})
        figure.add_annotation(x=x+.45, y=y+.42, text=f"区 {index+1}<br>{value:.0f} N", showarrow=False, font={"color":"white"})
    figure.add_scatter(x=[.1,.8,1.5,2.2,2.7], y=[1.65,1.4,1.55,1.3,.25], mode="lines+markers", name="足底 FBG 走线", line={"color":"#29c4d7","width":6})
    figure.update_layout(title=f"足底六区接触与光纤阵列：{terrain}", template="plotly_dark", height=390, xaxis={"visible":False,"range":[-.1,3.1]}, yaxis={"visible":False,"range":[-.1,2.1],"scaleanchor":"x"}, margin={"l":10,"r":10,"t":50,"b":10})
    return figure


def arm_health_figure(result: dict, diagnosis: dict) -> go.Figure:
    """Show an arm-link FBG array and its localised structural-health indication."""
    positions = np.asarray(result["sensor_positions_mm"], dtype=float)
    strain = np.asarray(result["strain"], dtype=float) * 1e6
    suspected = float(diagnosis["suspected_location_mm"])
    figure = go.Figure()
    figure.add_scatter(
        x=[0.0, 520.0], y=[0.0, 0.0], mode="lines", name="机械臂构件",
        line={"color": "#315064", "width": 24},
    )
    figure.add_scatter(
        x=positions, y=np.zeros_like(positions), mode="markers+text", name="FBG 阵列",
        text=[f"FBG {index + 1}<br>{value:.0f} με" for index, value in enumerate(strain)],
        textposition="top center", marker={"size": 18, "color": strain, "colorscale": "YlOrRd", "showscale": True, "colorbar": {"title": "应变 (με)"}},
    )
    figure.add_vline(x=suspected, line_dash="dash", line_color="#ff4d4f", annotation_text=f"可疑位置 {suspected:.0f} mm")
    figure.update_layout(
        title=f"机械臂结构健康：可疑位置 {suspected:.0f} mm",
        template="plotly_dark", height=360, xaxis_title="构件长度 (mm)",
        yaxis={"visible": False, "range": [-1.0, 1.8]}, margin={"l": 20, "r": 70, "t": 55, "b": 35},
    )
    return figure
