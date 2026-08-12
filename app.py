"""Streamlit entry point for the fiber robotics sensing lab."""

from __future__ import annotations

import csv
import io

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from fiber_robotics_sim import models, visuals


st.set_page_config(page_title="光纤机器人传感仿真实验室", page_icon="🦾", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: #101923;}
[data-testid="stSidebar"] * {color: #e8f1f7;}
[data-testid="stMetric"] {background: #172a3a; border: 1px solid #315064; border-radius: 12px; padding: 14px; min-height: 112px;}
[data-testid="stMetric"] * {color: #f5fbff !important;}
[data-testid="stMetricDelta"] {color: #65d6c3 !important;}
[data-testid="stAlert"] {border-radius: 12px;}
div[data-testid="stTabs"] button {font-size: 1rem; font-weight: 600;}
.element-container {margin-bottom: .55rem;}
h1 {letter-spacing: -.03em;}
h2, h3 {margin-top: .6rem;}
</style>""", unsafe_allow_html=True)
st.title("🦾 光纤机器人传感仿真实验室")
st.caption("用可解释的解析模型，学习 FBG 在机器人手指、触觉与连续体形状感知中的作用。")


def csv_bytes(labels: list[str], values: np.ndarray) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["sensor", "wavelength_shift_nm"])
    writer.writerows(zip(labels, values, strict=True))
    return buffer.getvalue().encode("utf-8-sig")


def module_directory(items: list[tuple[int, str, str]]) -> None:
    """Render a compact catalogue whose cards activate the matching Streamlit tab."""
    cards = "".join(
        f'<button class="module-card" data-tab="{index}"><strong>{title}</strong><span>{summary}</span></button>'
        for index, title, summary in items
    )
    markup = """<style>
    * { box-sizing: border-box; }
    .module-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; font-family: sans-serif; }
    .module-card { text-align: left; border: 1px solid #315064; border-radius: 10px; background: #132a3b; color: #eaf5fb; padding: 12px; cursor: pointer; min-height: 76px; }
    .module-card:hover { background: #1d4057; border-color: #44b8d5; }
    .module-card strong { display: block; font-size: 14px; margin-bottom: 6px; }
    .module-card span { display: block; color: #a9c0cf; font-size: 12px; line-height: 1.35; }
    </style><div class="module-grid">""" + cards + """</div>
    <script>
    document.querySelectorAll('.module-card').forEach((button) => {
      button.addEventListener('click', () => {
        const tabs = window.parent.document.querySelectorAll('[data-testid="stTabs"] button[role="tab"]');
        const target = tabs[Number(button.dataset.tab)];
        if (target) { target.click(); target.scrollIntoView({behavior: 'smooth', block: 'start'}); }
      });
    });
    </script>"""
    components.html(markup, height=274, scrolling=False)


with st.sidebar:
    st.header("公共光学与测量参数")
    temperature = st.slider("温度变化 ΔT (°C)", -20.0, 50.0, 0.0, 0.5, key="global_temperature")
    noise = st.slider("波长测量噪声 σ (nm)", 0.0, 0.020, 0.000, 0.0005, format="%.4f", key="global_noise")
    drift = st.slider("零点漂移 (nm)", 0.0, 0.020, 0.000, 0.0005, format="%.4f", key="global_drift")
    sample_rate = st.select_slider("采样率 (Hz)", options=[10, 25, 50, 100, 200], value=50, key="global_sample_rate")
    failed = st.selectbox("模拟失效通道", ["无", "手部 FBG 1", "手部 FBG 2", "手部 FBG 3", "足底区域 1"], key="global_failed_channel")
    seed = st.number_input("随机种子", min_value=0, value=7, step=1, key="global_seed")
    st.divider()
    st.info("FBG 模型：Δλᵦ = λᵦ[(1−pₑ)ε + kₜΔT]。所有页面均显示真实量与反演量。")

overview_tab, hand_tab, hand_3d_tab, calibration_tab, tactile_tab, foot_tab, shape_tab, health_tab, distributed_tab, fbg_simplus_tab, polarization_tab, chain_tab = st.tabs(
    ["① 系统总览", "② 二维手部抓取", "③ 三维抓取传感", "④ FBG 标定与诊断", "⑤ 多材质触觉识别", "⑥ 足底平衡与步态", "⑦ 连续体形状重建", "⑧ 机械臂健康监测", "⑨ 分布式光纤感知", "⑩ FBG-SimPlus 兼容", "⑪ 偏振与干涉传感", "⑫ 解调器与实验任务"]
)

with overview_tab:
    st.subheader("模块目录")
    module_directory([
        (1, "二维手部抓取", "平面姿态、接触与五路 FBG 抓取判定"),
        (2, "三维抓取传感", "独立三维接触、握持稳定度与手臂/手掌/手指光纤"),
        (3, "FBG 标定与诊断", "弯曲标定、温度补偿、冗余通道故障隔离"),
        (4, "多材质触觉识别", "五指/掌心接触分布与材料模式分类"),
        (5, "足底平衡与步态", "六区载荷、温补 CoP 与地形/相位影响"),
        (6, "连续体形状重建", "三芯光纤曲率、方向、扭转和中心线"),
        (7, "机械臂健康监测", "局部异常应变、位置定位与报警"),
        (8, "分布式光纤感知", "Rayleigh、DAS、Brillouin、Raman 的空间测量"),
        (9, "FBG-SimPlus 兼容", "检查 COMSOL FEM 导出数据，并进入原工具完成光谱仿真"),
        (10, "偏振与干涉传感", "Stokes 偏振态、Sagnac 陀螺与 EFPI 干涉谱"),
        (11, "解调器与实验任务", "波长流、滤波温补、控制输出与实验报告"),
    ])
    st.subheader("当前测量配置")
    config_a, config_b, config_c, config_d = st.columns(4)
    config_a.metric("温度变化", f"{temperature:.1f} °C")
    config_b.metric("采样率", f"{sample_rate} Hz")
    config_c.metric("波长噪声", f"{noise:.4f} nm")
    config_d.metric("模拟失效通道", failed)
    st.subheader("感知链与模块职责")
    st.markdown("| 环节 | 当前模块 | 输入 | 输出 |\n|---|---|---|---|\n| 机械交互 | 二维/三维抓取、触觉、足底 | 姿态、接触、载荷 | FBG 接触与应变读数 |\n| 光纤解调 | 标定与诊断、解调器链路 | 原始波长、温度、噪声 | 温补波长、异常通道 |\n| 状态估计 | 足底、形状、健康监测 | 多路 FBG | CoP、曲率、异常位置 |\n| 控制与任务 | 解调器与实验任务 | 估计状态 | 张开/闭合命令、实验报告 |")
    st.subheader("推荐实验路径")
    st.markdown("1. 先在 **FBG 标定与诊断** 对比原始波长与温补后的角度。\n2. 再用 **二维/三维抓取** 和 **多材质触觉** 观察接触如何改变多路 FBG。\n3. 最后在 **足底、形状重建、健康监测** 体验载荷、姿态与结构状态反演，并在 **解调器与实验任务** 导出当前报告。")
    st.info("所有页面均为可解释的教学解析模型。它们适合比较传感规律与算法流程，但真实系统仍须使用封装、温度场、动态载荷和设备标定数据进行验证。")

with hand_tab:
    st.subheader("机器人手：FBG 弯曲、指尖触觉与关节状态")
    action_defaults = {"抬臂": (15.0, 0.0), "伸手": (25.0, 0.0), "抓取": (55.0, 4.0), "按压": (35.0, 7.0), "松开": (5.0, 0.0), "复位": (0.0, 0.0)}
    action_poses = {
        "抬臂": ((72.0, -50.0, -12.0), (6.0, 8.0, 8.0, 8.0, 8.0)),
        "伸手": ((18.0, 2.0, 0.0), (3.0, 4.0, 4.0, 4.0, 4.0)),
        "抓取": ((38.0, -58.0, 18.0), (63.0, 84.0, 84.0, 84.0, 84.0)),
        "按压": ((20.0, -68.0, -18.0), (46.0, 62.0, 88.0, 62.0, 62.0)),
        "松开": ((32.0, -35.0, 10.0), (14.0, 18.0, 18.0, 18.0, 18.0)),
        "复位": ((45.0, -60.0, 15.0), (9.0, 12.0, 12.0, 12.0, 12.0)),
    }
    if "arm_action" not in st.session_state:
        st.session_state.arm_action = "伸手"
    if "can_world_center" not in st.session_state:
        st.session_state.can_world_center = np.asarray(visuals.dexterous_hand_pose("伸手")["target"])
        st.session_state.can_grasped = False
        st.session_state.can_relative_to_palm = np.zeros(2)
    if "can_position_x" not in st.session_state:
        st.session_state.can_position_x = float(st.session_state.can_world_center[0])
        st.session_state.can_position_y = float(st.session_state.can_world_center[1])
    if "can_depth_z" not in st.session_state:
        st.session_state.can_depth_z = 0.0
    for key in ("shoulder_translation_x", "shoulder_translation_y", "shoulder_translation_z"):
        st.session_state.setdefault(key, 0.0)
    if "two_d_task_phase" not in st.session_state:
        st.session_state.two_d_task_phase = "未启动"

    def apply_action_pose(action_name: str) -> None:
        st.session_state.arm_action = action_name
        arm_angles, finger_curls = action_poses[action_name]
        for key, value in zip(("manual_shoulder", "manual_elbow", "manual_wrist"), arm_angles):
            st.session_state[key] = value
        for key, value in zip(("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little"), finger_curls):
            st.session_state[key] = value

    def begin_two_d_grasp_task() -> None:
        remember_two_d_render_state()
        apply_action_pose("伸手")
        for key in ("shoulder_translation_x", "shoulder_translation_y", "shoulder_translation_z"):
            st.session_state[key] = 0.0
        st.session_state.two_d_task_phase = "寻找目标"

    def align_hand_to_two_d_target() -> None:
        """Move the hand's reach frame to the fixed world-space can position."""
        alignment_angles = action_poses["抓取"][0]
        alignment_curls = (20.0, 20.0, 20.0, 20.0, 20.0)
        st.session_state.arm_action = "伸手"
        for key, value in zip(("manual_shoulder", "manual_elbow", "manual_wrist"), alignment_angles):
            st.session_state[key] = value
        for key, value in zip(("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little"), alignment_curls):
            st.session_state[key] = value
        base_pose = visuals.dexterous_hand_pose("抓取", alignment_angles, alignment_curls)
        target = np.asarray(base_pose["target"])
        can_world = np.asarray(st.session_state.can_world_center)
        st.session_state.shoulder_translation_x = float(can_world[0] - target[0])
        st.session_state.shoulder_translation_z = float(can_world[1] - target[1])
        st.session_state.shoulder_translation_y = st.session_state.can_depth_z

    def current_two_d_grasp_is_verified() -> tuple[bool, dict, np.ndarray]:
        """Read the closed pose and resolve FBG grasp state before controls lock."""
        joint_angles = tuple(st.session_state[key] for key in ("manual_shoulder", "manual_elbow", "manual_wrist"))
        finger_curls = tuple(st.session_state[key] for key in ("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little"))
        pose = visuals.dexterous_hand_pose(
            st.session_state.arm_action,
            joint_angles,
            finger_curls,
            st.session_state.get("manual_wrist_rotation", 0.0),
            (st.session_state.shoulder_translation_x, st.session_state.shoulder_translation_z),
        )
        can_center = np.asarray(st.session_state.can_world_center, dtype=float)
        grasp = visuals.evaluate_can_grasp(pose, can_center)
        depth_aligned = abs(st.session_state.can_depth_z - st.session_state.shoulder_translation_y) <= .15
        sensing = models.simulate_planar_grasp_fbg(
            finger_curls,
            grasp["contact_fingers"] if depth_aligned else [],
            temperature,
        )
        decision = models.classify_planar_grasp_from_fbg(sensing, finger_curls, temperature)
        return bool(decision["is_grasped"]), pose, can_center

    def remember_two_d_render_state() -> None:
        """Keep the last complete scene so the next task command can animate from it."""
        st.session_state.two_d_previous_pose = visuals.dexterous_hand_pose(
            st.session_state.arm_action,
            tuple(st.session_state[key] for key in ("manual_shoulder", "manual_elbow", "manual_wrist")),
            tuple(st.session_state[key] for key in ("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little")),
            st.session_state.get("manual_wrist_rotation", 0.0),
            (st.session_state.shoulder_translation_x, st.session_state.shoulder_translation_z),
        )
        st.session_state.two_d_previous_can_center = np.asarray(st.session_state.can_world_center, dtype=float)
        st.session_state.two_d_previous_grasped = bool(st.session_state.can_grasped)

    def apply_two_d_transport_pose() -> None:
        """Lift the arm while maintaining the verified closed-finger grasp."""
        st.session_state.arm_action = "抬臂"
        for key, value in zip(("manual_shoulder", "manual_elbow", "manual_wrist"), (55.0, -35.0, 15.0)):
            st.session_state[key] = value

    def advance_two_d_grasp_task() -> None:
        phase = st.session_state.two_d_task_phase
        remember_two_d_render_state()
        if phase == "抓取失败":
            st.session_state.two_d_task_phase = "对准目标"
            return
        if phase == "寻找目标":
            st.session_state.two_d_found_target = np.asarray(st.session_state.can_world_center, dtype=float)
            st.session_state.two_d_task_phase = models.next_three_d_grasp_task_phase(phase, False)
            return
        if phase == "对准目标":
            align_hand_to_two_d_target()
            st.session_state.two_d_task_phase = models.next_three_d_grasp_task_phase(phase, False)
            return
        if phase == "闭合抓取":
            apply_action_pose("抓取")
            verified, pose, can_center = current_two_d_grasp_is_verified()
            if verified:
                st.session_state.can_grasped = True
                st.session_state.can_relative_to_palm = can_center - np.asarray(pose["palm_center"])
                apply_two_d_transport_pose()
                st.session_state.two_d_task_phase = models.next_three_d_grasp_task_phase("闭合抓取", True)
            else:
                st.session_state.two_d_task_phase = models.next_three_d_grasp_task_phase("闭合抓取", False)
            return
        if phase == "搬运目标":
            transport_pose = visuals.dexterous_hand_pose(
                st.session_state.arm_action,
                tuple(st.session_state[key] for key in ("manual_shoulder", "manual_elbow", "manual_wrist")),
                tuple(st.session_state[key] for key in ("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little")),
                st.session_state.get("manual_wrist_rotation", 0.0),
                (st.session_state.shoulder_translation_x, st.session_state.shoulder_translation_z),
            )
            released_center = np.asarray(transport_pose["palm_center"]) + st.session_state.can_relative_to_palm
            st.session_state.can_world_center = released_center
            st.session_state.can_position_x = float(released_center[0])
            st.session_state.can_position_y = float(released_center[1])
            apply_action_pose("松开")
            st.session_state.can_grasped = False
            st.session_state.two_d_task_phase = models.next_three_d_grasp_task_phase("搬运目标", True)
            return
        if phase == "松开并放置":
            st.session_state.two_d_task_phase = models.next_three_d_grasp_task_phase(phase, False)

    def release_can() -> None:
        release_pose = visuals.dexterous_hand_pose(
            st.session_state.arm_action,
            (st.session_state.manual_shoulder, st.session_state.manual_elbow, st.session_state.manual_wrist),
            (st.session_state.manual_thumb, st.session_state.manual_index, st.session_state.manual_middle, st.session_state.manual_ring, st.session_state.manual_little),
            st.session_state.manual_wrist_rotation,
            (st.session_state.get("shoulder_translation_x", 0.0), st.session_state.get("shoulder_translation_z", 0.0)),
        )
        released_center = np.asarray(release_pose["palm_center"]) + st.session_state.can_relative_to_palm if st.session_state.can_grasped else st.session_state.can_world_center
        st.session_state.can_world_center = np.asarray(released_center, dtype=float)
        st.session_state.can_position_x = float(released_center[0])
        st.session_state.can_position_y = float(released_center[1])
        st.session_state.can_grasped = False

    action = st.session_state.arm_action
    default_angle, default_force = action_defaults[action]
    route = "手指背侧"
    two_d_controls_unlocked = st.session_state.two_d_task_phase in ("未启动", "松开并放置", "完成")
    st.markdown("#### 二维任务指令")
    task_left, task_right = st.columns(2)
    task_left.button("开始二维寻找与抓取任务", key="start_two_d_grasp_task", on_click=begin_two_d_grasp_task, disabled=not two_d_controls_unlocked)
    task_right.button("执行下一步" if st.session_state.two_d_task_phase != "抓取失败" else "重新对准目标", key="advance_two_d_grasp_task", on_click=advance_two_d_grasp_task, disabled=st.session_state.two_d_task_phase in ("未启动", "完成"))
    if st.session_state.two_d_task_phase != "未启动":
        st.caption(f"二维任务状态：{st.session_state.two_d_task_phase}。物体世界坐标保持固定，手部向目标移动；抓取仅由 FBG 判定。")
    preset_rows = [*st.columns(3), *st.columns(3)]
    for column, action_name in zip(preset_rows, action_defaults):
        if column.button(action_name, key=f"action_{action_name}", disabled=not two_d_controls_unlocked):
            apply_action_pose(action_name)
            if action_name == "松开":
                st.session_state.can_grasped = False

    planar_controls, planar_display = st.columns([1, 2])
    with planar_controls:
        st.markdown("#### 二维姿态与目标")
        st.caption("可先用预设进入姿态，再单独调节每个关节。饮料罐只有在拇指与至少两根手指形成包络接触时才会绑定到掌心。")

        def controlled_slider(label: str, minimum: float, maximum: float, initial: float, key: str) -> float:
            if key not in st.session_state:
                st.session_state[key] = initial
            return st.slider(label, minimum, maximum, step=1.0, key=key, disabled=not two_d_controls_unlocked)

        shoulder = controlled_slider("肩关节 (°)", -20.0, 100.0, action_poses[action][0][0], "manual_shoulder")
        elbow = controlled_slider("肘关节 (°)", -100.0, 40.0, action_poses[action][0][1], "manual_elbow")
        wrist = controlled_slider("腕关节 (°)", -70.0, 70.0, action_poses[action][0][2], "manual_wrist")
        wrist_rotation = controlled_slider("腕部旋转 (°)", -90.0, 90.0, 0.0, "manual_wrist_rotation")
        thumb = controlled_slider("拇指屈曲 (°)", 0.0, 95.0, action_poses[action][1][0], "manual_thumb")
        index = controlled_slider("食指屈曲 (°)", 0.0, 95.0, action_poses[action][1][1], "manual_index")
        middle = controlled_slider("中指屈曲 (°)", 0.0, 95.0, action_poses[action][1][2], "manual_middle")
        ring = controlled_slider("无名指屈曲 (°)", 0.0, 95.0, action_poses[action][1][3], "manual_ring")
        little = controlled_slider("小指屈曲 (°)", 0.0, 95.0, action_poses[action][1][4], "manual_little")
        st.markdown("#### 物体世界坐标与肩部位移")
        can_x = st.slider("饮料罐水平位置", -8.0, 10.0, step=0.1, key="can_position_x", disabled=st.session_state.can_grasped or not two_d_controls_unlocked)
        can_y = st.slider("饮料罐垂直位置", -6.0, 8.0, step=0.1, key="can_position_y", disabled=st.session_state.can_grasped or not two_d_controls_unlocked)
        can_z = st.slider("饮料罐深度位置 (3D)", -4.0, 4.0, step=0.1, key="can_depth_z", disabled=st.session_state.can_grasped or not two_d_controls_unlocked)
        shoulder_x = st.slider("肩部 X 位移", -12.0, 12.0, step=0.1, key="shoulder_translation_x", disabled=not two_d_controls_unlocked)
        shoulder_y = st.slider("肩部 Y 位移", -12.0, 12.0, step=0.1, key="shoulder_translation_y", disabled=not two_d_controls_unlocked)
        shoulder_z = st.slider("肩部 Z 位移", -12.0, 12.0, step=0.1, key="shoulder_translation_z", disabled=not two_d_controls_unlocked)
        st.button("放下饮料罐", key="release_can", on_click=release_can, disabled=not two_d_controls_unlocked)

    joint_angles = (shoulder, elbow, wrist)
    finger_curls = (thumb, index, middle, ring, little)
    planar_translation = (st.session_state.get("shoulder_translation_x", 0.0), st.session_state.get("shoulder_translation_z", 0.0))
    pose = visuals.dexterous_hand_pose(action, joint_angles, finger_curls, wrist_rotation, planar_translation)
    if st.session_state.can_grasped and st.session_state.two_d_task_phase not in ("搬运目标", "松开并放置"):
        bound_center = np.asarray(pose["palm_center"]) + st.session_state.can_relative_to_palm
        if not visuals.evaluate_can_grasp(pose, bound_center)["is_grasped"]:
            st.session_state.can_grasped = False
            st.session_state.can_world_center = bound_center
            st.session_state.can_position_x = float(bound_center[0])
            st.session_state.can_position_y = float(bound_center[1])
    if not st.session_state.can_grasped:
        st.session_state.can_world_center = np.array([can_x, can_y])
    if st.session_state.can_grasped:
        can_center = np.asarray(pose["palm_center"]) + st.session_state.can_relative_to_palm
    else:
        can_center = np.asarray(st.session_state.can_world_center)
    grasp = visuals.evaluate_can_grasp(pose, can_center)
    depth_aligned = abs(can_z - shoulder_y) <= .15
    planar_fbg = models.simulate_planar_grasp_fbg(
        finger_curls,
        grasp["contact_fingers"] if depth_aligned else [],
        temperature,
    )
    planar_fbg_decision = models.classify_planar_grasp_from_fbg(planar_fbg, finger_curls, temperature)
    if planar_fbg_decision["is_grasped"] and not st.session_state.can_grasped:
        st.session_state.can_grasped = True
        st.session_state.can_relative_to_palm = can_center - np.asarray(pose["palm_center"])
    can_offset = (0.0, 0.0, 0.0) if st.session_state.can_grasped else (*tuple(visuals.can_offset_from_target(pose, can_center)), can_z)
    grasp_label = "FBG 已抓稳：饮料罐会跟随掌心移动" if st.session_state.can_grasped else "FBG 未抓稳：请让拇指与至少两根手指形成触觉接触"
    if st.session_state.can_grasped:
        st.success(grasp_label)
    else:
        st.warning(grasp_label)
    with planar_display:
        previous_pose = st.session_state.get("two_d_previous_pose", pose)
        previous_can_center = np.asarray(st.session_state.get("two_d_previous_can_center", can_center), dtype=float)
        previous_grasped = bool(st.session_state.get("two_d_previous_grasped", st.session_state.can_grasped))
        st.plotly_chart(
            visuals.planar_hand_transition_figure(
                previous_pose,
                pose,
                previous_can_center,
                can_center,
                previous_grasped,
                st.session_state.can_grasped,
            ),
            width="stretch",
        )
        st.plotly_chart(visuals.sensor_bar_figure(np.arange(1, 6), planar_fbg["wavelength_shifts_nm"], "二维抓取：五路 FBG 波长漂移"), width="stretch")
    st.session_state.two_d_previous_pose = pose
    st.session_state.two_d_previous_can_center = np.asarray(can_center, dtype=float)
    st.session_state.two_d_previous_grasped = bool(st.session_state.can_grasped)

with tactile_tab:
    st.subheader("多材质触觉识别：五指与掌心 FBG 接触分布")
    st.caption("选择目标材质后，模型用五指与掌心接触模式生成六路 FBG 读数，再按接触分布做基础材质分类。")
    tactile_left, tactile_right = st.columns([1, 2])
    with tactile_left:
        material = st.selectbox("目标材质", ["海绵", "硬块", "圆柱", "薄板"])
        grip_force = st.slider("握持力 (N)", 0.0, 12.0, 5.0, 0.1)
        contact_area = st.slider("接触面积 (%)", 0.0, 100.0, 20.0, 1.0)
    material_touch = models.simulate_material_touch(material, grip_force, contact_area, temperature)
    material_diagnosis = models.classify_tactile_material(
        material_touch["finger_touch_n"], material_touch["palm_touch_n"]
    )
    with tactile_right:
        st.plotly_chart(
            visuals.sensor_bar_figure(np.arange(1, 7), material_touch["wavelength_shifts_nm"], "五指与掌心：六路触觉 FBG 波长漂移"),
            width="stretch",
        )
    touch_a, touch_b, touch_c = st.columns(3)
    touch_a.metric("识别材质", str(material_diagnosis["material"]))
    touch_b.metric("模式置信度", f"{float(material_diagnosis['confidence']) * 100:.0f}%")
    touch_c.metric("掌心接触", f"{float(material_touch['palm_touch_n']):.2f} N")
    st.bar_chart({"五指接触力 (N)": material_touch["finger_touch_n"]})
    st.caption("这是基于预设接触模式的分类教学模型，用于比较软硬、曲面和薄板造成的接触分布；真实材质识别需要多次试验、不同握持力样本及训练/验证数据集。")

with foot_tab:
    st.subheader("机器人足：六区足底接触、地形与步态相位")
    terrain = st.selectbox("地形", ["平地", "前倾坡面", "后倾坡面", "柔软地面"])
    load = st.slider("总垂直载荷 (N)", 0.0, 400.0, 180.0, 5.0)
    phase = st.slider("步态相位 (%)", 0, 100, 55)
    support = st.selectbox("当前状态", ["支撑期", "摆动期"])
    terrain_weights = {"平地": np.ones(6), "前倾坡面": np.array([1.3, 1.2, 1.1, .8, .7, .6]), "后倾坡面": np.array([.6, .7, .8, 1.1, 1.2, 1.3]), "柔软地面": np.array([.9, 1.0, .9, 1.0, .9, 1.0])}[terrain]
    zones = load * terrain_weights / terrain_weights.sum() * (1 if support == "支撑期" else 0.03)
    if failed == "足底区域 1":
        zones[0] = 0.0
    foot_fbg = models.simulate_foot_fbg(zones, temperature, noise, int(seed))
    foot_estimate = models.estimate_foot_load_distribution(foot_fbg["wavelength_shifts_nm"], temperature)
    cop = foot_estimate["cop_region"]
    st.caption("先观察六个足底区域的受力颜色和青色光纤走线，再对照下方的数值柱状图。")
    foot_left, foot_right = st.columns([3, 2])
    with foot_left:
        st.plotly_chart(visuals.foot_schematic_figure(zones, terrain), width="stretch")
    with foot_right:
        st.plotly_chart(visuals.sensor_bar_figure(np.arange(1, 7), foot_fbg["wavelength_shifts_nm"], "足底六路 FBG 波长漂移"), width="stretch")
    a, b, c = st.columns(3)
    a.metric("总支撑力", f"{zones.sum():.1f} N")
    b.metric("压力中心 CoP", f"区域 {cop:.2f}")
    c.metric("步态相位", f"{phase}% · {support}")
    st.caption("六区读数采用独立线性标定，并先做共模温度补偿后反演区域载荷和 CoP；实际步态还需加入动态冲击与足部姿态补偿。")

with calibration_tab:
    st.subheader("FBG 标定与诊断")
    st.metric("当前配置", f"{sample_rate} Hz · {failed}")
    report = f"光纤机器人手足实验摘要\n温度变化：{temperature} °C\n波长噪声：{noise:.4f} nm\n零点漂移：{drift:.4f} nm\n采样率：{sample_rate} Hz\n模拟失效通道：{failed}\n模型边界：解析教学模型，真实系统需要封装、温度梯度和动态载荷标定。\n"
    st.download_button("下载中文实验摘要", report.encode("utf-8-sig"), "fiber_robotics_experiment_summary.txt", "text/plain")
    st.divider()
    st.subheader("单根手指 FBG 弯曲标定")
    left, right = st.columns([1, 2])
    with left:
        angle = st.slider("真实弯曲角 (°)", -100.0, 100.0, 0.0, 1.0, key="hand_bend_angle")
        length = st.slider("手指长度 (mm)", 40.0, 140.0, 80.0, 1.0)
        offset = st.slider("光纤距中性层偏置 (mm)", -2.0, 2.0, 1.0, 0.05)
        attachment = st.selectbox("光纤连接方式", ["嵌入式", "粘接式", "护套固定"])
        gain = {"嵌入式": 1.0, "粘接式": 0.78, "护套固定": 0.52}[attachment]
    finger = models.simulate_finger(angle, length, offset * gain, np.array([0.25, 0.50, 0.75]) * length, temperature, noise, int(seed))
    if failed.startswith("手部 FBG"):
        finger["wavelength_shifts_nm"][int(failed[-1]) - 1] = drift
    with right:
        st.plotly_chart(visuals.finger_figure(finger), width="stretch")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("真实弯曲角", f"{angle:.2f} °")
    metric_b.metric("FBG 融合角", f"{finger['estimated_angle_deg']:.2f} °", f"误差 {finger['estimated_angle_deg'] - angle:+.2f} °")
    metric_c.metric("连接传力系数", f"{gain:.2f}")
    st.plotly_chart(visuals.sensor_bar_figure(finger["sensor_positions_mm"], finger["wavelength_shifts_nm"], "三枚 FBG 的波长漂移"), width="stretch")
    st.download_button("下载手部 FBG 读数 CSV", csv_bytes(["FBG 1", "FBG 2", "FBG 3"], finger["wavelength_shifts_nm"]), "hand_fbg_readings.csv", "text/csv")
    thermal_only_shift = float(models.fbg_wavelength_shift_nm(np.array([0.0]), temperature)[0])
    diagnostic_left, diagnostic_right = st.columns(2)
    diagnostic_left.metric("预期共模温漂", f"{thermal_only_shift:.4f} nm")
    diagnostic_right.metric("通道诊断", "模拟失效" if failed.startswith("手部 FBG") else "通道正常", failed if failed.startswith("手部 FBG") else "三路参与融合")
    st.caption("诊断基础版将温度项视作三路共享的共模漂移；实际诊断还需要每路零点、封装差异、长期漂移与冗余通道的历史基线。")

    st.divider()
    st.subheader("冗余 FBG 故障诊断与容错反演")
    fault_left, fault_right = st.columns([1, 2])
    with fault_left:
        redundant_angle = st.slider("冗余通道真实弯曲角 (°)", -90.0, 90.0, 45.0, 1.0)
        fault_mode = st.selectbox("故障类型", ["无", "漂移", "断纤", "噪声增大"])
        fault_channel = st.slider("故障通道", 1, 4, 2, 1)
    redundant = models.simulate_redundant_finger_fbg(
        redundant_angle, 80.0, 1.0, temperature, fault_mode, fault_channel
    )
    redundant_diagnosis = models.diagnose_redundant_fbg(
        redundant["wavelength_shifts_nm"], 80.0, 1.0, temperature
    )
    with fault_right:
        st.plotly_chart(visuals.sensor_bar_figure(np.arange(1, 5), redundant["wavelength_shifts_nm"], "四路冗余 FBG：原始波长漂移"), width="stretch")
    fault_a, fault_b, fault_c = st.columns(3)
    fault_a.metric("诊断异常通道", "、".join(f"FBG {item}" for item in redundant_diagnosis["fault_channels"]) or "无")
    fault_b.metric("容错反演角度", f"{redundant_diagnosis['estimated_angle_deg']:.1f} °")
    fault_c.metric("共模温漂", f"{redundant_diagnosis['common_temperature_shift_nm']:.4f} nm")
    st.caption("容错反演会剔除偏离冗余中位数的通道；断纤、间歇失效和高噪声在真实系统中需要时间序列阈值与硬件自检共同确认。")

    st.divider()
    st.subheader("指尖接触位置与法向力反演")
    left, right = st.columns([1, 2])
    with left:
        contact_position = st.slider("真实接触位置 (mm)", 0.0, 70.0, 37.0, 0.5)
        force = st.slider("真实法向力 (N)", 0.0, 10.0, 4.0, 0.1)
        influence_width = st.slider("封装传力宽度 (mm)", 5.0, 25.0, 12.0, 0.5)
    contact_positions = np.array([15.0, 35.0, 55.0])
    contact = models.simulate_contact(contact_position, force, contact_positions, influence_width, 2e-4, temperature, noise, int(seed))
    estimated_position, estimated_force = models.estimate_contact(contact["wavelength_shifts_nm"], contact_positions, influence_width, 2e-4, temperature)
    with right:
        st.plotly_chart(visuals.contact_figure(contact, estimated_position), width="stretch")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("真实接触点", f"{contact_position:.1f} mm")
    metric_b.metric("反演接触点", f"{estimated_position:.1f} mm", f"误差 {estimated_position - contact_position:+.1f} mm")
    metric_c.metric("反演法向力", f"{estimated_force:.2f} N", f"误差 {estimated_force - force:+.2f} N")
    st.plotly_chart(visuals.sensor_bar_figure(contact_positions, contact["wavelength_shifts_nm"], "各 FBG 的波长漂移"), width="stretch")
    st.download_button("下载当前接触读数 CSV", csv_bytes(["FBG 1", "FBG 2", "FBG 3"], contact["wavelength_shifts_nm"]), "contact_fbg_readings.csv", "text/csv")
    with st.expander("模型边界"):
        st.write("这里用高斯传递函数表示接触力向光纤的传递。它能帮助理解传感器位置、封装刚度与可辨识性，但不能替代软材料非线性、胶层、摩擦和滞后的实验标定。")

with hand_3d_tab:
    st.subheader("三维抓取传感：独立接触与 FBG 读数")
    st.caption("本页不读取二维抓取的姿态、罐体位置或抓取结果。它以三维手自身的五指屈曲与罐体 X/Y/Z 偏移，独立估算指尖接触、握持稳定度和五路 FBG 读数。")

    if "three_d_action" not in st.session_state:
        st.session_state.three_d_action = "三维张开"
        for key, value in zip(
            ("three_d_shoulder", "three_d_elbow", "three_d_wrist"), (38.0, -58.0, 18.0)
        ):
            st.session_state[key] = value
        for key, value in zip(
            (
                "three_d_thumb_mcp", "three_d_thumb_ip",
                "three_d_index_mcp", "three_d_index_pip", "three_d_index_dip",
                "three_d_middle_mcp", "three_d_middle_pip", "three_d_middle_dip",
                "three_d_ring_mcp", "three_d_ring_pip", "three_d_ring_dip",
                "three_d_little_mcp", "three_d_little_pip", "three_d_little_dip",
            ),
            (0.0,) * 14,
        ):
            st.session_state[key] = value

    # 将此前“一键握拳”留下的拇指 IP 旧值迁移到双关节握拳预置。
    if st.session_state.three_d_action == "三维握拳" and st.session_state.get("three_d_fist_profile_version", 0) < 4:
        st.session_state.three_d_thumb_mcp = 90.0
        st.session_state.three_d_thumb_ip = 90.0
        st.session_state.three_d_fist_profile_version = 4

    def set_three_d_grasp_pose(closed: bool) -> None:
        st.session_state.three_d_action = "三维握拳" if closed else "三维张开"
        joint_angles = (90.0, 90.0, 72.0, 104.0, 74.0, 72.0, 104.0, 74.0, 72.0, 104.0, 74.0, 72.0, 104.0, 74.0) if closed else (0.0,) * 14
        for key, value in zip(
            (
                "three_d_thumb_mcp", "three_d_thumb_ip",
                "three_d_index_mcp", "three_d_index_pip", "three_d_index_dip",
                "three_d_middle_mcp", "three_d_middle_pip", "three_d_middle_dip",
                "three_d_ring_mcp", "three_d_ring_pip", "three_d_ring_dip",
                "three_d_little_mcp", "three_d_little_pip", "three_d_little_dip",
            ), joint_angles
        ):
            st.session_state[key] = value
        st.session_state.three_d_fist_profile_version = 4

    def reset_three_d_initial_pose() -> None:
        """Restore every independently controlled 3D node to its initial pose."""
        st.session_state.three_d_action = "三维初始"
        for key, value in zip(
            ("three_d_shoulder", "three_d_elbow", "three_d_wrist"), (38.0, -58.0, 18.0)
        ):
            st.session_state[key] = value
        for key in (
            "three_d_thumb_mcp", "three_d_thumb_ip",
            "three_d_index_mcp", "three_d_index_pip", "three_d_index_dip",
            "three_d_middle_mcp", "three_d_middle_pip", "three_d_middle_dip",
            "three_d_ring_mcp", "three_d_ring_pip", "three_d_ring_dip",
            "three_d_little_mcp", "three_d_little_pip", "three_d_little_dip",
        ):
            st.session_state[key] = 0.0
        for key in ("three_d_can_x", "three_d_can_y", "three_d_can_z"):
            st.session_state[key] = 0.0
        for key in ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"):
            st.session_state[key] = 0.0

    if "three_d_task_phase" not in st.session_state:
        st.session_state.three_d_task_phase = "未启动"
    for key in ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"):
        st.session_state.setdefault(key, 0.0)
    st.session_state.setdefault("three_d_previous_reach", (0.0, 0.0, 0.0))
    st.session_state.setdefault("three_d_previous_target", (0.0, 0.0, 0.0))
    st.session_state.setdefault("three_d_previous_arm_joints", (38.0, -58.0, 18.0))
    st.session_state.setdefault("three_d_previous_finger_joints", ((0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)))

    def remember_three_d_render_state() -> None:
        st.session_state.three_d_previous_reach = tuple(st.session_state[key] for key in ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"))
        st.session_state.three_d_previous_target = tuple(st.session_state[key] for key in ("three_d_can_x", "three_d_can_y", "three_d_can_z"))
        st.session_state.three_d_previous_arm_joints = tuple(st.session_state[key] for key in ("three_d_shoulder", "three_d_elbow", "three_d_wrist"))
        st.session_state.three_d_previous_finger_joints = (
            (st.session_state["three_d_thumb_mcp"], st.session_state["three_d_thumb_ip"]),
            (st.session_state["three_d_index_mcp"], st.session_state["three_d_index_pip"], st.session_state["three_d_index_dip"]),
            (st.session_state["three_d_middle_mcp"], st.session_state["three_d_middle_pip"], st.session_state["three_d_middle_dip"]),
            (st.session_state["three_d_ring_mcp"], st.session_state["three_d_ring_pip"], st.session_state["three_d_ring_dip"]),
            (st.session_state["three_d_little_mcp"], st.session_state["three_d_little_pip"], st.session_state["three_d_little_dip"]),
        )

    def current_three_d_grasp_is_verified() -> bool:
        """Evaluate the actual widget state immediately after hand closure."""
        finger_angles = (
            (st.session_state["three_d_thumb_mcp"], st.session_state["three_d_thumb_ip"]),
            (st.session_state["three_d_index_mcp"], st.session_state["three_d_index_pip"], st.session_state["three_d_index_dip"]),
            (st.session_state["three_d_middle_mcp"], st.session_state["three_d_middle_pip"], st.session_state["three_d_middle_dip"]),
            (st.session_state["three_d_ring_mcp"], st.session_state["three_d_ring_pip"], st.session_state["three_d_ring_dip"]),
            (st.session_state["three_d_little_mcp"], st.session_state["three_d_little_pip"], st.session_state["three_d_little_dip"]),
        )
        curls = tuple(float(np.mean(angles)) for angles in finger_angles)
        relative_target = models.relative_3d_target_offset(
            tuple(st.session_state[key] for key in ("three_d_can_x", "three_d_can_y", "three_d_can_z")),
            tuple(st.session_state[key] for key in ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z")),
        )
        sensing = models.evaluate_3d_grasp_sensing(
            curls,
            relative_target,
            temperature,
            finger_joint_angles_deg=finger_angles,
        )
        return bool(models.classify_3d_grasp_from_fbg(sensing, temperature)["is_grasped"])

    def start_three_d_grasp_task() -> None:
        remember_three_d_render_state()
        set_three_d_grasp_pose(False)
        for key in ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"):
            st.session_state[key] = 0.0
        st.session_state.three_d_task_phase = "寻找目标"

    def advance_three_d_grasp_task() -> None:
        phase = st.session_state.three_d_task_phase
        remember_three_d_render_state()
        if phase == "抓取失败":
            st.session_state.three_d_task_phase = "对准目标"
            return
        if phase == "寻找目标":
            st.session_state.three_d_found_target = tuple(
                st.session_state[key] for key in ("three_d_can_x", "three_d_can_y", "three_d_can_z")
            )
            st.session_state.three_d_task_phase = models.next_three_d_grasp_task_phase(phase, False)
            return
        if phase == "对准目标":
            # 保持物体世界坐标不变，移动手部抓取包络到已定位的目标坐标。
            for reach_key, target_key in zip(
                ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"),
                ("three_d_can_x", "three_d_can_y", "three_d_can_z"),
            ):
                st.session_state[reach_key] = st.session_state[target_key]
            st.session_state.three_d_task_phase = models.next_three_d_grasp_task_phase(phase, False)
            return
        if phase == "闭合抓取":
            set_three_d_grasp_pose(True)
            next_phase = models.next_three_d_grasp_task_phase("闭合抓取", current_three_d_grasp_is_verified())
            st.session_state.three_d_task_phase = next_phase
            if next_phase == "搬运目标":
                for key, value in zip(("three_d_shoulder", "three_d_elbow", "three_d_wrist"), (55.0, -35.0, 15.0)):
                    st.session_state[key] = value
            return
        if phase == "搬运目标":
            set_three_d_grasp_pose(False)
            st.session_state.three_d_can_x = st.session_state["three_d_reach_x"] + 1.4
            st.session_state.three_d_task_phase = models.next_three_d_grasp_task_phase(phase, True)
            return
        if phase == "松开并放置":
            st.session_state.three_d_task_phase = models.next_three_d_grasp_task_phase(phase, False)

    task_phase = st.session_state.three_d_task_phase
    task_left, task_right = st.columns(2)
    task_left.button("开始三维寻找与抓取任务", key="start_three_d_grasp_task", on_click=start_three_d_grasp_task, disabled=task_phase not in ("未启动", "完成"))
    task_right.button("执行下一步" if task_phase != "抓取失败" else "重新对准目标", key="advance_three_d_grasp_task", on_click=advance_three_d_grasp_task, disabled=task_phase in ("未启动", "完成"))
    if task_phase != "未启动":
        st.caption(f"三维任务状态：{task_phase}。流程：寻找目标 → 手部对准目标 → FBG 闭合抓取验证 → 搬运目标 → 松开并放置。")

    preset_left, preset_center, preset_right = st.columns(3)
    preset_left.button("三维张开手", key="three_d_open", on_click=set_three_d_grasp_pose, args=(False,))
    preset_center.button("恢复三维初始姿态", key="three_d_reset_initial", on_click=reset_three_d_initial_pose)
    preset_right.button("三维一键握拳", key="three_d_close", on_click=set_three_d_grasp_pose, args=(True,))

    controls, display = st.columns([1, 2])
    with controls:
        st.markdown("#### 三维姿态与目标")
        three_d_shoulder = st.slider("三维肩关节 (°)", -20.0, 100.0, step=1.0, key="three_d_shoulder")
        three_d_elbow = st.slider("三维肘关节 (°)", -100.0, 40.0, step=1.0, key="three_d_elbow")
        three_d_wrist = st.slider("三维腕关节 (°)", -70.0, 70.0, step=1.0, key="three_d_wrist")
        st.markdown("#### 三维手指 14 个独立关节")
        st.caption("拇指 2 个关节；食指、中指、无名指、小指各 3 个，第一项为与手掌连接的掌指关节（MCP）。")
        three_d_thumb_mcp = st.slider("拇指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_thumb_mcp")
        three_d_thumb_ip = st.slider("拇指 IP (°)", 0.0, 110.0, step=1.0, key="three_d_thumb_ip")
        three_d_index_mcp = st.slider("食指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_index_mcp")
        three_d_index_pip = st.slider("食指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_index_pip")
        three_d_index_dip = st.slider("食指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_index_dip")
        three_d_middle_mcp = st.slider("中指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_middle_mcp")
        three_d_middle_pip = st.slider("中指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_middle_pip")
        three_d_middle_dip = st.slider("中指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_middle_dip")
        three_d_ring_mcp = st.slider("无名指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_ring_mcp")
        three_d_ring_pip = st.slider("无名指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_ring_pip")
        three_d_ring_dip = st.slider("无名指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_ring_dip")
        three_d_little_mcp = st.slider("小指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_little_mcp")
        three_d_little_pip = st.slider("小指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_little_pip")
        three_d_little_dip = st.slider("小指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_little_dip")
        st.markdown("#### 物体世界坐标（任意可达位置）")
        three_d_can_x = st.slider("物体 X 位置", -3.0, 3.0, step=0.1, key="three_d_can_x")
        three_d_can_y = st.slider("物体 Y 位置", -3.0, 3.0, step=0.1, key="three_d_can_y")
        three_d_can_z = st.slider("物体 Z 位置", -3.0, 3.0, step=0.1, key="three_d_can_z")

    three_d_joints = (three_d_shoulder, three_d_elbow, three_d_wrist)
    three_d_finger_joints = (
        (three_d_thumb_mcp, three_d_thumb_ip),
        (three_d_index_mcp, three_d_index_pip, three_d_index_dip),
        (three_d_middle_mcp, three_d_middle_pip, three_d_middle_dip),
        (three_d_ring_mcp, three_d_ring_pip, three_d_ring_dip),
        (three_d_little_mcp, three_d_little_pip, three_d_little_dip),
    )
    three_d_curls = tuple(float(np.mean(angles)) for angles in three_d_finger_joints)
    three_d_target_world = (three_d_can_x, three_d_can_y, three_d_can_z)
    three_d_hand_reach = tuple(st.session_state[key] for key in ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"))
    three_d_can_offset = models.relative_3d_target_offset(three_d_target_world, three_d_hand_reach)
    three_d_sensing = models.evaluate_3d_grasp_sensing(
        three_d_curls, three_d_can_offset, temperature,
        arm_joint_angles_deg=three_d_joints,
        finger_joint_angles_deg=three_d_finger_joints,
    )
    three_d_fbg_decision = models.classify_3d_grasp_from_fbg(three_d_sensing, temperature)
    three_d_render_finger_joints = three_d_sensing["collision_limited_joint_angles_deg"]
    three_d_shifts = models.add_gaussian_noise(three_d_sensing["fbg_shifts_nm"], noise, int(seed) + 300)

    with display:
        st.caption("拖动模型可旋转视角；滚轮缩放保持关闭。物体保持世界坐标，寻找程序移动手部抓取包络至目标。")
        st.iframe(
            visuals.anthropomorphic_hand_html(
                st.session_state.three_d_action,
                three_d_joints,
                three_d_curls,
                bool(three_d_fbg_decision["is_grasped"]),
                can_offset=three_d_target_world,
                previous_can_offset=st.session_state.three_d_previous_target,
                shoulder_offset=three_d_hand_reach,
                previous_shoulder_offset=st.session_state.three_d_previous_reach,
                finger_curl_gain=1.0,
                finger_joint_angles_deg=three_d_render_finger_joints,
                previous_joint_angles_deg=st.session_state.three_d_previous_arm_joints,
                previous_finger_joint_angles_deg=st.session_state.three_d_previous_finger_joints,
            ),
            height=650,
        )

    three_d_metrics = st.columns(4)
    three_d_metrics[0].metric("FBG 触觉接触手指", f"{len(three_d_fbg_decision['contact_fingers'])} / 5")
    three_d_metrics[1].metric("FBG 反演接触合力", f"{np.asarray(three_d_fbg_decision['finger_touch_n']).sum():.2f} N")
    three_d_metrics[2].metric("握持稳定度", f"{float(three_d_sensing['stability']) * 100:.0f}%")
    three_d_metrics[3].metric("三维抓取状态", "FBG 已抓稳" if three_d_fbg_decision["is_grasped"] else "FBG 未抓稳")
    if three_d_fbg_decision["is_grasped"]:
        st.success("三维 FBG 判定：温度补偿后，掌心、拇指及至少两根手指的触觉通道均达到握持阈值。")
    else:
        st.warning("三维传感判定：请将罐体移回抓取包络，并提高拇指和至少两根手指的屈曲。")
    st.plotly_chart(
        visuals.sensor_bar_figure(np.arange(1, 6), three_d_shifts, "三维抓取：五指综合 FBG 波长漂移"),
        width="stretch",
    )
    st.bar_chart({"三维指尖接触力 (N)": np.asarray(three_d_sensing["contact_force_n"])})
    st.bar_chart({"14 个指节触觉 FBG 波长漂移 (nm)": three_d_sensing["tactile_fbg_shifts_nm"][:14]})
    tactile_left, tactile_right = st.columns(2)
    with tactile_left:
        st.bar_chart({"手掌与五指触觉 (N)": np.r_[three_d_sensing["palm_touch_n"], three_d_sensing["contact_force_n"]]})
    with tactile_right:
        st.bar_chart({"肩、肘、腕 FBG 弯曲应变 (με)": three_d_sensing["arm_bend_strain_ue"]})
    st.caption("青色光纤覆盖肩—肘—腕、两条掌部路线及全部 14 个手指指节：掌/指读数用于触觉，臂部读数用于弯曲。")
    st.download_button(
        "下载三维抓取 FBG 读数 CSV",
        csv_bytes(["拇指 FBG", "食指 FBG", "中指 FBG", "无名指 FBG", "小指 FBG"], three_d_shifts),
        "three_dimensional_grasp_fbg_readings.csv",
        "text/csv",
    )
    with st.expander("三维传感模型边界"):
        st.write("这里的接触力来自指尖到圆柱抓取包络的三维距离与屈曲角，作为光纤抓取传感教学模型。它不等同于刚体接触求解、摩擦锥或真实力控，需要结合传感器封装与实验数据标定。")

with shape_tab:
    st.subheader("三芯光纤的连续体机器人 3D 形状重建")
    left, right = st.columns([1, 2])
    with left:
        curvature = st.slider("真实曲率 (1/m)", 0.0, 20.0, 8.0, 0.1)
        direction = st.slider("弯曲方向 (°)", 0.0, 359.0, 35.0, 1.0)
        twist = st.slider("恒定扭转率 (1/m)", -20.0, 20.0, 0.0, 0.1)
        shape_length = st.slider("光纤长度 (mm)", 50.0, 300.0, 150.0, 1.0)
    shape = models.simulate_multicore_shape(curvature, direction, twist, shape_length, 125.0, temperature, noise, int(seed))
    with right:
        st.plotly_chart(visuals.multicore_figure(shape), width="stretch")
    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("反演曲率", f"{shape['estimated_curvature_per_m']:.3f} 1/m", f"误差 {shape['estimated_curvature_per_m'] - curvature:+.3f}")
    direction_error = (shape["estimated_direction_deg"] - direction + 180.0) % 360.0 - 180.0
    metric_b.metric("反演弯曲方向", f"{shape['estimated_direction_deg']:.1f} °", f"误差 {direction_error:+.1f} °")
    metric_c.metric("纤芯半径", "125 μm")
    st.plotly_chart(visuals.sensor_bar_figure(shape["core_angles_deg"], shape["wavelength_shifts_nm"], "三根纤芯的波长漂移"), width="stretch")
    st.download_button("下载当前多芯光纤读数 CSV", csv_bytes(["Core 1", "Core 2", "Core 3"], shape["wavelength_shifts_nm"]), "multicore_fbg_readings.csv", "text/csv")
    with st.expander("为何温度影响较小？"):
        st.write("三根纤芯共享近似相同的温度漂移。算法先减去三路读数的均值，再由差分应变求曲率向量，因此可以抑制共模温度项；实际系统仍需处理温度梯度与封装不对称。")

with health_tab:
    st.subheader("机械臂结构健康监测：四路 FBG 局部异常定位")
    st.caption("基础教学模型：四枚 FBG 布置在同一机械臂构件上，以温度补偿后的局部差分应变判断异常位置。")
    health_left, health_right = st.columns([1, 2])
    with health_left:
        arm_load = st.slider("机械臂载荷 (N)", 0.0, 160.0, 80.0, 5.0)
        anomaly_position = st.slider("局部异常位置 (mm)", 0.0, 520.0, 320.0, 5.0)
        anomaly_severity = st.slider("异常程度", 0.0, 1.0, 0.0, 0.05)
    arm_health = models.simulate_arm_health_fbg(
        arm_load, anomaly_position, anomaly_severity, temperature, noise, int(seed)
    )
    diagnosis = models.diagnose_arm_health(arm_health["wavelength_shifts_nm"], temperature)
    with health_right:
        st.plotly_chart(visuals.arm_health_figure(arm_health, diagnosis), width="stretch")
    health_a, health_b, health_c = st.columns(3)
    health_a.metric("诊断状态", str(diagnosis["status"]))
    health_b.metric("可疑位置", f"{diagnosis['suspected_location_mm']:.0f} mm")
    health_c.metric("局部异常指数", f"{diagnosis['damage_index']:.2f}")
    st.plotly_chart(visuals.sensor_bar_figure(arm_health["sensor_positions_mm"], arm_health["wavelength_shifts_nm"], "机械臂四路 FBG 波长漂移"), width="stretch")
    st.caption("“需检查”表示局部差分应变超过本教学模型阈值，不等同于真实裂纹结论；真实结构健康监测还需健康基线、载荷工况、温度场和无损检测交叉验证。")

with distributed_tab:
    st.subheader("分布式光纤感知：连续空间上的应变、振动与温度")
    st.caption("本页以四类教学解析模型对比不同散射机制的观测量：Rayleigh/OFDR 连续应变、φ-OTDR/DAS 振动事件、Brillouin 频移、Raman 分布式温度。")
    distributed_left, distributed_right = st.columns([1, 2])
    with distributed_left:
        distributed_mode = st.selectbox("分布式机制", ["Rayleigh/OFDR", "φ-OTDR / DAS", "Brillouin", "Raman"])
        fiber_length = st.slider("分布式光纤长度 (mm)", 100.0, 800.0, 300.0, 10.0)
        event_position = st.slider("局部事件位置 (mm)", 0.0, 800.0, 140.0, 5.0)
        event_strength = st.slider("局部应变 / 温度幅值", 0.0, 1000.0, 600.0, 10.0)
    event_position = min(event_position, fiber_length)
    with distributed_right:
        if distributed_mode == "Rayleigh/OFDR":
            distributed_result = models.simulate_rayleigh_ofdr(fiber_length, event_position, event_strength, 2.0)
            st.plotly_chart(visuals.distributed_curve_figure(distributed_result, "Rayleigh"), width="stretch")
            distributed_frame = models.build_sensor_frame("Rayleigh/OFDR", distributed_result["position_mm"], distributed_result["raw_strain_ue"], distributed_result["strain_ue"], .93)
        elif distributed_mode == "φ-OTDR / DAS":
            distributed_result = models.simulate_das_event(fiber_length, event_position, 60.0, int(sample_rate))
            st.plotly_chart(visuals.das_event_figure(distributed_result), width="stretch")
            distributed_frame = models.build_sensor_frame("φ-OTDR/DAS", distributed_result["position_mm"], distributed_result["amplitude"], distributed_result["amplitude"], .88)
        elif distributed_mode == "Brillouin":
            distributed_result = models.simulate_brillouin_distribution(fiber_length, min(event_strength, 100.0), event_strength)
            st.plotly_chart(visuals.distributed_curve_figure(distributed_result, "Brillouin"), width="stretch")
            distributed_frame = models.build_sensor_frame("Brillouin", distributed_result["position_mm"], distributed_result["brillouin_frequency_ghz"], distributed_result["strain_ue"], .90)
        else:
            distributed_result = models.simulate_raman_temperature(fiber_length, event_position, min(event_strength, 120.0))
            st.plotly_chart(visuals.distributed_curve_figure(distributed_result, "Raman"), width="stretch")
            distributed_frame = models.build_sensor_frame("Raman", distributed_result["position_mm"], distributed_result["anti_stokes_ratio"], distributed_result["temperature_c"], .86)
    distributed_a, distributed_b, distributed_c = st.columns(3)
    distributed_a.metric("传感机制", str(distributed_frame["sensor_type"]))
    distributed_b.metric("空间采样点", f"{len(np.asarray(distributed_frame['position_or_channel']))}")
    distributed_c.metric("数据质量", f"{float(distributed_frame['quality']) * 100:.0f}%")
    st.caption("不同机制的空间分辨率、测量距离、采样速度与温度—应变交叉敏感性不同；此处用于机制与数据形态比较，不代表具体商用解调设备指标。")

with fbg_simplus_tab:
    st.subheader("FBG-SimPlus 兼容：通用八列数据适配")
    st.caption("本页读取 FBG-SimPlus 所需的八列数值数据并进行标准化；不要求特定仿真软件，不包含、复制、执行或修改 FBG-SimPlus 源代码，也不在本网站生成其反射谱。")
    st.markdown(
        "**出处与许可：** [FBG-SimPlus V1.0（Ben Frey 等）](https://github.com/benfrey/FBG-SimPlus) "
        "采用 [GNU GPL-3.0](https://www.gnu.org/licenses/gpl-3.0.html)。本网站仅做独立的数据格式兼容；"
        "请从原仓库获取并独立运行该软件。"
    )
    st.markdown(
        "**请引用：** Frey, B., Snyder, P., Ziock, K., & Passian, A. (2021). "
        "*Semicomputational calculation of Bragg shift in stratified materials*. "
        "Physical Review E, 104(5), 055307."
    )
    st.markdown("""
**可输入并处理：**

- **空白分隔 `.txt` / `.dat`**：若已是八列数值，可直接上传并下载标准化版本；
- **逗号分隔 `.csv`** 或 **制表符文本**：选择相应分隔符、跳过表头行后上传；网站会转换为 FBG-SimPlus 读取的空白分隔文本；
- **任意 FEM 或自定义脚本的路径数据**：只要导出为上述文本形式且列含义一致即可；来源可以是任意仿真软件或实验预处理脚本。

**不能直接输入：** Excel `.xlsx`、原生模型文件（例如 `.mph`、`.odb`、`.rst`）和图像/PDF。请先在原软件或表格软件中导出为 CSV、制表符或空白分隔文本，再在本页处理。

固定八列依次为：`位置`、`εxx`、`εyy`、`εzz`、`σxx`、`σyy`、`σzz`、`温度`。位置统一使用 m 或 mm；三个应变无量纲；三个正应力为 Pa；温度为 K。
""")
    with st.expander("完整安装与使用说明（在本机独立运行 FBG-SimPlus）", expanded=False):
        st.markdown(r"""
### 1. 下载原项目

请从原作者仓库获取完整程序、许可证与教程。Windows 与 macOS/Linux 的命令分别如下。

#### Windows（PowerShell）

```powershell
git clone https://github.com/benfrey/FBG-SimPlus.git
cd FBG-SimPlus
```

#### macOS / Linux（Terminal / Bash）

```bash
git clone https://github.com/benfrey/FBG-SimPlus.git
cd FBG-SimPlus
```

也可在原仓库选择 **Code → Download ZIP**，解压后进入 `FBG-SimPlus` 目录。本网站不提供其源码或安装包。

### 2. 配置独立 Python 环境

FBG-SimPlus README 指定 Python 3.8。不要复用本网站的 Python 环境。

#### Windows（PowerShell）

确认已安装 Python 3.8 后，在 `FBG-SimPlus` 根目录复制执行：

```powershell
py -3.8 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install PyQt5 scipy matplotlib sympy six numpy
cd python
python run.py
```

若 PowerShell 禁止激活脚本，可仅对当前用户执行一次 `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`，再重新执行激活命令。

#### macOS / Linux（Terminal / Bash）

确认已安装 Python 3.8 后，在 `FBG-SimPlus` 根目录复制执行：

```bash
python3.8 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install PyQt5 scipy matplotlib sympy six numpy
cd python
python run.py
```

若找不到 Windows 的 `py -3.8`，或 macOS/Linux 的 `python3.8`，请先安装 Python 3.8。该原项目以 Python 3.8 为目标；更高版本的兼容性不在此页保证。

### 3. 准备并检查通用八列输入

从任意 FEM 软件、实验预处理脚本或表格软件导出数据。数据必须按如下顺序给出：位置、`εxx`、`εyy`、`εzz`、`σxx`、`σyy`、`σzz`、温度。支持空白分隔 `.txt/.dat`、CSV 与制表符文本；通过本页将其标准化为 FBG-SimPlus 所需的空白分隔八列 `.txt`。

先用本页的模板和上传框预检。预检仅检查八列、数值有效性和位置递增性，不替代 FEM 建模、材料参数或 FBG 标定验证。

### 4. 在应用内生成光谱

1. 在 **Select Stressed/Strained Path Files** 区域点击 **Add Files**，选中导出的 `.txt`。
2. 设置 **Skip Rows** 为文件开头元数据和表头的行数。原仓库教程 `tutorial/tut-export.txt` 的 `%` 开头元数据共有 7 行，教程文件填 `7`；你的文件按实际行数填写。
3. 在 **Path Distance Input Units** 选择第一列实际单位 `[m]` 或 `[mm]`。
4. 设置 FBG 数量、每个 FBG 的路径位置、FBG 长度和初始 Bragg 波长；位置单位必须与第 3 步一致。
5. 按你的模型设置均匀/非均匀应变、温度模拟、宿主热膨胀系数及其他光学参数。示例参数不等同于真实实验标定值。
6. 点击 **Generate** 计算模拟结果，点击 **Plot** 查看反射谱；完整参数含义请查阅原仓库的 `documentation.pdf` 和 `tutorial/`。

### 5. 已知限制与署名

原作者已说明：谱图绘制可能不稳定、macOS 退出时可能需要强制结束、图片保存可能不稳定。使用该软件的方法或结果时，请保留本页顶部的 Frey 等人论文引用、原项目链接与 GPL-3.0 许可说明。
""")
    template = (
        "% Generic eight-column input compatible with FBG-SimPlus\n"
        "% position_m exx eyy ezz sxx_pa syy_pa szz_pa temperature_k\n"
        "0.0000 0.002000 0.000100 -0.000200 100.0 20.0 -10.0 293.15\n"
        "0.0010 0.001000 0.000200 -0.000100 80.0 15.0 -8.0 293.15\n"
    )
    st.download_button("下载通用八列文本模板", template.encode("utf-8"), "fbg_simplus_eight_column_template.txt", "text/plain")
    input_left, input_right = st.columns(2)
    with input_left:
        input_delimiter = st.selectbox("输入分隔符", ["自动识别", "空白字符", "逗号（CSV）", "制表符（TSV）"], key="fbg_simplus_delimiter")
    with input_right:
        input_skip_rows = st.number_input("跳过文件开头行数", min_value=0, value=0, step=1, key="fbg_simplus_skip_rows")
    uploaded_export = st.file_uploader("上传八列数据（.txt / .dat / .csv）", type=["txt", "dat", "csv"], key="fbg_simplus_input")
    if uploaded_export is not None:
        try:
            parsed_export = models.parse_fbg_simplus_comsol_export(
                uploaded_export.getvalue().decode("utf-8-sig"), input_delimiter, int(input_skip_rows)
            )
        except (UnicodeDecodeError, ValueError) as error:
            st.error(f"无法作为 FBG-SimPlus 兼容输入读取：{error}")
        else:
            st.success(f"已通过格式检查：{parsed_export['sample_count']} 个采样点；识别为{parsed_export['source_delimiter']}。可下载标准化文本后导入 FBG-SimPlus。")
            st.plotly_chart(visuals.fbg_simplus_input_figure(parsed_export), width="stretch")
            st.download_button("下载标准化八列文本（供 FBG-SimPlus 导入）", models.fbg_simplus_normalised_text(parsed_export).encode("utf-8"), "fbg_simplus_normalised_input.txt", "text/plain")
            st.info("检查范围仅限文本结构、数值有效性和位置递增性；应变/应力分量的物理含义、单位和 FBG 参数仍需由数据来源和 FBG-SimPlus 文档确认。")

with polarization_tab:
    st.subheader("偏振与干涉传感：偏振态、旋转与微腔光程差")
    pol_left, pol_right = st.columns([1, 2])
    with pol_left:
        transverse_stress = st.slider("横向应力 (MPa)", 0.0, 250.0, 120.0, 5.0)
        twist_angle = st.slider("光纤扭转 (°)", -90.0, 90.0, 35.0, 1.0)
        gyro_rate = st.slider("角速度 (°/s)", -180.0, 180.0, 45.0, 1.0)
        cavity_pressure = st.slider("EFPI 压力 (MPa)", 0.0, 1.0, 0.4, 0.01)
    polarization = models.simulate_polarization_sensing(transverse_stress, twist_angle, temperature)
    gyro = models.simulate_sagnac_gyro(gyro_rate, 120.0)
    efpi = models.simulate_efpi_pressure(cavity_pressure, 28.0)
    with pol_right:
        st.plotly_chart(visuals.polarization_figure(polarization), width="stretch")
        st.plotly_chart(visuals.efpi_figure(efpi), width="stretch")
    pol_a, pol_b, pol_c = st.columns(3)
    pol_a.metric("偏振方位角", f"{polarization['azimuth_deg']:.1f} °")
    pol_b.metric("椭圆率角", f"{polarization['ellipticity_deg']:.1f} °")
    pol_c.metric("Sagnac 相位差", f"{gyro['phase_shift_rad']:.3e} rad")
    st.caption("偏振态模块用于理解双折射、扭转与温度对 Stokes 参数的影响；Sagnac 和 EFPI 用于对比旋转相位与腔长干涉。均为教学模型，不代表惯导或压力传感器精度。")

with chain_tab:
    st.subheader("解调器与实时数据链路：波长峰值 → 滤波温补 → 状态 → 控制")
    st.caption("基础版以两秒钟的单路 FBG 数据流演示解调后的原始波长、移动平均滤波、共模温度补偿、弯曲角反演和控制输出。")
    chain_left, chain_right = st.columns([1, 2])
    with chain_left:
        chain_angle = st.slider("链路真实弯曲角 (°)", 0.0, 90.0, 55.0, 1.0)
        chain_noise = st.slider("链路波长噪声 (nm)", 0.0, 0.030, 0.010, 0.001)
    chain = models.simulate_demodulation_chain(chain_angle, temperature, int(sample_rate), chain_noise, int(seed))
    with chain_right:
        st.plotly_chart(visuals.demodulation_figure(chain), width="stretch")
    chain_a, chain_b, chain_c = st.columns(3)
    chain_a.metric("反演弯曲角", f"{chain['estimated_angle_deg']:.1f} °", f"误差 {chain['estimated_angle_deg'] - chain_angle:+.1f} °")
    chain_b.metric("控制输出", str(chain["control_command"]))
    chain_c.metric("采样率", f"{sample_rate} Hz")

    fusion = models.fuse_robot_sensing(
        bool(three_d_fbg_decision["is_grasped"]),
        float(np.clip(1.0 - abs(cop - 2.5) / 2.5, 0.0, 1.0)),
        float(np.clip(1.0 - abs(shape["estimated_curvature_per_m"] - curvature) / 5.0, 0.0, 1.0)),
        str(diagnosis["status"]),
        float(distributed_frame["quality"]),
    )
    fusion_a, fusion_b = st.columns(2)
    fusion_a.metric("多模态任务状态", str(fusion["status"]))
    fusion_b.metric("融合置信度", f"{float(fusion['confidence']) * 100:.0f}%")

    st.divider()
    st.subheader("实验任务与报告")
    experiment = st.selectbox("实验任务", ["弯曲标定与温补", "冗余故障诊断", "多材质触觉识别", "足底平衡", "结构健康监测"])
    task_results = {
        "弯曲标定与温补": (abs(float(chain["estimated_angle_deg"]) - chain_angle), "角度反演误差"),
        "冗余故障诊断": (0.0 if fault_mode == "无" or redundant_diagnosis["fault_channels"] else 1.0, "故障识别残差"),
        "多材质触觉识别": (0.0 if material_diagnosis["material"] == material else 1.0, "材质分类误差"),
        "足底平衡": (abs(float(cop) - 2.5), "CoP 偏离"),
        "结构健康监测": (float(diagnosis["damage_index"]), "局部异常指数"),
    }
    task_value, task_label = task_results[experiment]
    st.metric(f"{experiment}：{task_label}", f"{task_value:.2f}")
    report = (
        "光纤机器人感知链实验报告（教学仿真）\n"
        f"任务：{experiment}\n采样率：{sample_rate} Hz\n温度变化：{temperature:.1f} °C\n"
        f"链路真实/反演弯曲角：{chain_angle:.1f}° / {chain['estimated_angle_deg']:.1f}°\n"
        f"控制输出：{chain['control_command']}\n{task_label}：{task_value:.3f}\n"
        f"故障诊断通道：{redundant_diagnosis['fault_channels'] or '无'}\n"
        f"材质识别：{material_diagnosis['material']}（{float(material_diagnosis['confidence']) * 100:.0f}%）\n"
        f"足底 CoP：区域 {cop:.2f}\n健康状态：{diagnosis['status']}，可疑位置 {diagnosis['suspected_location_mm']:.0f} mm\n"
        f"分布式机制：{distributed_frame['sensor_type']}；数据质量：{float(distributed_frame['quality']) * 100:.0f}%\n"
        f"多模态任务状态：{fusion['status']}；融合置信度：{float(fusion['confidence']) * 100:.0f}%\n"
        "说明：此报告基于解析教学模型，不可替代真实系统的标定、风险评估或安全决策。\n"
    )
    st.download_button("下载当前实验报告", report.encode("utf-8-sig"), "fiber_robotics_sensing_chain_report.txt", "text/plain")
