"""Streamlit entry point for the fiber robotics sensing lab."""

from __future__ import annotations

import csv
import io

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from fiber_robotics_sim import (
    eskin, eskin_experiments, eskin_visuals, experiments, models, visuals,
)


LAB_LABELS = (
    "① 系统总览",
    "② FBG 基础与解调",
    "③ 手部抓取与触觉",
    "④ 足底感知与装配",
    "⑤ 形状与结构监测",
    "⑥ 分布式光学与数据",
    "⑦ 电子皮肤与多模态感知",
)
LAB_SECTIONS = {
    "foundation": ("弯曲标定与诊断", "解调与实验任务"),
    "hand": ("二维抓取", "三维抓取", "材质识别"),
    "foot": ("平衡与步态", "装配校验"),
    "structure": ("连续体形状", "结构健康"),
    "optics": ("分布式感知", "偏振与干涉", "数据兼容"),
    "eskin": ("三轴触觉单元", "FBG 光学皮肤", "稀疏压力重建", "动态滑移与多模态"),
}

FBG_SIMPLUS_TEMPLATE = (
    "% Generic eight-column input compatible with FBG-SimPlus\n"
    "% position_m exx eyy ezz sxx_pa syy_pa szz_pa temperature_k\n"
    "0.0000 0.002000 0.000100 -0.000200 100.0 20.0 -10.0 293.15\n"
    "0.0010 0.001000 0.000200 -0.000100 80.0 15.0 -8.0 294.15\n"
)
FBG_SIMPLUS_EXAMPLES = {
    "标准空白八列": {
        "text": FBG_SIMPLUS_TEMPLATE, "delimiter": "自动识别", "skip_rows": 0,
        "expected": "预期结果：通过检查，并可比较八列数值范围。",
    },
    "CSV 含表头": {
        "text": (
            "position,exx,eyy,ezz,sxx,syy,szz,temperature\n"
            "0.0000,0.0020,0.0001,-0.0002,100,20,-10,293.15\n"
            "0.0010,0.0010,0.0002,-0.0001,80,15,-8,294.15\n"
        ),
        "delimiter": "自动识别", "skip_rows": 1,
        "expected": "预期结果：跳过一行表头后通过，并识别为 CSV。",
    },
    "列数不足（应失败）": {
        "text": "0.0000 0.0020 0.0001 -0.0002 100 20 293.15\n",
        "delimiter": "自动识别", "skip_rows": 0,
        "expected": "预期结果：拒绝不足八列的数据。",
    },
    "位置重复（应失败）": {
        "text": (
            "0.0000 0.0020 0.0001 -0.0002 100 20 -10 293.15\n"
            "0.0000 0.0010 0.0002 -0.0001 80 15 -8 294.15\n"
        ),
        "delimiter": "自动识别", "skip_rows": 0,
        "expected": "预期结果：拒绝位置重复的数据。",
    },
}


st.set_page_config(page_title="光纤机器人传感仿真实验室", page_icon="🦾", layout="wide")
st.markdown("""<style>
.block-container {max-width: 1450px; padding-top: 2rem; padding-bottom: 3rem;}
[data-testid="stSidebar"] {background: #101923;}
[data-testid="stSidebar"] * {color: #e8f1f7;}
[data-testid="stMetric"] {background: #172a3a; border: 1px solid #315064; border-radius: 12px; padding: 14px; min-height: 112px;}
[data-testid="stMetric"] * {color: #f5fbff !important;}
[data-testid="stMetricDelta"] {color: #65d6c3 !important;}
div.st-key-three_d_grasp_metrics [data-testid="stMetric"] {min-height: 64px; padding: 8px 12px;}
div.st-key-three_d_grasp_metrics [data-testid="stMetricLabel"] {font-size: .82rem;}
div.st-key-three_d_grasp_metrics [data-testid="stMetricValue"] {font-size: 1.75rem;}
[data-testid="stAlert"] {border-radius: 12px;}
div[data-testid="stTabs"] button {font-size: .95rem; font-weight: 600;}
div[data-testid="stTabs"] button p {font-size: .84rem;}
div[data-testid="stTabs"] [role="tablist"] {gap: .15rem;}
.element-container {margin-bottom: .55rem;}
[data-testid="stCaptionContainer"] p {text-align: center; text-align-last: left;}
h1 {letter-spacing: -.03em;}
h2, h3 {margin-top: .6rem;}
</style>""", unsafe_allow_html=True)
st.title("🦾 光纤机器人传感仿真实验室")
st.caption("用可解释的传感与重建模型，学习光纤机器人感知、触觉和电子皮肤。")


def csv_bytes(labels: list[str], values: np.ndarray) -> bytes:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["sensor", "wavelength_shift_nm"])
    writer.writerows(zip(labels, values, strict=True))
    return buffer.getvalue().encode("utf-8-sig")


def module_learning_frame(
    goal: str, action: str, observation: str, boundary: str
) -> None:
    """Render the same compact learning scaffold for content-rich modules."""
    with st.container(border=True):
        goal_column, action_column = st.columns(2)
        goal_column.markdown("**学习目标**")
        goal_column.write(goal)
        action_column.markdown("**建议操作**")
        action_column.write(action)
        observation_column, boundary_column = st.columns(2)
        observation_column.markdown("**重点观察**")
        observation_column.write(observation)
        boundary_column.markdown("**模型边界**")
        boundary_column.write(boundary)


def module_directory(groups: list[tuple[str, list[tuple[int, str, str]]]]) -> None:
    """Render native module buttons that update the tracked tab widget."""
    for group_name, items in groups:
        st.caption(group_name)
        columns = st.columns(3)
        for item_index, (tab_index, title, summary) in enumerate(items):
            with columns[item_index % 3]:
                with st.container(border=True):
                    st.button(
                        title,
                        key=f"module_jump_{tab_index}",
                        on_click=select_main_tab,
                        args=(LAB_LABELS[tab_index],),
                        use_container_width=True,
                    )
                    st.caption(summary)


def request_navigation(navigation_key: str, label: str) -> None:
    """Request a visible tab change by remounting only that tracked tab group."""
    st.session_state[navigation_key] = label
    target_key = f"{navigation_key}_target"
    revision_key = f"{navigation_key}_revision"
    st.session_state[target_key] = label
    st.session_state[revision_key] = int(st.session_state.get(revision_key, 0)) + 1


def sync_section_navigation(navigation_key: str, widget_key: str) -> None:
    """Copy a user's direct tab click into the stable navigation state."""
    st.session_state[navigation_key] = st.session_state[widget_key]


def sync_sidebar_navigation(widget_key: str) -> None:
    """Keep the sidebar aligned when the user clicks a top lab directly."""
    selected = st.session_state[widget_key]
    st.session_state.main_navigation = selected
    st.session_state.sidebar_module_navigation = selected


def tracked_tabs(labels: tuple[str, ...], navigation_key: str):
    """Render tabs that support both direct clicks and reliable callback jumps."""
    st.session_state.setdefault(navigation_key, labels[0])
    revision_key = f"{navigation_key}_revision"
    revision = int(st.session_state.get(revision_key, 0))
    widget_key = f"{navigation_key}_widget_{revision}"
    target_key = f"{navigation_key}_target"
    target = st.session_state.pop(target_key, None)
    if target is not None:
        if target not in labels:
            raise ValueError(f"unknown tab target for {navigation_key}: {target}")
        st.session_state.pop(widget_key, None)
    default = target or st.session_state[navigation_key]
    callback = sync_sidebar_navigation if navigation_key == "main_navigation" else sync_section_navigation
    callback_args = (widget_key,) if navigation_key == "main_navigation" else (navigation_key, widget_key)
    return st.tabs(labels, default=default, key=widget_key, on_change=callback, args=callback_args)


def select_main_tab(
    tab_label: str,
    section_key: str | None = None,
    section_label: str | None = None,
) -> None:
    """Select a tracked lab and, when supplied, its internal experiment."""
    request_navigation("main_navigation", tab_label)
    if section_key and section_label:
        request_navigation(section_key, section_label)
    if "sidebar_module_navigation" in st.session_state:
        st.session_state.sidebar_module_navigation = tab_label


def select_sidebar_module() -> None:
    """Use the compact sidebar selector to reach tabs hidden by horizontal scrolling."""
    request_navigation("main_navigation", st.session_state.sidebar_module_navigation)


def home_button() -> None:
    tab_jump_button(0, "← 返回主页")


def tab_jump_button(
    target_index: int,
    label: str,
    section_key: str | None = None,
    section_label: str | None = None,
) -> None:
    """Render a native button that switches a lab and optional experiment."""
    st.button(
        label,
        key=f"tab_jump_{target_index}_{label}",
        on_click=select_main_tab,
        args=(LAB_LABELS[target_index], section_key, section_label),
        use_container_width=True,
    )


GUIDED_ROUTES = {
    "基础标定与解调": {
        "id": "calibration",
        "duration": "约 25 分钟",
        "prerequisite": "无，建议从本路线开始",
        "goal": "从单根 FBG 的波长变化出发，完成温补、反演、故障诊断和解调链集成。",
        "deliverable": "一份可恢复标定 JSON，以及包含当前诊断与解调链摘要的中文报告。",
        "steps": (
            (1, "foundation_navigation", "弯曲标定与诊断", "建立理想基线", "载入“理想标定”，确认融合角接近真实角，再保存为基线 A。", "三路理想读数接近，温补前后角度一致。"),
            (1, "foundation_navigation", "弯曲标定与诊断", "比较温漂、故障与接触反演", "载入温漂或噪声对照，再在冗余诊断和指尖接触区各改变一个输入。", "温补修正共模温漂；冗余异常通道与接触位置反演使用不同判据。"),
            (1, "foundation_navigation", "解调与实验任务", "接入解调与控制链", "在实验任务页选择“弯曲标定与温补”或“冗余故障诊断”，核对控制输出并下载报告。", "报告同时包含当前任务记录和共享感知链摘要。"),
        ),
    },
    "抓取与触觉": {
        "id": "grasp",
        "duration": "约 35 分钟",
        "prerequisite": "建议先完成“基础标定与解调”",
        "goal": "观察物体位置、手指接触与多路 FBG 如何共同形成抓稳判断。",
        "deliverable": "二维、三维抓取记录与一份触觉材质 A/B 对照报告。",
        "steps": (
            (2, "hand_navigation", "二维抓取", "完成二维寻找与抓取", "启动二维任务并逐步执行，再提高波长噪声做 50 次重复采样。", "只有 FBG 接触条件满足后任务才进入搬运；重复统计会显示判定翻转。"),
            (2, "hand_navigation", "三维抓取", "核对三维抓稳条件", "启动三维任务，逐项查看三个达标条件，再比较三种传感器布置。", "握拳不等于抓稳；缺少必要通道的布置无法完整执行当前规则。"),
            (2, "hand_navigation", "材质识别", "比较触觉模式", "选择材质并改变握持力或接触扰动，比较五指与掌心的概率分布。", "噪声或扰动会降低分类集中度，结果不代表真实材料鉴定。"),
            (1, "foundation_navigation", "解调与实验任务", "汇总抓取相关状态", "在实验任务页选择“多材质触觉识别”，对照抓取、触觉与多模态融合状态。", "共享报告会保留当前触觉记录，但不会把教学判定提升为真实安全结论。"),
        ),
    },
    "足底与装配": {
        "id": "foot_assembly",
        "duration": "约 30 分钟",
        "prerequisite": "建议理解已知温度补偿和六路反演",
        "goal": "把六区足底载荷、CoP 反演与可更换足底的空载复装筛查联系起来。",
        "deliverable": "一份足底载荷/CoP 报告和一份装配验证参数摘要。",
        "steps": (
            (3, "foot_navigation", "平衡与步态", "比较步态与低载荷边界", "载入平地中期并保存基线，再切换到摆动期低载荷或通道失效。", "低载荷或失效会降低 CoP 可靠性，真实载荷不会因传感器失效而消失。"),
            (3, "foot_navigation", "装配校验", "执行空载装配筛查", "比较正常装配、压入不足和单侧错位，再观察容差、温差和使用载荷干扰。", "装配检查应在卸载条件下进行；密封与寿命仍需实物试验。"),
            (1, "foundation_navigation", "解调与实验任务", "归档足底实验", "在实验任务页选择“足底平衡”，下载当前报告并与装配参数摘要一起保存。", "两份记录分别描述使用载荷感知与空载装配筛查，不能混作同一验收结论。"),
        ),
    },
    "形状与结构健康": {
        "id": "structure",
        "duration": "约 40 分钟",
        "prerequisite": "建议先完成基础标定，理解点式与连续测量的区别",
        "goal": "比较点式、分布式与多芯光纤对形状和局部异常的表达能力。",
        "deliverable": "形状、异常位置和分布式响应的对照观察，以及综合报告。",
        "steps": (
            (4, "structure_navigation", "连续体形状", "重建连续体形状", "改变曲率、方向、已知扭转先验和温度梯度，对比真实中心线与反演中心线。", "芯间温度梯度会破坏理想差分补偿。"),
            (4, "structure_navigation", "结构健康", "定位结构异常", "改变异常位置、程度与阵列密度，记录定位区间如何变化。", "加密点式阵列会缩小区间，但仍不是连续测量。"),
            (5, "optics_navigation", "分布式感知", "比较分布式机制", "调节事件位置与空间采样，比较 Rayleigh、DAS、Brillouin 和 Raman。", "采样间隔过大会低估或漏掉局部峰。"),
            (1, "foundation_navigation", "解调与实验任务", "查看综合报告", "在实验任务页选择结构健康监测，并下载当前感知链报告。", "报告汇总当前条件和教学模型结果，不替代安全评估。"),
        ),
    },
    "光学机制与数据兼容": {
        "id": "optics_data",
        "duration": "约 35 分钟",
        "prerequisite": "建议先理解波长、温度和空间采样的基本概念",
        "goal": "区分分布式散射、偏振、Sagnac、EFPI 的观测量，并完成八列数据兼容检查。",
        "deliverable": "一份分布式或偏振实验报告，以及一份通过检查的标准化八列文本。",
        "steps": (
            (5, "optics_navigation", "分布式感知", "识别四种分布式观测量", "载入 Rayleigh、DAS、Brillouin 或 Raman 场景，保存基线并比较定位误差。", "不同机制输出曲线、热图、频移或温度；空间采样会改变峰值定位。"),
            (5, "optics_navigation", "偏振与干涉", "比较偏振与干涉机制", "依次载入偏振基线、温度交叉敏感和旋转压力场景。", "Stokes、Sagnac 相位和 EFPI 腔长是三类独立观测量。"),
            (5, "optics_navigation", "数据兼容", "完成八列数据预检", "载入内置八列示例，核对列顺序、位置递增和标准化下载，再阅读外部工具边界。", "本页只验证数据结构，不生成反射谱，也不验证物理单位或 FEM 模型。"),
            (1, "foundation_navigation", "解调与实验任务", "生成光学综合记录", "在实验任务页选择“分布式事件定位”或“偏振与干涉传感”并下载报告。", "综合报告保留当前参数和模型边界，便于与标准化八列文本配套归档。"),
        ),
    },
    "电子皮肤与多模态": {
        "id": "electronic_skin",
        "duration": "约 45 分钟",
        "prerequisite": "建议先理解温度补偿、阵列采样和误差指标",
        "goal": "从三轴触觉单元出发，完成 FBG 光学皮肤、稀疏压力重建和动态滑移判别。",
        "deliverable": "四类可复现实验数据、A/B 对照和带模型边界的电子皮肤报告。",
        "steps": (
            (6, "eskin_navigation", "三轴触觉单元", "校正三轴触觉单元", "载入‘三轴温漂校正’，保存基线后改变参考匹配度。", "参考结构只在匹配充分时降低共模误差；矩阵秩和条件数描述可辨识性。"),
            (6, "eskin_navigation", "FBG 光学皮肤", "比较光学感受野", "载入‘双点光学触觉’，比较 4、8、16 个 FBG。", "通道数量、感受野和温补共同影响载荷质心；双点结果不代表唯一分离。"),
            (6, "eskin_navigation", "稀疏压力重建", "评价稀疏压力场", "从 4×4 切换到 8×8 采样，同时查看五项误差指标。", "通道节省会伴随峰值、质心或总载荷信息损失，热图相似不是唯一标准。"),
            (6, "eskin_navigation", "动态滑移与多模态", "判断动态事件", "比较稳定按压、即将滑移和热物体，再观察重复告警率。", "剪切比和质心速度共同触发教学告警；温度升高本身不应被解释为滑移。"),
        ),
    },
}


def normalise_guided_progress(existing: dict | None) -> dict[str, list[bool]]:
    """Preserve matching progress while adding routes or changing step counts."""
    progress = dict(existing or {})
    for route in GUIDED_ROUTES.values():
        stored = list(progress.get(route["id"], ()))
        step_count = len(route["steps"])
        progress[route["id"]] = (stored + [False] * step_count)[:step_count]
    return progress


def guided_route_report(route_name: str, route: dict, progress: list[bool]) -> bytes:
    """Export the visible route, manual completion marks, and expected observations."""
    covered_modules = " → ".join(LAB_LABELS[step[0]] for step in route["steps"])
    lines = [
        "光纤机器人仿真实验室 · 引导式学习记录",
        f"路线：{route_name}",
        f"预计时间：{route['duration']}；前置建议：{route['prerequisite']}",
        f"学习目标：{route['goal']}",
        f"覆盖模块：{covered_modules}",
        f"最终产物：{route['deliverable']}",
        f"手动完成进度：{sum(progress)} / {len(progress)}",
        "步骤记录：",
    ]
    for index, ((lab_index, section_key, section_label, title, action, observation), completed) in enumerate(
        zip(route["steps"], progress, strict=True), start=1
    ):
        lines.extend((
            f"{index}. [{'已完成' if completed else '未完成'}] {LAB_LABELS[lab_index]} / {section_label} · {title}",
            f"   操作：{action}",
            f"   预期观察：{observation}",
        ))
    lines.append("说明：完成标记由学习者手动勾选，不代表模型正确或真实系统验证通过。")
    return "\n".join(lines).encode("utf-8-sig")


def update_guided_progress(route_id: str, step_index: int, widget_key: str) -> None:
    """Copy a visible checkbox into the route store that survives route changes."""
    progress = dict(st.session_state["guided_route_progress"])
    route_progress = list(progress[route_id])
    route_progress[step_index] = bool(st.session_state[widget_key])
    progress[route_id] = route_progress
    st.session_state.guided_route_progress = progress


def reset_guided_route(route_id: str, step_count: int) -> None:
    """Only clear the manually confirmed progress for one learning route."""
    progress = dict(st.session_state["guided_route_progress"])
    progress[route_id] = [False] * step_count
    st.session_state.guided_route_progress = progress
    for index in range(step_count):
        widget_key = f"guided_progress_{route_id}_{index}"
        if widget_key in st.session_state:
            st.session_state[widget_key] = False


with st.sidebar:
    home_button()
    st.selectbox(
        "快速模块导航",
        LAB_LABELS,
        key="sidebar_module_navigation",
        on_change=select_sidebar_module,
    )
    st.caption("用于直接跳转 7 个领域实验室；进入后可切换具体实验。")
    st.divider()
    st.header("公共光学与测量参数")
    public_defaults = {
        "global_temperature": 0.0,
        "global_noise": 0.0,
        "global_drift": 0.0,
        "global_sample_rate": 50,
        "global_failed_channel": "无",
        "sole_assembly_case": "正常装配",
        "global_seed": 7,
    }
    for key, value in public_defaults.items():
        st.session_state.setdefault(key, value)
    temperature = st.slider("温度变化 ΔT (°C)", -20.0, 50.0, step=0.5, key="global_temperature")
    noise = st.slider("波长测量噪声 σ (nm)", 0.0, 0.020, step=0.0005, format="%.4f", key="global_noise")
    drift = st.slider("零点漂移 (nm)", 0.0, 0.020, step=0.0005, format="%.4f", key="global_drift")
    sample_rate = st.select_slider("采样率 (Hz)", options=[10, 25, 50, 100, 200], key="global_sample_rate")
    failed = st.selectbox("模拟失效通道", ["无", "手部 FBG 1", "手部 FBG 2", "手部 FBG 3", "足底区域 1"], key="global_failed_channel")
    sole_assembly_case = st.selectbox(
        "可更换足底复装工况（教学设置）",
        ["正常装配", "压入不足", "单侧错位"],
        key="sole_assembly_case",
    )
    seed = st.number_input("随机种子", min_value=0, max_value=2**32 - 1, step=1, key="global_seed")
    def apply_demo_preset() -> None:
        for key, value in public_defaults.items():
            st.session_state[key] = value
    st.button("演示预设（恢复公共参数）", key="demo_preset", on_click=apply_demo_preset)
    st.caption("仅恢复上方公共参数；不重置各页局部参数、手部姿态或任务进度。")
    st.checkbox("平滑过渡动画（2D/3D 手）", value=True, key="smooth_animation")
    st.divider()
    st.info("FBG 模型：Δλᵦ = λᵦ[(1−pₑ)ε + kₜΔT]。所有页面均显示真实量与反演量。")

overview_tab, foundation_lab, hand_lab, foot_lab, structure_lab, optics_lab, eskin_lab = tracked_tabs(
    LAB_LABELS, "main_navigation"
)
with foundation_lab:
    calibration_tab, chain_tab = tracked_tabs(
        LAB_SECTIONS["foundation"], "foundation_navigation"
    )
with hand_lab:
    hand_tab, hand_3d_tab, tactile_tab = tracked_tabs(
        LAB_SECTIONS["hand"], "hand_navigation"
    )
with foot_lab:
    foot_tab, assembly_tab = tracked_tabs(
        LAB_SECTIONS["foot"], "foot_navigation"
    )
with structure_lab:
    shape_tab, health_tab = tracked_tabs(
        LAB_SECTIONS["structure"], "structure_navigation"
    )
with optics_lab:
    distributed_tab, polarization_tab, fbg_simplus_tab = tracked_tabs(
        LAB_SECTIONS["optics"], "optics_navigation"
    )

with overview_tab:
    st.subheader("引导式实验路线")
    st.caption("六条路线覆盖 6 个专业领域内的全部实验。先选目标，再按步骤进入对应子实验；路线只记录学习进度，不修改实验参数、姿态或任务状态。")
    st.session_state.guided_route_progress = normalise_guided_progress(
        st.session_state.get("guided_route_progress")
    )
    guided_route_name = st.selectbox(
        "选择实验路线", list(GUIDED_ROUTES), key="guided_route"
    )
    guided_route = GUIDED_ROUTES[guided_route_name]
    guided_steps = guided_route["steps"]
    route_progress = st.session_state.guided_route_progress[guided_route["id"]]
    completed_steps = sum(route_progress)
    covered_modules = list(dict.fromkeys(LAB_LABELS[step[0]] for step in guided_steps))
    route_summary, route_reset = st.columns([3.5, 1.5])
    with route_summary:
        st.markdown(f"**学习目标：** {guided_route['goal']}")
        st.markdown(f"**最终产物：** {guided_route['deliverable']}")
        st.caption(f"预计时间：{guided_route['duration']} · 前置建议：{guided_route['prerequisite']}")
        st.caption("覆盖模块：" + " → ".join(covered_modules))
    with route_reset:
        st.button(
            "重置本路线进度",
            key="reset_guided_route",
            on_click=reset_guided_route,
            args=(guided_route["id"], len(guided_steps)),
            width="stretch",
        )
        st.download_button(
            "下载当前路线学习记录",
            guided_route_report(guided_route_name, guided_route, route_progress),
            f"guided_route_{guided_route['id']}.txt",
            "text/plain",
            width="stretch",
        )
    st.progress(completed_steps / len(guided_steps))
    st.caption(f"进度 {completed_steps} / {len(guided_steps)}。完成勾选仅记录学习进度，不代表模型结果正确或真实系统验证通过。")
    for step_index, (lab_index, section_key, section_label, title, action, observation) in enumerate(guided_steps):
        with st.container(border=True):
            step_text, step_action = st.columns([4, 1])
            with step_text:
                st.markdown(f"**第 {step_index + 1} 步 · {title}**")
                st.write(action)
                st.caption(f"预期观察：{observation}")
                progress_key = f"guided_progress_{guided_route['id']}_{step_index}"
                st.session_state.setdefault(
                    progress_key,
                    st.session_state.guided_route_progress[guided_route["id"]][step_index],
                )
                st.checkbox(
                    "我已完成并记录观察",
                    key=progress_key,
                    on_change=update_guided_progress,
                    args=(guided_route["id"], step_index, progress_key),
                )
            with step_action:
                tab_jump_button(lab_index, f"进入第 {step_index + 1} 步", section_key, section_label)
    st.divider()
    st.subheader("模块目录")
    module_directory([
        ("领域实验室", [
            (1, "FBG 基础与解调", "标定、温补、冗余诊断、接触反演与控制报告"),
            (2, "手部抓取与触觉", "二维与三维抓取、六路接触感知和材质识别"),
            (3, "足底感知与装配", "六区载荷、CoP、通道失效和空载复装筛查"),
            (4, "形状与结构监测", "多芯连续体重建和点式阵列异常定位"),
            (5, "分布式光学与数据", "分布式、偏振与干涉观测及八列数据兼容"),
            (6, "电子皮肤与多模态感知", "三轴触觉、光学皮肤、压力重建和动态滑移判别"),
        ]),
    ])
    st.subheader("系统架构示意")
    st.iframe(visuals.sensing_chain_svg(), height=220)
    st.subheader("当前测量配置")
    config_a, config_b, config_c, config_d, config_e, config_f = st.columns(6)
    config_a.metric("温度变化", f"{temperature:.1f} °C")
    config_b.metric("采样率", f"{sample_rate} Hz")
    config_c.metric("波长噪声", f"{noise:.4f} nm")
    config_d.metric("随机种子", f"{int(seed)}")
    config_e.metric("模拟失效通道", failed)
    config_f.metric("复装工况", sole_assembly_case)
    st.subheader("感知链与模块职责")
    st.markdown(
        "| 环节 | 当前模块 | 输入 | 输出 |\n"
        "|---|---|---|---|\n"
        "| 机械交互 | 二维/三维抓取、多材质触觉、足底 | 姿态、接触、载荷 | FBG 接触与应变读数 |\n"
        "| 光纤解调 | FBG 标定与诊断、解调器链路 | 原始波长、温度、噪声 | 温补波长、异常通道 |\n"
        "| 状态估计 | 足底、连续体形状、机械臂健康 | 多路 FBG | CoP、曲率、异常位置 |\n"
        "| 分布式与偏振 | 分布式感知、偏振与干涉 | 空间/时间观测量 | 应变、振动、温度、偏振态 |\n"
        "| 控制与任务 | 解调器与实验任务 | 估计状态 | 张开/闭合命令、实验报告 |\n"
        "| 辅助与兼容 | 装配校验、FBG-SimPlus 兼容 | 结构/文件输入 | 装配预测、标准化八列文本 |\n"
        "| 电子皮肤 | 三轴单元、光学皮肤、压力重建、动态判别 | 多轴力、压力场、时间序列 | 力分量、质心、重建误差、滑移风险 |"
    )
    st.caption("感知链各环节对应上方目录卡片，可点击卡片直接跳转到对应页面。")
    st.subheader("推荐实验路径")
    st.markdown(
        "1. **基础标定与解调**：理解波长、温补、故障诊断和解调输出。\n"
        "2. **抓取与触觉**：从二维接触推进到三维抓稳与材质模式。\n"
        "3. **足底与装配**：区分使用载荷感知和空载装配筛查。\n"
        "4. **形状与结构健康**：比较多芯重建、点式阵列与分布式测量。\n"
        "5. **光学机制与数据兼容**：比较分布式、偏振和干涉观测量，并完成八列数据预检。\n"
        "6. **电子皮肤与多模态**：从三轴单元推进到压力场和动态滑移判别。"
    )
    st.info("所有页面均为可解释的教学解析模型。它们适合比较传感规律与算法流程，但真实系统仍须使用封装、温度场、动态载荷和设备标定数据进行验证。")

with hand_tab:
    st.subheader("机器人手：FBG 弯曲、指尖触觉与关节状态")
    module_learning_frame(
        "理解五指屈曲、接触力与六路 FBG 读数如何共同形成二维抓取判定。",
        "先用‘抓取’预设建立接触，再改变目标位置或侧栏噪声，比较单次结果与重复采样统计。",
        "同时检查抓稳率、判定一致率、翻转次数和不同传感器布置的通道覆盖。",
        "二维判定要求拇指与至少两根其余手指达到阈值；掌心通道提供接触背景，但不参与当前二维抓稳规则。",
    )
    action_order = ("抬臂", "伸手", "抓取", "按压", "松开", "复位")
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
        st.session_state.can_world_center = np.asarray(models.dexterous_hand_pose("抓取")["target"])
        st.session_state.can_grasped = False
        st.session_state.can_relative_to_palm = np.zeros(2)
    if "can_position_x" not in st.session_state:
        st.session_state.can_position_x = float(st.session_state.can_world_center[0])
        st.session_state.can_position_y = float(st.session_state.can_world_center[1])
    for key in ("shoulder_translation_x", "shoulder_translation_z"):
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
        for key in ("shoulder_translation_x", "shoulder_translation_z"):
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
        base_pose = models.dexterous_hand_pose("抓取", alignment_angles, alignment_curls)
        target = np.asarray(base_pose["target"])
        can_world = np.asarray(st.session_state.can_world_center)
        st.session_state.shoulder_translation_x = float(can_world[0] - target[0])
        st.session_state.shoulder_translation_z = float(can_world[1] - target[1])

    def current_two_d_grasp_is_verified() -> tuple[bool, dict, np.ndarray]:
        """Read the closed pose and resolve FBG grasp state before controls lock."""
        joint_angles = tuple(st.session_state[key] for key in ("manual_shoulder", "manual_elbow", "manual_wrist"))
        finger_curls = tuple(st.session_state[key] for key in ("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little"))
        pose = models.dexterous_hand_pose(
            st.session_state.arm_action,
            joint_angles,
            finger_curls,
            0.0,
            (st.session_state.shoulder_translation_x, st.session_state.shoulder_translation_z),
        )
        can_center = np.asarray(st.session_state.can_world_center, dtype=float)
        grasp = models.evaluate_can_grasp(pose, can_center)
        sensing = models.simulate_planar_grasp_fbg(
            finger_curls,
            grasp["contact_force_n"],
            temperature,
        )
        decision = models.classify_planar_grasp_from_fbg(sensing, finger_curls, temperature)
        return bool(decision["is_grasped"]), pose, can_center

    def remember_two_d_render_state() -> None:
        """Keep the last complete scene so the next task command can animate from it."""
        st.session_state.two_d_previous_pose = models.dexterous_hand_pose(
            st.session_state.arm_action,
            tuple(st.session_state[key] for key in ("manual_shoulder", "manual_elbow", "manual_wrist")),
            tuple(st.session_state[key] for key in ("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little")),
            0.0,
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
            st.session_state.two_d_task_phase = models.next_grasp_task_phase(phase, False)
            return
        if phase == "对准目标":
            align_hand_to_two_d_target()
            st.session_state.two_d_task_phase = models.next_grasp_task_phase(phase, False)
            return
        if phase == "闭合抓取":
            apply_action_pose("抓取")
            verified, pose, can_center = current_two_d_grasp_is_verified()
            if verified:
                st.session_state.can_grasped = True
                st.session_state.can_relative_to_palm = can_center - np.asarray(pose["palm_center"])
                apply_two_d_transport_pose()
                st.session_state.two_d_task_phase = models.next_grasp_task_phase("闭合抓取", True)
            else:
                st.session_state.two_d_task_phase = models.next_grasp_task_phase("闭合抓取", False)
            return
        if phase == "搬运目标":
            transport_pose = models.dexterous_hand_pose(
                st.session_state.arm_action,
                tuple(st.session_state[key] for key in ("manual_shoulder", "manual_elbow", "manual_wrist")),
                tuple(st.session_state[key] for key in ("manual_thumb", "manual_index", "manual_middle", "manual_ring", "manual_little")),
                0.0,
                (st.session_state.shoulder_translation_x, st.session_state.shoulder_translation_z),
            )
            released_center = np.asarray(transport_pose["palm_center"]) + st.session_state.can_relative_to_palm
            st.session_state.can_world_center = released_center
            st.session_state.can_position_x = float(released_center[0])
            st.session_state.can_position_y = float(released_center[1])
            apply_action_pose("松开")
            st.session_state.can_grasped = False
            st.session_state.two_d_task_phase = models.next_grasp_task_phase("搬运目标", True)
            return
        if phase == "松开并放置":
            st.session_state.two_d_task_phase = models.next_grasp_task_phase(phase, False)

    def release_can() -> None:
        release_pose = models.dexterous_hand_pose(
            st.session_state.arm_action,
            (st.session_state.manual_shoulder, st.session_state.manual_elbow, st.session_state.manual_wrist),
            (st.session_state.manual_thumb, st.session_state.manual_index, st.session_state.manual_middle, st.session_state.manual_ring, st.session_state.manual_little),
            0.0,
            (st.session_state.get("shoulder_translation_x", 0.0), st.session_state.get("shoulder_translation_z", 0.0)),
        )
        released_center = np.asarray(release_pose["palm_center"]) + st.session_state.can_relative_to_palm if st.session_state.can_grasped else st.session_state.can_world_center
        st.session_state.can_world_center = np.asarray(released_center, dtype=float)
        st.session_state.can_position_x = float(released_center[0])
        st.session_state.can_position_y = float(released_center[1])
        st.session_state.can_grasped = False

    action = st.session_state.arm_action
    two_d_controls_unlocked = st.session_state.two_d_task_phase in ("未启动", "松开并放置", "完成")
    planar_controls, planar_display = st.columns([1, 2])
    with planar_controls:
        st.markdown("#### 指令")
        preset_rows = [*st.columns(3), *st.columns(3)]
        for column, action_name in zip(preset_rows, action_order):
            if column.button(action_name, key=f"action_{action_name}", disabled=not two_d_controls_unlocked):
                apply_action_pose(action_name)
                if action_name == "松开":
                    st.session_state.can_grasped = False
        task_left, task_right = st.columns(2)
        task_left.button("开始二维寻找与抓取任务", key="start_two_d_grasp_task", on_click=begin_two_d_grasp_task, disabled=not two_d_controls_unlocked)
        task_right.button("执行下一步" if st.session_state.two_d_task_phase != "抓取失败" else "重新对准目标", key="advance_two_d_grasp_task", on_click=advance_two_d_grasp_task, disabled=st.session_state.two_d_task_phase in ("未启动", "完成"))
        if st.session_state.two_d_task_phase != "未启动":
            st.caption(f"二维任务状态：{st.session_state.two_d_task_phase}。物体世界坐标保持固定，手部向目标移动；抓取仅由 FBG 判定。")
        st.markdown("#### 姿态与目标")
        st.caption("可先用上方预设进入姿态，再单独调节每个关节。手指碰到罐体后会产生接触力并反映到 FBG 读数；只有拇指与至少两根手指的接触力都达到阈值才会绑定到掌心。")

        def controlled_slider(label: str, minimum: float, maximum: float, initial: float, key: str) -> float:
            if key not in st.session_state:
                st.session_state[key] = initial
            return st.slider(label, minimum, maximum, step=1.0, key=key, disabled=not two_d_controls_unlocked)

        arm_a, arm_b, arm_c = st.columns(3)
        with arm_a:
            shoulder = controlled_slider("肩关节 (°)", -20.0, 100.0, action_poses[action][0][0], "manual_shoulder")
        with arm_b:
            elbow = controlled_slider("肘关节 (°)", -100.0, 40.0, action_poses[action][0][1], "manual_elbow")
        with arm_c:
            wrist = controlled_slider("腕关节 (°)", -70.0, 70.0, action_poses[action][0][2], "manual_wrist")
        finger_a, finger_b = st.columns(2)
        with finger_a:
            thumb = controlled_slider("拇指屈曲 (°)", 0.0, 95.0, action_poses[action][1][0], "manual_thumb")
            middle = controlled_slider("中指屈曲 (°)", 0.0, 95.0, action_poses[action][1][2], "manual_middle")
            little = controlled_slider("小指屈曲 (°)", 0.0, 95.0, action_poses[action][1][4], "manual_little")
        with finger_b:
            index = controlled_slider("食指屈曲 (°)", 0.0, 95.0, action_poses[action][1][1], "manual_index")
            ring = controlled_slider("无名指屈曲 (°)", 0.0, 95.0, action_poses[action][1][3], "manual_ring")
        st.markdown("#### 物体世界坐标与肩部位移")
        target_a, target_b = st.columns(2)
        with target_a:
            can_x = st.slider("饮料罐水平位置", -8.0, 10.0, step=0.1, key="can_position_x", disabled=st.session_state.can_grasped or not two_d_controls_unlocked)
            st.slider("肩部水平位移", -12.0, 12.0, step=0.1, key="shoulder_translation_x", disabled=not two_d_controls_unlocked)
        with target_b:
            can_y = st.slider("饮料罐垂直位置", -6.0, 8.0, step=0.1, key="can_position_y", disabled=st.session_state.can_grasped or not two_d_controls_unlocked)
            st.slider("肩部垂直位移", -12.0, 12.0, step=0.1, key="shoulder_translation_z", disabled=not two_d_controls_unlocked)
        st.button("放下饮料罐", key="release_can", on_click=release_can, disabled=not two_d_controls_unlocked)

    joint_angles = (shoulder, elbow, wrist)
    finger_curls = (thumb, index, middle, ring, little)
    planar_translation = (st.session_state.get("shoulder_translation_x", 0.0), st.session_state.get("shoulder_translation_z", 0.0))
    pose = models.dexterous_hand_pose(action, joint_angles, finger_curls, 0.0, planar_translation)
    if st.session_state.can_grasped and st.session_state.two_d_task_phase not in ("搬运目标", "松开并放置"):
        bound_center = np.asarray(pose["palm_center"]) + st.session_state.can_relative_to_palm
        if not models.evaluate_can_grasp(pose, bound_center)["is_grasped"]:
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
    grasp = models.evaluate_can_grasp(pose, can_center)
    planar_fbg = models.simulate_planar_grasp_fbg(
        finger_curls,
        grasp["contact_force_n"],
        temperature,
    )
    planar_fbg_decision = models.classify_planar_grasp_from_fbg(planar_fbg, finger_curls, temperature)
    if planar_fbg_decision["is_grasped"] and not st.session_state.can_grasped:
        st.session_state.can_grasped = True
        st.session_state.can_relative_to_palm = can_center - np.asarray(pose["palm_center"])
    grasp_label = "FBG 已抓稳：饮料罐会跟随掌心移动" if st.session_state.can_grasped else "FBG 未抓稳：请让拇指与至少两根手指形成触觉接触"
    if st.session_state.can_grasped:
        display_curls = finger_curls
    else:
        display_curls = tuple(float(value) for value in grasp["limited_curls_deg"])
    display_pose = models.dexterous_hand_pose(action, joint_angles, display_curls, 0.0, planar_translation)
    previous_pose = st.session_state.get("two_d_previous_pose", display_pose)
    previous_can_center = np.asarray(st.session_state.get("two_d_previous_can_center", can_center), dtype=float)
    previous_grasped = bool(st.session_state.get("two_d_previous_grasped", st.session_state.can_grasped))
    previous_contact_fingers = st.session_state.get("two_d_previous_contact_fingers", list(planar_fbg_decision["contact_fingers"]))
    with planar_display:
        if st.session_state.can_grasped:
            st.success(grasp_label)
        else:
            st.warning(grasp_label)
        planar_metrics = st.columns(3)
        planar_metrics[0].metric("FBG 触觉接触手指", f"{len(planar_fbg_decision['contact_fingers'])} / 5")
        planar_metrics[1].metric("FBG 反演接触合力", f"{np.asarray(planar_fbg_decision['contact_force_n']).sum():.2f} N")
        planar_metrics[2].metric("掌心 FBG 接触力", f"{planar_fbg_decision['palm_touch_n']:.2f} N")
        st.caption("动画自动播放手部从上一状态到当前状态的过渡；手指碰到罐体会停在接触面，接触力随屈曲增大。")
        st.iframe(
            visuals.planar_hand_animation_html(
                previous_pose,
                display_pose,
                previous_can_center,
                can_center,
                previous_grasped,
                st.session_state.can_grasped,
                previous_contact_fingers,
                list(planar_fbg_decision["contact_fingers"]),
                animate=st.session_state.get("smooth_animation", True),
            ),
            height=620,
        )
        st.plotly_chart(visuals.sensor_bar_figure(np.arange(1, 7), planar_fbg["wavelength_shifts_nm"], "二维抓取：五指与掌心六路 FBG 波长漂移"), width="stretch")
        st.caption(f"第 6 路为掌心 FBG：{'检测到掌心接触力' if planar_fbg_decision['palm_contact'] else '当前掌心接触力较弱'}。抓稳判定仍保持“拇指＋至少两根其余手指”规则，且需要接触力达到阈值。")
    finger_names = ["拇指", "食指", "中指", "无名指", "小指"]
    planar_record = {
        "dimension": "二维",
        "task_phase": st.session_state.two_d_task_phase,
        "is_grasped": bool(planar_fbg_decision["is_grasped"]),
        "contact_fingers": [finger_names[index] for index in planar_fbg_decision["contact_fingers"]],
        "contact_force_n": np.asarray(planar_fbg_decision["contact_force_n"], dtype=float).tolist(),
        "palm_touch_n": float(planar_fbg_decision["palm_touch_n"]),
        "wavelength_shifts_nm": np.asarray(planar_fbg["wavelength_shifts_nm"], dtype=float).tolist(),
        "temperature_c": temperature,
        "noise_nm": noise,
        "seed": int(seed),
        "target_position": np.asarray(can_center, dtype=float).tolist(),
    }
    planar_download_a, planar_download_b = st.columns(2)
    planar_download_a.download_button(
        "下载二维抓取 FBG 读数 CSV",
        csv_bytes(["拇指 FBG", "食指 FBG", "中指 FBG", "无名指 FBG", "小指 FBG", "掌心 FBG"], np.asarray(planar_fbg["wavelength_shifts_nm"])),
        "planar_grasp_fbg_readings.csv", "text/csv",
    )
    planar_download_b.download_button(
        "下载二维抓取实验报告", experiments.grasp_report(planar_record).encode("utf-8-sig"),
        "planar_grasp_report.txt", "text/plain",
    )
    st.markdown("#### 重复采样与传感器布置实验")
    planar_repeat_samples = st.select_slider(
        "二维重复采样次数", options=[20, 50, 100, 200], value=50, key="planar_repeat_samples"
    )
    planar_study = experiments.run_planar_grasp_noise_study(
        planar_fbg, finger_curls, temperature, noise, int(planar_repeat_samples), (int(seed) + 330) % 2**32
    )
    planar_layouts = experiments.compare_grasp_sensor_layouts(
        planar_study["baseline_contact_force_n"],
        planar_study["baseline_palm_touch_n"],
        requires_palm=False,
    )
    planar_stats = st.columns(4)
    planar_stats[0].metric("无噪声基准", "抓稳" if planar_study["baseline_is_grasped"] else "未抓稳")
    planar_stats[1].metric("重复采样抓稳率", f"{planar_study['grasped_rate_percent']:.1f}%")
    planar_stats[2].metric("判定一致率", f"{planar_study['decision_consistency_percent']:.1f}%")
    planar_stats[3].metric("判定翻转", f"{planar_study['decision_flip_count']} 次")
    st.caption(
        f"当前 σ={noise:.4f} nm：五指反演合力 {planar_study['total_force_mean_n']:.3f} ± "
        f"{planar_study['total_force_std_n']:.3f} N；掌心反演力 {planar_study['palm_force_mean_n']:.3f} ± "
        f"{planar_study['palm_force_std_n']:.3f} N。判定一致率表示重复结果与无噪声当前姿态判定相同的比例。"
    )
    st.dataframe(planar_layouts, hide_index=True, width="stretch")
    st.caption("布置表的受力覆盖率只针对当前姿态，无接触力时记为 0%；‘可完整执行当前判定’表示所列通道能否观察二维规则所需的全部五指。")
    with st.expander("查看二维逐次采样记录", expanded=False):
        st.dataframe(planar_study["samples"], hide_index=True, width="stretch")
    planar_repeat_a, planar_repeat_b = st.columns(2)
    planar_repeat_a.download_button(
        "下载二维重复采样 CSV", experiments.grasp_noise_study_csv(planar_study),
        "planar_grasp_repeatability.csv", "text/csv",
    )
    planar_repeat_b.download_button(
        "下载二维抓取稳健性报告",
        experiments.grasp_robustness_report("二维", planar_study, planar_layouts).encode("utf-8-sig"),
        "planar_grasp_robustness_report.txt", "text/plain",
    )
    st.session_state.two_d_previous_pose = display_pose
    st.session_state.two_d_previous_can_center = np.asarray(can_center, dtype=float)
    st.session_state.two_d_previous_grasped = bool(st.session_state.can_grasped)
    st.session_state.two_d_previous_contact_fingers = list(planar_fbg_decision["contact_fingers"])

with tactile_tab:
    st.subheader("多材质触觉识别：五指与掌心 FBG 接触分布")
    module_learning_frame(
        "理解六路 FBG 接触分布如何区分软体、硬块、曲面与薄板。",
        "先载入标准海绵并保存基线 A，再载入其他场景或只提高扰动，比较当前 B。",
        "关注类别间隔、模板偏差和 24 次重复识别一致率，而不只看单次标签。",
        "这是预设模板的余弦相似度教学模型；没有使用真实材料训练集，不能替代实物鉴定。",
    )
    tactile_defaults = (
        ("tactile_material", "海绵"),
        ("tactile_grip_force", 5.0),
        ("tactile_contact_area", 35.0),
        ("tactile_pattern_noise", 0.0),
    )
    for key, value in tactile_defaults:
        st.session_state.setdefault(key, value)

    def load_tactile_preset() -> None:
        preset = experiments.TACTILE_PRESETS[st.session_state.tactile_preset]
        st.session_state.tactile_material = preset["material"]
        st.session_state.tactile_grip_force = preset["grip_force_n"]
        st.session_state.tactile_contact_area = preset["contact_area_percent"]
        st.session_state.tactile_pattern_noise = preset["pattern_noise"]
        st.session_state.global_temperature = preset["temperature_c"]
        st.session_state.global_noise = preset["noise_nm"]
        st.session_state.global_seed = preset["seed"]

    preset_column, load_column = st.columns([2, 1])
    with preset_column:
        st.selectbox("推荐实验场景", list(experiments.TACTILE_PRESETS), key="tactile_preset")
    with load_column:
        st.button(
            "载入触觉场景", key="load_tactile_preset", on_click=load_tactile_preset,
            width="stretch",
        )
    st.caption("载入场景会同步本页输入以及侧栏温度、波长噪声和随机种子；不会修改其他模块的局部参数或已有基线。")
    tactile_left, tactile_right = st.columns([1, 2])
    with tactile_left:
        material = st.selectbox(
            "目标材质", ["海绵", "硬块", "圆柱", "薄板"], key="tactile_material"
        )
        grip_force = st.slider(
            "握持力 (N)", 0.0, 12.0, step=0.1, key="tactile_grip_force"
        )
        contact_area = st.slider(
            "接触面积 (%)", 0.0, 100.0, step=1.0, key="tactile_contact_area"
        )
        pattern_noise = st.slider(
            "接触模式扰动", 0.0, 0.60, step=0.01, format="%.2f",
            key="tactile_pattern_noise",
        )
    tactile_record = experiments.run_tactile_experiment({
        "material": material,
        "grip_force_n": grip_force,
        "contact_area_percent": contact_area,
        "pattern_noise": pattern_noise,
        "temperature_c": temperature,
        "noise_nm": noise,
        "seed": int(seed),
    })
    tactile_results = tactile_record["results"]
    material_diagnosis = {
        "material": tactile_results["diagnosed_material"],
        "confidence": tactile_results["confidence"],
        "probabilities": tactile_results["probabilities"],
    }
    material_touch = {
        "wavelength_shifts_nm": np.asarray(tactile_results["wavelength_shifts_nm"]),
        "finger_touch_n": np.asarray(tactile_results["estimated_touch_n"][:5]),
        "palm_touch_n": float(tactile_results["estimated_touch_n"][5]),
    }
    with tactile_right:
        st.plotly_chart(
            visuals.sensor_bar_figure(np.arange(1, 7), material_touch["wavelength_shifts_nm"], "五指与掌心：六路触觉 FBG 波长漂移"),
            width="stretch",
        )
        st.caption("六路波长先按已知温度补偿并反演为接触力，再进入分类；侧栏波长噪声因此会真实影响本页分类结果。")
    touch_a, touch_b, touch_c, touch_d = st.columns(4)
    touch_a.metric("识别材质", str(material_diagnosis["material"]))
    touch_b.metric(
        "类别间隔",
        f"{tactile_results['probability_margin'] * 100:.2f} 个百分点" if tactile_results["valid_contact"] else "不适用",
    )
    touch_c.metric(
        "模板偏差",
        f"{tactile_results['pattern_error_percent']:.1f}%" if tactile_results["valid_contact"] else "不适用",
    )
    touch_d.metric(
        "重复一致率",
        f"{tactile_results['repeat_match_rate'] * 100:.0f}%" if tactile_results["valid_contact"] else "不适用",
    )
    if not tactile_results["valid_contact"]:
        st.warning(f"当前理想或反演总接触力低于 {experiments.TACTILE_VALID_CONTACT_N:.1f} N，材料标签与概率不应作为识别结论。请提高握持力或检查六路 FBG。")
    elif material_diagnosis["material"] == material:
        st.success(f"识别一致：{material}。")
    else:
        st.warning(f"识别不一致：实际为 {material}，模型识别为 {material_diagnosis['material']}。接触模式扰动或噪声使分布偏离模板。")
    st.plotly_chart(visuals.material_probability_figure(material_diagnosis["probabilities"]), width="stretch")
    st.caption(f"最高类别与次高类别“{tactile_results['runner_up_material']}”的概率差就是类别间隔；间隔越小，单次标签越不稳定。")
    touch_comparison = go.Figure()
    channel_labels = ["拇指", "食指", "中指", "无名指", "小指", "掌心"]
    touch_comparison.add_bar(name="理想模板", x=channel_labels, y=tactile_results["ideal_touch_n"])
    touch_comparison.add_bar(name="FBG 反演", x=channel_labels, y=tactile_results["estimated_touch_n"])
    touch_comparison.update_layout(
        title="理想模板与当前 FBG 反演接触力", barmode="group",
        yaxis_title="接触力 (N)", legend_orientation="h",
    )
    st.plotly_chart(touch_comparison, width="stretch")
    st.caption("模板偏差比较归一化后的六路形状，因此主要反映接触分布变化，而不是握持力整体变大或变小。重复一致率使用相同条件和连续随机种子重复 24 次。")

    if st.button("保存当前为触觉基线 A", key="save_tactile_baseline"):
        st.session_state.tactile_baseline = tactile_record
    tactile_baseline = st.session_state.get("tactile_baseline")
    if tactile_baseline is not None:
        baseline_results = tactile_baseline["results"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": [
                "目标材质", "握持力", "接触面积", "模式扰动", "温度变化",
                "波长噪声", "随机种子", "识别结果", "类别间隔", "模板偏差",
                "重复一致率",
            ],
            "基线 A": [
                tactile_baseline["parameters"]["material"],
                f"{tactile_baseline['parameters']['grip_force_n']:.2f} N",
                f"{tactile_baseline['parameters']['contact_area_percent']:.1f}%",
                f"{tactile_baseline['parameters']['pattern_noise']:.2f}",
                f"{tactile_baseline['parameters']['temperature_c']:.2f}°C",
                f"{tactile_baseline['parameters']['noise_nm']:.4f} nm",
                str(tactile_baseline["parameters"]["seed"]),
                baseline_results["diagnosed_material"],
                f"{baseline_results['probability_margin'] * 100:.2f} 个百分点" if baseline_results["valid_contact"] else "不适用",
                f"{baseline_results['pattern_error_percent']:.1f}%" if baseline_results["valid_contact"] else "不适用",
                f"{baseline_results['repeat_match_rate'] * 100:.0f}%" if baseline_results["valid_contact"] else "不适用",
            ],
            "当前 B": [
                material, f"{grip_force:.2f} N", f"{contact_area:.1f}%",
                f"{pattern_noise:.2f}", f"{temperature:.2f}°C", f"{noise:.4f} nm",
                str(int(seed)), material_diagnosis["material"],
                f"{tactile_results['probability_margin'] * 100:.2f} 个百分点" if tactile_results["valid_contact"] else "不适用",
                f"{tactile_results['pattern_error_percent']:.1f}%" if tactile_results["valid_contact"] else "不适用",
                f"{tactile_results['repeat_match_rate'] * 100:.0f}%" if tactile_results["valid_contact"] else "不适用",
            ],
        }, hide_index=True, width="stretch")
    tactile_download_a, tactile_download_b = st.columns(2)
    tactile_download_a.download_button(
        "下载触觉实验 CSV", experiments.tactile_csv_bytes(tactile_record),
        "tactile_experiment.csv", "text/csv",
    )
    tactile_download_b.download_button(
        "下载触觉实验报告", experiments.tactile_report(tactile_record),
        "tactile_experiment_report.txt", "text/plain",
    )

with foot_tab:
    st.subheader("机器人足：六区足底接触、地形与步态相位")
    module_learning_frame(
        "理解六区载荷如何形成压力中心，并量化 FBG 反演误差。",
        "先载入平地中期并保存基线 A，再比较脚跟、前掌、柔软地面或摆动期。",
        "同时查看真实载荷、反演载荷、区域 MAE 与 CoP 位置误差；低载荷时先看可靠性提示。",
        "六区采用独立线性标定，未包含动态冲击、足部姿态、材料迟滞和实物封装标定。",
    )
    foot_defaults = (
        ("foot_terrain", "平地"), ("foot_load", 180.0),
        ("foot_phase", 55), ("foot_support", "支撑期"),
    )
    for key, value in foot_defaults:
        st.session_state.setdefault(key, value)

    def load_foot_preset() -> None:
        preset = experiments.FOOT_PRESETS[st.session_state.foot_preset]
        st.session_state.foot_terrain = preset["terrain"]
        st.session_state.foot_load = preset["load_n"]
        st.session_state.foot_phase = int(preset["phase_percent"])
        st.session_state.foot_support = preset["support"]
        st.session_state.global_temperature = preset["temperature_c"]
        st.session_state.global_noise = preset["noise_nm"]
        st.session_state.global_seed = preset["seed"]
        st.session_state.global_failed_channel = "无"
        st.session_state.global_drift = 0.0

    foot_preset_column, foot_load_column = st.columns([2, 1])
    with foot_preset_column:
        st.selectbox("推荐实验场景", list(experiments.FOOT_PRESETS), key="foot_preset")
    with foot_load_column:
        st.button(
            "载入足底场景", key="load_foot_preset", on_click=load_foot_preset,
            width="stretch",
        )
    st.caption("载入场景会同步本页输入以及侧栏温度、波长噪声和随机种子，并清除模拟失效与替代漂移；不会修改复装工况或其他模块的局部参数。")
    foot_control_a, foot_control_b = st.columns(2)
    with foot_control_a:
        terrain = st.selectbox(
            "地形", ["平地", "前倾坡面", "后倾坡面", "柔软地面"], key="foot_terrain"
        )
        load = st.slider(
            "总垂直载荷 (N)", 0.0, 400.0, step=5.0, key="foot_load"
        )
    with foot_control_b:
        phase = st.slider("步态相位 (%)", 0, 100, key="foot_phase")
        support = st.selectbox("当前状态", ["支撑期", "摆动期"], key="foot_support")
    foot_record = experiments.run_foot_experiment({
        "terrain": terrain, "load_n": load, "phase_percent": float(phase),
        "support": support, "temperature_c": temperature, "noise_nm": noise,
        "seed": int(seed), "failed_zone": 1 if failed == "足底区域 1" else None,
        "drift_nm": drift,
    })
    foot_results = foot_record["results"]
    zones = np.asarray(foot_results["true_zone_loads_n"], dtype=float)
    estimated_zones = np.asarray(foot_results["estimated_zone_loads_n"], dtype=float)
    foot_fbg = {"wavelength_shifts_nm": np.asarray(foot_results["wavelength_shifts_nm"], dtype=float)}
    estimated_total = max(float(estimated_zones.sum()), 1e-9)
    cop = float(np.dot(np.arange(6), estimated_zones) / estimated_total)
    cop_xy = np.asarray(foot_results["estimated_cop_xy"], dtype=float)
    foot_estimate = {"zone_loads_n": estimated_zones, "cop_region": cop, "cop_xy": cop_xy}
    cop = foot_estimate["cop_region"]
    cop_xy = foot_estimate["cop_xy"]
    assembly_overview = models.simulate_replaceable_sole_assembly(sole_assembly_case, temperature)
    sole_readiness = models.assess_replaceable_sole_sensing_readiness(sole_assembly_case, 0.0)
    if not sole_readiness["can_enter_foot_sensing_flow"]:
        st.warning(f"装配自检：当前工况“{sole_assembly_case}”未通过空载筛查，足底读数仅作教学演示，不能作为感知流程输入。")
    if foot_results["true_total_load_n"] < 20.0:
        st.info("当前为低载荷状态，CoP 仅供参考；请优先观察六区读数是否接近噪声量级。")
    elif not foot_results["reliable_cop"]:
        st.info("当前载荷充足，但存在模拟通道失效，CoP 仅供参考；请结合区域 MAE 和真实/反演载荷对照。")
    if failed == "足底区域 1":
        st.warning("足底区域 1 的真实载荷仍保留，但该通道读数已替换为侧栏漂移值；误差指标会反映这次传感失效。")
    st.subheader("实时 FBG 和足底载荷结果")
    st.plotly_chart(
        visuals.foot_fbg_dashboard_figure(foot_fbg["wavelength_shifts_nm"], zones),
        width="stretch",
    )
    st.caption("紫色柱表示六路实际显示的 FBG 波长漂移；橙色线表示与之对应的六区教学载荷。两者随地形、支撑状态、步态相位、温度和噪声设置联动变化；相位 0% 偏脚跟、100% 偏前掌。")
    a, b, c, d = st.columns(4)
    a.metric("真实 / 反演支撑力", f"{foot_results['true_total_load_n']:.1f} / {foot_results['estimated_total_load_n']:.1f} N")
    b.metric("区域 MAE", f"{foot_results['zone_mae_n']:.2f} N")
    c.metric("CoP 位置误差", f"{foot_results['cop_error']:.3f}")
    d.metric("步态相位", f"{phase}% · {support}")
    foot_left, foot_right = st.columns([3, 2])
    with foot_left:
        st.plotly_chart(visuals.foot_schematic_figure(estimated_zones, terrain, cop_xy), width="stretch")
    with foot_right:
        st.plotly_chart(visuals.sensor_bar_figure(np.arange(1, 7), foot_fbg["wavelength_shifts_nm"], "足底六路 FBG 波长漂移"), width="stretch")
    load_comparison = go.Figure()
    zone_labels = [f"区域 {index}" for index in range(1, 7)]
    load_comparison.add_bar(name="真实载荷", x=zone_labels, y=zones)
    load_comparison.add_bar(name="FBG 反演载荷", x=zone_labels, y=estimated_zones)
    load_comparison.update_layout(
        title="六区真实载荷与 FBG 反演载荷", barmode="group",
        yaxis_title="载荷 (N)", legend_orientation="h",
    )
    st.plotly_chart(load_comparison, width="stretch")
    st.caption("足底示意图的颜色和 CoP 标记都来自 FBG 反演载荷；下方分组柱图再与真实载荷对照。区域 MAE 是六区绝对误差的平均值，CoP 位置误差是在当前归一化足底坐标中的欧氏距离。低于 20 N 的支撑力会标记为低载荷观察。")

    if st.button("保存当前为足底基线 A", key="save_foot_baseline"):
        st.session_state.foot_baseline = foot_record
    foot_baseline = st.session_state.get("foot_baseline")
    if foot_baseline is not None:
        baseline_results = foot_baseline["results"]
        baseline_parameters = foot_baseline["parameters"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": [
                "地形", "当前状态", "步态相位", "输入载荷", "温度变化",
                "波长噪声", "随机种子", "模拟失效", "替代漂移", "真实支撑力",
                "反演支撑力", "区域 MAE", "CoP 位置误差",
            ],
            "基线 A": [
                baseline_parameters["terrain"], baseline_parameters["support"],
                f"{baseline_parameters['phase_percent']:.0f}%",
                f"{baseline_parameters['load_n']:.1f} N",
                f"{baseline_parameters['temperature_c']:.2f}°C",
                f"{baseline_parameters['noise_nm']:.4f} nm",
                str(baseline_parameters["seed"]),
                "无" if baseline_parameters.get("failed_zone") is None else f"足底区域 {baseline_parameters['failed_zone']}",
                f"{baseline_parameters.get('drift_nm', 0.0):.4f} nm",
                f"{baseline_results['true_total_load_n']:.1f} N",
                f"{baseline_results['estimated_total_load_n']:.1f} N",
                f"{baseline_results['zone_mae_n']:.2f} N",
                f"{baseline_results['cop_error']:.3f}",
            ],
            "当前 B": [
                terrain, support, f"{phase}%", f"{load:.1f} N", f"{temperature:.2f}°C",
                f"{noise:.4f} nm", str(int(seed)), failed,
                f"{drift:.4f} nm", f"{foot_results['true_total_load_n']:.1f} N",
                f"{foot_results['estimated_total_load_n']:.1f} N",
                f"{foot_results['zone_mae_n']:.2f} N", f"{foot_results['cop_error']:.3f}",
            ],
        }, hide_index=True, width="stretch")
    foot_download_a, foot_download_b = st.columns(2)
    foot_download_a.download_button(
        "下载足底实验 CSV", experiments.foot_csv_bytes(foot_record),
        "foot_experiment.csv", "text/csv",
    )
    foot_download_b.download_button(
        "下载足底实验报告", experiments.foot_report(foot_record),
        "foot_experiment_report.txt", "text/plain",
    )
    with st.expander("复装与结构说明（辅助）"):
        empty_load_gate = models.assess_replaceable_sole_sensing_readiness(sole_assembly_case, 0.0)
        status_a, status_b = st.columns(2)
        status_a.metric("当前复装工况", sole_assembly_case)
        status_b.metric("空载筛查（仿真候选）", str(empty_load_gate["assembly_prediction"]))
        st.caption("装配自检必须在卸载状态下完成；下方结构图用于说明模块关系，不替代实物装配、密封或耐久验证。")
        selected_sole_component = st.selectbox(
            "点读部件（红框高亮）",
            ["可更换耐磨外底", "分区传力模块", "柔性隔离膜", "周向密封圈", "定位柱与锁止件", "固定光纤感知芯", "基板与限位台阶"],
            key="selected_sole_component",
        )
        st.plotly_chart(visuals.sole_component_explorer_figure(assembly_overview, selected_sole_component), width="stretch")
        st.plotly_chart(visuals.replaceable_sole_explainer_figure(assembly_overview, zones, terrain, cop, support), width="stretch")

with calibration_tab:
    st.subheader("FBG 标定与诊断")
    calibration_defaults = (
        ("hand_bend_angle", 0.0),
        ("calibration_length", 80.0),
        ("calibration_offset", 1.0),
        ("calibration_attachment", "嵌入式"),
        ("calibration_redundant_angle", 45.0),
        ("calibration_fault_mode", "无"),
        ("calibration_fault_channel", 2),
        ("calibration_contact_position", 37.0),
        ("calibration_contact_force", 4.0),
        ("calibration_influence_width", 12.0),
    )
    for key, value in calibration_defaults:
        st.session_state.setdefault(key, value)
    def reset_calibration() -> None:
        # 只恢复标定页的局部输入，保留公共测量条件和其他页面状态。
        for key, value in calibration_defaults:
            st.session_state[key] = value
    st.button("重置本页参数", key="reset_calibration", on_click=reset_calibration)
    st.caption("只重置本页的弯曲、冗余故障与接触输入；保留侧栏公共温度、噪声及模拟失效设置。恢复完整默认条件时，请同时点击侧栏“演示预设（恢复公共参数）”。")
    st.metric("当前配置", f"{sample_rate} Hz · {failed}")
    st.divider()
    st.subheader("单根手指 FBG 弯曲标定")
    st.caption("① 载入示例 → ② 保存基线 A → ③ 改一个条件，观察当前 B → ④ 下载实验记录。")
    parameter_keys = {
        "angle_deg": "hand_bend_angle", "length_mm": "calibration_length",
        "fiber_offset_mm": "calibration_offset", "attachment": "calibration_attachment",
        "temperature_c": "global_temperature", "noise_nm": "global_noise",
        "drift_nm": "global_drift", "failed_channel": "global_failed_channel", "seed": "global_seed",
    }
    def apply_calibration_parameters(parameters: dict) -> None:
        # 显式白名单：只恢复单指输入及其测量条件，不写入其他页面状态。
        for parameter, key in parameter_keys.items():
            value = parameters[parameter]
            st.session_state[key] = value if parameter in ("attachment", "failed_channel", "seed") else float(value)
    def load_calibration_preset() -> None:
        apply_calibration_parameters(experiments.PRESETS[st.session_state.calibration_preset])
    def save_calibration_baseline(record: dict) -> None:
        st.session_state.calibration_baseline = experiments.run_calibration(record["parameters"])
    def restore_calibration_record(record: dict, baseline: dict | None) -> None:
        apply_calibration_parameters(record["parameters"])
        st.session_state.calibration_baseline = baseline
    preset_left, preset_right = st.columns([2, 1])
    preset_left.selectbox("示例实验", list(experiments.PRESETS), key="calibration_preset")
    preset_right.button("载入示例", key="load_calibration_preset", on_click=load_calibration_preset)
    st.caption("载入示例、恢复 A 或导入记录会同步侧栏温度、噪声、漂移、失效通道与随机种子，影响其他模块的测量条件；不会改动姿态、任务、采样率或复装工况。选择示例后需点击“载入示例”才生效。")
    left, right = st.columns([1, 2])
    with left:
        angle = st.slider("真实弯曲角 (°)", -100.0, 100.0, step=1.0, key="hand_bend_angle")
        with st.expander("进阶：几何与连接方式"):
            length = st.slider("手指长度 (mm)", 40.0, 140.0, step=1.0, key="calibration_length")
            offset = st.slider("光纤距中性层偏置 (mm)", -2.0, 2.0, step=0.05, key="calibration_offset")
            attachment = st.selectbox("光纤连接方式", list(experiments.ATTACHMENT_GAINS), key="calibration_attachment")
        st.write(f"当前测量条件：温差 {temperature:g} °C · 噪声 {noise:.4f} nm · 种子 {int(seed)}")
        st.caption(f"几何：{length:g} mm · 偏置 {offset:g} mm · {attachment}。温度和噪声在侧栏调节。")
    current_experiment = experiments.run_calibration({parameter: st.session_state[key] for parameter, key in parameter_keys.items()})
    result = current_experiment["results"]
    gain = result["gain"]
    # 中心线仍展示真实形状；反演读数与报告共用同一组实验结果。
    finger = models.simulate_finger(angle, length, offset * gain, np.array([0.25, 0.50, 0.75]) * length, temperature, noise, int(seed))
    finger["wavelength_shifts_nm"] = np.asarray(result["raw_shifts_nm"])
    finger["estimated_angle_deg"] = result["estimated_angle_deg"]
    with right:
        st.plotly_chart(visuals.finger_figure(finger), width="stretch")
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("真实弯曲角", f"{angle:.2f} °")
    metric_b.metric("未温补角", "不可反演" if not result["identifiable"] else f"{result['uncompensated_angle_deg']:.2f} °")
    metric_c.metric("FBG 融合角", "不可反演" if not result["identifiable"] else f"{result['estimated_angle_deg']:.2f} °",
                    None if not result["identifiable"] else f"误差 {result['error_deg']:+.2f} °")
    metric_d.metric("连接传力系数", f"{gain:.2f}")
    if not result["identifiable"]:
        st.warning("光纤位于中性层（偏置为 0），本模型没有弯曲应变信号，无法反演角度；零读数不代表手指未弯曲。请在进阶设置中改为非零偏置。")
    calibration_chart = go.Figure()
    calibration_chart.add_bar(name="原始读数", x=["FBG 1", "FBG 2", "FBG 3"], y=result["raw_shifts_nm"])
    calibration_chart.add_bar(name="扣除已知温漂", x=["FBG 1", "FBG 2", "FBG 3"], y=result["compensated_shifts_nm"])
    calibration_chart.update_layout(title="三枚 FBG：温补前后", yaxis_title="波长漂移 (nm)", barmode="group", template="plotly_white", height=360, legend=dict(orientation="h"))
    st.plotly_chart(calibration_chart, width="stretch")
    st.download_button("下载手部 FBG 读数 CSV", csv_bytes(["FBG 1", "FBG 2", "FBG 3"], finger["wavelength_shifts_nm"]), "hand_fbg_readings.csv", "text/csv")
    st.caption("上图为真实手指中心线与 FBG 位置，下方柱图为当前三路波长漂移。恒曲率模型中三路理想应变相同；反演先按已知温差扣除共模温漂，再由三路平均应变计算弯曲角，并非由通道间差异反演。")
    if failed.startswith("手部 FBG"):
        st.warning("普通三路融合未剔除故障通道：所选通道的读数被替换为侧栏“零点漂移”值，并参与当前角度计算。下方四路冗余模块单独演示故障剔除与容错反演。")
    thermal_only_shift = float(models.fbg_wavelength_shift_nm(np.array([0.0]), temperature)[0])
    diagnostic_left, diagnostic_right = st.columns(2)
    diagnostic_left.metric("预期共模温漂", f"{thermal_only_shift:.4f} nm")
    diagnostic_right.metric("通道诊断", "模拟失效" if failed.startswith("手部 FBG") else "通道正常", failed if failed.startswith("手部 FBG") else "三路参与融合")
    st.caption("此处“通道诊断”反映侧栏的手动失效设置，不是自动故障识别。温补将温度项视作三路共享的共模漂移；实际诊断还需要每路零点、封装差异、长期漂移与冗余通道的历史基线。")

    st.markdown("#### 基线 A 与当前 B")
    baseline = st.session_state.get("calibration_baseline")
    baseline_actions = st.columns(2)
    baseline_actions[0].button("保存当前为基线 A" if baseline is None else "用当前结果更新基线 A", key="save_calibration_baseline", on_click=save_calibration_baseline, args=(current_experiment,))
    baseline_actions[1].button("恢复基线 A 的参数", key="restore_calibration_baseline", disabled=baseline is None,
                               on_click=apply_calibration_parameters, args=((baseline or current_experiment)["parameters"],))
    if baseline is None:
        st.info("先保存基线 A，再改变温度或噪声。A 将保持不变，当前参数与结果作为 B。")
    else:
        comparison_rows = []
        comparison_labels = {
            "angle_deg": "真实角度 (°)", "length_mm": "长度 (mm)", "fiber_offset_mm": "偏置 (mm)",
            "attachment": "连接方式", "temperature_c": "温差 (°C)", "noise_nm": "噪声 (nm)",
            "drift_nm": "故障替代漂移 (nm)", "failed_channel": "失效通道", "seed": "随机种子",
        }
        for parameter, label in comparison_labels.items():
            old, new = baseline["parameters"][parameter], current_experiment["parameters"][parameter]
            if old != new:
                comparison_rows.append({"已改变的条件": label, "基线 A": str(old), "当前 B": str(new)})
        if comparison_rows:
            st.dataframe(comparison_rows, hide_index=True, width="stretch")
        else:
            st.caption("A 与 B 的输入条件相同；相同随机种子下读数可复现。")
        result_rows = []
        for key, label in (("estimated_angle_deg", "温补反演角 (°)"), ("error_deg", "相对真实角误差 (°)")):
            old, new = baseline["results"][key], result[key]
            result_rows.append({"结果": label, "基线 A": "不可反演" if old is None else f"{old:.4f}",
                                "当前 B": "不可反演" if new is None else f"{new:.4f}",
                                "B − A": "不可比较" if old is None or new is None else f"{new - old:+.4f}"})
        st.dataframe(result_rows, hide_index=True, width="stretch")
        st.caption("差值是两次教学实验的结果变化，不是准确度提升的证明。本页重置保留 A；刷新或关闭会话可能丢失，请下载记录留存。")
    with st.expander("保存与恢复实验记录"):
        st.caption("记录只包含单指弯曲标定的输入、随机种子、结果及可选基线 A，不包含其他页面状态。导入时重新计算结果；文件中的旧结果不作为测量依据。")
        st.download_button("下载可恢复记录 JSON", experiments.export_record(current_experiment, baseline), "calibration_experiment.json", "application/json")
        st.download_button("下载中文实验摘要", experiments.calibration_report(current_experiment, baseline).encode("utf-8-sig"), "calibration_experiment_summary.txt", "text/plain")
        uploaded_record = st.file_uploader("选择实验记录 JSON（最大 128 KB）", type=["json"], key="calibration_record_upload", max_upload_size=1)
        if uploaded_record is not None:
            try:
                imported_current, imported_baseline = experiments.import_record(uploaded_record.getvalue())
            except ValueError as exc:
                st.error(f"未导入：{exc}")
            else:
                st.caption("文件校验通过。点击后将替换当前单指参数、相关公共测量条件和基线 A；没有基线的记录会清空当前 A。")
                st.button("导入参数并替换基线 A", key="import_calibration_record", on_click=restore_calibration_record, args=(imported_current, imported_baseline))

    st.divider()
    st.subheader("冗余 FBG 故障诊断与容错反演")
    fault_left, fault_right = st.columns([1, 2])
    with fault_left:
        redundant_angle = st.slider("冗余通道真实弯曲角 (°)", -90.0, 90.0, step=1.0, key="calibration_redundant_angle")
        fault_mode = st.selectbox("故障类型", ["无", "漂移", "断纤", "噪声增大"], key="calibration_fault_mode")
        fault_channel = st.slider("故障通道", 1, 4, step=1, key="calibration_fault_channel")
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
    st.caption("四路柱高应基本一致，故障通道会明显偏离中位数；诊断剔除该通道后反演角度。")

    st.divider()
    st.subheader("指尖接触位置与法向力反演")
    left, right = st.columns([1, 2])
    with left:
        contact_position = st.slider("真实接触位置 (mm)", 0.0, 70.0, step=0.5, key="calibration_contact_position")
        force = st.slider("真实法向力 (N)", 0.0, 10.0, step=0.1, key="calibration_contact_force")
        influence_width = st.slider("封装传力宽度 (mm)", 5.0, 25.0, step=0.5, key="calibration_influence_width")
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
    st.caption("曲线为三枚 FBG 的应变分布，接触点附近的 FBG 应变最高；反演接触位置与法向力由网格最小二乘恢复。")
    with st.expander("模型边界"):
        st.write("这里用高斯传递函数表示接触力向光纤的传递。它能帮助理解传感器位置、封装刚度与可辨识性，但不能替代软材料非线性、胶层、摩擦和滞后的实验标定。")
    tab_jump_button(2, "下一步 → 手部：三维抓取", "hand_navigation", "三维抓取")

with hand_3d_tab:
    st.subheader("三维抓取传感：独立接触与 FBG 读数")
    module_learning_frame(
        "理解三维位置、14 个指节触觉通道和掌心通道如何共同决定抓稳状态。",
        "完成寻找与抓取任务后，提高侧栏噪声并做多次采样，对照单次模型状态与统计稳定性。",
        "先确认掌心、拇指和其余手指三个条件，再看判定一致率、翻转次数和布置可观测性。",
        "重复实验只扰动波长读数；布置对照不包含真实封装、串扰、摩擦和动态力控。",
    )
    st.caption("本页不读取二维抓取的姿态、罐体位置或抓取结果。它以三维手自身的五指屈曲与罐体 X/Y/Z 偏移，独立估算指尖接触、握持稳定度和五指＋掌心六路 FBG 读数。")

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
            # 先移动到目标附近的扫描点，再在“对准目标”精确靠近，让寻找过程有可见动作。
            for reach_key, target_key in zip(
                ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"),
                ("three_d_can_x", "three_d_can_y", "three_d_can_z"),
            ):
                st.session_state[reach_key] = st.session_state[target_key] + (2.0 if reach_key == "three_d_reach_x" else 0.0)
            st.session_state.three_d_task_phase = models.next_grasp_task_phase(phase, False)
            return
        if phase == "对准目标":
            # 保持物体世界坐标不变，移动手部抓取包络到已定位的目标坐标。
            for reach_key, target_key in zip(
                ("three_d_reach_x", "three_d_reach_y", "three_d_reach_z"),
                ("three_d_can_x", "three_d_can_y", "three_d_can_z"),
            ):
                st.session_state[reach_key] = st.session_state[target_key]
            st.session_state.three_d_task_phase = models.next_grasp_task_phase(phase, False)
            return
        if phase == "闭合抓取":
            set_three_d_grasp_pose(True)
            next_phase = models.next_grasp_task_phase("闭合抓取", current_three_d_grasp_is_verified())
            st.session_state.three_d_task_phase = next_phase
            if next_phase == "搬运目标":
                for key, value in zip(("three_d_shoulder", "three_d_elbow", "three_d_wrist"), (55.0, -35.0, 15.0)):
                    st.session_state[key] = value
            return
        if phase == "搬运目标":
            set_three_d_grasp_pose(False)
            st.session_state.three_d_can_x = st.session_state["three_d_reach_x"] + 1.4
            st.session_state.three_d_task_phase = models.next_grasp_task_phase(phase, True)
            return
        if phase == "松开并放置":
            st.session_state.three_d_task_phase = models.next_grasp_task_phase(phase, False)

    task_phase = st.session_state.three_d_task_phase
    controls, display = st.columns([1.15, 2.1], gap="large")
    with controls:
        st.markdown("#### 抓取指令")
        preset_left, preset_right = st.columns(2)
        preset_left.button(
            "三维张开手", key="three_d_open", on_click=set_three_d_grasp_pose,
            args=(False,), width="stretch",
        )
        preset_right.button(
            "三维一键握拳", key="three_d_close", on_click=set_three_d_grasp_pose,
            args=(True,), width="stretch",
        )
        st.button(
            "恢复三维初始姿态", key="three_d_reset_initial",
            on_click=reset_three_d_initial_pose, width="stretch",
        )
        st.button(
            "开始三维寻找与抓取任务", key="start_three_d_grasp_task",
            on_click=start_three_d_grasp_task,
            disabled=task_phase not in ("未启动", "完成"), width="stretch",
        )
        st.button(
            "执行下一步" if task_phase != "抓取失败" else "重新对准目标",
            key="advance_three_d_grasp_task", on_click=advance_three_d_grasp_task,
            disabled=task_phase in ("未启动", "完成"), width="stretch",
        )
        st.caption(
            f"任务状态：{task_phase}。寻找目标 → 对准 → FBG 抓取验证 → 搬运 → 松开放置。"
        )
        with st.expander("高级姿态与目标参数", expanded=False):
            st.caption("用于手动实验；自动任务可直接使用上方指令。拇指含 2 个关节，其余手指各含 3 个关节。")
            st.markdown("##### 手臂姿态")
            arm_a, arm_b, arm_c = st.columns(3)
            with arm_a:
                three_d_shoulder = st.slider("三维肩关节 (°)", -20.0, 100.0, step=1.0, key="three_d_shoulder")
            with arm_b:
                three_d_elbow = st.slider("三维肘关节 (°)", -100.0, 40.0, step=1.0, key="three_d_elbow")
            with arm_c:
                three_d_wrist = st.slider("三维腕关节 (°)", -70.0, 70.0, step=1.0, key="three_d_wrist")
            st.markdown("##### 手指关节")
            finger_a, finger_b = st.columns(2)
            with finger_a:
                three_d_thumb_mcp = st.slider("拇指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_thumb_mcp")
                three_d_thumb_ip = st.slider("拇指 IP (°)", 0.0, 110.0, step=1.0, key="three_d_thumb_ip")
                three_d_index_mcp = st.slider("食指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_index_mcp")
                three_d_index_pip = st.slider("食指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_index_pip")
                three_d_index_dip = st.slider("食指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_index_dip")
                three_d_middle_mcp = st.slider("中指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_middle_mcp")
                three_d_middle_pip = st.slider("中指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_middle_pip")
            with finger_b:
                three_d_middle_dip = st.slider("中指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_middle_dip")
                three_d_ring_mcp = st.slider("无名指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_ring_mcp")
                three_d_ring_pip = st.slider("无名指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_ring_pip")
                three_d_ring_dip = st.slider("无名指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_ring_dip")
                three_d_little_mcp = st.slider("小指 MCP (°)", 0.0, 110.0, step=1.0, key="three_d_little_mcp")
                three_d_little_pip = st.slider("小指 PIP (°)", 0.0, 110.0, step=1.0, key="three_d_little_pip")
                three_d_little_dip = st.slider("小指 DIP (°)", 0.0, 110.0, step=1.0, key="three_d_little_dip")
            st.markdown("##### 物体世界坐标")
            can_a, can_b, can_c = st.columns(3)
            with can_a:
                three_d_can_x = st.slider("物体 X 位置", -3.0, 3.0, step=0.1, key="three_d_can_x")
            with can_b:
                three_d_can_y = st.slider("物体 Y 位置", -3.0, 3.0, step=0.1, key="three_d_can_y")
            with can_c:
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
    three_d_palm_shift = models.add_gaussian_noise(
        np.asarray([three_d_sensing["tactile_fbg_shifts_nm"][-1]]), noise, int(seed) + 301
    )
    three_d_display_shifts = np.r_[three_d_shifts, three_d_palm_shift]

    grasp_calibration = models.THREE_D_GRASP_CALIBRATION
    grasp_contacts = three_d_fbg_decision["contact_fingers"]
    other_contacts = len([index for index in grasp_contacts if index != 0])
    grasp_conditions = [
        ("掌心支撑", float(three_d_fbg_decision["palm_touch_n"]) >= grasp_calibration.palm_contact_threshold_n,
         f"{three_d_fbg_decision['palm_touch_n']:.3f} N", f"≥ {grasp_calibration.palm_contact_threshold_n:g} N"),
        ("拇指接触", 0 in grasp_contacts,
         f"{three_d_fbg_decision['contact_force_n'][0]:.3f} N", f"≥ {grasp_calibration.contact_force_threshold_n:g} N"),
        ("其余手指接触", other_contacts >= 2,
         f"{other_contacts} 根", f"至少 2 根，每根 ≥ {grasp_calibration.contact_force_threshold_n:g} N"),
    ]
    with display:
        st.markdown("#### 三维交互视图")
        if three_d_fbg_decision["is_grasped"]:
            st.success("FBG 已抓稳：掌心、拇指及至少两根其余手指达到触觉阈值。")
        else:
            st.warning("尚未满足：" + "、".join(name for name, passed, _, _ in grasp_conditions if not passed) + "。")
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
                animate=st.session_state.get("smooth_animation", True),
            ),
            height=560,
        )
        st.caption("拖动模型可旋转视角；滚轮缩放保持关闭。物体保持世界坐标，寻找程序移动手部抓取包络至目标。")
        with st.container(key="three_d_grasp_metrics"):
            three_d_metrics = st.columns(4)
            three_d_metrics[0].metric("FBG 触觉接触手指", f"{len(three_d_fbg_decision['contact_fingers'])} / 5")
            three_d_metrics[1].metric("FBG 反演接触合力", f"{np.asarray(three_d_fbg_decision['contact_force_n']).sum():.2f} N")
            three_d_metrics[2].metric("握持稳定度", f"{float(three_d_sensing['stability']) * 100:.0f}%")
            three_d_metrics[3].metric("三维抓取状态", "FBG 已抓稳" if three_d_fbg_decision["is_grasped"] else "FBG 未抓稳")
        with st.expander("查看抓稳条件与当前读数", expanded=False):
            st.dataframe([{"判定条件": name, "当前值": value, "要求": threshold, "状态": "已满足" if passed else "未满足"}
                          for name, passed, value, threshold in grasp_conditions], hide_index=True, width="stretch")
            st.caption("三个条件全部满足才判定抓稳；读数按显示精度取整，是否达标以未取整的温补结果为准。握持稳定度是独立教学指标，不代替这三个条件。")
        with st.expander("查看 FBG 路径说明", expanded=False):
            st.info("青色发光线表示 FBG 封装/走线路径：肩—肘—腕为弯曲监测；掌部两条短线为掌心接触区域；每根手指上的分段线为指节触觉区域。青色只表示传感路径，不表示受力大小；接触后对应路径会变为黄色。")
            st.caption("先查看物体是否位于手部抓取范围，再调节缺少接触的手指。关节屈曲和柱高本身不能证明抓稳。")
    st.markdown("#### 传感通道对照")
    inspected_channel = st.selectbox("查看哪个部位的通道", ["拇指", "食指", "中指", "无名指", "小指", "掌心"], key="three_d_inspect_channel")
    channel_map = {
        "拇指": (0, [0, 1]), "食指": (1, [2, 3, 4]), "中指": (2, [5, 6, 7]),
        "无名指": (3, [8, 9, 10]), "小指": (4, [11, 12, 13]), "掌心": (5, [14]),
    }
    overview_index, tactile_indices = channel_map[inspected_channel]
    st.caption(f"{inspected_channel}：总览第 {overview_index + 1} 路；细分 FBG " + "、".join(str(index + 1) for index in tactile_indices) + "。橙色柱仅标记当前查看的通道，不表示故障或达标；选择不会改变模型姿态。")
    st.caption("六路总览加入了侧栏噪声；细分触觉读数与当前抓稳判定使用未叠加该噪声的模型信号。侧栏噪声不参与当前抓稳判定，不能用本页验证抗噪性能。")
    three_d_result_chart, three_d_result_notes = st.columns([3, 2])
    with three_d_result_chart:
        overview_chart = visuals.sensor_bar_figure(np.arange(1, 7), three_d_display_shifts, "三维抓取：五指与掌心六路 FBG 波长漂移")
        overview_chart.update_traces(marker_color=[visuals.COLORS["estimate"] if index == overview_index else visuals.COLORS["sensor"] for index in range(6)])
        st.plotly_chart(overview_chart, width="stretch")
    with three_d_result_notes:
        st.markdown("#### 图表结果说明")
        st.markdown(
            "- **第 1–5 路**：拇指至小指的综合弯曲／接触通道。\n"
            "- **第 6 路**：掌心接触通道；它与五指通道分开显示，不等同于任一手指。\n"
            "- **柱高**：当前温度、噪声、关节屈曲与接触状态共同作用后的波长漂移。\n"
            "- **抓稳判定**：温度补偿后，掌心、拇指和至少两根其余手指的触觉条件共同满足时，才显示“FBG 已抓稳”。\n"
            "- **阅读顺序**：先看六路柱状分布，再对照下方指尖接触力、细分指节／掌心通道和稳定度。"
        )
    force_chart = visuals.sensor_bar_figure(np.arange(1, 6), three_d_sensing["contact_force_n"], "三维指尖接触力 (N)")
    force_chart.update_yaxes(title_text="接触力 (N)")
    force_chart.update_xaxes(tickvals=[f"FBG {i}" for i in range(1, 6)], ticktext=["拇指", "食指", "中指", "无名指", "小指"])
    st.plotly_chart(force_chart, width="stretch")
    tactile_chart = visuals.sensor_bar_figure(np.arange(1, 16), three_d_sensing["tactile_fbg_shifts_nm"], "细分触觉 FBG：14 个指节＋第 15 路掌心")
    tactile_chart.update_traces(marker_color=[visuals.COLORS["estimate"] if index in tactile_indices else visuals.COLORS["sensor"] for index in range(15)])
    st.plotly_chart(tactile_chart, width="stretch")
    tactile_left, tactile_right = st.columns(2)
    with tactile_left:
        palm_finger_chart = visuals.sensor_bar_figure(np.arange(1, 7), np.r_[three_d_sensing["palm_touch_n"], three_d_sensing["contact_force_n"]], "手掌与五指触觉 (N)")
        palm_finger_chart.update_yaxes(title_text="接触力 (N)")
        palm_finger_chart.update_xaxes(tickvals=[f"FBG {i}" for i in range(1, 7)], ticktext=["掌心", "拇指", "食指", "中指", "无名指", "小指"])
        st.plotly_chart(palm_finger_chart, width="stretch")
    with tactile_right:
        arm_strain_chart = visuals.sensor_bar_figure(np.arange(1, 4), three_d_sensing["arm_bend_strain_ue"], "肩、肘、腕 FBG 弯曲应变 (με)")
        arm_strain_chart.update_yaxes(title_text="弯曲应变 (με)")
        arm_strain_chart.update_xaxes(tickvals=["FBG 1", "FBG 2", "FBG 3"], ticktext=["肩", "肘", "腕"])
        st.plotly_chart(arm_strain_chart, width="stretch")
    st.caption("通道对应关系：六路总览的第 1–5 路为拇指至小指综合通道，第 6 路为掌心；细分图第 1–14 路为指节，第 15 路为掌心。青色光纤覆盖肩—肘—腕、两条掌部路线及全部 14 个手指指节；下方指尖接触力、掌心＋五指触觉和手臂弯曲应变柱状图用于对照 FBG 读数判断接触与弯曲。")
    three_d_record = {
        "dimension": "三维",
        "task_phase": task_phase,
        "is_grasped": bool(three_d_fbg_decision["is_grasped"]),
        "contact_fingers": [["拇指", "食指", "中指", "无名指", "小指"][index] for index in three_d_fbg_decision["contact_fingers"]],
        "contact_force_n": np.asarray(three_d_fbg_decision["contact_force_n"], dtype=float).tolist(),
        "palm_touch_n": float(three_d_fbg_decision["palm_touch_n"]),
        "wavelength_shifts_nm": np.asarray(three_d_display_shifts, dtype=float).tolist(),
        "temperature_c": temperature,
        "noise_nm": noise,
        "seed": int(seed),
        "target_position": list(three_d_target_world),
    }
    three_d_download_a, three_d_download_b = st.columns(2)
    three_d_download_a.download_button(
        "下载三维抓取 FBG 读数 CSV",
        csv_bytes(["拇指 FBG", "食指 FBG", "中指 FBG", "无名指 FBG", "小指 FBG", "掌心 FBG"], three_d_display_shifts),
        "three_dimensional_grasp_fbg_readings.csv",
        "text/csv",
    )
    three_d_download_b.download_button(
        "下载三维抓取实验报告", experiments.grasp_report(three_d_record).encode("utf-8-sig"),
        "three_dimensional_grasp_report.txt", "text/plain",
    )
    st.markdown("#### 重复采样与传感器布置实验")
    three_d_repeat_samples = st.select_slider(
        "三维重复采样次数", options=[20, 50, 100, 200], value=50, key="three_d_repeat_samples"
    )
    three_d_study = experiments.run_three_d_grasp_noise_study(
        three_d_sensing, temperature, noise, int(three_d_repeat_samples), (int(seed) + 430) % 2**32
    )
    three_d_layouts = experiments.compare_grasp_sensor_layouts(
        three_d_study["baseline_contact_force_n"],
        three_d_study["baseline_palm_touch_n"],
        requires_palm=True,
    )
    three_d_stats = st.columns(4)
    three_d_stats[0].metric("无噪声基准", "抓稳" if three_d_study["baseline_is_grasped"] else "未抓稳")
    three_d_stats[1].metric("重复采样抓稳率", f"{three_d_study['grasped_rate_percent']:.1f}%")
    three_d_stats[2].metric("判定一致率", f"{three_d_study['decision_consistency_percent']:.1f}%")
    three_d_stats[3].metric("判定翻转", f"{three_d_study['decision_flip_count']} 次")
    st.caption(
        f"当前 σ={noise:.4f} nm：15 路触觉 FBG 逐次加噪后，五指反演合力 "
        f"{three_d_study['total_force_mean_n']:.3f} ± {three_d_study['total_force_std_n']:.3f} N；"
        f"掌心反演力 {three_d_study['palm_force_mean_n']:.3f} ± {three_d_study['palm_force_std_n']:.3f} N。"
    )
    st.dataframe(three_d_layouts, hide_index=True, width="stretch")
    st.caption("三维当前规则同时需要掌心、拇指与其余四指信息；缺少掌心或部分其余手指时会标为无法完整执行判定，无接触力时受力覆盖率记为 0%。")
    with st.expander("查看三维逐次采样记录", expanded=False):
        st.dataframe(three_d_study["samples"], hide_index=True, width="stretch")
    three_d_repeat_a, three_d_repeat_b = st.columns(2)
    three_d_repeat_a.download_button(
        "下载三维重复采样 CSV", experiments.grasp_noise_study_csv(three_d_study),
        "three_dimensional_grasp_repeatability.csv", "text/csv",
    )
    three_d_repeat_b.download_button(
        "下载三维抓取稳健性报告",
        experiments.grasp_robustness_report("三维", three_d_study, three_d_layouts).encode("utf-8-sig"),
        "three_dimensional_grasp_robustness_report.txt", "text/plain",
    )
    with st.expander("分布式光纤视角（点式 FBG 对比）"):
        distributed_finger = models.simulate_distributed_sensing(
            np.asarray(three_d_curls, dtype=float), three_d_fbg_decision["contact_fingers"]
        )
        distributed_finger_left, distributed_finger_right = st.columns(2)
        with distributed_finger_left:
            st.plotly_chart(
                visuals.distributed_finger_figure(distributed_finger, three_d_fbg_decision["contact_fingers"]),
                width="stretch",
            )
        with distributed_finger_right:
            st.plotly_chart(visuals.das_event_figure(distributed_finger), width="stretch")
        st.caption("同一抓取状态：点式 FBG 输出 14 个指节离散通道；分布式光纤沿五指给出连续应变分布（Rayleigh）和时空振动事件（DAS），接触手指对应的光纤段出现应变峰。")
    with st.expander("三维传感模型边界"):
        st.write("这里的接触力来自指尖到圆柱抓取包络的三维距离与屈曲角，作为光纤抓取传感教学模型。它不等同于刚体接触求解、摩擦锥或真实力控，需要结合传感器封装与实验数据标定。")
    tab_jump_button(4, "下一步 → 结构健康", "structure_navigation", "结构健康")

with shape_tab:
    st.subheader("三芯光纤的连续体机器人 3D 形状重建")
    module_learning_frame(
        "理解三芯差分波长如何反演曲率与弯曲方向，并量化整条中心线误差。",
        "先载入理想恒曲率并保存基线 A，再比较已知扭转先验、波长噪声或芯间温差场景。",
        "同时查看曲率误差、方向误差、中心线 RMSE 和末端误差，不只判断两条曲线是否重合。",
        "采用恒曲率教学模型；扭转率是已知重建先验，并非由当前三芯波长读数估计。",
    )
    for key, value in (
        ("shape_curvature", 8.0), ("shape_direction", 35.0),
        ("shape_twist", 0.0), ("shape_length", 150.0),
        ("shape_temperature_gradient", 0.0),
    ):
        st.session_state.setdefault(key, value)

    def load_shape_preset() -> None:
        preset = experiments.SHAPE_PRESETS[st.session_state.shape_preset]
        st.session_state.shape_curvature = preset["curvature_per_m"]
        st.session_state.shape_direction = preset["direction_deg"]
        st.session_state.shape_twist = preset["twist_per_m"]
        st.session_state.shape_length = preset["length_mm"]
        st.session_state.shape_temperature_gradient = preset["core_temperature_gradient_c"]
        st.session_state.global_temperature = preset["temperature_c"]
        st.session_state.global_noise = preset["noise_nm"]
        st.session_state.global_seed = preset["seed"]

    shape_preset_column, shape_load_column = st.columns([2, 1])
    with shape_preset_column:
        st.selectbox("推荐实验场景", list(experiments.SHAPE_PRESETS), key="shape_preset")
    with shape_load_column:
        st.button("载入形状场景", key="load_shape_preset", on_click=load_shape_preset, width="stretch")
    st.caption("载入场景会同步本页输入以及侧栏温度、波长噪声和随机种子；不会改动其他模块的局部参数或已有基线。")
    left, right = st.columns([1, 2])
    with left:
        curvature = st.slider("真实曲率 (1/m)", 0.0, 20.0, step=0.1, key="shape_curvature")
        direction = st.slider("弯曲方向 (°)", 0.0, 359.0, step=1.0, key="shape_direction")
        twist = st.slider("已知恒定扭转率（重建先验）(1/m)", -20.0, 20.0, step=0.1, key="shape_twist")
        shape_length = st.slider("光纤长度 (mm)", 50.0, 300.0, step=1.0, key="shape_length")
        core_temperature_gradient = st.slider(
            "芯间温度梯度 (°C/芯)", -20.0, 20.0, step=0.1,
            key="shape_temperature_gradient",
        )
    shape_record = experiments.run_shape_experiment({
        "curvature_per_m": curvature, "direction_deg": direction,
        "twist_per_m": twist, "length_mm": shape_length, "core_radius_um": 125.0,
        "temperature_c": temperature, "noise_nm": noise,
        "core_temperature_gradient_c": core_temperature_gradient, "seed": int(seed),
    })
    shape_results = shape_record["results"]
    shape = {
        "core_angles_deg": np.asarray(shape_results["core_angles_deg"]),
        "strain": np.asarray(shape_results["strain"]),
        "wavelength_shifts_nm": np.asarray(shape_results["wavelength_shifts_nm"]),
        "centerline_xyz_mm": np.asarray(shape_results["true_centerline_xyz_mm"]),
        "estimated_centerline_xyz_mm": np.asarray(shape_results["estimated_centerline_xyz_mm"]),
        "estimated_curvature_per_m": shape_results["estimated_curvature_per_m"],
        "estimated_direction_deg": shape_results["estimated_direction_deg"],
    }
    with right:
        st.plotly_chart(visuals.multicore_figure(shape), width="stretch")
    metric_a, metric_b, metric_c, metric_d = st.columns(4)
    metric_a.metric("反演曲率", f"{shape['estimated_curvature_per_m']:.3f} 1/m", f"误差 {shape['estimated_curvature_per_m'] - curvature:+.3f}")
    metric_b.metric(
        "反演弯曲方向",
        f"{shape['estimated_direction_deg']:.1f} °" if shape_results["direction_identifiable"] else "不适用",
        f"误差 {shape_results['direction_error_deg']:+.1f} °" if shape_results["direction_identifiable"] else "零曲率",
    )
    metric_c.metric("中心线 RMSE", f"{shape_results['centerline_rmse_mm']:.3f} mm")
    metric_d.metric("末端误差", f"{shape_results['tip_error_mm']:.3f} mm")
    if abs(core_temperature_gradient) > 0.0:
        st.warning(f"芯间温度梯度 {core_temperature_gradient:+.1f} °C/芯：差分应变只能消除共模温度，芯间温差会让反演曲率/方向偏离真实值。")
    st.plotly_chart(visuals.sensor_bar_figure(shape["core_angles_deg"], shape["wavelength_shifts_nm"], "三根纤芯的波长漂移"), width="stretch")
    st.caption("三维图实线为真实中心线、虚线为反演中心线；中心线 RMSE 汇总整条曲线的逐点空间误差，末端误差只看最后一个点。柱状图是三芯波长漂移，芯间差异决定弯曲方向；扭转率作为已知先验参与中心线重建。")
    st.download_button("下载当前多芯光纤读数 CSV", csv_bytes(["Core 1", "Core 2", "Core 3"], shape["wavelength_shifts_nm"]), "multicore_fbg_readings.csv", "text/csv")
    if st.button("保存当前为形状基线 A", key="save_shape_baseline"):
        st.session_state.shape_baseline = shape_record
    shape_baseline = st.session_state.get("shape_baseline")
    if shape_baseline is not None:
        baseline_parameters = shape_baseline["parameters"]
        baseline_results = shape_baseline["results"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": ["真实曲率", "弯曲方向", "扭转率（已知先验）", "光纤长度", "温度变化", "芯间温度梯度", "波长噪声", "随机种子", "曲率误差", "方向误差", "中心线 RMSE", "末端误差"],
            "基线 A": [
                f"{baseline_parameters['curvature_per_m']:.2f} 1/m", f"{baseline_parameters['direction_deg']:.1f}°",
                f"{baseline_parameters['twist_per_m']:.2f} 1/m", f"{baseline_parameters['length_mm']:.1f} mm",
                f"{baseline_parameters['temperature_c']:.1f}°C", f"{baseline_parameters['core_temperature_gradient_c']:+.1f}°C/芯",
                f"{baseline_parameters['noise_nm']:.4f} nm", str(baseline_parameters["seed"]), f"{baseline_results['curvature_error_per_m']:+.3f} 1/m",
                f"{baseline_results['direction_error_deg']:+.1f}°" if baseline_results["direction_identifiable"] else "不适用", f"{baseline_results['centerline_rmse_mm']:.3f} mm",
                f"{baseline_results['tip_error_mm']:.3f} mm",
            ],
            "当前 B": [
                f"{curvature:.2f} 1/m", f"{direction:.1f}°", f"{twist:.2f} 1/m", f"{shape_length:.1f} mm",
                f"{temperature:.1f}°C", f"{core_temperature_gradient:+.1f}°C/芯",
                f"{noise:.4f} nm", str(int(seed)), f"{shape_results['curvature_error_per_m']:+.3f} 1/m", f"{shape_results['direction_error_deg']:+.1f}°" if shape_results["direction_identifiable"] else "不适用",
                f"{shape_results['centerline_rmse_mm']:.3f} mm", f"{shape_results['tip_error_mm']:.3f} mm",
            ],
        }, hide_index=True, width="stretch")
    shape_download_a, shape_download_b = st.columns(2)
    shape_download_a.download_button(
        "下载形状重建 CSV", experiments.shape_csv_bytes(shape_record),
        "shape_reconstruction_experiment.csv", "text/csv",
    )
    shape_download_b.download_button(
        "下载形状重建报告", experiments.shape_report(shape_record),
        "shape_reconstruction_report.txt", "text/plain",
    )
    with st.expander("与分布式传感的联动"):
        distributed_link = models.simulate_shape_distributed_link(shape_length, shape["strain"], curvature)
        st.plotly_chart(visuals.shape_distributed_link_figure(distributed_link), width="stretch")
        st.caption("三芯光纤沿长度的应变接近均匀（恒定曲率），把应变沿长度积分即可还原形状；虚线是分布式 Rayleigh 能看到的局部事件峰，用于对比“点式三芯”与“连续分布”两种视角。")
    with st.expander("为何温度影响较小？"):
        st.write("三根纤芯共享近似相同的温度漂移。算法先减去三路读数的均值，再由差分应变求曲率向量，因此可以抑制共模温度项；但芯间温度梯度、封装不对称会让差分补偿失效——你可以把“芯间温度梯度”滑杆调大观察误差如何增长。")

with health_tab:
    st.subheader("机械臂结构健康监测：点式 FBG 阵列局部异常定位")
    module_learning_frame(
        "区分‘检测到异常’与‘异常定位准确’，并理解阵列密度对定位区间的影响。",
        "先载入健康基线，再载入局部异常；分别比较稀疏与高密度阵列。",
        "核对真实异常位置、可疑位置、定位误差、定位区间和检测是否符合设定。",
        "‘需检查’是教学阈值触发，不是裂纹确认、寿命预测或结构安全结论。",
    )
    for key, value in (
        ("arm_load", 80.0), ("anomaly_position", 320.0),
        ("anomaly_severity", 0.0), ("arm_fbg_count", 6),
    ):
        st.session_state.setdefault(key, value)

    def load_health_preset() -> None:
        preset = experiments.HEALTH_PRESETS[st.session_state.health_preset]
        st.session_state.arm_load = preset["load_n"]
        st.session_state.anomaly_position = preset["anomaly_position_mm"]
        st.session_state.anomaly_severity = preset["anomaly_severity"]
        st.session_state.arm_fbg_count = preset["sensor_count"]
        st.session_state.global_temperature = preset["temperature_c"]
        st.session_state.global_noise = preset["noise_nm"]
        st.session_state.global_seed = preset["seed"]

    health_preset_column, health_load_column = st.columns([2, 1])
    with health_preset_column:
        st.selectbox("推荐实验场景", list(experiments.HEALTH_PRESETS), key="health_preset")
    with health_load_column:
        st.button("载入健康场景", key="load_health_preset", on_click=load_health_preset, width="stretch")
    st.caption("载入场景会同步本页输入以及侧栏温度、波长噪声和随机种子；异常位置参数仅在异常程度大于 0 且形成有效定位时参与对照。")
    health_left, health_right = st.columns([1, 2])
    with health_left:
        arm_load = st.slider("机械臂载荷 (N)", 0.0, 160.0, step=5.0, key="arm_load")
        anomaly_position = st.slider("局部异常位置 (mm)", 0.0, 520.0, step=5.0, key="anomaly_position")
        anomaly_severity = st.slider("异常程度", 0.0, 1.0, step=0.05, key="anomaly_severity")
        fbg_count = st.select_slider("FBG 阵列密度（数量）", options=[4, 6, 8], key="arm_fbg_count")
    health_record = experiments.run_health_experiment({
        "load_n": arm_load, "anomaly_position_mm": anomaly_position,
        "anomaly_severity": anomaly_severity, "sensor_count": int(fbg_count),
        "temperature_c": temperature, "noise_nm": noise, "seed": int(seed),
    })
    health_results = health_record["results"]
    sensor_positions = np.asarray(health_results["sensor_positions_mm"], dtype=float)
    arm_health = {
        "sensor_positions_mm": sensor_positions,
        "strain": np.asarray(health_results["strain_ue"], dtype=float) * 1e-6,
        "wavelength_shifts_nm": np.asarray(health_results["wavelength_shifts_nm"], dtype=float),
    }
    diagnosis = {
        "status": health_results["status"],
        "suspected_location_mm": health_results["suspected_location_mm"],
        "location_uncertainty_mm": health_results["location_uncertainty_mm"],
        "sensor_spacing_mm": health_results["sensor_spacing_mm"],
        "damage_index": health_results["damage_index"],
        "localization_valid": health_results["localization_valid"],
    }
    with health_right:
        health_figure = visuals.arm_health_figure(arm_health, diagnosis)
        if anomaly_severity > 0.0:
            health_figure.add_vline(
                x=anomaly_position, line_dash="dot", line_color=visuals.COLORS["truth"],
                annotation_text=f"真实设定 {anomaly_position:.0f} mm",
                annotation_position="bottom right",
            )
        st.plotly_chart(health_figure, width="stretch")
    health_a, health_b, health_c, health_d = st.columns(4)
    health_a.metric("诊断状态", str(diagnosis["status"]))
    health_b.metric(
        "可疑位置",
        f"{diagnosis['suspected_location_mm']:.0f} ± {diagnosis['location_uncertainty_mm']:.0f} mm"
        if health_results["localization_valid"] else "未形成定位",
    )
    health_c.metric(
        "定位误差",
        f"{health_results['localization_error_mm']:.2f} mm"
        if health_results["localization_error_mm"] is not None else "不适用",
    )
    health_d.metric("局部异常指数", f"{diagnosis['damage_index']:.2f}")
    if not health_results["localization_valid"]:
        st.info("当前没有形成有效异常定位，页面与报告不显示可疑位置或定位区间。")
    elif health_results["localization_error_mm"] <= health_results["location_uncertainty_mm"]:
        st.success("当前可疑位置覆盖真实设定位置；继续比较不同阵列密度下的定位区间。")
    else:
        st.warning("当前定位误差已超出教学区间，请检查波长噪声、温度条件与阵列密度。")
    def reset_arm_health_demo() -> None:
        for key, value in (
            ("arm_load", 80.0),
            ("anomaly_position", 320.0),
            ("anomaly_severity", 0.0),
            ("arm_fbg_count", 6),
        ):
            st.session_state[key] = value
    st.button("重置本页演示参数", key="reset_arm_health", on_click=reset_arm_health_demo)
    st.plotly_chart(visuals.sensor_bar_figure(arm_health["sensor_positions_mm"], arm_health["wavelength_shifts_nm"], f"机械臂 {fbg_count} 路 FBG 波长漂移"), width="stretch")
    st.caption("“需检查”表示局部差分应变超过本教学模型阈值，不等同于真实裂纹结论；定位误差比较可疑位置与仿真真值，± 区间约为 FBG 间距的一半。真实结构健康监测还需健康基线、载荷工况、温度场和无损检测交叉验证。")
    if st.button("保存当前为健康基线 A", key="save_health_baseline"):
        st.session_state.health_baseline = health_record
    health_baseline = st.session_state.get("health_baseline")
    if health_baseline is not None:
        baseline_parameters = health_baseline["parameters"]
        baseline_results = health_baseline["results"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": ["异常程度", "真实异常位置", "阵列数量", "传感器间距", "波长噪声", "诊断状态", "可疑位置", "定位误差", "定位区间"],
            "基线 A": [
                f"{baseline_parameters['anomaly_severity']:.2f}",
                f"{baseline_parameters['anomaly_position_mm']:.1f} mm" if baseline_parameters["anomaly_severity"] > 0.0 else "未启用",
                str(baseline_parameters["sensor_count"]), f"{baseline_results['sensor_spacing_mm']:.1f} mm",
                f"{baseline_parameters['noise_nm']:.4f} nm", baseline_results["status"],
                f"{baseline_results['suspected_location_mm']:.1f} mm" if baseline_results["localization_valid"] else "未形成定位",
                "不适用" if baseline_results["localization_error_mm"] is None else f"{baseline_results['localization_error_mm']:.2f} mm",
                f"± {baseline_results['location_uncertainty_mm']:.1f} mm" if baseline_results["localization_valid"] else "不适用",
            ],
            "当前 B": [
                f"{anomaly_severity:.2f}", f"{anomaly_position:.1f} mm" if anomaly_severity > 0.0 else "未启用", str(fbg_count),
                f"{health_results['sensor_spacing_mm']:.1f} mm", f"{noise:.4f} nm", health_results["status"],
                f"{health_results['suspected_location_mm']:.1f} mm" if health_results["localization_valid"] else "未形成定位",
                "不适用" if health_results["localization_error_mm"] is None else f"{health_results['localization_error_mm']:.2f} mm",
                f"± {health_results['location_uncertainty_mm']:.1f} mm" if health_results["localization_valid"] else "不适用",
            ],
        }, hide_index=True, width="stretch")
    health_download_a, health_download_b = st.columns(2)
    health_download_a.download_button(
        "下载健康监测 CSV", experiments.health_csv_bytes(health_record),
        "structural_health_experiment.csv", "text/csv",
    )
    health_download_b.download_button(
        "下载健康监测报告", experiments.health_report(health_record),
        "structural_health_report.txt", "text/plain",
    )
    with st.expander("分布式 vs 点式 FBG（定位对比）"):
        if health_results["localization_valid"]:
            distributed_arm = models.simulate_rayleigh_ofdr(
                520.0,
                float(diagnosis["suspected_location_mm"]),
                float(diagnosis["damage_index"]) * 800.0,
                2.0,
            )
            st.plotly_chart(
                visuals.arm_distributed_vs_fbg_figure(
                    distributed_arm,
                    arm_health["sensor_positions_mm"],
                    arm_health["strain"] * 1e6,
                    float(diagnosis["suspected_location_mm"]),
                    float(diagnosis["location_uncertainty_mm"]),
                ),
                width="stretch",
            )
            st.caption("连续 Rayleigh 曲线能直接读出峰的位置与宽度；点式 FBG 只能给出最近传感器的定位区间（约为间距一半），间距越密区间越小。")
        else:
            st.info("当前未形成有效异常定位，因此不生成基于伪峰值的分布式定位对比。")
    tab_jump_button(5, "下一步 → 分布式感知", "optics_navigation", "分布式感知")

with distributed_tab:
    st.subheader("分布式光纤感知：连续空间上的应变、振动与温度")
    st.caption("本页以四类教学解析模型对比不同散射机制的观测量：Rayleigh/OFDR 连续应变、φ-OTDR/DAS 振动事件、Brillouin 频移、Raman 分布式温度。")
    distributed_widget_keys = {
        "mode": "distributed_mode",
        "fiber_length_mm": "distributed_fiber_length",
        "event_position_mm": "distributed_event_position",
        "event_strength": "distributed_event_strength",
        "spatial_spacing_mm": "distributed_spatial_spacing",
        "sample_rate_hz": "global_sample_rate",
    }
    for parameter, key in distributed_widget_keys.items():
        st.session_state.setdefault(
            key, experiments.DISTRIBUTED_PRESETS["Rayleigh 局部应变"][parameter]
        )

    def load_distributed_preset() -> None:
        for parameter, key in distributed_widget_keys.items():
            st.session_state[key] = experiments.DISTRIBUTED_PRESETS[
                st.session_state.distributed_preset
            ][parameter]

    distributed_preset_column, distributed_load_column = st.columns([2, 1])
    distributed_preset_column.selectbox(
        "推荐实验场景", list(experiments.DISTRIBUTED_PRESETS), key="distributed_preset"
    )
    distributed_load_column.button(
        "载入分布式场景", key="load_distributed_preset",
        on_click=load_distributed_preset, width="stretch",
    )
    st.caption("载入场景会同步机制、事件参数、空间采样间隔和侧栏采样率；保存基线后可量化采样变化带来的定位误差。")
    distributed_left, distributed_right = st.columns([1, 2])
    with distributed_left:
        distributed_mode = st.selectbox("分布式机制", ["Rayleigh/OFDR", "φ-OTDR / DAS", "Brillouin", "Raman"], key="distributed_mode")
        fiber_length = st.slider("分布式光纤长度 (mm)", 100.0, 800.0, step=10.0, key="distributed_fiber_length")
        event_position = st.slider("局部事件位置 (mm)", 0.0, 800.0, step=5.0, key="distributed_event_position")
        event_strength = st.slider("局部应变 / 温度幅值", 0.0, 1000.0, step=10.0, key="distributed_event_strength")
        spatial_spacing = st.slider("空间采样间隔 (mm)", 2, 40, step=1, key="distributed_spatial_spacing")
    event_position = min(event_position, fiber_length)
    distributed_parameters = {
        "mode": distributed_mode,
        "fiber_length_mm": fiber_length,
        "event_position_mm": event_position,
        "event_strength": event_strength,
        "spatial_spacing_mm": int(spatial_spacing),
        "sample_rate_hz": int(sample_rate),
    }
    distributed_record = experiments.run_distributed_experiment(distributed_parameters)
    distributed_result, distributed_frame = models.simulate_distributed_mechanism(
        distributed_mode, fiber_length, event_position, event_strength, int(sample_rate)
    )
    distributed_result = models.decimate_distributed_result(distributed_result, spatial_spacing)
    with distributed_right:
        if distributed_mode == "φ-OTDR / DAS":
            st.plotly_chart(visuals.das_event_figure(distributed_result), width="stretch")
        else:
            curve_kind = {"Rayleigh/OFDR": "Rayleigh", "Brillouin": "Brillouin", "Raman": "Raman"}[distributed_mode]
            st.plotly_chart(visuals.distributed_curve_figure(distributed_result, curve_kind), width="stretch")
        st.caption("曲线峰值（或热图亮斑）所在位置即事件位置；把“空间采样间隔”调大可看到峰被低估或漏掉。")
    compare_all = st.checkbox("四机制对比视图（同一事件参数）", value=False)
    if compare_all:
        distributed_modes = ["Rayleigh/OFDR", "φ-OTDR / DAS", "Brillouin", "Raman"]
        for row_start in range(0, 4, 2):
            comparison_columns = st.columns(2)
            for column, mode in zip(comparison_columns, distributed_modes[row_start:row_start + 2]):
                result, frame = models.simulate_distributed_mechanism(
                    mode, fiber_length, event_position, event_strength, int(sample_rate)
                )
                result = models.decimate_distributed_result(result, spatial_spacing)
                with column:
                    if mode == "φ-OTDR / DAS":
                        st.plotly_chart(visuals.das_event_figure(result), width="stretch")
                    else:
                        kind = {"Rayleigh/OFDR": "Rayleigh", "Brillouin": "Brillouin", "Raman": "Raman"}[mode]
                        st.plotly_chart(visuals.distributed_curve_figure(result, kind), width="stretch")
                    st.caption(f"{mode} · 数据质量 {float(frame['quality']) * 100:.0f}%")
        st.markdown(
            "| 机制 | 观测量 | 教学特点 |\n"
            "|---|---|---|\n"
            "| Rayleigh/OFDR | 连续应变曲线 | 高空间分辨率，适合静态/准静态分布 |\n"
            "| φ-OTDR / DAS | 时间-距离振动热图 | 动态事件定位，依赖采样率 |\n"
            "| Brillouin | 频移分布 | 应变与温度交叉敏感 |\n"
            "| Raman | 温度分布 | 主要对温度敏感，可作温补参考 |"
        )
    distributed_a, distributed_b, distributed_c = st.columns(3)
    distributed_a.metric("传感机制", str(distributed_frame["sensor_type"]))
    sampled_points = len(np.asarray(distributed_result.get("das_distance_mm", distributed_result["position_mm"])))
    distributed_b.metric("显示采样点", f"{sampled_points}")
    distributed_c.metric("数据质量", f"{float(distributed_frame['quality']) * 100:.0f}%")
    distributed_location = distributed_record["results"]
    location_a, location_b = st.columns(2)
    location_a.metric("估计事件位置", f"{distributed_location['estimated_event_position_mm']:.1f} mm")
    location_b.metric("事件定位误差", f"{distributed_location['location_error_mm']:.1f} mm")
    def reset_distributed_demo() -> None:
        for key, value in (
            ("distributed_mode", "Rayleigh/OFDR"),
            ("distributed_fiber_length", 300.0),
            ("distributed_event_position", 140.0),
            ("distributed_event_strength", 600.0),
            ("distributed_spatial_spacing", 10),
        ):
            st.session_state[key] = value
    st.button("重置本页演示参数", key="reset_distributed", on_click=reset_distributed_demo)
    if st.button("保存当前为分布式基线 A", key="save_distributed_baseline"):
        st.session_state.distributed_baseline = distributed_record
    distributed_baseline = st.session_state.get("distributed_baseline")
    if distributed_baseline is not None:
        baseline_parameters = distributed_baseline["parameters"]
        baseline_results = distributed_baseline["results"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": ["机制", "光纤长度", "事件位置", "事件幅值", "空间采样间隔", "采样率", "估计位置", "定位误差", "数据质量"],
            "基线 A": [
                baseline_parameters["mode"], f"{baseline_parameters['fiber_length_mm']:.1f} mm",
                f"{baseline_parameters['event_position_mm']:.1f} mm", f"{baseline_parameters['event_strength']:.1f}",
                f"{baseline_parameters['spatial_spacing_mm']} mm", f"{baseline_parameters['sample_rate_hz']} Hz",
                f"{baseline_results['estimated_event_position_mm']:.1f} mm", f"{baseline_results['location_error_mm']:.1f} mm",
                f"{baseline_results['quality'] * 100:.0f}%",
            ],
            "当前 B": [
                distributed_parameters["mode"], f"{distributed_parameters['fiber_length_mm']:.1f} mm",
                f"{distributed_parameters['event_position_mm']:.1f} mm", f"{distributed_parameters['event_strength']:.1f}",
                f"{distributed_parameters['spatial_spacing_mm']} mm", f"{distributed_parameters['sample_rate_hz']} Hz",
                f"{distributed_location['estimated_event_position_mm']:.1f} mm", f"{distributed_location['location_error_mm']:.1f} mm",
                f"{distributed_location['quality'] * 100:.0f}%",
            ],
        }, hide_index=True, width="stretch")
    distributed_download_a, distributed_download_b = st.columns(2)
    distributed_download_a.download_button(
        "下载分布式实验 CSV", experiments.distributed_csv_bytes(distributed_record),
        "distributed_fiber_experiment.csv", "text/csv",
    )
    distributed_download_b.download_button(
        "下载分布式实验报告", experiments.distributed_report(distributed_record).encode("utf-8-sig"),
        "distributed_fiber_experiment_report.txt", "text/plain",
    )
    st.caption("不同机制的空间分辨率、测量距离、采样速度与温度—应变交叉敏感性不同；此处用于机制与数据形态比较，不代表具体商用解调设备指标。把“空间采样间隔”调大可以看到事件峰被低估或漏掉。")
    with st.expander("机制能力与分辨率对比"):
        st.markdown(
            "| 机制 | 应变 | 振动 | 温度 | 分辨率特点 |\n"
            "|---|---|---|---|---|\n"
            "| Rayleigh/OFDR | ✓ 静态/准静态 | 动态受限 | ✗ | 高（亚 mm～cm 级）|\n"
            "| φ-OTDR / DAS | 动态应变 | ✓ | ✗ | 中（受脉冲与采样限制）|\n"
            "| Brillouin | ✓ | ✗ | ✓（与应变交叉敏感） | 中（m 级）|\n"
            "| Raman | ✗ | ✗ | ✓ | 中 |"
        )
        st.caption("可测性为教学相对特点，不是具体商用设备指标；Brillouin 的温度—应变交叉敏感需要用 Raman 或参考段解耦。")
    st.divider()
    with st.expander("Brillouin × Raman：温度伪装成应变的温补解耦"):
        compensation_left, compensation_right = st.columns([1, 2])
        with compensation_left:
            strain_peak = st.slider("真实应变峰值 (με)", 0.0, 800.0, 400.0, 10.0)
            ambient_temperature = st.slider("环境温度变化 (°C)", -10.0, 10.0, 5.0, 0.5)
            local_temperature_rise = st.slider("局部温升 (°C)", 0.0, 30.0, 15.0, 1.0)
            raman_temperature_noise = st.slider("Raman 温度噪声 (°C)", 0.0, 2.0, 0.2, 0.05)
        compensation = models.simulate_brillouin_raman_compensation(
            fiber_length, event_position, strain_peak, ambient_temperature,
            local_temperature_rise, temperature_noise_c=raman_temperature_noise, seed=int(seed),
        )
        with compensation_right:
            st.plotly_chart(visuals.brillouin_raman_compensation_figure(compensation), width="stretch")
        compensation_a, compensation_b, compensation_c = st.columns(3)
        compensation_a.metric("未温补峰值误差", f"{float(compensation['naive_peak_error_ue']):.0f} με")
        compensation_b.metric("温补后峰值误差", f"{float(compensation['compensated_peak_error_ue']):.0f} με")
        compensation_c.metric("温度估计 RMSE", f"{float(compensation['temperature_rms_error_c']):.3f} °C")
        st.caption(f"Brillouin 对温度和应变都敏感（本例 1 °C 会伪装成约 22 με 应变）；用 Raman 测出温度后从频移中扣除，才能还原真实应变。{compensation['validation_boundary']}。")

with fbg_simplus_tab:
    st.subheader("FBG-SimPlus 兼容：通用八列数据适配")
    st.caption("本页读取 FBG-SimPlus 所需的八列数值数据并进行标准化；不要求特定仿真软件，不包含、复制、执行或修改 FBG-SimPlus 源代码，也不在本网站生成其反射谱。")
    module_learning_frame(
        "识别通用八列输入的列顺序、分隔符、表头跳过规则和位置递增要求。",
        "依次载入标准文本、带表头 CSV、列数不足和位置重复练习，比较通过与拒绝原因。",
        "记录采样跨度、最小/最大间隔、间隔变异以及八列数值范围。",
        "本页只判断文本结构和数值条件，不判断单位、材料参数、FEM 模型或反射谱是否正确。",
    )
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
    def load_fbg_simplus_example() -> None:
        example = FBG_SIMPLUS_EXAMPLES[st.session_state.fbg_simplus_example]
        st.session_state.fbg_simplus_use_example = True
        st.session_state.fbg_simplus_delimiter = example["delimiter"]
        st.session_state.fbg_simplus_skip_rows = example["skip_rows"]

    example_choice, template_download, example_load = st.columns([2, 1, 1])
    example_choice.selectbox(
        "内置数据练习", list(FBG_SIMPLUS_EXAMPLES), key="fbg_simplus_example"
    )
    template_download.download_button(
        "下载通用八列文本模板", FBG_SIMPLUS_TEMPLATE.encode("utf-8"),
        "fbg_simplus_eight_column_template.txt", "text/plain", width="stretch",
    )
    example_load.button(
        "载入内置八列示例", key="load_fbg_simplus_example", on_click=load_fbg_simplus_example,
        width="stretch",
    )
    selected_example = FBG_SIMPLUS_EXAMPLES[st.session_state.fbg_simplus_example]
    st.caption(f"{selected_example['expected']} 上传文件后优先检查上传内容。")
    input_left, input_right = st.columns(2)
    with input_left:
        input_delimiter = st.selectbox("输入分隔符", ["自动识别", "空白字符", "逗号（CSV）", "制表符（TSV）"], key="fbg_simplus_delimiter")
    with input_right:
        input_skip_rows = st.number_input("跳过文件开头行数", min_value=0, step=1, key="fbg_simplus_skip_rows")
    uploaded_export = st.file_uploader(
        "上传八列数据（.txt / .dat / .csv，最大 8 MB）",
        type=["txt", "dat", "csv"], key="fbg_simplus_input", max_upload_size=8,
    )
    st.caption(f"单次最多处理 {models.MAX_FBG_SIMPLUS_ROWS:,} 个有效采样行；更大的数据请先分段或降采样。")
    use_builtin_example = bool(st.session_state.get("fbg_simplus_use_example", False))
    source_text = None
    source_delimiter = input_delimiter
    source_skip_rows = int(input_skip_rows)
    source_label = "上传文件"
    if uploaded_export is not None:
        try:
            source_text = uploaded_export.getvalue().decode("utf-8-sig")
        except UnicodeDecodeError as error:
            st.error(f"无法作为 FBG-SimPlus 兼容输入读取：{error}")
    elif use_builtin_example:
        selected_example = FBG_SIMPLUS_EXAMPLES[st.session_state.fbg_simplus_example]
        source_text = selected_example["text"]
        source_delimiter = selected_example["delimiter"]
        source_skip_rows = int(selected_example["skip_rows"])
        source_label = "内置示例"
    if source_text is not None:
        try:
            parsed_export = models.parse_fbg_simplus_comsol_export(
                source_text, source_delimiter, source_skip_rows
            )
        except ValueError as error:
            st.error(f"无法作为 FBG-SimPlus 兼容输入读取：{error}")
        else:
            st.success(f"{source_label}已通过格式检查：{parsed_export['sample_count']} 个采样点；识别为{parsed_export['source_delimiter']}。可下载标准化文本后导入 FBG-SimPlus。")
            st.plotly_chart(visuals.fbg_simplus_input_figure(parsed_export), width="stretch")
            st.caption("上方为位置-应变/应力-温度剖面，仅用于结构检查；单位与物理含义需按数据来源和 FBG-SimPlus 文档确认。")
            quality_summary = models.summarise_fbg_simplus_input(parsed_export)
            quality_a, quality_b, quality_c, quality_d = st.columns(4)
            quality_a.metric("位置跨度", f"{quality_summary['position_span_mm']:.3f} mm")
            quality_b.metric("最小采样间隔", f"{quality_summary['minimum_spacing_mm']:.3f} mm")
            quality_c.metric("最大采样间隔", f"{quality_summary['maximum_spacing_mm']:.3f} mm")
            quality_d.metric("采样间隔变异", f"{quality_summary['spacing_cv_percent']:.2f}%")
            with st.expander("查看八列数值范围", expanded=True):
                column_ranges = quality_summary["column_ranges"]
                st.dataframe({
                    "列与约定单位": list(column_ranges),
                    "最小值": [limits[0] for limits in column_ranges.values()],
                    "最大值": [limits[1] for limits in column_ranges.values()],
                }, hide_index=True, width="stretch")
                st.caption("间隔变异为采样间隔的变异系数；不均匀采样不一定无效，但导入前应确认外部工具与后续算法是否支持。数值范围只用于发现明显量级或列映射问题。")
            download_label = "下载内置示例的标准化八列文本" if source_label == "内置示例" else "下载标准化八列文本（供 FBG-SimPlus 导入）"
            st.download_button(download_label, models.fbg_simplus_normalised_text(parsed_export).encode("utf-8"), "fbg_simplus_normalised_input.txt", "text/plain")
            st.info("检查范围仅限文本结构、数值有效性和位置递增性；应变/应力分量的物理含义、单位和 FBG 参数仍需由数据来源和 FBG-SimPlus 文档确认。")

with polarization_tab:
    st.subheader("偏振与干涉传感：偏振态、旋转与微腔光程差")
    module_learning_frame(
        "区分偏振 Stokes 状态、Sagnac 旋转相位和 EFPI 微腔长度三类光学观测量。",
        "先载入偏振基线，再分别比较横向应力、光纤扭转、温度交叉敏感和旋转压力场景。",
        "记录 S1/S2/S3、方位角、椭圆率、Sagnac 相位差、EFPI 腔长变化及温度偏移。",
        "三部分是彼此独立的解析教学模型，不代表实际器件的解调精度、交叉敏感性或量程。",
    )
    for key, value in (
        ("optical_stress", 120.0), ("optical_twist", 35.0),
        ("optical_gyro_rate", 45.0), ("optical_pressure", 0.4),
        ("optical_cavity_um", 28.0),
    ):
        st.session_state.setdefault(key, value)

    def load_optical_preset() -> None:
        preset = experiments.OPTICAL_PRESETS[st.session_state.optical_preset]
        st.session_state.optical_stress = preset["stress_mpa"]
        st.session_state.optical_twist = preset["twist_deg"]
        st.session_state.optical_gyro_rate = preset["gyro_rate_deg_s"]
        st.session_state.optical_pressure = preset["pressure_mpa"]
        st.session_state.optical_cavity_um = preset["cavity_um"]
        st.session_state.global_temperature = preset["temperature_c"]

    optical_preset_column, optical_load_column = st.columns([2, 1])
    with optical_preset_column:
        st.selectbox("推荐实验场景", list(experiments.OPTICAL_PRESETS), key="optical_preset")
    with optical_load_column:
        st.button("载入光学场景", key="load_optical_preset", on_click=load_optical_preset, width="stretch")
    st.caption("载入场景会同步本页输入和侧栏温度；偏振、Sagnac 与 EFPI 参数保持分区显示，避免把三种机制误认为同一传感链。")
    pol_left, pol_right = st.columns([1, 2])
    with pol_left:
        transverse_stress = st.slider("横向应力 (MPa)", 0.0, 250.0, step=5.0, key="optical_stress")
        twist_angle = st.slider("光纤扭转 (°)", -90.0, 90.0, step=1.0, key="optical_twist")
        gyro_rate = st.slider("角速度 (°/s)", -180.0, 180.0, step=1.0, key="optical_gyro_rate")
        cavity_pressure = st.slider("EFPI 压力 (MPa)", 0.0, 1.0, step=0.01, key="optical_pressure")
        cavity_length = st.slider("EFPI 初始腔长 (μm)", 10.0, 60.0, step=0.5, key="optical_cavity_um")
    polarization = models.simulate_polarization_sensing(transverse_stress, twist_angle, temperature)
    gyro = models.simulate_sagnac_gyro(gyro_rate, 120.0)
    efpi = models.simulate_efpi_pressure(cavity_pressure, cavity_length)
    optical_record = experiments.run_optical_experiment({
        "stress_mpa": transverse_stress, "twist_deg": twist_angle,
        "gyro_rate_deg_s": gyro_rate, "pressure_mpa": cavity_pressure,
        "cavity_um": cavity_length, "temperature_c": temperature,
    })
    optical_results = optical_record["results"]
    with pol_right:
        st.plotly_chart(visuals.polarization_figure(polarization), width="stretch")
        st.plotly_chart(visuals.efpi_figure(efpi), width="stretch")
        st.caption("庞加莱球上的向量端点表示偏振态（方位角/椭圆率）；EFPI 谱条纹周期随腔长变化，压力改变腔长使条纹疏密变化。")
    pol_a, pol_b, pol_c = st.columns(3)
    pol_a.metric("偏振方位角", f"{polarization['azimuth_deg']:.1f} °")
    pol_b.metric("椭圆率角", f"{polarization['ellipticity_deg']:.1f} °")
    pol_c.metric("Sagnac 相位差", f"{gyro['phase_shift_rad']:.3e} rad")
    optical_a, optical_b, optical_c = st.columns(3)
    optical_a.metric("Stokes 向量", "(" + ", ".join(f"{value:+.2f}" for value in optical_results["stokes"]) + ")")
    optical_b.metric("EFPI 腔长变化", f"{optical_results['cavity_change_nm']:+.1f} nm")
    optical_c.metric("温度引起的椭圆率偏移", f"{optical_results['temperature_ellipticity_offset_deg']:+.2f} °")
    st.caption("偏振态模块用于理解双折射、扭转与温度对 Stokes 参数的影响；Sagnac 相位正负号表示旋转方向，EFPI 展示腔长干涉。均为教学模型，不代表惯导或压力传感器精度。")
    if st.button("保存当前为光学基线 A", key="save_optical_baseline"):
        st.session_state.optical_baseline = optical_record
    optical_baseline = st.session_state.get("optical_baseline")
    if optical_baseline is not None:
        baseline_parameters = optical_baseline["parameters"]
        baseline_results = optical_baseline["results"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": ["横向应力", "扭转", "温度变化", "角速度", "EFPI 压力", "偏振方位角", "椭圆率角", "Sagnac 相位差", "EFPI 腔长变化"],
            "基线 A": [
                f"{baseline_parameters['stress_mpa']:.1f} MPa", f"{baseline_parameters['twist_deg']:.1f}°",
                f"{baseline_parameters['temperature_c']:.1f}°C", f"{baseline_parameters['gyro_rate_deg_s']:.1f} °/s",
                f"{baseline_parameters['pressure_mpa']:.2f} MPa", f"{baseline_results['azimuth_deg']:.2f}°",
                f"{baseline_results['ellipticity_deg']:.2f}°", f"{baseline_results['sagnac_phase_shift_rad']:.3e} rad",
                f"{baseline_results['cavity_change_nm']:+.2f} nm",
            ],
            "当前 B": [
                f"{transverse_stress:.1f} MPa", f"{twist_angle:.1f}°", f"{temperature:.1f}°C",
                f"{gyro_rate:.1f} °/s", f"{cavity_pressure:.2f} MPa", f"{optical_results['azimuth_deg']:.2f}°",
                f"{optical_results['ellipticity_deg']:.2f}°", f"{optical_results['sagnac_phase_shift_rad']:.3e} rad",
                f"{optical_results['cavity_change_nm']:+.2f} nm",
            ],
        }, hide_index=True, width="stretch")
    with st.expander("机制响应曲线与可辨识性", expanded=True):
        st.plotly_chart(visuals.optical_response_scan_figure(120.0, cavity_length), width="stretch")
        st.markdown(
            "| 机制 | 独立输入 | 直接观测量 | 本页可验证关系 |\n"
            "|---|---|---|---|\n"
            "| 偏振 | 横向应力、扭转、温度 | Stokes、方位角、椭圆率 | 多输入会共同改变偏振态，需要基线或额外信息解耦 |\n"
            "| Sagnac | 角速度 | 相位差 | 相位正负对应旋转方向，幅值随角速度线性变化 |\n"
            "| EFPI | 压力 | 光谱与有效腔长 | 本教学模型中压力升高使腔长单调减小 |"
        )
    with st.expander("双折射—扭转—温度联合视图"):
        polarization_map = models.simulate_polarization_map(temperature_change_c=temperature)
        st.plotly_chart(visuals.polarization_map_figure(polarization_map), width="stretch")
        st.caption("横轴是光纤扭转、纵轴是横向应力，颜色表示该组合下的偏振方位角（左）与椭圆率角（右）；把上方滑块调到图中任意一点，单点图会显示对应的偏振态。")
    optical_download_a, optical_download_b = st.columns(2)
    optical_download_a.download_button(
        "下载偏振与干涉 CSV", experiments.optical_csv_bytes(optical_record),
        "polarization_interference_experiment.csv", "text/csv",
    )
    optical_download_b.download_button(
        "下载偏振与干涉报告", experiments.optical_report(optical_record),
        "polarization_interference_report.txt", "text/plain",
    )

with chain_tab:
    st.subheader("解调器与实时数据链路：波长峰值 → 滤波温补 → 状态 → 控制")
    module_learning_frame(
        "理解同一条 FBG 波长流如何依次经过滤波、温补、角度反演和控制阈值判断。",
        "先载入阈值附近噪声，再载入强滤波对照；比较噪声 RMS、理论延迟和指令一致率。",
        "滤波窗口变大通常降低噪声，但增加响应延迟；阈值裕量越小，噪声越容易改变控制指令。",
        "延迟按居中移动平均的理论群延迟估算；本页不模拟真实解调器峰值搜索、通信延迟或执行器动力学。",
    )
    for key, value in (
        ("chain_angle", 55.0), ("chain_noise", 0.010),
        ("chain_filter_window", 5), ("chain_control_threshold", 35.0),
    ):
        st.session_state.setdefault(key, value)

    def load_chain_preset() -> None:
        preset = experiments.DEMODULATION_PRESETS[st.session_state.chain_preset]
        st.session_state.chain_angle = preset["angle_deg"]
        st.session_state.chain_noise = preset["noise_nm"]
        st.session_state.chain_filter_window = preset["filter_window"]
        st.session_state.chain_control_threshold = preset["control_threshold_deg"]
        st.session_state.global_temperature = preset["temperature_c"]
        st.session_state.global_sample_rate = preset["sample_rate_hz"]

    chain_preset_column, chain_load_column = st.columns([2, 1])
    chain_preset_column.selectbox(
        "推荐链路实验", list(experiments.DEMODULATION_PRESETS), key="chain_preset"
    )
    chain_load_column.button(
        "载入链路场景", key="load_chain_preset", on_click=load_chain_preset,
        width="stretch",
    )
    st.caption("载入场景会同步弯曲角、噪声、滤波窗口、控制阈值，以及侧栏温度和采样率。")
    chain_left, chain_right = st.columns([1, 2])
    with chain_left:
        chain_angle = st.slider("链路真实弯曲角 (°)", 0.0, 90.0, step=1.0, key="chain_angle")
        chain_noise = st.slider("链路波长噪声 (nm)", 0.0, 0.030, step=0.001, key="chain_noise")
        chain_filter_window = st.select_slider(
            "移动平均滤波窗口（采样点）", options=[1, 3, 5, 9, 15, 21],
            key="chain_filter_window",
        )
        chain_control_threshold = st.slider(
            "闭合控制阈值 (°)", 5.0, 80.0, step=1.0, key="chain_control_threshold"
        )
    chain = models.simulate_demodulation_chain(
        chain_angle, temperature, int(sample_rate), chain_noise, int(seed),
        filter_window=int(chain_filter_window),
        control_threshold_deg=chain_control_threshold,
    )
    with chain_right:
        st.plotly_chart(visuals.demodulation_figure(chain), width="stretch")
        st.caption("三条曲线依次为原始、滤波、温补后的波长流；改变窗口可直接比较降噪与响应延迟的取舍。")
    chain_a, chain_b, chain_c, chain_d = st.columns(4)
    chain_a.metric("反演弯曲角", f"{chain['estimated_angle_deg']:.1f} °", f"误差 {chain['estimated_angle_deg'] - chain_angle:+.1f} °")
    chain_b.metric("控制输出", str(chain["control_command"]))
    chain_c.metric("滤波后噪声 RMS", f"{chain['filtered_noise_rms_nm']:.4f} nm", f"原始 {chain['raw_noise_rms_nm']:.4f} nm")
    chain_d.metric("理论滤波延迟", f"{chain['filter_delay_ms']:.1f} ms")
    chain_e, chain_f, chain_g = st.columns(3)
    chain_e.metric("控制阈值", f"{chain_control_threshold:.1f} °")
    chain_f.metric("阈值裕量", f"{chain['control_margin_deg']:+.1f} °")
    chain_g.metric("指令一致率", f"{chain['command_consistency_percent']:.1f}%")
    if st.button("保存当前为链路基线 A", key="save_chain_baseline"):
        st.session_state.chain_baseline = {
            "angle_deg": chain_angle, "temperature_c": temperature,
            "noise_nm": chain_noise, "sample_rate_hz": int(sample_rate),
            "filter_window": int(chain_filter_window),
            "control_threshold_deg": chain_control_threshold, "results": chain,
        }
    chain_baseline = st.session_state.get("chain_baseline")
    if chain_baseline is not None:
        baseline_chain = chain_baseline["results"]
        st.markdown("#### 基线 A 与当前 B")
        st.dataframe({
            "指标": ["弯曲角", "温度变化", "波长噪声", "采样率", "滤波窗口", "控制阈值", "反演角", "滤波后噪声 RMS", "理论延迟", "阈值裕量", "指令一致率"],
            "基线 A": [
                f"{chain_baseline['angle_deg']:.1f}°", f"{chain_baseline['temperature_c']:.1f}°C",
                f"{chain_baseline['noise_nm']:.4f} nm", f"{chain_baseline['sample_rate_hz']} Hz",
                str(chain_baseline["filter_window"]), f"{chain_baseline['control_threshold_deg']:.1f}°",
                f"{baseline_chain['estimated_angle_deg']:.2f}°", f"{baseline_chain['filtered_noise_rms_nm']:.4f} nm",
                f"{baseline_chain['filter_delay_ms']:.1f} ms", f"{baseline_chain['control_margin_deg']:+.2f}°",
                f"{baseline_chain['command_consistency_percent']:.1f}%",
            ],
            "当前 B": [
                f"{chain_angle:.1f}°", f"{temperature:.1f}°C", f"{chain_noise:.4f} nm", f"{sample_rate} Hz",
                str(chain_filter_window), f"{chain_control_threshold:.1f}°", f"{chain['estimated_angle_deg']:.2f}°",
                f"{chain['filtered_noise_rms_nm']:.4f} nm", f"{chain['filter_delay_ms']:.1f} ms",
                f"{chain['control_margin_deg']:+.2f}°", f"{chain['command_consistency_percent']:.1f}%",
            ],
        }, hide_index=True, width="stretch")
    with st.expander("解调链各环节如何影响结果"):
        st.markdown(
            "| 环节 | 本页输入或输出 | 主要作用 | 观察重点 |\n"
            "|---|---|---|---|\n"
            "| 波长采样 | 噪声、采样率 | 获得两秒钟原始波长流 | 原始噪声 RMS |\n"
            "| 移动平均 | 滤波窗口 | 压低随机波动 | 噪声降低与理论延迟同时增加 |\n"
            "| 温度补偿 | 侧栏温度 | 扣除已知共模热漂移 | 温补后机械信号回到弯曲分量 |\n"
            "| 状态反演 | 角度估计 | 把机械波长变化换算为弯曲角 | 反演误差 |\n"
            "| 控制判定 | 闭合阈值 | 输出张开或闭合 | 阈值裕量与指令一致率 |"
        )

    fusion_qualities = {
        "grasp": models.ModuleQuality(1.0 if three_d_fbg_decision["is_grasped"] else 0.0, 1.0, "三维 FBG 抓稳判定"),
        "foot": models.assess_foot_quality(cop),
        "shape": models.assess_shape_quality(shape["estimated_curvature_per_m"], curvature),
        "health": models.assess_health_quality(diagnosis["status"], diagnosis["damage_index"]),
        "distributed": models.ModuleQuality(float(distributed_frame["quality"]), 1.0, str(distributed_frame["sensor_type"])),
    }
    fusion = models.fuse_robot_sensing(fusion_qualities)
    fusion_a, fusion_b = st.columns(2)
    fusion_a.metric("多模态任务状态", str(fusion["status"]))
    fusion_b.metric("融合置信度", f"{float(fusion['confidence']) * 100:.0f}%")
    with st.expander("各模块质量"):
        for name, quality in fusion_qualities.items():
            st.metric(name, f"{float(quality.score) * 100:.0f}%", quality.note)

    st.divider()
    st.subheader("实验任务与报告")
    experiment = st.selectbox("实验任务", list(experiments.CHAIN_TASK_GUIDES), key="chain_experiment")
    task_steps = experiments.CHAIN_TASK_GUIDES[experiment]
    st.markdown(
        "**任务步骤：**\n" + "\n".join(
            f"{index}. {step}" for index, step in enumerate(task_steps, start=1)
        )
    )
    calibration_error = current_experiment["results"]["error_deg"]
    task_results = {
        "弯曲标定与温补": (
            "不可反演" if calibration_error is None else f"{abs(float(calibration_error)):.2f}",
            "角度反演误差",
        ),
        "冗余故障诊断": (f"{float(len(redundant_diagnosis['fault_channels'])):.2f}", "已隔离通道数"),
        "多材质触觉识别": (
            f"{float(tactile_results['pattern_error_percent']):.2f}" if tactile_results["valid_contact"] else "接触不足",
            "模板偏差 (%)",
        ),
        "足底平衡": (f"{float(foot_results['cop_error']):.2f}", "CoP 位置误差"),
        "连续体形状重建": (f"{float(shape_results['centerline_rmse_mm']):.2f}", "中心线 RMSE (mm)"),
        "结构健康监测": (
            f"{float(health_results['localization_error_mm']):.2f}"
            if health_results["localization_error_mm"] is not None
            else f"{float(health_results['damage_index']):.2f}",
            "定位误差 (mm)" if health_results["localization_error_mm"] is not None else "局部异常指数",
        ),
        "分布式事件定位": (f"{float(distributed_location['location_error_mm']):.2f}", "事件定位误差 (mm)"),
        "偏振与干涉传感": (f"{abs(float(optical_results['temperature_ellipticity_offset_deg'])):.2f}", "温度椭圆率偏移 (°)"),
    }
    task_value, task_label = task_results[experiment]
    st.metric(f"{experiment}：{task_label}", task_value)
    redundant_report = experiments.redundant_fbg_report(
        true_angle_deg=redundant_angle,
        temperature_c=temperature,
        fault_mode=fault_mode,
        injected_channel=fault_channel,
        wavelength_shifts_nm=redundant["wavelength_shifts_nm"],
        diagnosed_channels=redundant_diagnosis["fault_channels"],
        estimated_angle_deg=redundant_diagnosis["estimated_angle_deg"],
    )
    task_reports = {
        "弯曲标定与温补": experiments.calibration_report(current_experiment, baseline),
        "冗余故障诊断": redundant_report,
        "多材质触觉识别": experiments.tactile_report(tactile_record),
        "足底平衡": experiments.foot_report(foot_record),
        "连续体形状重建": experiments.shape_report(shape_record),
        "结构健康监测": experiments.health_report(health_record),
        "分布式事件定位": experiments.distributed_report(distributed_record),
        "偏振与干涉传感": experiments.optical_report(optical_record),
    }
    chain_summary = "\n".join((
        f"采样率：{sample_rate} Hz；温度变化：{temperature:.1f}°C",
        f"链路真实/反演弯曲角：{chain_angle:.1f}° / {chain['estimated_angle_deg']:.1f}°",
        f"控制输出：{chain['control_command']}",
        f"分布式机制：{distributed_frame['sensor_type']}；数据质量：{float(distributed_frame['quality']) * 100:.0f}%",
        f"多模态任务状态：{fusion['status']}；融合置信度：{float(fusion['confidence']) * 100:.0f}%",
    ))
    report = experiments.sensing_chain_report(experiment, task_reports[experiment], chain_summary)
    st.download_button("下载当前实验报告", report.encode("utf-8-sig"), "fiber_robotics_sensing_chain_report.txt", "text/plain")

with assembly_tab:
    st.subheader("可更换式足底组件：二维装配状态预测")
    st.caption("固定光纤感知芯与可更换耐磨外底/分区传力模块分离；以下是空载、恒温条件下的解析仿真预测，不是实物验收、密封或耐久结论。")
    module_learning_frame(
        "理解空载复装筛查如何利用工作光栅、参考光栅和左右差异区分压入不足与单侧错位。",
        "依次选择三种复装工况，再改变噪声、参考温差、密封错位和预应变保持率。",
        "比较基线残差、左右差异、混淆矩阵、最低密封压缩率和预应变敏感性。",
        "所有阈值与材料参数均为教学假设；密封、疲劳、耐磨和 IP 等级必须通过实物试验。",
    )
    assembly = models.simulate_replaceable_sole_assembly(sole_assembly_case, temperature)
    parameters = assembly["case_parameters"]
    assembly_left, assembly_right = st.columns([1, 2])
    with assembly_left:
        st.metric("当前复装工况", sole_assembly_case)
        st.metric("空载装配预测", str(assembly["assembly_prediction"]))
        st.metric("平均基线残差", f"{float(assembly['mean_baseline_residual_ue']):+.1f} με")
        st.metric("左右工作光栅差异", f"{float(assembly['left_right_difference_ue']):.1f} με")
    with assembly_right:
        st.plotly_chart(visuals.replaceable_sole_transfer_figure(assembly), width="stretch")
        st.caption("热图是相对传力场，亮区为传力集中处；错位时亮区中心横向偏移，由两枚工作光栅的左右差异反映。")
    st.markdown("**计算流程：** 设定复装工况 → 生成二维相对传力场 → 比较两枚工作光栅与一枚参考光栅 → 温度补偿 → 比较空载基线残差和左右差异 → 输出仿真筛查结果。")
    st.markdown("**边界：** 定位柱、锁止件、轴向限位、周向密封圈和柔性隔离膜在此作为结构方案边界；不计算接触应力、泄漏、材料疲劳、耐磨或 IP 等级。")
    st.divider()
    st.subheader("装配公差与阈值敏感性")
    tolerance_left, tolerance_right = st.columns([1, 2])
    with tolerance_left:
        tolerance_samples = st.slider("每工况仿真样本数", 20, 300, 100, 20)
        tolerance_noise = st.slider("装配筛查波长噪声 (nm)", 0.0, 0.010, 0.002, 0.0005, format="%.4f")
    tolerance_scan = models.simulate_replaceable_sole_tolerance_scan(int(tolerance_samples), temperature, tolerance_noise, int(seed))
    with tolerance_right:
        st.plotly_chart(visuals.assembly_tolerance_confusion_figure(tolerance_scan), width="stretch")
        st.caption("混淆矩阵对角线越高越好，非对角格表示被误判成其他工况；样本越多、噪声越小，结果越稳定。")
    thermal_left, thermal_right = st.columns(2)
    with thermal_left:
        reference_temperature_offset = st.slider("参考光栅相对温差 (°C)", -5.0, 5.0, 0.0, 0.1)
        thermal_mismatch = models.simulate_reference_temperature_mismatch(temperature, temperature + reference_temperature_offset)
        st.metric("温度失配引入的基线偏置", f"{float(thermal_mismatch['baseline_bias_ue']):+.1f} με")
        st.caption(str(thermal_mismatch["validation_boundary"]))
    with thermal_right:
        operational_load = st.slider("比较用使用载荷 (N)", 0.0, 400.0, 180.0, 5.0)
        operational = models.simulate_assembly_operational_load_interference(operational_load, temperature)
        st.metric("使用载荷机械信号均值", f"{float(np.mean(operational['operational_signal_ue'])):.1f} με")
        st.caption(f"装配自检条件：{operational['assembly_check_condition']}；{operational['validation_boundary']}。")
    st.divider()
    st.subheader("密封与预应变保持：试验规划敏感性")
    seal_left, retention_right = st.columns(2)
    with seal_left:
        seal_nominal_compression = st.slider("名义密封压缩率", 0.05, 0.40, 0.20, 0.01)
        seal_lateral_offset = st.slider("密封分析横向错位 (mm)", 0.0, 2.0, 0.80, 0.05)
        seal = models.simulate_seal_compression_screen(seal_nominal_compression, seal_lateral_offset)
        st.plotly_chart(visuals.seal_compression_screen_figure(seal), width="stretch")
        st.caption(f"最低相对压缩率：{float(seal['minimum_compression_ratio']) * 100:.1f}%；{seal['validation_boundary']}。")
    with retention_right:
        retention_cycles = st.slider("规划最大循环次数", 1000, 20000, 5000, 1000)
        assumed_retention = st.slider("假设每千次预应变保持率", 0.90, 1.00, 0.985, 0.001)
        retention = models.simulate_preload_retention_sensitivity(retention_cycles, assumed_retention)
        st.plotly_chart(visuals.preload_retention_sensitivity_figure(retention), width="stretch")
        st.caption(f"{retention['validation_boundary']}；保持率为试验规划假设，不是寿命预测。")
    verification_summary = (
        "可更换式足底组件验证参数摘要（仿真输入，不是实物结论）\n"
        f"装配工况：{sole_assembly_case}\n温度变化：{temperature:.1f} °C\n随机种子：{int(seed)}\n"
        f"名义预应变：{parameters['nominal_preload_ue']:.0f} με\n压入不足：{parameters['insertion_deficit_mm']:.1f} mm\n"
        f"横向错位：{parameters['lateral_offset_mm']:.1f} mm\n每工况样本数：{int(tolerance_samples)}\n波长噪声：{tolerance_noise:.4f} nm\n"
        "边界：仍需材料参数标定、有限元复核、空载复装、温度梯度、步态载荷、密封及循环实物试验。\n"
    )
    st.download_button("下载装配验证参数摘要", verification_summary.encode("utf-8-sig"), "replaceable_sole_verification_parameters.txt", "text/plain")

with eskin_lab:
    st.subheader("电子皮肤系统总览与机制对照")
    st.caption(
        "从单个触觉单元推进到光学阵列、压力场重建和动态事件判别。"
        "本页所有数值来自透明教学模型，不能替代器件标定、实物测试或安全认证。"
    )
    mechanism_rows = [
        {"机制": "压阻", "主要观测": "电阻变化", "适合观察": "静态/准静态压力", "常见干扰": "温漂、迟滞、材料蠕变"},
        {"机制": "电容", "主要观测": "电容变化", "适合观察": "微小位移与多轴力", "常见干扰": "寄生电容、曲率、湿度"},
        {"机制": "压电/摩擦电", "主要观测": "电荷或电压", "适合观察": "动态接触和振动", "常见干扰": "静态保持能力有限、负载阻抗"},
        {"机制": "离子", "主要观测": "界面电容/离子迁移", "适合观察": "柔软界面与大形变", "常见干扰": "频率、含水量、封装"},
        {"机制": "FBG 光学", "主要观测": "Bragg 波长变化", "适合观察": "抗电磁干扰、多点复用", "常见干扰": "温度、封装传力、解调带宽"},
    ]
    st.dataframe(mechanism_rows, width="stretch", hide_index=True)
    with st.expander("如何选择机制与评价指标", expanded=False):
        st.markdown(
            "- **先定义任务**：静态压力、动态滑移、温度或曲率需要不同观测量。\n"
            "- **再看可辨识性**：通道数多不等于三轴力一定可分离，应检查灵敏度矩阵的秩和条件数。\n"
            "- **最后看系统链路**：封装、参考结构、采样率、解调器和重建算法都会改变最终性能。\n"
            "- **统一报告误差**：同时给出载荷、位置、压力场和重复性指标，不用单一准确率概括全部能力。"
        )

    def load_eskin_preset() -> None:
        preset = eskin_experiments.ESKIN_PRESETS[st.session_state.eskin_preset]
        for state_key, state_value in preset["state"].items():
            st.session_state[state_key] = state_value
        request_navigation("eskin_navigation", {
            "三轴单元": "三轴触觉单元",
            "光学皮肤": "FBG 光学皮肤",
            "压力重建": "稀疏压力重建",
            "动态判别": "动态滑移与多模态",
        }[preset["mode"]])

    preset_column, preset_action = st.columns([3, 1])
    with preset_column:
        eskin_preset_name = st.selectbox(
            "电子皮肤场景预设",
            list(eskin_experiments.ESKIN_PRESETS),
            key="eskin_preset",
        )
        st.caption(eskin_experiments.ESKIN_PRESETS[eskin_preset_name]["description"])
    with preset_action:
        st.write("")
        st.button(
            "载入场景",
            key="load_eskin_preset",
            on_click=load_eskin_preset,
            width="stretch",
        )

    taxel_tab, optical_skin_tab, pressure_tab, dynamic_tab = tracked_tabs(
        LAB_SECTIONS["eskin"], "eskin_navigation"
    )

    with taxel_tab:
        st.subheader("三轴触觉单元：主动/参考信号与力反演")
        module_learning_frame(
            "理解五个电容通道如何分离 Fx、Fy、Fz，并观察参考结构对温度、曲率和应变共模的校正作用。",
            "先保存默认工况为基线 A，再提高温度或降低参考匹配度，比较原始与校正后的力误差。",
            "关注灵敏度矩阵的秩、条件数、三轴分量误差，以及参考校正是否真的降低共模误差。",
            "灵敏度矩阵为透明教学参数，不代表某款电子皮肤器件的实测标定矩阵。",
        )
        taxel_controls, taxel_display = st.columns([1.0, 1.7], gap="large")
        with taxel_controls:
            eskin_fx = st.slider("切向力 Fx (N)", -5.0, 5.0, 2.0, 0.1, key="eskin_fx_n")
            eskin_fy = st.slider("切向力 Fy (N)", -5.0, 5.0, -1.5, 0.1, key="eskin_fy_n")
            eskin_fz = st.slider("法向力 Fz (N)", 0.0, 20.0, 8.0, 0.2, key="eskin_fz_n")
            eskin_curvature = st.slider("表面曲率 (1/m)", 0.0, 8.0, 3.0, 0.1, key="eskin_curvature")
            eskin_strain_milli = st.slider("基底应变 (‰)", -2.0, 3.0, 1.0, 0.1, key="eskin_strain_milli")
            eskin_taxel_temperature = st.slider("触觉单元温度 (°C)", 0.0, 60.0, 38.0, 0.5, key="eskin_taxel_temperature")
            eskin_noise_pf = st.slider("电容噪声 σ (pF)", 0.0, 0.10, 0.01, 0.005, key="eskin_noise_pf")
            eskin_reference_match = st.slider("参考结构匹配度", 0.80, 1.00, 0.98, 0.01, key="eskin_reference_match")
        taxel_result = eskin.simulate_triaxial_taxel(
            fx_n=eskin_fx, fy_n=eskin_fy, fz_n=eskin_fz,
            curvature_per_m=eskin_curvature,
            strain_fraction=eskin_strain_milli / 1000.0,
            temperature_c=eskin_taxel_temperature,
            noise_pf=eskin_noise_pf,
            reference_match=eskin_reference_match,
            seed=int(seed),
        )
        with taxel_display:
            taxel_metrics = st.columns(4)
            taxel_metrics[0].metric("校正后力 MAE", f"{taxel_result['corrected_mae_n']:.3f} N")
            taxel_metrics[1].metric("校正前力 MAE", f"{taxel_result['raw_mae_n']:.3f} N")
            taxel_metrics[2].metric("矩阵秩", f"{taxel_result['matrix_rank']} / 3")
            taxel_metrics[3].metric("矩阵条件数", f"{taxel_result['condition_number']:.2f}")
            st.plotly_chart(eskin_visuals.taxel_channel_figure(taxel_result), width="stretch")
        force_table = [
            {
                "分量": label,
                "真实力 (N)": float(taxel_result["forces_true_n"][index]),
                "未校正反演 (N)": float(taxel_result["raw_estimate_n"][index]),
                "参考校正反演 (N)": float(taxel_result["corrected_estimate_n"][index]),
                "校正后误差 (N)": float(taxel_result["corrected_error_n"][index]),
            }
            for index, label in enumerate(("Fx", "Fy", "Fz"))
        ]
        st.dataframe(force_table, width="stretch", hide_index=True)
        current_taxel_metrics = {
            "校正后力 MAE": round(taxel_result["corrected_mae_n"], 4),
            "校正前力 MAE": round(taxel_result["raw_mae_n"], 4),
            "矩阵条件数": round(taxel_result["condition_number"], 3),
        }
        if st.button("保存三轴单元基线 A", key="save_eskin_taxel_baseline"):
            st.session_state.eskin_taxel_baseline = current_taxel_metrics
        if st.session_state.get("eskin_taxel_baseline"):
            st.dataframe([
                {"指标": name, "基线 A": st.session_state.eskin_taxel_baseline[name], "当前 B": value}
                for name, value in current_taxel_metrics.items()
            ], width="stretch", hide_index=True)
        with st.expander("查看灵敏度矩阵与可辨识性", expanded=False):
            st.dataframe(
                [{"通道": f"C{i + 1}", "Fx": row[0], "Fy": row[1], "Fz": row[2]}
                 for i, row in enumerate(taxel_result["sensitivity_matrix"])],
                width="stretch", hide_index=True,
            )
            st.caption("满列秩表示三轴分量在该线性模型中可反演；条件数越高，噪声越容易被放大。")
        taxel_parameters = {
            "真实力": f"({eskin_fx:.1f}, {eskin_fy:.1f}, {eskin_fz:.1f}) N",
            "温度": f"{eskin_taxel_temperature:.1f} °C",
            "参考匹配度": f"{eskin_reference_match * 100:.0f}%",
            "随机种子": int(seed),
        }
        taxel_report_metrics = {
            "校正后力 MAE": f"{taxel_result['corrected_mae_n']:.4f} N",
            "校正前力 MAE": f"{taxel_result['raw_mae_n']:.4f} N",
            "矩阵秩": taxel_result["matrix_rank"],
            "条件数": f"{taxel_result['condition_number']:.3f}",
        }
        taxel_downloads = st.columns(2)
        taxel_downloads[0].download_button(
            "下载三轴单元 CSV",
            eskin_experiments.eskin_csv_bytes("三轴单元", taxel_result),
            "eskin_triaxial_taxel.csv", "text/csv", width="stretch",
        )
        taxel_downloads[1].download_button(
            "下载三轴单元报告",
            eskin_experiments.eskin_report_bytes("三轴触觉单元", taxel_parameters, taxel_report_metrics, taxel_result["boundary"]),
            "eskin_triaxial_taxel.md", "text/markdown", width="stretch",
        )
        st.info(taxel_result["boundary"])

    with optical_skin_tab:
        st.subheader("FBG 光学皮肤：感受野、温补与压力质心")
        module_learning_frame(
            "理解有限数量的 FBG 如何通过重叠感受野编码接触位置与合力。",
            "比较 4、8、16 个传感点，并在单点和双点之间切换；再提高温度或噪声观察补偿结果。",
            "观察波长空间分布、温度共模、估计载荷和压力质心定位误差。",
            "双点接触只评价载荷质心，不声称仅凭当前响应唯一分离两个接触点。",
        )
        fbg_controls, fbg_display = st.columns([1.0, 1.7], gap="large")
        with fbg_controls:
            fbg_sensor_count = st.select_slider("FBG 数量", options=[4, 8, 16], value=8, key="eskin_fbg_sensor_count")
            contact_mode = st.selectbox("接触模式", ["单点", "双点"], key="eskin_contact_mode")
            skin_width = st.slider("皮肤宽度 (mm)", 40.0, 120.0, 80.0, 5.0, key="eskin_skin_width")
            skin_height = st.slider("皮肤高度 (mm)", 30.0, 100.0, 60.0, 5.0, key="eskin_skin_height")
            receptive_width = st.slider("感受野宽度 (mm)", 6.0, 35.0, 18.0, 1.0, key="eskin_receptive_width")
            touch1_x = st.slider("接触 1：x (mm)", 0.0, skin_width, min(24.0, skin_width), 1.0, key="eskin_touch1_x")
            touch1_y = st.slider("接触 1：y (mm)", 0.0, skin_height, min(30.0, skin_height), 1.0, key="eskin_touch1_y")
            touch1_force = st.slider("接触 1：载荷 (N)", 0.0, 15.0, 6.0, 0.2, key="eskin_touch1_force")
            touch2_x = st.slider("接触 2：x (mm)", 0.0, skin_width, min(58.0, skin_width), 1.0, key="eskin_touch2_x")
            touch2_y = st.slider("接触 2：y (mm)", 0.0, skin_height, min(22.0, skin_height), 1.0, key="eskin_touch2_y")
            touch2_force = st.slider("接触 2：载荷 (N)", 0.0, 15.0, 4.0, 0.2, key="eskin_touch2_force", disabled=contact_mode == "单点")
            fbg_temperature = st.slider("光学皮肤温度 (°C)", 0.0, 60.0, 36.0, 0.5, key="eskin_fbg_temperature")
            fbg_noise_nm = st.slider("解调噪声 σ (nm)", 0.0, 0.010, 0.001, 0.0005, format="%.4f", key="eskin_noise_nm")
        touches = [(touch1_x, touch1_y, touch1_force)]
        if contact_mode == "双点":
            touches.append((touch2_x, touch2_y, touch2_force))
        fbg_skin_result = eskin.simulate_fbg_skin(
            sensor_count=fbg_sensor_count, touch_points=touches,
            skin_width_mm=skin_width, skin_height_mm=skin_height,
            receptive_width_mm=receptive_width, temperature_c=fbg_temperature,
            noise_nm=fbg_noise_nm, seed=int(seed),
        )
        with fbg_display:
            fbg_metrics = st.columns(4)
            fbg_metrics[0].metric("载荷误差", f"{fbg_skin_result['load_error_n']:.3f} N")
            fbg_metrics[1].metric("质心定位误差", f"{fbg_skin_result['location_error_mm']:.2f} mm")
            fbg_metrics[2].metric("传感点数", str(fbg_sensor_count))
            fbg_metrics[3].metric("温度共模", f"{fbg_skin_result['temperature_shift_nm']:.3f} nm")
            st.plotly_chart(eskin_visuals.fbg_skin_figure(fbg_skin_result, skin_width, skin_height), width="stretch")
        st.dataframe(eskin_experiments.eskin_result_records("光学皮肤", fbg_skin_result), width="stretch", hide_index=True)
        current_fbg_metrics = {
            "载荷误差 (N)": round(fbg_skin_result["load_error_n"], 4),
            "质心定位误差 (mm)": round(fbg_skin_result["location_error_mm"], 3),
            "传感点数": fbg_sensor_count,
        }
        if st.button("保存光学皮肤基线 A", key="save_eskin_fbg_baseline"):
            st.session_state.eskin_fbg_baseline = current_fbg_metrics
        if st.session_state.get("eskin_fbg_baseline"):
            st.dataframe([
                {"指标": name, "基线 A": st.session_state.eskin_fbg_baseline[name], "当前 B": value}
                for name, value in current_fbg_metrics.items()
            ], width="stretch", hide_index=True)
        fbg_parameters = {
            "传感点": fbg_sensor_count, "接触模式": contact_mode,
            "感受野宽度": f"{receptive_width:.1f} mm", "温度": f"{fbg_temperature:.1f} °C",
            "随机种子": int(seed),
        }
        fbg_report_metrics = {
            "真实/估计合力": f"{fbg_skin_result['true_total_force_n']:.2f} / {fbg_skin_result['estimated_total_force_n']:.2f} N",
            "载荷误差": f"{fbg_skin_result['load_error_n']:.4f} N",
            "质心定位误差": f"{fbg_skin_result['location_error_mm']:.3f} mm",
            "解释": fbg_skin_result["interpretation"],
        }
        fbg_downloads = st.columns(2)
        fbg_downloads[0].download_button(
            "下载光学皮肤 CSV", eskin_experiments.eskin_csv_bytes("光学皮肤", fbg_skin_result),
            "eskin_fbg_skin.csv", "text/csv", width="stretch",
        )
        fbg_downloads[1].download_button(
            "下载光学皮肤报告",
            eskin_experiments.eskin_report_bytes("FBG 光学皮肤", fbg_parameters, fbg_report_metrics, fbg_skin_result["boundary"]),
            "eskin_fbg_skin.md", "text/markdown", width="stretch",
        )
        st.info(fbg_skin_result["interpretation"] + " " + fbg_skin_result["boundary"])

    with pressure_tab:
        st.subheader("稀疏压力重建：采样、插值与误差")
        module_learning_frame(
            "理解稀疏通道如何映射为致密压力场，并用多项指标评价信息损失。",
            "从 4×4 切换到 8×8 采样，比较单点、双点、边缘接触和滑动前兆。",
            "同时观察 RMSE、峰值、质心、总载荷和通道节省率，不只看热图是否相似。",
            "当前为透明高斯核插值，不代表神经网络或超维计算算法的实测性能。",
        )
        pressure_controls, pressure_display = st.columns([1.0, 1.9], gap="large")
        with pressure_controls:
            pressure_scenario = st.selectbox("压力场场景", ["单点接触", "双点接触", "边缘接触", "滑动前兆"], key="eskin_pressure_scenario")
            sparse_size = st.select_slider("稀疏采样网格", options=[4, 8], value=4, format_func=lambda value: f"{value}×{value}", key="eskin_sparse_size")
            output_size = st.select_slider("输出压力网格", options=[16, 32], value=16, format_func=lambda value: f"{value}×{value}", key="eskin_output_size")
            peak_pressure = st.slider("峰值压力 (kPa)", 20.0, 160.0, 80.0, 5.0, key="eskin_peak_pressure")
            kernel_bandwidth = st.slider("核插值带宽", 0.06, 0.30, 0.16, 0.01, key="eskin_kernel_bandwidth")
            pressure_noise = st.slider("稀疏通道噪声 σ (kPa)", 0.0, 5.0, 0.5, 0.1, key="eskin_pressure_noise")
        pressure_result = eskin.simulate_pressure_reconstruction(
            pressure_scenario, sparse_size, output_size,
            peak_pressure_kpa=peak_pressure, bandwidth=kernel_bandwidth,
            noise_kpa=pressure_noise, seed=int(seed),
        )
        with pressure_display:
            pressure_metrics = st.columns(5)
            pressure_metrics[0].metric("压力场 RMSE", f"{pressure_result['rmse_kpa']:.2f} kPa")
            pressure_metrics[1].metric("峰值误差", f"{pressure_result['peak_error_pct']:.1f}%")
            pressure_metrics[2].metric("质心误差", f"{pressure_result['centroid_error_pct']:.1f}%")
            pressure_metrics[3].metric("总载荷误差", f"{pressure_result['total_force_error_pct']:.1f}%")
            pressure_metrics[4].metric("通道节省率", f"{pressure_result['channel_saving_pct']:.1f}%")
            st.plotly_chart(eskin_visuals.pressure_reconstruction_figure(pressure_result), width="stretch")
        current_pressure_metrics = {
            "RMSE (kPa)": round(pressure_result["rmse_kpa"], 3),
            "质心误差 (%)": round(pressure_result["centroid_error_pct"], 2),
            "总载荷误差 (%)": round(pressure_result["total_force_error_pct"], 2),
            "通道节省率 (%)": round(pressure_result["channel_saving_pct"], 2),
        }
        if st.button("保存压力重建基线 A", key="save_eskin_pressure_baseline"):
            st.session_state.eskin_pressure_baseline = current_pressure_metrics
        if st.session_state.get("eskin_pressure_baseline"):
            st.dataframe([
                {"指标": name, "基线 A": st.session_state.eskin_pressure_baseline[name], "当前 B": value}
                for name, value in current_pressure_metrics.items()
            ], width="stretch", hide_index=True)
        pressure_parameters = {
            "场景": pressure_scenario, "稀疏/输出网格": f"{sparse_size}×{sparse_size} / {output_size}×{output_size}",
            "峰值压力": f"{peak_pressure:.1f} kPa", "核带宽": f"{kernel_bandwidth:.2f}",
            "随机种子": int(seed),
        }
        pressure_report_metrics = {
            "压力场 RMSE": f"{pressure_result['rmse_kpa']:.3f} kPa",
            "峰值误差": f"{pressure_result['peak_error_pct']:.2f}%",
            "质心误差": f"{pressure_result['centroid_error_pct']:.2f}%",
            "总载荷误差": f"{pressure_result['total_force_error_pct']:.2f}%",
            "通道节省率": f"{pressure_result['channel_saving_pct']:.2f}%",
        }
        pressure_downloads = st.columns(2)
        pressure_downloads[0].download_button(
            "下载压力重建 CSV", eskin_experiments.eskin_csv_bytes("压力重建", pressure_result),
            "eskin_pressure_reconstruction.csv", "text/csv", width="stretch",
        )
        pressure_downloads[1].download_button(
            "下载压力重建报告",
            eskin_experiments.eskin_report_bytes("稀疏压力重建", pressure_parameters, pressure_report_metrics, pressure_result["boundary"]),
            "eskin_pressure_reconstruction.md", "text/markdown", width="stretch",
        )
        st.warning("透明教学模型：当前重建使用高斯核插值，可检查每一步；它不等同于公开论文中的学习模型或超维计算实现。")

    with dynamic_tab:
        st.subheader("动态滑移与多模态决策")
        module_learning_frame(
            "把法向力、剪切力、压力质心运动和温度放到同一时间轴上，理解多条件告警。",
            "比较稳定按压、载荷爬升、横向滑动、即将滑移、热物体和温漂，再提高噪声进行重复采样。",
            "观察剪切比是否越阈、质心速度是否同时升高，以及重复试验中告警率是否稳定。",
            "阈值是教学设置，实际系统必须按材料、封装、接触速度与采样链重新标定。",
        )
        dynamic_controls, dynamic_display = st.columns([1.0, 1.8], gap="large")
        with dynamic_controls:
            dynamic_event = st.selectbox("动态事件", ["稳定按压", "载荷爬升", "横向滑动", "即将滑移", "热物体", "温漂"], key="eskin_dynamic_event")
            dynamic_sample_rate = st.select_slider("动态采样率 (Hz)", options=[20, 50, 100, 200], value=100, key="eskin_dynamic_sample_rate")
            dynamic_duration = st.slider("记录时长 (s)", 2.0, 8.0, 4.0, 0.5, key="eskin_dynamic_duration")
            dynamic_force = st.slider("目标法向力 (N)", 1.0, 25.0, 12.0, 0.5, key="eskin_dynamic_force")
            slip_threshold = st.slider("滑移剪切比阈值", 0.15, 0.70, 0.35, 0.01, key="eskin_slip_threshold")
            dynamic_temperature = st.slider("初始温度 (°C)", 0.0, 50.0, 25.0, 0.5, key="eskin_dynamic_temperature")
            dynamic_noise = st.slider("相对噪声", 0.0, 0.08, 0.01, 0.005, key="eskin_dynamic_noise")
            repeat_count = st.slider("重复试验次数", 20, 200, 50, 10, key="eskin_repeat_count")
        dynamic_result = eskin.simulate_dynamic_skin_event(
            dynamic_event, sample_rate_hz=dynamic_sample_rate,
            duration_s=dynamic_duration, normal_force_n=dynamic_force,
            slip_threshold=slip_threshold, temperature_c=dynamic_temperature,
            noise_ratio=dynamic_noise, seed=int(seed),
        )
        repeat_result = eskin.repeat_dynamic_event(
            dynamic_event, repeats=repeat_count,
            sample_rate_hz=dynamic_sample_rate, duration_s=dynamic_duration,
            normal_force_n=dynamic_force, slip_threshold=slip_threshold,
            temperature_c=dynamic_temperature, noise_ratio=dynamic_noise,
            seed=int(seed),
        )
        with dynamic_display:
            dynamic_metrics = st.columns(5)
            dynamic_metrics[0].metric("滑移判定", dynamic_result["status"])
            dynamic_metrics[1].metric("峰值剪切比", f"{dynamic_result['peak_shear_ratio']:.3f}")
            dynamic_metrics[2].metric("阈值裕度", f"{dynamic_result['threshold_margin']:+.3f}")
            dynamic_metrics[3].metric("质心峰值速度", f"{dynamic_result['peak_centroid_speed_mm_s']:.1f} mm/s")
            dynamic_metrics[4].metric("重复告警率", f"{repeat_result['alert_rate_pct']:.1f}%")
            st.plotly_chart(eskin_visuals.dynamic_event_figure(dynamic_result), width="stretch")
        if dynamic_result["alert"]:
            st.error("当前教学规则判定为滑移风险：剪切比与压力质心速度同时超过阈值。")
        else:
            st.success("当前教学规则未触发滑移风险；仍需结合任务允许的漏报与误报代价选择阈值。")
        repeat_rows = [
            {"统计项": "重复次数", "结果": repeat_result["repeat_count"]},
            {"统计项": "告警率", "结果": f"{repeat_result['alert_rate_pct']:.1f}%"},
            {"统计项": "峰值剪切比均值", "结果": f"{repeat_result['mean_peak_ratio']:.3f}"},
            {"统计项": "峰值剪切比标准差", "结果": f"{repeat_result['peak_ratio_std']:.4f}"},
        ]
        st.dataframe(repeat_rows, width="stretch", hide_index=True)
        current_dynamic_metrics = {
            "滑移判定": dynamic_result["status"],
            "峰值剪切比": round(dynamic_result["peak_shear_ratio"], 4),
            "质心峰值速度 (mm/s)": round(dynamic_result["peak_centroid_speed_mm_s"], 3),
            "重复告警率 (%)": round(repeat_result["alert_rate_pct"], 2),
        }
        if st.button("保存动态事件基线 A", key="save_eskin_dynamic_baseline"):
            st.session_state.eskin_dynamic_baseline = current_dynamic_metrics
        if st.session_state.get("eskin_dynamic_baseline"):
            st.dataframe([
                {"指标": name, "基线 A": st.session_state.eskin_dynamic_baseline[name], "当前 B": value}
                for name, value in current_dynamic_metrics.items()
            ], width="stretch", hide_index=True)
        dynamic_parameters = {
            "事件": dynamic_event, "采样率": f"{dynamic_sample_rate} Hz",
            "记录时长": f"{dynamic_duration:.1f} s", "目标法向力": f"{dynamic_force:.1f} N",
            "滑移剪切比阈值": f"{slip_threshold:.2f}", "重复次数": repeat_count,
            "随机种子": int(seed),
        }
        dynamic_report_metrics = {
            "判定": dynamic_result["status"],
            "峰值剪切比": f"{dynamic_result['peak_shear_ratio']:.4f}",
            "阈值裕度": f"{dynamic_result['threshold_margin']:+.4f}",
            "质心峰值速度": f"{dynamic_result['peak_centroid_speed_mm_s']:.3f} mm/s",
            "重复告警率": f"{repeat_result['alert_rate_pct']:.2f}%",
            "判据": dynamic_result["rule"],
        }
        dynamic_downloads = st.columns(2)
        dynamic_downloads[0].download_button(
            "下载动态事件 CSV", eskin_experiments.eskin_csv_bytes("动态判别", dynamic_result),
            "eskin_dynamic_event.csv", "text/csv", width="stretch",
        )
        dynamic_downloads[1].download_button(
            "下载动态事件报告",
            eskin_experiments.eskin_report_bytes("动态滑移与多模态决策", dynamic_parameters, dynamic_report_metrics, dynamic_result["boundary"]),
            "eskin_dynamic_event.md", "text/markdown", width="stretch",
        )
        st.info(dynamic_result["rule"] + " " + dynamic_result["boundary"])

    st.divider()
    st.markdown("#### 公开研究入口与使用边界")
    st.markdown(
        "- [Ruffini-inspired FBG optical skin（Nature Machine Intelligence）](https://www.nature.com/articles/s42256-022-00487-3)：用于理解光纤感受野和机器人触觉研究方向。\n"
        "- [Polymer FBG skin（Journal of Lightwave Technology）](https://opg.optica.org/jlt/abstract.cfm?uri=jlt-42-8-3022)：用于了解聚合物 FBG 皮肤与多点感知。\n"
        "- [Electronic skin with hyperdimensional computing（PubMed）](https://pubmed.ncbi.nlm.nih.gov/42284403/)：用于比较感知与计算协同的研究方向。\n"
        "- [Multimodal electronic skin（npj Flexible Electronics）](https://www.nature.com/articles/s41528-023-00252-5)：用于了解多参数柔性传感。"
    )
    st.caption(
        "以上链接只作为延伸阅读。本网站未复制论文模型、数据或性能结论；"
        "当前实现用于展示可检查的传感链和评价方法。"
    )
