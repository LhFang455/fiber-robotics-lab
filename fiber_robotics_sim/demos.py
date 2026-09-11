"""Parameter sweeps for explanatory models; playback time is not physical time."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import eskin, experiments, model_assets


ROOT = Path(__file__).parent


def foot_demo(parameters: dict) -> dict:
    parameters = dict(parameters)
    selected = parameters.pop("inspection_channel", None)
    if selected is not None:
        selected = max(0, min(int(selected), 5))
    phases = sorted(set(np.linspace(0, 100, 121).tolist() + [float(parameters["phase_percent"])]))
    frames = []
    for phase in phases:
        result = experiments.run_foot_experiment({**parameters, "phase_percent": phase})["results"]
        has_contact = result["true_total_load_n"] > 1e-9 and result["estimated_total_load_n"] > 1e-9
        reliable = bool(result["reliable_cop"])
        caption = (
            "脚跟承载：先看后排区域的信号，压力中心靠近脚跟。" if phase < 30
            else "全掌过渡：载荷逐步向前排转移，观察压力中心沿足底移动。" if phase < 70
            else "前掌承载：前排信号增强，压力中心靠近前掌。"
        )
        if not reliable:
            caption += " 当前低载荷或存在通道失效，压力中心不作为可靠定位。"
        frames.append({
            "progress": phase / 100, "phase": phase,
            "loads": result["estimated_zone_loads_n"], "true_loads": result["true_zone_loads_n"],
            "valid_loads": [parameters.get("failed_zone") != i + 1 and bool(np.isfinite(v)) for i, v in enumerate(result["estimated_zone_loads_n"])],
            "cop": result["estimated_cop_xy"] if has_contact else None,
            "true_cop": result["true_cop_xy"] if result["true_total_load_n"] > 1e-9 else None,
            "reliable": reliable, "model_input_signals": result["wavelength_shifts_nm"],
            "signals": [None if parameters.get("failed_zone") == i + 1 else v for i, v in enumerate(result["wavelength_shifts_nm"])],
            "caption": caption,
            "metrics": [
                ["相位", f"{phase:.0f}%"],
                ["真实 / 反演载荷", f"{result['true_total_load_n']:.1f} / {result['estimated_total_load_n']:.1f} N"],
                ["区域误差 MAE", f"{result['zone_mae_n']:.2f} N"],
                ["CoP 可靠性", "可观察" if reliable else "仅供参考"],
            ],
        })
    if selected is not None:
        for frame in frames:
            observed = f"{frame['loads'][selected]:.2f} N" if frame['valid_loads'][selected] else "无有效观测（失效或缺测）"
            frame['caption'] += (f" 所选 SOLE-{selected+1:02d} / FBG {selected+1}：本动画帧反演载荷 {observed}。"
                                 "白色边框是所选区域，不是 CoP；点击‘当前参数’与下方相位和读数对照。")
    return {
        "inspection_channel": selected,
        "kind": "foot", "title": "看见脚底的载荷迁移",
        "subtitle": f"{parameters['terrain']} · {parameters['support']} · 六区 FBG" + (f" · SOLE-{selected+1:02d}" if selected is not None else ''),
        "legend": "区域颜色：反演载荷 · 灰色：失效/缺测 · 白框：所选区域 · 白点：参考 CoP · 橙环：反演 CoP",
        "boundary": "扫描相位，不模拟完整行走动力学。外形与区域位置为示意；CoP 使用原实验归一化坐标。颜色为反演载荷，白点为真实 CoP、橙环为反演 CoP。",
        "frames": frames, "initial_index": phases.index(float(parameters["phase_percent"])),
        "labels": [f"区域 {i}" for i in range(1, 7)],
        "color_max": max(1.0, max(max(f["loads"]) for f in frames)),
    }


def skin_demo(parameters: dict) -> dict:
    parameters = dict(parameters)
    selected = parameters.pop("inspection_channel", None)
    if selected is not None:
        selected = max(0, min(int(selected), parameters["sensor_count"] - 1))
    width = float(parameters["skin_width_mm"])
    height = float(parameters["skin_height_mm"])
    frames = []
    for progress in np.linspace(0, 1, 121):
        gain = min(1.0, progress * 4, (1 - progress) * 4)
        travel = np.clip((progress - .25) / .5, 0, 1) * width * .22
        touches = [[float(min(width, x + travel)), float(y), float(force * gain)]
                   for x, y, force in parameters["touch_points"]]
        result = eskin.simulate_fbg_skin(**{**parameters, "touch_points": touches})
        centroid = result["estimated_centroid_mm"]
        valid = result["true_total_force_n"] > 1e-9 and np.all(np.isfinite(centroid))
        error = result["location_error_mm"]
        frames.append({
            "progress": float(progress), "touches": touches,
            "force": result["true_total_force_n"], "estimated_force": result["estimated_total_force_n"],
            "centroid": centroid.tolist() if valid else None,
            "signals": result["compensated_shift_nm"].tolist(),
            "caption": (
                "按下：载荷增加，附近 FBG 的温补响应增强。" if progress < .25
                else "平移：相邻感受野接力响应，估计质心随接触移动。" if progress < .75
                else "释放：机械信号减弱；无接触时不显示无定义的压力质心。"
            ) + (" 双点时显示的是合力质心，并非分离出两个接触点。" if len(touches) == 2 else ""),
            "metrics": [
                ["真实合力", f"{result['true_total_force_n']:.2f} N"],
                ["反演合力", f"{result['estimated_total_force_n']:.2f} N"],
                ["质心误差", f"{error:.2f} mm" if valid and np.isfinite(error) else "无有效接触"],
                ["接触数量", str(sum(force > 1e-9 for _, _, force in touches))],
            ],
        })
    if selected is not None:
        channel_id = f"SKIN-N{parameters['sensor_count']:02d}-{selected+1:02d}"
        for frame in frames:
            frame['caption'] += (f" 所选 {channel_id}：本动画帧温补响应 {frame['signals'][selected]:+.5f} nm。"
                                 "白环跟随该测点；点击‘当前参数’才与下方静态读数对照，播放不更新观察器读数。")
    return {
        "inspection_channel": selected,
        "kind": "skin", "title": "让皮肤感知一次触碰",
        "subtitle": f"{parameters['sensor_count']} 个 FBG · 按压 → 平移 → 释放" + (f" · {channel_id}" if selected is not None else ""),
        "boundary": "表面凹陷与色带仅为接触示意，不能读作真实位移或压力场。FBG 光点、温补波长和反演载荷来自本页模型；橙环表示响应加权质心。",
        "frames": frames, "initial_index": 30, "width": width, "height": height,
        "sensors": result["sensor_positions_mm"].tolist(), "labels": result["sensor_labels"],
        "legend": "光点颜色：温补响应 · 橙环：估计质心 · 白环：观察器所选 FBG（不是新增测点）",
        "receptive_width": float(parameters["receptive_width_mm"]),
        "color_max": max(.001, max(max(f["signals"]) for f in frames)),
    }


def shape_demo(parameters: dict) -> dict:
    parameters = dict(parameters)
    selected = parameters.pop("inspection_node", None)
    frames = []
    for progress in np.linspace(0, 1, 41):
        curvature = float(parameters["curvature_per_m"] * progress)
        result = experiments.run_shape_experiment({**parameters, "curvature_per_m": curvature})["results"]
        if selected is not None:
            selected = max(0, min(int(selected), len(result["point_error_mm"]) - 1))
        frames.append({
            "selected_error_mm": float(result["point_error_mm"][selected]) if selected is not None else None,
            "progress": float(progress), "curvature": curvature,
            "truth": result["true_centerline_xyz_mm"], "estimate": result["estimated_centerline_xyz_mm"],
            "rmse": result["centerline_rmse_mm"], "tip_error": result["tip_error_mm"],
            "signals": result["wavelength_shifts_nm"],
            "caption": (
                "从直线开始：逐渐增大曲率，三根纤芯出现不同波长响应。" if progress < .35
                else "弯曲形成：青色实体是真实形状，橙色轨迹是波长反演的中心线。" if progress < .8
                else "到达当前参数：比较整条中心线误差与末端误差；两者反映不同问题。"
            ),
            "metrics": [
                ["真实 / 反演曲率", f"{curvature:.2f} / {result['estimated_curvature_per_m']:.2f} 1/m"],
                ["中心线 RMSE", f"{result['centerline_rmse_mm']:.2f} mm"],
                ["末端误差", f"{result['tip_error_mm']:.2f} mm"],
                ["已知扭转先验", f"{parameters['twist_per_m']:.1f} 1/m"],
            ],
        })
    return {
        "inspection_node": selected,
        "kind": "shape", "title": "看见弯曲，也看见重建误差",
        "legend": "青色实体：真实中心线 · 橙线：重建中心线 · 白/金点：所选计算节点 · 粉线：未放大局部误差",
        "inspection_text": (f"所选 S-{selected:03d} 为计算节点，不是额外测点。播放仅改变演示帧；点击‘当前参数’与下方观察器对照。" if selected is not None else ''),
        "subtitle": f"三芯差分重建 · 长度 {parameters['length_mm']:.0f} mm" + (f" · S-{selected:03d}" if selected is not None else ''),
        "boundary": "沿用恒曲率教学模型，扭转率为已知先验。管径为示意，中心线与误差使用实际模型坐标；播放进度不是物理时间。",
        "frames": frames, "initial_index": len(frames) - 1,
        "labels": ["纤芯 1", "纤芯 2", "纤芯 3"], "length": float(parameters["length_mm"]),
    }


def demo_html(payload: dict) -> str:
    payload = dict(payload)
    try:
        payload['asset'] = model_assets.scene_asset(payload)
    except (model_assets.AssetError, OSError, KeyError, ValueError, TypeError):
        payload['asset'] = None
        payload['asset_warning'] = '结构资产校验未通过，当前使用原程序模型；数据计算不受影响。'
    config = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    template = (ROOT / "demo_scene.html").read_text(encoding="utf-8")
    runtime = (ROOT / "vendor" / "three.min.js").read_text(encoding="utf-8")
    return template.replace("__CONFIG__", config).replace("__THREE_RUNTIME__", runtime).replace("__READOUT_RUNTIME__", (ROOT / "demo_readout.js").read_text(encoding="utf-8")).replace("__ASSET_RUNTIME__", (ROOT / "model_assets.js").read_text(encoding="utf-8"))


def assembly_demo(parameters: dict) -> dict:
    """Explode/reseat a concept assembly; do not invent readings in transit."""
    from . import models

    result = models.simulate_replaceable_sole_assembly(parameters['assembly_case'], parameters['temperature_c'])
    case = result['case_parameters']
    frames = []
    for progress in np.linspace(0, 1, 121):
        seated = bool(progress >= .75)
        frames.append({
            'progress': float(progress), 'explosion': float(max(0, 1 - progress / .75)),
            'seated': seated, 'prediction': result['assembly_prediction'] if seated else None,
            'signals': [*result['working_wavelength_shifts_nm'].tolist(), result['reference_wavelength_shift_nm']]
            if seated else [None, None, None],
            'caption': ('分层观察：外底、传力模块和隔离膜与固定感知芯分离；结构仅为方案示意。' if progress < .35
                        else '复位：观察定位与锁止关系；分离、移动阶段不生成装配筛查结果。' if not seated
                        else f"空载筛查：{result['assembly_prediction']}。参考光栅补偿共模温漂，左右差异用于识别错位。"),
            'metrics': [
                ['复装工况', parameters['assembly_case']],
                ['平均基线残差', f"{result['mean_baseline_residual_ue']:+.1f} με" if seated else '就位后读取'],
                ['左右应变差', f"{result['left_right_difference_ue']:.1f} με" if seated else '就位后读取'],
                ['状态', '筛查预测通过' if seated and parameters['assembly_case'] == '正常装配'
                 else '筛查预测不通过' if seated else '结构展示中'],
            ],
        })
    return {
        'pending_signal_text': '就位后读取',
        'kind': 'assembly', 'title': '拆开看结构，装好看信号',
        'subtitle': '可更换外底 · 固定感知芯 · 空载复装筛查',
        'boundary': '分层距离和外形为示意；错位与压入不足在模型中放大 4 倍。读数仅对应就位后的当前工况，不是装配运动、接触应力、密封或耐久计算。',
        'legend': '自下而上：外底 / 传力模块 / 隔离膜 / 密封圈 / 定位锁止 / 感知芯 / 基板',
        'frames': frames, 'initial_index': 120, 'labels': ['左工作 FBG', '右工作 FBG', '参考 FBG'],
        'lateral_offset_mm': case['lateral_offset_mm'], 'insertion_deficit_mm': case['insertion_deficit_mm'],
    }


def health_demo(parameters: dict) -> dict:
    parameters = dict(parameters)
    selected = parameters.pop('inspection_channel', None)
    if selected is not None:
        selected = max(0, min(int(selected), parameters['sensor_count'] - 1))
    frames = []
    for progress in np.linspace(0, 1, 121):
        severity = float(parameters['anomaly_severity'] * progress)
        result = experiments.run_health_experiment({**parameters, 'anomaly_severity': severity})['results']
        valid = result['localization_valid']
        frames.append({
            'progress': float(progress), 'severity': severity,
            'truth_location': float(parameters['anomaly_position_mm']) if severity > 0 else None,
            'suspected': result['suspected_location_mm'] if valid else None,
            'uncertainty': result['location_uncertainty_mm'] if valid else None,
            'signals': result['wavelength_shifts_nm'],
            'caption': ('当前未设定异常，可在下方载入“局部异常”后播放。' if parameters['anomaly_severity'] == 0
                        else '逐渐增加设定的局部异常程度，观察附近点式 FBG 的响应。')
            + (' 橙色区间为本页有效定位范围；红色标记是仿真真值，两者不能混淆。' if valid
               else ' 尚未形成有效定位，不显示可疑位置或定位区间。'),
            'metrics': [
                ['异常程度', f'{severity:.2f}'], ['诊断状态', result['status']],
                ['可疑位置', f"{result['suspected_location_mm']:.0f} ± {result['location_uncertainty_mm']:.0f} mm" if valid else '未形成定位'],
                ['定位误差', f"{result['localization_error_mm']:.1f} mm" if valid else '不适用'],
            ],
        })
    if selected is not None:
        channel_id = f"ARM-N{parameters['sensor_count']:02d}-{selected+1:02d}"
        position = float(result['sensor_positions_mm'][selected])
        for frame in frames:
            frame['caption'] += (f" 所选 {channel_id}，x={position:.1f} mm；本动画帧波长 {frame['signals'][selected]:+.5f} nm。"
                                 "白环仅标出观察器测点，不是异常定位；点击‘当前参数’与下方静态值对照。")
    return {
        'inspection_channel': selected,
        'kind': 'health', 'title': '看见异常，也看见定位范围',
        'subtitle': f"{parameters['sensor_count']} 点阵列 · 当前载荷 {parameters['load_n']:.0f} N" + (f" · {channel_id}" if selected is not None else ''),
        'boundary': '梁体与标记为示意，横向位置采用本页毫米坐标。红色区域不是裂纹形貌；橙色区间是教学定位区间，不是统计置信区间或安全验收结论。',
        'legend': '光点颜色：FBG 波长响应 · 红环：真实异常设定 · 橙色带：有效定位区间 · 白环：所选测点',
        'frames': frames, 'initial_index': 120,
        'labels': [f'FBG {i + 1}' for i in range(parameters['sensor_count'])],
        'sensors': result['sensor_positions_mm'],
    }


def fbg_demo(parameters: dict) -> dict:
    from . import models

    frames = []
    for progress in np.linspace(0, 1, 121):
        angle = float(parameters['angle_deg'] * min(1, progress * 2))
        temperature = float(parameters['temperature_c'] * max(0, progress * 2 - 1))
        current = {**parameters, 'angle_deg': angle, 'temperature_c': temperature}
        result = experiments.run_calibration(current)['results']
        strain = float(models.simulate_finger(
            angle, parameters['length_mm'], parameters['fiber_offset_mm'] * result['gain'],
            np.array([parameters['length_mm'] / 2]), temperature, 0.0, parameters['seed'],
        )['strain'][0])
        estimated = result['estimated_angle_deg']
        frames.append({
            'progress': float(progress), 'strain': strain, 'temperature': temperature,
            'estimated_angle': estimated,
            'signals': result['raw_shifts_nm'] + result['compensated_shifts_nm'],
            'caption': ('机械阶段：改变当前弯曲对应的轴向应变，观察光栅间距示意与波长响应。' if progress < .5
                        else '温度阶段：保持机械应变不变，逐渐施加当前温差；比较原始与温补波长。')
            + (' 当前偏置为零，位于中性层，无法由此信号反演弯曲角。' if estimated is None else '')
            + (' 当前弯曲角为零；可在下方载入“温漂对照”演示两类响应。' if parameters['angle_deg'] == 0 else ''),
            'metrics': [
                ['机械应变', f'{strain * 1e6:+.0f} με'], ['温度变化', f'{temperature:+.1f} °C'],
                ['未温补角', f"{result['uncompensated_angle_deg']:.1f}°" if estimated is not None else '不可反演'],
                ['温补反演角', f'{estimated:.1f}°' if estimated is not None else '不可反演'],
            ],
        })
    amplification = min(25.0, .45 / max(1e-12, max(abs(f['strain']) for f in frames)))
    return {
        'kind': 'fbg', 'title': '分清拉伸信号与温度漂移',
        'subtitle': '机械应变 → 温度变化 → 温补对照',
        'boundary': f'直线光纤展示局部轴向应变，形变放大 {amplification:.1f} 倍；光栅尺寸、间距与外形为示意，不是实际光栅周期。温补使用已知温差，不模拟真实温度参考测量。',
        'legend': '环状纹理：三段工作光栅 · 灰线：零应变长度 · 环境色：温度变化示意',
        'frames': frames, 'initial_index': 120, 'strain_amplification': amplification,
        'labels': ['原始 FBG 1', '原始 FBG 2', '原始 FBG 3', '温补 FBG 1', '温补 FBG 2', '温补 FBG 3'],
    }
