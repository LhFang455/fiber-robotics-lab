"""Presentation adapters for spatial sensing, optical mechanisms and touch templates."""

import numpy as np

from . import eskin, experiments, models


def distributed_demo(parameters: dict) -> dict:
    parameters = dict(parameters)
    selected = parameters.pop('inspection_index', None)
    result = experiments.run_distributed_experiment(parameters)['results']
    positions, profile = result['profile_position_mm'], result['profile_value']
    if selected is not None:
        selected = max(0, min(int(selected), len(positions) - 1))
    background = float(np.median(profile))
    valid = result['localization_valid']
    estimate = result['estimated_event_position_mm'] if valid else None
    frames = []
    for i, (position, value) in enumerate(zip(positions, profile, strict=True)):
        frames.append({
            'progress': i / max(1, len(positions) - 1), 'sample_index': i,
            'signals': [value, value - background],
            'caption': '沿光纤依次读取当前空间剖面。移动光点是讲解游标，不是实际光脉冲或事件传播。'
            + (' 扫描完成后比较白色真值与橙色峰值定位。' if valid else ' 当前不适用单事件定位，不报告位置。') + result['model_note'],
            'metrics': [['读取位置', f'{position:.1f} mm'], ['采样点数', f'{i + 1} / {len(positions)}'],
                        ['峰值定位', f'{estimate:.1f} mm' if valid else '不适用'],
                        ['定位误差', f"{result['location_error_mm']:.1f} mm" if valid else '不适用']],
        })
    return {
        'kind': 'distributed', 'title': '沿着光纤找到局部事件' if parameters['mode'] != 'Brillouin' else '观察温度与应变的叠加响应', 'subtitle': parameters['mode'] + ' · 空间剖面读取' + (f' · D-{positions[selected]:.3f}mm' if selected is not None else ''),
        'current_label': '观察器位置' if selected is not None else '当前参数',
        'inspection_index': selected,
        'inspection_text': ('紫色方标为观察器所选空间采样位置，不是事件定位；播放只改变读取游标，不回写观察器。点击‘观察器位置’返回选点。' if selected is not None else ''),
        'boundary': '空间位置沿用原实验毫米坐标，曲线高度按相对背景变化归一化，仅用于看清剖面。DAS 展示时间窗内最大振幅，不是瞬时波形；播放时间不代表光传播时间。',
        'legend': '青线：完整剖面 · 橙线：已读取部分 · 白标：真实位置 · 橙标：峰值定位',
        'frames': frames, 'initial_index': selected if selected is not None else len(frames) - 1, 'positions': positions, 'profile': profile,
        'background': background, 'length': float(parameters['fiber_length_mm']), 'unit': result['unit'],
        'truth': float(parameters['event_position_mm']) if valid else None, 'estimate': estimate,
        'signal_title': f"{result['observable']} · {result['unit']}", 'labels': ['当前采样值', '相对背景变化'],
    }


def tactile_demo(parameters: dict) -> dict:
    materials = ['海绵', '硬块', '圆柱', '薄板']
    frames = []
    for progress in np.linspace(0, 1, 121):
        gain = float(min(progress * 2, (1 - progress) * 2))
        force = parameters['grip_force_n'] * gain
        results = [experiments.run_tactile_experiment(
            {**parameters, 'material': name, 'grip_force_n': force}, repeat_count=1,
        )['results'] for name in materials]
        selected = results[materials.index(parameters['material'])]
        frames.append({
            'progress': float(progress), 'gain': gain,
            'touches': [result['estimated_touch_n'] for result in results],
            'signals': selected['wavelength_shifts_nm'], 'diagnosis': selected['diagnosed_material'],
            'caption': '四块台面使用相同握持力、接触面积和扰动条件；每块周围六根柱表示反演接触力。'
            + (' 当前处于无有效接触状态；残余响应可能来自噪声。' if not selected['valid_contact']
               else ' 右侧读数对应高亮的当前材质；比较分布形状，不把相似度当成实物鉴定准确率。'),
            'metrics': [['当前材质', parameters['material']], ['握持力输入', f'{force:.1f} N'],
                        ['识别结果', selected['diagnosed_material']],
                        ['类别间隔', f"{selected['probability_margin'] * 100:.2f} 个百分点" if selected['valid_contact'] else '不适用']],
        })
    return {
        'kind': 'tactile', 'title': '相同输入，不同接触分布', 'subtitle': '四种模板 · 五指与掌心 FBG 对照',
        'boundary': '物体与柱高用于解释预设接触模板，不计算材料刚度、压痕或真实抓取动力学；每块台面的六柱从左到右为拇指、食指、中指、无名指、小指、掌心，柱高共用同一比例。',
        'legend': '金色台面：当前材质 · 六柱：反演接触力 · 物体不展示未经计算的压缩形变',
        'frames': frames, 'initial_index': 60, 'materials': materials, 'selected': materials.index(parameters['material']),
        'force_max': max(1e-9, max(v for f in frames for values in f['touches'] for v in values)),
        'labels': ['拇指', '食指', '中指', '无名指', '小指', '掌心'],
    }


def optical_demo(parameters: dict) -> dict:
    mechanism = parameters['mechanism']
    params = {key: value for key, value in parameters.items() if key != 'mechanism'}
    frames = []
    for progress in np.linspace(0, 1, 121):
        current = dict(params)
        for key in {'偏振态': ['stress_mpa', 'twist_deg'], 'Sagnac 环路': ['gyro_rate_deg_s'], 'EFPI 微腔': ['pressure_mpa']}[mechanism]:
            current[key] *= float(progress)
        result = experiments.run_optical_experiment(current)['results']
        spectrum = models.simulate_efpi_pressure(current['pressure_mpa'], current['cavity_um'])['intensity'][::10].tolist()
        if mechanism == '偏振态':
            signals = result['stokes']
            metrics = [['横向应力', f"{current['stress_mpa']:.1f} MPa"], ['扭转角', f"{current['twist_deg']:.1f}°"],
                       ['方位角', f"{result['azimuth_deg']:.1f}°"], ['椭圆率角', f"{result['ellipticity_deg']:.1f}°"]]
            caption = '逐渐施加当前应力和扭转，保持温度不变。球面端点表示 Stokes 状态，旁侧椭圆表示电场横截面的偏振轨迹。'
        elif mechanism == 'Sagnac 环路':
            signals = [result['sagnac_phase_shift_rad']]
            metrics = [['角速度输入', f"{current['gyro_rate_deg_s']:+.1f} °/s"], ['相位差', f"{signals[0]:+.3e} rad"],
                       ['反演角速度', f"{result['estimated_rate_deg_s']:+.1f} °/s"], ['模型线圈长度', '120 m']]
            caption = '青色与橙色光点说明环路的两个相反方向；相位差正负随旋转方向改变，真实相位读右侧数值。光点运动不表示真实光速或到达时差。'
        else:
            signals = [spectrum[25]]
            metrics = [['施加压力', f"{current['pressure_mpa']:.2f} MPa"], ['有效腔长', f"{result['effective_cavity_um']:.3f} μm"],
                       ['腔长变化', f"{result['cavity_change_nm']:+.1f} nm"], ['1550 nm 光强', f'{signals[0]:.4f}']]
            caption = '压力改变两反射面之间的腔长，下方曲线展示 1500–1600 nm 的干涉光谱。固定波长光强会周期变化，不能把单点光强当成唯一压力读数。'
        frames.append({
            'progress': float(progress), 'stokes': result['stokes'], 'phase': result['sagnac_phase_shift_rad'],
            'azimuth': result['azimuth_deg'], 'ellipticity': result['ellipticity_deg'],
            'rate': current['gyro_rate_deg_s'], 'cavity': result['effective_cavity_um'],
            'spectrum': spectrum, 'signals': signals, 'metrics': metrics, 'caption': caption,
        })
    return {
        'kind': 'optical', 'title': {'偏振态': '看见偏振态怎样改变', 'Sagnac 环路': '双向环路，读出旋转', 'EFPI 微腔': '微腔变化，光谱随之移动'}[mechanism],
        'subtitle': mechanism + ' · 独立机制演示',
        'boundary': {'偏振态': '球面和电场椭圆为归一化状态表示，不是机械形变或真实光线轨迹；电场描点进度不是光频。',
                     'Sagnac 环路': '采用原模型的固定线圈几何；环路和光点为光路说明，不绘制未经验证的传播时差。',
                     'EFPI 微腔': '镜面外形为示意，腔长相对变化放大 20 倍，真实微米/纳米读数见右侧；光谱为解析模型，未加入实际仪器噪声。'}[mechanism],
        'legend': {'偏振态': '左：归一化 Stokes 球 · 右：电场偏振椭圆', 'Sagnac 环路': '青点 / 橙点：双向光路 · 白色箭头：旋转方向示意',
                   'EFPI 微腔': '两反射面之间为微腔 · 下方：归一化干涉光谱（左 1500 → 右 1600 nm）'}[mechanism],
        'frames': frames, 'initial_index': 120, 'mechanism': mechanism, 'initial_cavity': params['cavity_um'],
        'signal_title': {'偏振态': '归一化 Stokes 分量 · 无量纲', 'Sagnac 环路': 'Sagnac 相位差 · rad', 'EFPI 微腔': '归一化光强 · a.u.'}[mechanism],
        'signal_format': 'scientific' if mechanism == 'Sagnac 环路' else 'fixed',
        'labels': {'偏振态': ['S1', 'S2', 'S3'], 'Sagnac 环路': ['相位差'], 'EFPI 微腔': ['1550 nm']}[mechanism],
    }


def dynamic_demo(parameters: dict) -> dict:
    """Replay original offline samples; accumulate the original window-peak rule."""
    parameters = dict(parameters)
    selected = parameters.pop('inspection_index', None)
    result = eskin.simulate_dynamic_skin_event(**parameters)
    if selected is not None:
        selected = max(0, min(int(selected), len(result['time_s']) - 1))
    peak_ratio = peak_speed = 0.0
    frames = []
    for i, time in enumerate(result['time_s']):
        fields = {name: float(result[key][i]) for name, key in [
            ('time', 'time_s'), ('normal', 'normal_force_n'), ('shear', 'shear_force_n'),
            ('x', 'centroid_x_mm'), ('temperature', 'temperature_c'),
            ('ratio', 'shear_ratio'), ('speed', 'centroid_velocity_mm_s'),
        ]}
        settled = time / parameters['duration_s'] >= .20
        if settled:
            peak_ratio = max(peak_ratio, fields['ratio'])
            peak_speed = max(peak_speed, abs(fields['speed']))
        alert = bool(peak_ratio > parameters['slip_threshold'] and peak_speed > 2) if settled else None
        status = '建立接触，尚未进入判定窗口' if alert is None else '窗口峰值触发风险' if alert else '窗口峰值未触发风险'
        contact = fields['normal'] > max(.05 * parameters['normal_force_n'], 1e-9)
        ratio_text = f"{fields['ratio']:.3f}" if contact else '未定义（低法向力）'
        frames.append({
            **fields, 'progress': i / (len(result['time_s']) - 1),
            'peak_ratio': peak_ratio if settled else None, 'peak_speed': peak_speed if settled else None,
            'alert': alert, 'signals': [fields['normal'], fields['shear']],
            'contact': fields['normal'] > max(.05 * parameters['normal_force_n'], 1e-9),
            'caption': f"{status}。当前剪切比 {ratio_text}，质心速度 {fields['speed']:+.2f} mm/s。"
            + f"从记录 20% 起累计峰值：剪切比 > {parameters['slip_threshold']:.2f} 且速度 > 2 mm/s，不要求同一时刻越阈。",
            'metrics': [['记录时间', f"{time:.3f} s"], ['当前温度', f"{fields['temperature']:.1f} °C"],
                        ['窗口剪切比峰值', f'{peak_ratio:.3f}' if settled else '等待接触'],
                        ['窗口速度峰值', f'{peak_speed:.2f} mm/s' if settled else '等待接触']],
        })
    return {
        'kind': 'dynamic', 'title': '看见接触变化，读懂滑移判据',
        'subtitle': parameters['event'] + ' · 多模态离线记录回放' + (f" · 观察器 FRAME-{selected:04d}" if selected is not None else ''),
        'current_label': '观察器时刻' if selected is not None else '当前参数',
        'inspection_text': ('观察器定位到所选采样帧；播放仅改变本面板，不回写观察器。灯为截至回放时刻的累计判据，下方结论为整段记录判定。' if selected is not None else ''),
        'boundary': '质心标记不是物体轨迹；箭头长度示意力大小，方向仅作说明，面板颜色表示温升，不重建压力场。离线速度含相邻采样信息，不是因果实时检测；阈值需实测标定。',
        'legend': '橙环：质心 · 青/黄箭：法向/剪切力 · 面板青→红：温升 · 判据灯灰/青/红：等待/未触发/风险',
        'frames': frames, 'initial_index': selected if selected is not None else len(frames) - 1,
        'force_max': max(1e-9, max(max(f['signals']) for f in frames)),
        'initial_temperature': parameters['temperature_c'],
        'playback_duration_ms': float(result['time_s'][-1]) * 1000,
        'signal_title': '当前力读数 · N', 'labels': ['法向力', '剪切力'],
    }
