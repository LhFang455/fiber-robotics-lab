"""Original-model adapters for triaxial correction and sparse pressure readout."""

import numpy as np

from . import eskin


def taxel_demo(parameters: dict) -> dict:
    frames = []
    for progress in np.linspace(0, 1, 121):
        load = min(1., float(progress) * 2)
        interference = max(0., float(progress) * 2 - 1)
        current = {**parameters,
                   **{key: parameters[key] * load for key in ('fx_n', 'fy_n', 'fz_n')},
                   'curvature_per_m': parameters['curvature_per_m'] * interference,
                   'strain_fraction': parameters['strain_fraction'] * interference,
                   'temperature_c': 25 + (parameters['temperature_c'] - 25) * interference}
        result = eskin.simulate_triaxial_taxel(**current)
        truth = result['forces_true_n'].tolist()
        raw = result['raw_estimate_n'].tolist()
        corrected = result['corrected_estimate_n'].tolist()
        frames.append({
            'progress': float(progress), 'truth': truth, 'raw': raw, 'corrected': corrected,
            'temperature': current['temperature_c'], 'common_mode': result['common_mode_pf'],
            'raw_mae': result['raw_mae_n'], 'corrected_mae': result['corrected_mae_n'],
            'signals': truth + raw + corrected,
            'metrics': [['当前温度', f"{current['temperature_c']:.1f} °C"],
                        ['共模干扰', f"{result['common_mode_pf']:+.3f} pF"],
                        ['校正前 MAE', f"{result['raw_mae_n']:.3f} N"],
                        ['校正后 MAE', f"{result['corrected_mae_n']:.3f} N"]],
            'caption': ('加载阶段：先将三轴力从零增加到当前输入，温度为 25°C，曲率与基底应变为零。'
                        if progress <= .5 else '干扰阶段：保持三轴力，逐渐施加当前温度、曲率和基底应变。')
            + ' 比较三组力向量；参考失配和噪声会影响校正效果，改善程度以 MAE 为准。',
        })
    return {
        'kind': 'taxel', 'title': '三轴受力与参考校正',
        'subtitle': '真实输入 → 未校正反演 → 参考校正反演',
        'boundary': '三块台面是同一触觉单元的结果对照，不是三个独立器件；箭头为力分量坐标向量，长度共用比例，不表示位移。五通道电容与参考信号见下方原实验。固定种子的噪声用于对照，播放时间不是物理时间。',
        'legend': 'X 青色 · Y 金色 · Z 紫色；正 X 向右、正 Y 向后、正 Z 向上，负分量反向',
        'frames': frames, 'initial_index': 120,
        'force_max': max(1e-9, max(abs(v) for f in frames for v in f['signals'])),
        'signal_title': '三轴力分量 · N',
        'labels': [f'{group} {axis}' for group in ['真实', '未校正', '已校正'] for axis in ['Fx', 'Fy', 'Fz']],
    }


def pressure_demo(parameters: dict) -> dict:
    result = eskin.simulate_pressure_reconstruction(**parameters)
    size, output = parameters['sparse_size'], parameters['output_size']
    samples = result['sparse_samples_kpa'].ravel()
    truth_peak = float(result['truth_kpa'].max())
    reconstruction_peak = float(result['reconstruction_kpa'].max())
    frames = []
    for progress in np.linspace(0, 1, 121):
        read_count = min(size**2, int(round(min(1., progress * 2) * size**2)))
        row_count = min(output, int(max(0., (progress - .5) / .35) * output))
        complete = row_count == output
        frames.append({
            'progress': float(progress), 'read_count': read_count, 'row_count': row_count,
            'rmse': result['rmse_kpa'] if complete else None,
            'signals': [truth_peak, float(samples[:read_count].max()) if read_count else None,
                        reconstruction_peak if complete else None],
            'metrics': [['读取通道', f'{read_count} / {size**2}'], ['显示重建行', f'{row_count} / {output}'],
                        ['压力 RMSE', f"{result['rmse_kpa']:.2f} kPa" if complete else '完整显示后读取'],
                        ['峰值误差', f"{result['peak_error_pct']:.1f}%" if complete else '完整显示后读取']],
            'caption': ('采样：左侧是真实压力分布，中间逐点亮起原稀疏采样值，观察窄峰与边缘是否被采到。'
                        if read_count < size**2 else '重建：使用全部采样完成高斯核插值，再逐行显示结果；这是显示顺序，不是逐行求解。'
                        if not complete else '对照：真实与重建面使用同一压力高度和色标；右侧红线连接最大绝对误差点的真实值与重建值。')
            + (' “滑动前兆”仅为静态压力模板，不构成滑移风险判据。' if parameters['scenario'] == '滑动前兆' else ''),
        })
    return {
        'kind': 'pressure', 'title': '从稀疏采样还原压力分布',
        'subtitle': f"{parameters['scenario']} · {size}×{size} 采样 → {output}×{output} 重建",
        'boundary': '高度和颜色均表示压力，不是皮肤形变；位置为 0–1 归一化坐标。演示是空间读取与结果显示，不是算法迭代，不生成部分采样的重建或误差。完整压力差图与评价指标见下方实验。',
        'legend': '左：真实压力 · 中：稀疏测量 · 右：核插值重建 · 青→橙：压力增大（共用比例）',
        'frames': frames, 'initial_index': 120,
        'truth': result['truth_kpa'].tolist(), 'samples': result['sparse_samples_kpa'].tolist(),
        'reconstruction': result['reconstruction_kpa'].tolist(),
        'pressure_max': max(float(result[key].max()) for key in ['truth_kpa', 'sparse_samples_kpa', 'reconstruction_kpa']),
        'max_error_index': int(np.argmax(np.abs(result['error_kpa']))),
        'signal_title': '已显示数据的峰值 · kPa', 'labels': ['真实压力', '已读采样', '完整重建'],
        'pending_signal_text': '尚未显示',
    }
