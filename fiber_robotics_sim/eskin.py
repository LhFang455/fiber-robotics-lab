"""Transparent teaching models for electronic-skin sensing experiments.

The functions in this module generate deterministic synthetic signals.  They are
not calibrated hardware models and their metrics must not be presented as
measured device performance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


_TAXEL_SENSITIVITY_PF_PER_N = np.array(
    [
        [0.85, 0.12, 0.08],
        [-0.78, 0.10, 0.09],
        [0.08, 0.82, 0.16],
        [0.06, -0.76, 0.15],
        [0.02, 0.02, 1.15],
    ],
    dtype=float,
)


def simulate_triaxial_taxel(
    *,
    fx_n: float,
    fy_n: float,
    fz_n: float,
    curvature_per_m: float = 0.0,
    strain_fraction: float = 0.0,
    temperature_c: float = 25.0,
    noise_pf: float = 0.0,
    reference_match: float = 1.0,
    seed: int = 7,
) -> dict:
    """Simulate five capacitive channels and a matched reference structure."""
    if noise_pf < 0:
        raise ValueError("noise_pf must be nonnegative")
    if not 0.0 <= reference_match <= 1.0:
        raise ValueError("reference_match must be between 0 and 1")

    forces = np.array([fx_n, fy_n, fz_n], dtype=float)
    common_mode_pf = (
        0.035 * float(curvature_per_m)
        + 28.0 * float(strain_fraction)
        + 0.012 * (float(temperature_c) - 25.0)
    )
    rng = np.random.default_rng(int(seed))
    active_noise = rng.normal(0.0, noise_pf, 5)
    reference_noise = rng.normal(0.0, noise_pf, 5)
    mechanical_signal = _TAXEL_SENSITIVITY_PF_PER_N @ forces
    active = mechanical_signal + common_mode_pf + active_noise
    reference = reference_match * common_mode_pf + reference_noise
    corrected = active - reference

    raw_estimate = np.linalg.lstsq(
        _TAXEL_SENSITIVITY_PF_PER_N, active, rcond=None
    )[0]
    corrected_estimate = np.linalg.lstsq(
        _TAXEL_SENSITIVITY_PF_PER_N, corrected, rcond=None
    )[0]
    raw_error = raw_estimate - forces
    corrected_error = corrected_estimate - forces

    return {
        "forces_true_n": forces,
        "sensitivity_matrix": _TAXEL_SENSITIVITY_PF_PER_N.copy(),
        "active_pf": active,
        "reference_pf": np.full(5, reference, dtype=float) if np.ndim(reference) == 0 else reference,
        "corrected_pf": corrected,
        "raw_estimate_n": raw_estimate,
        "corrected_estimate_n": corrected_estimate,
        "raw_error_n": raw_error,
        "corrected_error_n": corrected_error,
        "raw_mae_n": float(np.mean(np.abs(raw_error))),
        "corrected_mae_n": float(np.mean(np.abs(corrected_error))),
        "common_mode_pf": float(common_mode_pf),
        "matrix_rank": int(np.linalg.matrix_rank(_TAXEL_SENSITIVITY_PF_PER_N)),
        "condition_number": float(np.linalg.cond(_TAXEL_SENSITIVITY_PF_PER_N)),
        "units": {"force": "N", "capacitance_change": "pF", "curvature": "1/m"},
        "boundary": "透明线性教学模型；灵敏度矩阵和参考匹配度不是器件标定数据。",
    }


def _sensor_layout(sensor_count: int, width_mm: float, height_mm: float) -> np.ndarray:
    layout = {4: (2, 2), 8: (4, 2), 16: (4, 4)}
    if sensor_count not in layout:
        raise ValueError("sensor_count must be one of 4, 8, or 16")
    columns, rows = layout[sensor_count]
    xs = np.linspace(width_mm * 0.12, width_mm * 0.88, columns)
    ys = np.linspace(height_mm * 0.15, height_mm * 0.85, rows)
    return np.array([(x, y) for y in ys for x in xs], dtype=float)


def simulate_fbg_skin(
    *,
    sensor_count: int,
    touch_points: Sequence[tuple[float, float, float]],
    skin_width_mm: float = 80.0,
    skin_height_mm: float = 60.0,
    receptive_width_mm: float = 18.0,
    temperature_c: float = 25.0,
    noise_nm: float = 0.0,
    seed: int = 7,
) -> dict:
    """Simulate an FBG skin using normalized Gaussian receptive fields."""
    if skin_width_mm <= 0 or skin_height_mm <= 0:
        raise ValueError("skin dimensions must be positive")
    if receptive_width_mm <= 0:
        raise ValueError("receptive_width_mm must be positive")
    if noise_nm < 0:
        raise ValueError("noise_nm must be nonnegative")
    if not 1 <= len(touch_points) <= 2:
        raise ValueError("touch_points must contain one or two contacts")

    positions = _sensor_layout(sensor_count, skin_width_mm, skin_height_mm)
    optical_sensitivity_nm_per_n = 0.012
    mechanical = np.zeros(sensor_count, dtype=float)
    total_force = 0.0
    weighted_location = np.zeros(2, dtype=float)
    receptive_fields = []
    for x_mm, y_mm, force_n in touch_points:
        if not (0.0 <= x_mm <= skin_width_mm and 0.0 <= y_mm <= skin_height_mm):
            raise ValueError("touch point must lie inside the skin")
        if force_n < 0:
            raise ValueError("touch force must be nonnegative")
        distance_sq = (positions[:, 0] - x_mm) ** 2 + (positions[:, 1] - y_mm) ** 2
        weights = np.exp(-distance_sq / (2.0 * receptive_width_mm**2))
        weight_sum = float(weights.sum())
        if weight_sum > 0:
            weights /= weight_sum
        mechanical += optical_sensitivity_nm_per_n * force_n * weights
        receptive_fields.append(weights)
        total_force += float(force_n)
        weighted_location += float(force_n) * np.array([x_mm, y_mm])

    true_centroid = weighted_location / total_force if total_force > 0 else np.array([np.nan, np.nan])
    temperature_shift = 0.010 * (float(temperature_c) - 25.0)
    rng = np.random.default_rng(int(seed))
    measured = mechanical + temperature_shift + rng.normal(0.0, noise_nm, sensor_count)
    compensated = measured - temperature_shift
    positive_response = np.clip(compensated, 0.0, None)
    response_sum = float(positive_response.sum())
    if response_sum > 0:
        estimated_centroid = np.average(positions, axis=0, weights=positive_response)
        estimated_force = response_sum / optical_sensitivity_nm_per_n
    else:
        estimated_centroid = np.array([np.nan, np.nan])
        estimated_force = 0.0
    location_error = (
        float(np.linalg.norm(estimated_centroid - true_centroid))
        if np.all(np.isfinite(estimated_centroid)) and np.all(np.isfinite(true_centroid))
        else float("nan")
    )

    interpretation = (
        "双点接触按载荷质心评价，不声称唯一分离两个接触点。"
        if len(touch_points) == 2
        else "单点接触按响应加权质心评价。"
    )
    return {
        "sensor_positions_mm": positions,
        "sensor_labels": [f"FBG {index + 1}" for index in range(sensor_count)],
        "receptive_fields": np.asarray(receptive_fields),
        "measured_shift_nm": measured,
        "compensated_shift_nm": compensated,
        "true_centroid_mm": true_centroid,
        "estimated_centroid_mm": estimated_centroid,
        "true_total_force_n": float(total_force),
        "estimated_total_force_n": float(estimated_force),
        "load_error_n": float(abs(estimated_force - total_force)),
        "location_error_mm": location_error,
        "temperature_shift_nm": float(temperature_shift),
        "interpretation": interpretation,
        "boundary": "感受野为归一化高斯教学模型；定位和载荷结果不代表封装后的实物标定精度。",
    }


def _pressure_components(scenario: str):
    scenarios = {
        "单点接触": [(0.50, 0.52, 1.00, 0.10)],
        "双点接触": [(0.32, 0.55, 0.90, 0.09), (0.70, 0.43, 0.72, 0.11)],
        "边缘接触": [(0.10, 0.58, 1.00, 0.12)],
        "滑动前兆": [(0.43, 0.52, 0.75, 0.13), (0.61, 0.52, 0.48, 0.18)],
    }
    if scenario not in scenarios:
        raise ValueError(f"unknown pressure scenario: {scenario}")
    return scenarios[scenario]


def _evaluate_pressure(x, y, components, peak_pressure_kpa: float):
    field = np.zeros(np.broadcast_shapes(np.shape(x), np.shape(y)), dtype=float)
    for center_x, center_y, amplitude, width in components:
        field += peak_pressure_kpa * amplitude * np.exp(
            -((x - center_x) ** 2 + (y - center_y) ** 2) / (2.0 * width**2)
        )
    return field


def _field_centroid(field: np.ndarray) -> np.ndarray:
    coordinate = np.linspace(0.0, 1.0, field.shape[0])
    grid_x, grid_y = np.meshgrid(coordinate, coordinate)
    total = float(field.sum())
    if total <= 0:
        return np.array([np.nan, np.nan])
    return np.array([(field * grid_x).sum() / total, (field * grid_y).sum() / total])


def simulate_pressure_reconstruction(
    scenario: str,
    sparse_size: int,
    output_size: int,
    *,
    peak_pressure_kpa: float = 80.0,
    bandwidth: float = 0.16,
    noise_kpa: float = 0.0,
    seed: int = 7,
) -> dict:
    """Reconstruct a dense pressure field from sparse grid samples."""
    if sparse_size not in (4, 8):
        raise ValueError("sparse_size must be 4 or 8")
    if output_size not in (16, 32):
        raise ValueError("output_size must be 16 or 32")
    if peak_pressure_kpa <= 0 or bandwidth <= 0 or noise_kpa < 0:
        raise ValueError("pressure and bandwidth must be positive and noise nonnegative")

    components = _pressure_components(scenario)
    dense_axis = np.linspace(0.0, 1.0, output_size)
    dense_x, dense_y = np.meshgrid(dense_axis, dense_axis)
    truth = _evaluate_pressure(dense_x, dense_y, components, peak_pressure_kpa)

    sparse_axis = np.linspace(0.0, 1.0, sparse_size)
    sparse_x, sparse_y = np.meshgrid(sparse_axis, sparse_axis)
    samples = _evaluate_pressure(sparse_x, sparse_y, components, peak_pressure_kpa)
    rng = np.random.default_rng(int(seed))
    samples = np.clip(samples + rng.normal(0.0, noise_kpa, samples.shape), 0.0, None)

    sample_points = np.column_stack([sparse_x.ravel(), sparse_y.ravel()])
    target_points = np.column_stack([dense_x.ravel(), dense_y.ravel()])
    distance_sq = (
        (target_points[:, None, 0] - sample_points[None, :, 0]) ** 2
        + (target_points[:, None, 1] - sample_points[None, :, 1]) ** 2
    )
    weights = np.exp(-distance_sq / (2.0 * bandwidth**2))
    reconstruction = (
        weights @ samples.ravel() / np.maximum(weights.sum(axis=1), np.finfo(float).eps)
    ).reshape(output_size, output_size)
    reconstruction = np.clip(reconstruction, 0.0, None)
    error = reconstruction - truth

    true_centroid = _field_centroid(truth)
    reconstructed_centroid = _field_centroid(reconstruction)
    true_total = float(truth.mean())
    reconstructed_total = float(reconstruction.mean())
    true_peak = float(truth.max())
    reconstructed_peak = float(reconstruction.max())
    return {
        "scenario": scenario,
        "sparse_samples_kpa": samples,
        "truth_kpa": truth,
        "reconstruction_kpa": reconstruction,
        "error_kpa": error,
        "true_centroid_normalized": true_centroid,
        "reconstructed_centroid_normalized": reconstructed_centroid,
        "rmse_kpa": float(np.sqrt(np.mean(error**2))),
        "peak_error_pct": float(abs(reconstructed_peak - true_peak) / true_peak * 100.0),
        "centroid_error_pct": float(np.linalg.norm(reconstructed_centroid - true_centroid) / np.sqrt(2.0) * 100.0),
        "total_force_error_pct": float(abs(reconstructed_total - true_total) / true_total * 100.0),
        "channel_saving_pct": float((1.0 - sparse_size**2 / output_size**2) * 100.0),
        "boundary": "使用高斯核插值的透明教学重建，不代表神经网络或超维计算模型的实测性能。",
    }


def simulate_dynamic_skin_event(
    event: str,
    *,
    sample_rate_hz: int = 100,
    duration_s: float = 4.0,
    normal_force_n: float = 12.0,
    slip_threshold: float = 0.35,
    temperature_c: float = 25.0,
    noise_ratio: float = 0.01,
    seed: int = 7,
) -> dict:
    """Generate a multimodal event and apply an explicit slip-risk rule."""
    valid_events = {"稳定按压", "载荷爬升", "横向滑动", "即将滑移", "热物体", "温漂"}
    if event not in valid_events:
        raise ValueError(f"unknown dynamic event: {event}")
    if sample_rate_hz <= 0 or duration_s <= 0 or normal_force_n < 0:
        raise ValueError("sample rate and duration must be positive; force must be nonnegative")
    if slip_threshold <= 0 or noise_ratio < 0:
        raise ValueError("slip threshold must be positive and noise nonnegative")

    sample_count = max(2, int(round(sample_rate_hz * duration_s)))
    time_s = np.arange(sample_count, dtype=float) / float(sample_rate_hz)
    phase = time_s / max(duration_s, np.finfo(float).eps)
    contact = np.clip(phase / 0.16, 0.0, 1.0)
    normal = normal_force_n * contact
    ratio = np.full(sample_count, 0.12, dtype=float)
    centroid_x = np.zeros(sample_count, dtype=float)
    temperature = np.full(sample_count, float(temperature_c), dtype=float)

    if event == "载荷爬升":
        normal = normal_force_n * np.clip(phase, 0.0, 1.0)
        ratio[:] = 0.15
    elif event == "横向滑动":
        ratio = 0.16 + 0.34 * np.clip((phase - 0.42) / 0.45, 0.0, 1.0)
        centroid_x = 18.0 * np.clip((phase - 0.42) / 0.45, 0.0, 1.0)
    elif event == "即将滑移":
        ratio = 0.16 + (slip_threshold * 1.22 - 0.16) * np.clip((phase - 0.48) / 0.38, 0.0, 1.0)
        centroid_x = 7.0 * np.clip((phase - 0.52) / 0.36, 0.0, 1.0)
    elif event == "热物体":
        temperature = temperature_c + 16.0 * np.clip((phase - 0.18) / 0.65, 0.0, 1.0)
    elif event == "温漂":
        temperature = temperature_c + 8.0 * phase
        normal = normal + 0.03 * normal_force_n * phase

    rng = np.random.default_rng(int(seed))
    normal_noise = rng.normal(0.0, noise_ratio * max(normal_force_n, 1.0), sample_count)
    shear_noise = rng.normal(0.0, noise_ratio * max(normal_force_n, 1.0), sample_count)
    centroid_noise = rng.normal(0.0, noise_ratio * 0.2, sample_count)
    observed_normal = np.clip(normal + normal_noise, 0.0, None)
    observed_shear = np.clip(ratio * normal + shear_noise, 0.0, None)
    observed_centroid = centroid_x + centroid_noise
    observed_ratio = np.divide(
        observed_shear,
        observed_normal,
        out=np.zeros_like(observed_shear),
        where=observed_normal > max(0.05 * normal_force_n, 1e-9),
    )
    centroid_velocity = np.gradient(observed_centroid, 1.0 / sample_rate_hz)
    settled = phase >= 0.20
    peak_ratio = float(np.max(observed_ratio[settled]))
    peak_speed = float(np.max(np.abs(centroid_velocity[settled])))
    threshold_margin = peak_ratio - slip_threshold
    alert = bool(threshold_margin > 0.0 and peak_speed > 2.0)

    return {
        "event": event,
        "time_s": time_s,
        "normal_force_n": observed_normal,
        "shear_force_n": observed_shear,
        "shear_ratio": observed_ratio,
        "centroid_x_mm": observed_centroid,
        "centroid_velocity_mm_s": centroid_velocity,
        "temperature_c": temperature,
        "peak_shear_ratio": peak_ratio,
        "peak_centroid_speed_mm_s": peak_speed,
        "threshold_margin": float(threshold_margin),
        "alert": alert,
        "status": "滑移风险" if alert else "稳定",
        "rule": "剪切/法向力比超过阈值，且压力质心速度超过 2 mm/s。",
        "boundary": "告警阈值用于教学对照，必须用目标材料、封装和采样链实测数据重新标定。",
    }


def repeat_dynamic_event(event: str, *, repeats: int = 50, seed: int = 7, **kwargs) -> dict:
    """Summarize seeded repeats of one dynamic event."""
    if not 1 <= repeats <= 200:
        raise ValueError("repeats must be between 1 and 200")
    results = [
        simulate_dynamic_skin_event(event, seed=int(seed) + index, **kwargs)
        for index in range(repeats)
    ]
    alerts = np.array([item["alert"] for item in results], dtype=float)
    peaks = np.array([item["peak_shear_ratio"] for item in results], dtype=float)
    margins = np.array([item["threshold_margin"] for item in results], dtype=float)
    return {
        "event": event,
        "repeat_count": int(repeats),
        "alert_rate_pct": float(alerts.mean() * 100.0),
        "mean_peak_ratio": float(peaks.mean()),
        "peak_ratio_std": float(peaks.std()),
        "mean_threshold_margin": float(margins.mean()),
    }
