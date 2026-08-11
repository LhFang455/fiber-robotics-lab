"""与界面无关的光纤传感正向与反演模型。"""

from __future__ import annotations

import numpy as np


def auto_grasp_phase(elapsed_seconds: float) -> str:
    """Return the current phase of the non-blocking teaching grasp demonstration."""
    if elapsed_seconds < 1.0:
        return "寻找目标"
    if elapsed_seconds < 2.0:
        return "对准目标"
    if elapsed_seconds < 3.0:
        return "抓取"
    if elapsed_seconds < 4.2:
        return "搬运"
    if elapsed_seconds < 5.0:
        return "放下"
    return "完成"


def next_three_d_grasp_task_phase(current_phase: str, grasp_verified: bool) -> str:
    """Advance one observable 3D grasp-task step.

    The UI owns pose changes, while this small state transition keeps the task
    order deterministic.  Transport is deliberately impossible before the
    closed-hand contact model has confirmed a stable grasp.
    """
    if current_phase == "寻找目标":
        return "对准目标"
    if current_phase == "对准目标":
        return "闭合抓取"
    if current_phase == "闭合抓取":
        return "搬运目标" if grasp_verified else "抓取失败"
    if current_phase == "搬运目标":
        return "松开并放置"
    if current_phase == "松开并放置":
        return "完成"
    return current_phase


WAVELENGTH_NM = 1550.0
PHOTOELASTIC_COEFFICIENT = 0.22
THERMAL_SENSITIVITY_PER_C = 6.7e-6


def fbg_wavelength_shift_nm(
    strain: np.ndarray,
    temperature_change_c: float,
    wavelength_nm: float = WAVELENGTH_NM,
    photoelastic_coefficient: float = PHOTOELASTIC_COEFFICIENT,
    thermal_sensitivity_per_c: float = THERMAL_SENSITIVITY_PER_C,
) -> np.ndarray:
    """Return FBG Bragg wavelength shifts in nm for the supplied strains."""
    strain = np.asarray(strain, dtype=float)
    return wavelength_nm * (
        (1.0 - photoelastic_coefficient) * strain
        + thermal_sensitivity_per_c * temperature_change_c
    )


def add_gaussian_noise(values: np.ndarray, standard_deviation: float, seed: int) -> np.ndarray:
    """Add reproducible zero-mean Gaussian measurement noise."""
    values = np.asarray(values, dtype=float)
    if standard_deviation == 0:
        return values.copy()
    return values + np.random.default_rng(seed).normal(0.0, standard_deviation, values.shape)


def simulate_redundant_finger_fbg(
    bend_angle_deg: float,
    length_mm: float,
    fiber_offset_mm: float,
    temperature_change_c: float,
    fault_mode: str,
    fault_channel: int,
) -> dict[str, np.ndarray]:
    """Create four redundant FBG readings with one selectable teaching fault."""
    strain = np.deg2rad(bend_angle_deg) / length_mm * fiber_offset_mm
    shifts = fbg_wavelength_shift_nm(np.full(4, strain), temperature_change_c)
    channel = int(np.clip(fault_channel, 1, 4)) - 1
    if fault_mode == "漂移":
        shifts[channel] += .070
    elif fault_mode == "断纤":
        shifts[channel] = 0.0
    elif fault_mode == "噪声增大":
        shifts[channel] += .035
    return {"wavelength_shifts_nm": shifts, "fault_channel": np.array([channel])}


def diagnose_redundant_fbg(
    wavelength_shifts_nm: np.ndarray,
    length_mm: float,
    fiber_offset_mm: float,
    temperature_change_c: float,
) -> dict[str, float | list[int]]:
    """Detect outliers against the redundant median and estimate angle from healthy channels."""
    shifts = np.asarray(wavelength_shifts_nm, dtype=float)
    if shifts.shape != (4,):
        raise ValueError("冗余手指诊断需要四路 FBG 读数")
    thermal_shift = float(fbg_wavelength_shift_nm(np.array([0.0]), temperature_change_c)[0])
    median = float(np.median(shifts))
    faulty_indices = np.flatnonzero(np.abs(shifts - median) > .025).astype(int).tolist()
    healthy = np.delete(shifts, faulty_indices) if faulty_indices else shifts
    if healthy.size == 0:
        healthy = shifts
    angle = estimate_finger_angle_deg(healthy, length_mm, fiber_offset_mm, temperature_change_c)
    return {
        "fault_channels": [index + 1 for index in faulty_indices],
        "estimated_angle_deg": angle,
        "common_temperature_shift_nm": thermal_shift,
    }


_TACTILE_PROFILES = {
    "海绵": np.array([.52, .62, .68, .62, .52, .92]),
    "硬块": np.array([.88, .76, .28, .18, .12, .16]),
    "圆柱": np.array([.72, .84, .86, .58, .38, .42]),
    "薄板": np.array([.35, .40, .42, .40, .35, .96]),
}


def simulate_material_touch(material: str, grip_force_n: float, contact_area_percent: float, temperature_change_c: float) -> dict[str, np.ndarray | float]:
    """Generate five finger and one palm contact forces for four teaching materials."""
    if material not in _TACTILE_PROFILES:
        raise ValueError("未知触觉材料")
    profile = _TACTILE_PROFILES[material]
    area_gain = .45 + .55 * np.clip(contact_area_percent, 0.0, 100.0) / 100.0
    touch = profile * max(0.0, grip_force_n) * area_gain
    shifts = fbg_wavelength_shift_nm(touch * 4.0e-5, temperature_change_c)
    return {
        "finger_touch_n": touch[:5],
        "palm_touch_n": float(touch[5]),
        "wavelength_shifts_nm": shifts,
    }


def classify_tactile_material(finger_touch_n: np.ndarray, palm_touch_n: float) -> dict[str, float | str]:
    """Classify a teaching material from normalized five-finger and palm contact pattern."""
    observed = np.r_[np.asarray(finger_touch_n, dtype=float), float(palm_touch_n)]
    if observed.shape != (6,) or np.allclose(observed, 0.0):
        return {"material": "未接触", "confidence": 0.0}
    normalized = observed / np.linalg.norm(observed)
    scores = {name: float(np.dot(normalized, profile / np.linalg.norm(profile))) for name, profile in _TACTILE_PROFILES.items()}
    material = max(scores, key=scores.get)
    return {"material": material, "confidence": scores[material]}


def simulate_demodulation_chain(
    bend_angle_deg: float,
    temperature_change_c: float,
    sample_rate_hz: int,
    noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Produce a short raw-to-filtered-to-compensated FBG teaching data stream."""
    count = max(20, int(sample_rate_hz * 2))
    time_s = np.arange(count) / sample_rate_hz
    strain = np.deg2rad(bend_angle_deg) / 80.0
    clean = float(fbg_wavelength_shift_nm(np.array([strain]), temperature_change_c)[0])
    raw = clean + add_gaussian_noise(np.zeros(count), noise_nm, seed)
    kernel = np.ones(5) / 5.0
    filtered = np.convolve(np.pad(raw, (2, 2), mode="edge"), kernel, mode="valid")
    thermal_shift = float(fbg_wavelength_shift_nm(np.array([0.0]), temperature_change_c)[0])
    compensated = filtered - thermal_shift
    estimated = estimate_finger_angle_deg(compensated, 80.0, 1.0, 0.0)
    return {
        "time_s": time_s,
        "raw_wavelength_nm": raw,
        "filtered_wavelength_nm": filtered,
        "compensated_wavelength_nm": compensated,
        "estimated_angle_deg": estimated,
        "control_command": "闭合" if estimated >= 35.0 else "张开",
    }


def simulate_planar_grasp_fbg(
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    contact_fingers: list[int],
    temperature_change_c: float,
) -> dict[str, np.ndarray]:
    """Generate one contact-sensitive FBG channel per planar finger."""
    curls = np.asarray(finger_curls_deg, dtype=float)
    if curls.shape != (5,):
        raise ValueError("finger_curls_deg 必须包含五根手指")
    contact = np.zeros(5, dtype=float)
    contact[np.asarray(contact_fingers, dtype=int)] = 1.0
    bend_strain = curls * 4.0e-5
    contact_strain = contact * 2.1e-4
    return {
        "wavelength_shifts_nm": fbg_wavelength_shift_nm(bend_strain + contact_strain, temperature_change_c),
        "contact_strain": contact_strain,
    }


def classify_planar_grasp_from_fbg(
    sensing: dict[str, np.ndarray],
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    temperature_change_c: float,
) -> dict[str, np.ndarray | list[int] | bool]:
    """Recover tactile contacts from FBG readings and decide a planar grasp."""
    curls = np.asarray(finger_curls_deg, dtype=float)
    shifts = np.asarray(sensing["wavelength_shifts_nm"], dtype=float)
    if curls.shape != (5,) or shifts.shape != (5,):
        raise ValueError("二维 FBG 判定需要五根手指的屈曲和波长读数")
    strain = _strain_from_shift(shifts, temperature_change_c)
    contact_strain = np.maximum(0.0, strain - curls * 4.0e-5)
    contact_fingers = np.flatnonzero(contact_strain >= 1.0e-4).astype(int).tolist()
    is_grasped = bool(0 in contact_fingers and len([index for index in contact_fingers if index != 0]) >= 2)
    return {"is_grasped": is_grasped, "contact_fingers": contact_fingers, "contact_strain": contact_strain}


def _strain_from_shift(shifts_nm: np.ndarray, temperature_change_c: float) -> np.ndarray:
    thermal_shift = WAVELENGTH_NM * THERMAL_SENSITIVITY_PER_C * temperature_change_c
    return (np.asarray(shifts_nm, dtype=float) - thermal_shift) / (
        WAVELENGTH_NM * (1.0 - PHOTOELASTIC_COEFFICIENT)
    )


def _constant_curvature_centerline(length_mm: float, curvature_per_mm: float) -> np.ndarray:
    s = np.linspace(0.0, length_mm, 161)
    if abs(curvature_per_mm) < 1e-12:
        return np.column_stack((s, np.zeros_like(s)))
    x = np.sin(curvature_per_mm * s) / curvature_per_mm
    y = (1.0 - np.cos(curvature_per_mm * s)) / curvature_per_mm
    return np.column_stack((x, y))


def estimate_finger_angle_deg(
    wavelength_shifts_nm: np.ndarray,
    length_mm: float,
    fiber_offset_mm: float,
    temperature_change_c: float,
) -> float:
    """Estimate a planar constant-curvature finger angle from FBG shifts."""
    if abs(fiber_offset_mm) < 1e-12:
        return 0.0
    strain = np.mean(_strain_from_shift(wavelength_shifts_nm, temperature_change_c))
    return float(np.rad2deg(strain / fiber_offset_mm * length_mm))


def simulate_finger(
    bend_angle_deg: float,
    length_mm: float,
    fiber_offset_mm: float,
    sensor_positions_mm: np.ndarray,
    temperature_change_c: float,
    noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Simulate three axial FBGs embedded in a constant-curvature finger."""
    sensor_positions_mm = np.asarray(sensor_positions_mm, dtype=float)
    curvature_per_mm = np.deg2rad(bend_angle_deg) / length_mm
    strains = np.full(sensor_positions_mm.shape, curvature_per_mm * fiber_offset_mm)
    clean_shifts = fbg_wavelength_shift_nm(strains, temperature_change_c)
    shifts = add_gaussian_noise(clean_shifts, noise_nm, seed)
    return {
        "sensor_positions_mm": sensor_positions_mm,
        "strain": strains,
        "wavelength_shifts_nm": shifts,
        "clean_wavelength_shifts_nm": clean_shifts,
        "centerline_xy_mm": _constant_curvature_centerline(length_mm, curvature_per_mm),
        "estimated_angle_deg": estimate_finger_angle_deg(
            shifts, length_mm, fiber_offset_mm, temperature_change_c
        ),
    }


def simulate_contact(
    contact_position_mm: float,
    force_n: float,
    sensor_positions_mm: np.ndarray,
    influence_width_mm: float,
    strain_per_newton: float,
    temperature_change_c: float,
    noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Simulate a Gaussian contact-to-strain transfer in a compliant fingertip."""
    positions = np.asarray(sensor_positions_mm, dtype=float)
    influence = np.exp(-0.5 * ((positions - contact_position_mm) / influence_width_mm) ** 2)
    strains = force_n * strain_per_newton * influence
    clean_shifts = fbg_wavelength_shift_nm(strains, temperature_change_c)
    shifts = add_gaussian_noise(clean_shifts, noise_nm, seed)
    return {
        "sensor_positions_mm": positions,
        "strain": strains,
        "wavelength_shifts_nm": shifts,
        "clean_wavelength_shifts_nm": clean_shifts,
        "contact_position_mm": contact_position_mm,
        "force_n": force_n,
    }


def estimate_contact(
    wavelength_shifts_nm: np.ndarray,
    sensor_positions_mm: np.ndarray,
    influence_width_mm: float,
    strain_per_newton: float,
    temperature_change_c: float,
) -> tuple[float, float]:
    """Recover contact location and non-negative force with a grid least-squares fit."""
    positions = np.asarray(sensor_positions_mm, dtype=float)
    observed_strain = _strain_from_shift(wavelength_shifts_nm, temperature_change_c)
    if np.allclose(observed_strain, 0.0):
        return float(np.mean(positions)), 0.0
    candidates = np.linspace(float(positions.min()), float(positions.max()), 401)
    best_position, best_force, best_error = candidates[0], 0.0, np.inf
    for candidate in candidates:
        basis = strain_per_newton * np.exp(-0.5 * ((positions - candidate) / influence_width_mm) ** 2)
        force = max(0.0, float(np.dot(observed_strain, basis) / np.dot(basis, basis)))
        error = float(np.sum((observed_strain - force * basis) ** 2))
        if error < best_error:
            best_position, best_force, best_error = candidate, force, error
    return float(best_position), float(best_force)


def simulate_foot_fbg(
    zone_loads_n: np.ndarray,
    temperature_change_c: float,
    noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """Generate six temperature-affected FBG readings for sole contact zones.

    This deliberately uses an independent, linear teaching calibration for
    each sole zone; it is not a replacement for shoe/sole calibration data.
    """
    loads = np.asarray(zone_loads_n, dtype=float)
    if loads.shape != (6,):
        raise ValueError("足底 FBG 模型需要六个区域载荷")
    if np.any(loads < 0.0):
        raise ValueError("足底区域载荷不能为负值")
    strains = loads * 4.0e-6
    clean_shifts = fbg_wavelength_shift_nm(strains, temperature_change_c)
    return {
        "zone_loads_n": loads,
        "strain": strains,
        "wavelength_shifts_nm": add_gaussian_noise(clean_shifts, noise_nm, seed),
        "clean_wavelength_shifts_nm": clean_shifts,
    }


def estimate_foot_load_distribution(
    wavelength_shifts_nm: np.ndarray, temperature_change_c: float
) -> dict[str, np.ndarray | float]:
    """Temperature-compensate six FBG readings and recover the teaching CoP."""
    shifts = np.asarray(wavelength_shifts_nm, dtype=float)
    if shifts.shape != (6,):
        raise ValueError("足底载荷反演需要六路 FBG 读数")
    loads = np.maximum(0.0, _strain_from_shift(shifts, temperature_change_c) / 4.0e-6)
    return {
        "zone_loads_n": loads,
        "cop_region": float(np.dot(np.arange(6), loads) / max(loads.sum(), 1e-9)),
    }


def simulate_arm_health_fbg(
    load_n: float,
    damage_position_mm: float,
    damage_severity: float,
    temperature_change_c: float,
    noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Simulate four FBGs on an arm link with a local damage-sensitive peak."""
    positions = np.array([80.0, 200.0, 320.0, 440.0])
    severity = float(np.clip(damage_severity, 0.0, 1.0))
    baseline_strain = np.full(positions.shape, max(0.0, load_n) * 2.0e-6)
    damage_profile = severity * 8.0e-4 * np.exp(-0.5 * ((positions - damage_position_mm) / 55.0) ** 2)
    strains = baseline_strain + damage_profile
    clean_shifts = fbg_wavelength_shift_nm(strains, temperature_change_c)
    return {
        "sensor_positions_mm": positions,
        "strain": strains,
        "damage_profile": damage_profile,
        "wavelength_shifts_nm": add_gaussian_noise(clean_shifts, noise_nm, seed),
        "clean_wavelength_shifts_nm": clean_shifts,
    }


def diagnose_arm_health(
    wavelength_shifts_nm: np.ndarray, temperature_change_c: float
) -> dict[str, float | str]:
    """Find the strongest local differential strain after common-mode compensation."""
    shifts = np.asarray(wavelength_shifts_nm, dtype=float)
    positions = np.array([80.0, 200.0, 320.0, 440.0])
    if shifts.shape != positions.shape:
        raise ValueError("机械臂健康诊断需要四路 FBG 读数")
    strain = _strain_from_shift(shifts, temperature_change_c)
    local_excess = np.maximum(0.0, strain - np.median(strain))
    index = int(np.argmax(local_excess))
    damage_index = float(local_excess[index] / 8.0e-4)
    return {
        "suspected_location_mm": float(positions[index]),
        "damage_index": damage_index,
        "status": "需检查" if damage_index >= 0.35 else "正常",
    }


def _core_angles_rad() -> np.ndarray:
    return np.deg2rad(np.array([0.0, 120.0, 240.0]))


def estimate_multicore_curvature(
    wavelength_shifts_nm: np.ndarray, core_radius_um: float
) -> tuple[float, float]:
    """Estimate curvature (1/m) and bend direction (degrees) from three FBG cores."""
    radius_m = core_radius_um * 1e-6
    if radius_m <= 0:
        return 0.0, 0.0
    strain = _strain_from_shift(wavelength_shifts_nm, 0.0)
    differential_strain = strain - np.mean(strain)
    angles = _core_angles_rad()
    matrix = radius_m * np.column_stack((np.cos(angles), np.sin(angles)))
    kx, ky = np.linalg.lstsq(matrix, differential_strain, rcond=None)[0]
    curvature = float(np.hypot(kx, ky))
    direction = float(np.rad2deg(np.arctan2(ky, kx)) % 360.0) if curvature else 0.0
    return curvature, direction


def _multicore_centerline(
    curvature_per_m: float, bend_direction_deg: float, twist_per_m: float, length_mm: float
) -> np.ndarray:
    """Integrate a centreline whose curvature-vector direction rotates with twist."""
    count = 161
    ds_m = length_mm * 1e-3 / (count - 1)
    point = np.zeros(3)
    tangent = np.array([0.0, 0.0, 1.0])
    normal_x = np.array([1.0, 0.0, 0.0])
    normal_y = np.array([0.0, 1.0, 0.0])
    points = [point.copy()]
    base_direction = np.deg2rad(bend_direction_deg)
    for index in range(1, count):
        direction = base_direction + twist_per_m * (index - 0.5) * ds_m
        normal = np.cos(direction) * normal_x + np.sin(direction) * normal_y
        tangent = tangent + curvature_per_m * normal * ds_m
        tangent /= np.linalg.norm(tangent)
        point = point + tangent * ds_m
        points.append(point.copy())
    return np.asarray(points) * 1e3


def simulate_multicore_shape(
    curvature_per_m: float,
    bend_direction_deg: float,
    twist_per_m: float,
    length_mm: float,
    core_radius_um: float,
    temperature_change_c: float,
    noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray | float]:
    """Simulate a three-core FBG shape sensor and reconstruct its centreline."""
    direction_rad = np.deg2rad(bend_direction_deg)
    angles = _core_angles_rad()
    radius_m = core_radius_um * 1e-6
    strains = radius_m * curvature_per_m * np.cos(angles - direction_rad)
    clean_shifts = fbg_wavelength_shift_nm(strains, temperature_change_c)
    shifts = add_gaussian_noise(clean_shifts, noise_nm, seed)
    estimated_curvature, estimated_direction = estimate_multicore_curvature(shifts, core_radius_um)
    return {
        "core_angles_deg": np.rad2deg(angles),
        "strain": strains,
        "wavelength_shifts_nm": shifts,
        "clean_wavelength_shifts_nm": clean_shifts,
        "centerline_xyz_mm": _multicore_centerline(
            curvature_per_m, bend_direction_deg, twist_per_m, length_mm
        ),
        "estimated_centerline_xyz_mm": _multicore_centerline(
            estimated_curvature, estimated_direction, twist_per_m, length_mm
        ),
        "estimated_curvature_per_m": estimated_curvature,
        "estimated_direction_deg": estimated_direction,
    }


def simulate_distributed_sensing(
    finger_curls_deg: np.ndarray,
    contact_fingers: list[int],
) -> dict[str, np.ndarray]:
    """Create deterministic Rayleigh/OFDR and DAS teaching signals for five fingers.

    The values are explanatory synthetic signals, not calibrated instrument readings.
    """
    curls = np.asarray(finger_curls_deg, dtype=float)
    if curls.shape != (5,):
        raise ValueError("finger_curls_deg 必须包含五根手指的屈曲角")
    distance_mm = np.linspace(0.0, 100.0, 121)
    normalized = distance_mm / distance_mm.max()
    bend_profile = .25 + .75 * np.sin(np.pi * normalized / 2.0) ** 1.5
    rayleigh = np.empty((5, len(distance_mm)))
    for index, curl in enumerate(curls):
        contact_peak = 190.0 * np.exp(-.5 * ((normalized - .83) / .10) ** 2) if index in contact_fingers else 0.0
        rayleigh[index] = curl * 5.0 * bend_profile + contact_peak
    das_time_ms = np.linspace(-120.0, 120.0, 121)
    das_distance_mm = np.arange(5 * len(distance_mm)) * (distance_mm[1] - distance_mm[0])
    das = np.zeros((len(das_time_ms), len(das_distance_mm)))
    event = np.exp(-.5 * (das_time_ms / 18.0) ** 2)[:, None]
    for index in contact_fingers:
        start = index * len(distance_mm)
        local_peak = np.exp(-.5 * ((normalized - .83) / .09) ** 2)
        das[:, start : start + len(distance_mm)] += event * local_peak * (0.25 + curls[index] / 180.0)
    return {
        "distance_mm": distance_mm,
        "rayleigh_strain_ue": rayleigh,
        "das_time_ms": das_time_ms,
        "das_distance_mm": das_distance_mm,
        "das_amplitude": das,
    }


def build_sensor_frame(
    sensor_type: str,
    position_or_channel: np.ndarray,
    raw_signal: np.ndarray,
    compensated_signal: np.ndarray,
    quality: float,
) -> dict[str, str | np.ndarray | float]:
    """Store different fibre-sensing mechanisms in one teaching data contract."""
    return {
        "sensor_type": sensor_type,
        "position_or_channel": np.asarray(position_or_channel, dtype=float),
        "raw_signal": np.asarray(raw_signal, dtype=float),
        "compensated_signal": np.asarray(compensated_signal, dtype=float),
        "quality": float(np.clip(quality, 0.0, 1.0)),
    }


def simulate_rayleigh_ofdr(length_mm: float, event_position_mm: float, peak_strain_ue: float, noise_ue: float) -> dict[str, np.ndarray]:
    """Teaching Rayleigh/OFDR spatial strain profile with a local deformation."""
    position = np.linspace(0.0, max(10.0, length_mm), 241)
    strain = peak_strain_ue * np.exp(-.5 * ((position - event_position_mm) / 18.0) ** 2)
    raw = strain + noise_ue * np.sin(position * .37)
    return {"position_mm": position, "strain_ue": strain, "raw_strain_ue": raw}


def simulate_das_event(length_mm: float, event_position_mm: float, frequency_hz: float, sample_rate_hz: int) -> dict[str, np.ndarray]:
    """Teaching phi-OTDR/DAS time-distance response to a local vibration event."""
    position = np.linspace(0.0, max(10.0, length_mm), 121)
    time = np.linspace(-.25, .25, max(40, int(sample_rate_hz)), endpoint=False)
    spatial = np.exp(-.5 * ((position - event_position_mm) / 16.0) ** 2)
    temporal = np.sin(2 * np.pi * frequency_hz * time) * np.exp(-.5 * (time / .11) ** 2)
    return {"position_mm": position, "time_s": time, "amplitude": temporal[:, None] * spatial[None, :]}


def simulate_brillouin_distribution(length_mm: float, temperature_c: float, peak_strain_ue: float) -> dict[str, np.ndarray]:
    """Teaching Brillouin frequency shift from distributed temperature and strain."""
    position = np.linspace(0.0, max(10.0, length_mm), 161)
    temperature = 20.0 + (temperature_c - 20.0) * np.exp(-.5 * ((position - length_mm * .68) / 42.0) ** 2)
    strain = peak_strain_ue * np.exp(-.5 * ((position - length_mm * .35) / 24.0) ** 2)
    frequency = 10.80 + .0011 * (temperature - 20.0) + 5e-5 * strain
    return {"position_mm": position, "temperature_c": temperature, "strain_ue": strain, "brillouin_frequency_ghz": frequency}


def simulate_raman_temperature(length_mm: float, heater_position_mm: float, peak_temperature_c: float) -> dict[str, np.ndarray]:
    """Teaching Raman DTS temperature profile from an anti-Stokes ratio proxy."""
    position = np.linspace(0.0, max(10.0, length_mm), 161)
    temperature = 20.0 + max(0.0, peak_temperature_c - 20.0) * np.exp(-.5 * ((position - heater_position_mm) / 25.0) ** 2)
    ratio = np.exp(-850.0 / (temperature + 273.15))
    return {"position_mm": position, "temperature_c": temperature, "anti_stokes_ratio": ratio}


def simulate_polarization_sensing(transverse_stress_mpa: float, twist_deg: float, temperature_change_c: float) -> dict[str, np.ndarray | float]:
    """Teaching polarimetric sensor: stress and twist rotate a normalized Stokes state."""
    azimuth = np.deg2rad(18.0 + .16 * transverse_stress_mpa + .42 * twist_deg)
    ellipticity = np.deg2rad(np.clip(.12 * temperature_change_c + .03 * twist_deg, -40.0, 40.0))
    stokes = np.array([
        np.cos(2 * azimuth) * np.cos(2 * ellipticity),
        np.sin(2 * azimuth) * np.cos(2 * ellipticity),
        np.sin(2 * ellipticity),
    ])
    return {"stokes": stokes / np.linalg.norm(stokes), "azimuth_deg": float(np.rad2deg(azimuth) % 180.0), "ellipticity_deg": float(np.rad2deg(ellipticity))}


def simulate_sagnac_gyro(angular_rate_deg_s: float, coil_length_m: float) -> dict[str, float]:
    """Teaching Sagnac phase response for a fibre-optic gyro coil of fixed diameter."""
    radius_m = .08
    area_m2 = np.pi * radius_m ** 2
    turns = max(1.0, coil_length_m / (2 * np.pi * radius_m))
    angular_rate_rad_s = np.deg2rad(angular_rate_deg_s)
    phase = abs(8 * np.pi * area_m2 * turns * angular_rate_rad_s / (1.55e-6 * 299_792_458.0))
    return {"phase_shift_rad": float(phase), "estimated_rate_deg_s": float(angular_rate_deg_s)}


def simulate_efpi_pressure(pressure_mpa: float, cavity_um: float) -> dict[str, np.ndarray | float]:
    """Teaching EFPI spectrum from pressure-induced microcavity length change."""
    wavelength_nm = np.linspace(1500.0, 1600.0, 501)
    effective_cavity_um = max(2.0, cavity_um - max(0.0, pressure_mpa) * .35)
    phase = 4 * np.pi * effective_cavity_um * 1e3 / wavelength_nm
    intensity = .5 + .45 * np.cos(phase)
    return {"wavelength_nm": wavelength_nm, "intensity": intensity, "effective_cavity_um": effective_cavity_um}


def fuse_robot_sensing(grasp_verified: bool, balance_quality: float, shape_quality: float, health_status: str, distributed_quality: float) -> dict[str, float | str]:
    """Summarise independent teaching estimators without treating them as a safety controller."""
    score = float(np.mean([float(grasp_verified), balance_quality, shape_quality, distributed_quality]))
    ready = grasp_verified and health_status == "正常" and min(balance_quality, shape_quality, distributed_quality) >= .75
    return {"status": "任务就绪" if ready else "需人工复核", "confidence": score}


_HAND_FINGER_BASES = np.array([
    (.18, -.98, .04), (.92, -.56, .08), (1.04, -.18, .10), (.98, .23, .08), (.84, .59, .02),
])
_HAND_FINGER_LENGTHS = ((1.31, .85), (1.384, 1.0, .72), (1.512, 1.072, .768), (1.36, .976, .688), (1.104, .768, .56))
_HAND_FINGER_RADII = ((.19, .16), (.17, .145, .12), (.17, .145, .12), (.17, .145, .12), (.17, .145, .12))
_HAND_FINGER_SPREADS = (-.83, -.09, -.03, .05, .15)
_CAN_GRASP_CENTER = np.array((.30, -.20, .76))
_CAN_RADIUS = .48
_CAN_HALF_LENGTH = .86


def relative_3d_target_offset(
    target_world_xyz: tuple[float, float, float] | np.ndarray,
    hand_reach_xyz: tuple[float, float, float] | np.ndarray,
) -> np.ndarray:
    """Return the object position in the moving hand's grasp-frame coordinates."""
    target = np.asarray(target_world_xyz, dtype=float)
    reach = np.asarray(hand_reach_xyz, dtype=float)
    if target.shape != (3,) or reach.shape != (3,):
        raise ValueError("目标与手部到达坐标均必须包含 X/Y/Z 三个值")
    return target - reach


def _rotation_y(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(((cosine, 0.0, sine), (0.0, 1.0, 0.0), (-sine, 0.0, cosine)))


def _rotation_z(angle_rad: float) -> np.ndarray:
    cosine, sine = np.cos(angle_rad), np.sin(angle_rad)
    return np.array(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))


def _finger_capsules(finger_index: int, joint_angles_deg: tuple[float, ...]) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """Return the same finger-link capsule axes as the Three.js hand renderer."""
    point = _HAND_FINGER_BASES[finger_index].copy()
    transform = _rotation_z(_HAND_FINGER_SPREADS[finger_index]) @ _rotation_y(-np.deg2rad(joint_angles_deg[0]))
    capsules: list[tuple[np.ndarray, np.ndarray, float]] = []
    for segment_index, length in enumerate(_HAND_FINGER_LENGTHS[finger_index]):
        end = point + transform @ np.array((length, 0.0, 0.0))
        capsules.append((point, end, _HAND_FINGER_RADII[finger_index][segment_index]))
        point = end
        if segment_index + 1 < len(joint_angles_deg):
            transform = transform @ _rotation_y(-np.deg2rad(joint_angles_deg[segment_index + 1]))
    return capsules


def _capsule_cylinder_clearance(start: np.ndarray, end: np.ndarray, radius: float, center: np.ndarray) -> float:
    """Approximate a capsule-to-vertical-cylinder signed clearance by axial samples."""
    samples = np.linspace(0.0, 1.0, 33)[:, None]
    points = start + samples * (end - start)
    radial = np.linalg.norm(points[:, (0, 2)] - center[[0, 2]], axis=1) - (_CAN_RADIUS + radius)
    axial = np.abs(points[:, 1] - center[1]) - _CAN_HALF_LENGTH
    outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
    signed = np.where((radial <= 0.0) & (axial <= 0.0), np.maximum(radial, axial), outside)
    return float(np.min(signed))


def _finger_collision_clearance(finger_index: int, joint_angles_deg: tuple[float, ...], center: np.ndarray) -> float:
    return min(_capsule_cylinder_clearance(start, end, radius, center) for start, end, radius in _finger_capsules(finger_index, joint_angles_deg))


def _collision_limited_finger_angles(finger_index: int, desired_angles: tuple[float, ...], center: np.ndarray) -> tuple[np.ndarray, float]:
    """Stop a finger at first cylinder contact instead of rendering interpenetration."""
    desired = np.asarray(desired_angles, dtype=float)
    if _finger_collision_clearance(finger_index, tuple(desired), center) >= 0.0:
        return desired, _finger_collision_clearance(finger_index, tuple(desired), center)
    if _finger_collision_clearance(finger_index, tuple(np.zeros_like(desired)), center) < 0.0:
        return np.zeros_like(desired), _finger_collision_clearance(finger_index, tuple(np.zeros_like(desired)), center)
    lower, upper = 0.0, 1.0
    for _ in range(28):
        middle = (lower + upper) / 2.0
        if _finger_collision_clearance(finger_index, tuple(desired * middle), center) >= 0.0:
            lower = middle
        else:
            upper = middle
    limited = desired * lower
    return limited, _finger_collision_clearance(finger_index, tuple(limited), center)


def classify_3d_grasp_from_fbg(sensing: dict[str, np.ndarray | list[int] | float | bool], temperature_change_c: float) -> dict[str, np.ndarray | list[int] | float | bool]:
    """Classify holding state from compensated tactile FBG data, not geometry flags."""
    tactile_shift = np.asarray(sensing["tactile_fbg_shifts_nm"], dtype=float)
    arm_shift = np.asarray(sensing["arm_fbg_shifts_nm"], dtype=float)
    if tactile_shift.shape != (15,) or arm_shift.shape != (3,):
        raise ValueError("三维 FBG 判定需要 14 个指节、1 个掌心和 3 个臂部通道")
    tactile_strain = _strain_from_shift(tactile_shift, temperature_change_c)
    finger_touch = tactile_strain[:14] / 2.4e-4
    palm_touch = max(0.0, float(tactile_strain[14] / 1.6e-4))
    counts = (2, 3, 3, 3, 3)
    index = 0
    finger_force = []
    for count in counts:
        finger_force.append(float(np.sum(np.maximum(finger_touch[index : index + count], 0.0))))
        index += count
    finger_force_array = np.asarray(finger_force)
    contacts = np.flatnonzero(finger_force_array >= .12).astype(int).tolist()
    arm_strain_ue = _strain_from_shift(arm_shift, temperature_change_c) * 1.0e6
    is_grasped = bool(palm_touch >= .20 and 0 in contacts and len([item for item in contacts if item != 0]) >= 2)
    return {
        "is_grasped": is_grasped,
        "contact_fingers": contacts,
        "finger_touch_n": finger_force_array,
        "palm_touch_n": palm_touch,
        "arm_bend_strain_ue": arm_strain_ue,
    }


def evaluate_3d_grasp_sensing(
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    can_offset_xyz: tuple[float, float, float] | np.ndarray,
    temperature_change_c: float = 0.0,
    arm_joint_angles_deg: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    finger_joint_angles_deg: tuple[tuple[float, ...], ...] | None = None,
) -> dict[str, np.ndarray | list[int] | float | bool]:
    """Estimate the 3D grasp state and separated arm/tactile fibre readings.

    ``can_offset_xyz`` is the can displacement from the 3D hand's nominal grasp
    envelope.  This teaching model deliberately has no dependency on the 2D pose
    or its contact state.
    """
    curls = np.asarray(finger_curls_deg, dtype=float)
    offset = np.asarray(can_offset_xyz, dtype=float)
    if curls.shape != (5,):
        raise ValueError("finger_curls_deg 必须包含五根手指的屈曲角")
    if offset.shape != (3,):
        raise ValueError("can_offset_xyz 必须包含三维位置")
    arm_angles = np.asarray(arm_joint_angles_deg, dtype=float)
    if arm_angles.shape != (3,):
        raise ValueError("arm_joint_angles_deg 必须包含肩、肘、腕三个角")

    joint_counts = (2, 3, 3, 3, 3)
    if finger_joint_angles_deg is None:
        # 用符合握持姿态的关节分配还原五指，而不是把总屈曲平均摊到
        # 每一节；后者会令中节无法触到位于掌心上方的圆柱。
        finger_joint_angles_deg = (
            (curls[0], curls[0]),
            *((curl * .85, curl * 1.25, curl * .90) for curl in curls[1:]),
        )
    if len(finger_joint_angles_deg) != 5 or any(len(angles) != count for angles, count in zip(finger_joint_angles_deg, joint_counts)):
        raise ValueError("finger_joint_angles_deg 必须包含 14 个手指关节")
    can_center = _CAN_GRASP_CENTER + offset
    limited_results = [_collision_limited_finger_angles(index, tuple(map(float, angles)), can_center) for index, angles in enumerate(finger_joint_angles_deg)]
    limited_angles = tuple(tuple(float(value) for value in angles) for angles, _ in limited_results)
    collision_clearance = np.asarray([clearance for _, clearance in limited_results])
    commanded_bend = np.asarray([np.mean(angles) for angles in finger_joint_angles_deg])
    limited_bend = np.asarray([np.mean(angles) for angles in limited_angles])
    contact_force_n = np.maximum(0.0, commanded_bend - limited_bend) * .045
    contact_fingers = np.flatnonzero(contact_force_n > .08).astype(int).tolist()
    non_thumb_contacts = [index for index in contact_fingers if index != 0]
    palm_support = bool(
        -1.12 <= can_center[0] <= 1.04
        and -.98 <= can_center[1] <= 1.00
        and .24 <= can_center[2] - _CAN_RADIUS <= .36
    )
    is_grasped = bool(palm_support and 0 in contact_fingers and len(non_thumb_contacts) >= 2)
    stability = min(1.0, .12 * len(contact_fingers) + .08 * float(np.minimum(contact_force_n, 1.0).sum()) + (.25 if palm_support else 0.0))
    tip_distances = collision_clearance
    # 五路读数保留为每根手指的综合弯曲/接触通道；分区通道在下方单独
    # 给出，避免把手臂弯曲误解释成触觉。
    strain = curls * 4.0e-5 + contact_force_n * 1.7e-4
    segment_angles = np.concatenate([np.asarray(angles, dtype=float) for angles in finger_joint_angles_deg])
    finger_segment_touch = np.repeat(contact_force_n, joint_counts) / np.repeat(joint_counts, joint_counts)
    palm_touch = float(contact_force_n.sum() * (.34 if is_grasped else .08))
    arm_bend_strain = np.abs(arm_angles) * np.array([3.2, 4.1, 2.6])
    tactile_strain = np.r_[finger_segment_touch * 2.4e-4, palm_touch * 1.6e-4]
    arm_strain = arm_bend_strain * 1.0e-6
    return {
        "tip_distances": tip_distances,
        "contact_force_n": contact_force_n,
        "contact_fingers": contact_fingers,
        "stability": stability,
        "is_grasped": is_grasped,
        "fbg_shifts_nm": fbg_wavelength_shift_nm(strain, temperature_change_c),
        "finger_segment_bend_deg": segment_angles,
        "finger_segment_touch_n": finger_segment_touch,
        "palm_touch_n": palm_touch,
        "arm_bend_strain_ue": arm_bend_strain,
        "tactile_fbg_shifts_nm": fbg_wavelength_shift_nm(tactile_strain, temperature_change_c),
        "arm_fbg_shifts_nm": fbg_wavelength_shift_nm(arm_strain, temperature_change_c),
        "can_center_xyz": can_center,
        "collision_limited_joint_angles_deg": limited_angles,
        "collision_clearance": collision_clearance,
        "palm_support_contact": palm_support,
    }
