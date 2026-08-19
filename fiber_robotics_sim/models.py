"""与界面无关的光纤传感正向与反演模型。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def parse_fbg_simplus_comsol_export(
    text: str, delimiter: str = "自动识别", skip_rows: int = 0
) -> dict[str, np.ndarray | int | str]:
    """Parse a generic FBG-SimPlus-compatible eight-column text data export.

    Accept whitespace, CSV, or tab-separated text.  This independent reader
    does not import, execute, reproduce, or modify FBG-SimPlus, and it stops
    before any reflection-spectrum calculation.
    """
    separators = {"空白字符": None, "逗号（CSV）": ",", "制表符（TSV）": "\t"}
    if delimiter not in {"自动识别", *separators}:
        raise ValueError("不支持的分隔符选项")
    if skip_rows < 0:
        raise ValueError("跳过行数不能为负数")
    source_lines = text.splitlines()[skip_rows:]
    data_lines = [line.strip() for line in source_lines if line.strip() and not line.lstrip().startswith(("%", "#"))]
    if not data_lines:
        raise ValueError("FBG-SimPlus 兼容输入未包含数据行")
    selected_delimiter = delimiter
    if delimiter == "自动识别":
        first_line = data_lines[0]
        selected_delimiter = "逗号（CSV）" if "," in first_line else "制表符（TSV）" if "\t" in first_line else "空白字符"

    rows: list[list[float]] = []
    separator = separators[selected_delimiter]
    for line in data_lines:
        fields = line.split() if separator is None else [field.strip() for field in line.split(separator)]
        if len(fields) != 8:
            raise ValueError("FBG-SimPlus 兼容输入的每个数据行必须包含八列")
        try:
            rows.append([float(field) for field in fields])
        except ValueError as error:
            raise ValueError("FBG-SimPlus 兼容输入包含无法读取的数值") from error
    values = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("FBG-SimPlus 兼容输入不能包含 NaN 或无穷值")
    if np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError("FBG-SimPlus 兼容输入的位置列必须严格递增")
    return {
        "position_m": values[:, 0],
        "longitudinal_strain": values[:, 1],
        "transverse_stress_pa": values[:, 5:7],
        "temperature_k": values[:, 7],
        "input_values": values,
        "sample_count": int(values.shape[0]),
        "source_delimiter": selected_delimiter,
    }


def fbg_simplus_normalised_text(result: dict[str, np.ndarray | int | str]) -> str:
    """Write validated data as the whitespace-separated eight-column input FBG-SimPlus reads."""
    values = np.asarray(result["input_values"], dtype=float)
    return "\n".join(" ".join(f"{value:.12g}" for value in row) for row in values) + "\n"

def next_grasp_task_phase(current_phase: str, grasp_verified: bool) -> str:
    """Advance one observable grasp-task step shared by the 2D and 3D pages.

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


def simulate_material_touch(
    material: str,
    grip_force_n: float,
    contact_area_percent: float,
    temperature_change_c: float,
    pattern_noise: float = 0.0,
    noise_nm: float = 0.0,
    seed: int = 7,
) -> dict[str, np.ndarray | float]:
    """Generate five finger and one palm contact forces for four teaching materials.

    ``pattern_noise`` perturbs the normalised contact pattern so the classifier
    no longer recovers the material by construction, and ``noise_nm`` adds
    wavelength measurement noise like the other modules.
    """
    if material not in _TACTILE_PROFILES:
        raise ValueError("未知触觉材料")
    if pattern_noise < 0.0 or noise_nm < 0.0:
        raise ValueError("接触模式扰动与波长噪声不能为负")
    profile = _TACTILE_PROFILES[material]
    area_gain = .45 + .55 * np.clip(contact_area_percent, 0.0, 100.0) / 100.0
    touch = profile * max(0.0, grip_force_n) * area_gain
    if pattern_noise > 0.0:
        generator = np.random.default_rng(int(seed))
        touch = np.maximum(touch * (1.0 + generator.normal(0.0, pattern_noise, touch.shape)), 0.0)
    clean_shifts = fbg_wavelength_shift_nm(touch * TACTILE_TOUCH_STRAIN_PER_N, temperature_change_c)
    shifts = add_gaussian_noise(clean_shifts, noise_nm, int(seed) + 1)
    return {
        "finger_touch_n": touch[:5],
        "palm_touch_n": float(touch[5]),
        "wavelength_shifts_nm": shifts,
        "clean_wavelength_shifts_nm": clean_shifts,
        "pattern_noise": float(pattern_noise),
    }


def classify_tactile_material(finger_touch_n: np.ndarray, palm_touch_n: float) -> dict[str, float | str]:
    """Classify a teaching material from normalized five-finger and palm contact pattern."""
    observed = np.r_[np.asarray(finger_touch_n, dtype=float), float(palm_touch_n)]
    if observed.shape != (6,) or np.allclose(observed, 0.0):
        return {
            "material": "未接触",
            "confidence": 0.0,
            "probabilities": {name: 0.0 for name in _TACTILE_PROFILES},
        }
    normalized = observed / np.linalg.norm(observed)
    scores = {name: float(np.dot(normalized, profile / np.linalg.norm(profile))) for name, profile in _TACTILE_PROFILES.items()}
    material = max(scores, key=scores.get)
    exp_scores = {name: float(np.exp(score)) for name, score in scores.items()}
    total = sum(exp_scores.values())
    probabilities = {name: exp_scores[name] / total for name in _TACTILE_PROFILES}
    return {
        "material": material,
        "confidence": scores[material],
        "probabilities": probabilities,
    }


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


@dataclass(frozen=True)
class GraspCalibration:
    """Per-channel FBG sensitivities shared by the 2D and 3D grasp models.

    Defaults reproduce the teaching constants; a custom instance lets a lesson
    change one channel's sensitivity or decision threshold without touching the
    model math.
    """

    bend_strain_per_deg: float = 4.0e-5
    contact_strain_per_n: float = 1.7e-4
    finger_touch_strain_per_n: float = 2.4e-4
    palm_touch_strain_per_n: float = 1.6e-4
    contact_force_threshold_n: float = 0.12
    palm_contact_threshold_n: float = 0.20


PLANAR_GRASP_CALIBRATION = GraspCalibration(
    contact_force_threshold_n=0.10,
    palm_contact_threshold_n=0.08,
)

GRASP_FORCE_PER_DEGREE = 0.045
GRASP_PALM_ACTIVE_FACTOR = 0.34
GRASP_PALM_PASSIVE_FACTOR = 0.08
GRASP_STABILITY_CONTACT_WEIGHT = 0.12
GRASP_STABILITY_FORCE_WEIGHT = 0.05
GRASP_STABILITY_FORCE_WEIGHT_3D = 0.08
GRASP_STABILITY_PALM_BONUS = 0.25
TACTILE_TOUCH_STRAIN_PER_N = 4.0e-5
FOOT_SWING_LOAD_RATIO = 0.03


def simulate_planar_grasp_fbg(
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    contact_force_n: tuple[float, float, float, float, float] | np.ndarray,
    temperature_change_c: float,
    calibration: GraspCalibration = PLANAR_GRASP_CALIBRATION,
) -> dict[str, np.ndarray | float]:
    """Generate five finger channels plus one palm FBG channel from contact forces."""
    curls = np.asarray(finger_curls_deg, dtype=float)
    forces = np.maximum(np.asarray(contact_force_n, dtype=float), 0.0)
    if curls.shape != (5,) or forces.shape != (5,):
        raise ValueError("二维 FBG 需要五根手指的屈曲角与接触力")
    bend_strain = curls * calibration.bend_strain_per_deg
    contact_strain = forces * calibration.contact_strain_per_n
    finger_contacts = np.flatnonzero(forces >= calibration.contact_force_threshold_n).astype(int).tolist()
    palm_active = bool(0 in finger_contacts and len([index for index in finger_contacts if index != 0]) >= 2)
    palm_touch_n = float(forces.sum() * (GRASP_PALM_ACTIVE_FACTOR if palm_active else GRASP_PALM_PASSIVE_FACTOR))
    palm_contact_strain = palm_touch_n * calibration.palm_touch_strain_per_n
    return {
        "wavelength_shifts_nm": fbg_wavelength_shift_nm(
            np.r_[bend_strain + contact_strain, palm_contact_strain], temperature_change_c
        ),
        "contact_force_n": forces,
        "contact_strain": contact_strain,
        "palm_touch_n": palm_touch_n,
        "palm_contact_strain": np.array([palm_contact_strain]),
    }


def classify_planar_grasp_from_fbg(
    sensing: dict[str, np.ndarray],
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    temperature_change_c: float,
    calibration: GraspCalibration = PLANAR_GRASP_CALIBRATION,
) -> dict[str, np.ndarray | list[int] | bool | float]:
    """Recover per-finger contact forces from FBG readings and decide a planar grasp."""
    curls = np.asarray(finger_curls_deg, dtype=float)
    shifts = np.asarray(sensing["wavelength_shifts_nm"], dtype=float)
    if curls.shape != (5,) or shifts.shape != (6,):
        raise ValueError("二维 FBG 判定需要五根手指和一枚掌心 FBG 的波长读数")
    strain = _strain_from_shift(shifts, temperature_change_c)
    contact_strain = np.maximum(0.0, strain[:5] - curls * calibration.bend_strain_per_deg)
    contact_force_n = contact_strain / calibration.contact_strain_per_n
    contact_fingers = np.flatnonzero(contact_force_n >= calibration.contact_force_threshold_n).astype(int).tolist()
    palm_touch_n = max(0.0, float(strain[5])) / calibration.palm_touch_strain_per_n
    is_grasped = bool(0 in contact_fingers and len([index for index in contact_fingers if index != 0]) >= 2)
    return {
        "is_grasped": is_grasped,
        "contact_fingers": contact_fingers,
        "contact_force_n": contact_force_n,
        "contact_strain": contact_strain,
        "palm_touch_n": palm_touch_n,
        "palm_contact": bool(palm_touch_n >= calibration.palm_contact_threshold_n),
    }


_PLANAR_CAN_HALF_WIDTH = .24
_PLANAR_CAN_HALF_HEIGHT = .37
_PLANAR_FINGER_CONTACT_MARGIN = .08
# 几何接触判定与 FBG 判定共用同一阈值，避免两处数字漂移。
_PLANAR_CONTACT_FORCE_THRESHOLD_N = PLANAR_GRASP_CALIBRATION.contact_force_threshold_n


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


def _planar_finger_polyline(
    base: np.ndarray,
    lengths: tuple[float, ...],
    start_direction_rad: float,
    curl_deg: float,
    joints: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Return the same planar finger polyline as the 2D renderer for one curl value."""
    points = [np.asarray(base, dtype=float)]
    direction = float(start_direction_rad)
    if joints is None:
        joints = (0.0, curl_deg, curl_deg)
    for length, joint_angle in zip(lengths, joints):
        direction += np.deg2rad(joint_angle)
        points.append(points[-1] + length * np.array([np.cos(direction), np.sin(direction)]))
    return np.asarray(points)


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
    palm_fiber_route = np.vstack([
        palm_center - .33 * palm_length * forward - .18 * palm_width * lateral,
        palm_center + .08 * palm_length * forward,
        palm_center + .33 * palm_length * forward + .18 * palm_width * lateral,
    ])
    curl_by_action = {"抬臂": 8.0, "伸手": 4.0, "抓取": 84.0, "按压": 62.0, "松开": 18.0, "复位": 12.0}
    curl = curl_by_action.get(action, 12.0)
    curls = finger_curls_deg if finger_curls_deg is not None else (curl * .75, curl, curl, curl, curl)
    finger_offsets = (-.275, -.10, .10, .275)
    finger_lengths = ((.62, .48, .36), (.70, .52, .40), (.66, .50, .38), (.58, .43, .32))
    fingers: list[np.ndarray] = []
    fiber_routes: list[np.ndarray] = []
    finger_builds: list[dict[str, object]] = []
    for index, (offset, lengths) in enumerate(zip(finger_offsets, finger_lengths)):
        base = palm_center + .50 * palm_length * forward + offset * lateral
        splay = np.deg2rad(offset * 18.0)
        local_curl = 88.0 if action == "按压" and index == 1 and finger_curls_deg is None else curls[index + 1]
        start_direction = np.arctan2(forward[1], forward[0]) + splay
        # 四指从 MCP 开始屈曲，使闭合时指尖回到掌心包络，而不是先直伸再弯。
        finger_joints = (local_curl, local_curl, local_curl)
        finger = _planar_finger_polyline(base, lengths, start_direction, local_curl, finger_joints)
        fingers.append(finger)
        fiber_routes.append(finger + .055 * lateral)
        finger_builds.append({
            "base": base,
            "lengths": lengths,
            "start_direction_rad": float(start_direction),
            "commanded_curl_deg": float(local_curl),
            "joints": finger_joints,
        })
    thumb_base = palm_center + .10 * palm_length * forward - .60 * palm_width * lateral
    thumb_direction = np.arctan2(forward[1], forward[0]) + np.deg2rad(50.0)
    thumb_curl = curls[0]
    thumb_joints = (0.0, thumb_curl, thumb_curl)
    thumb = _planar_finger_polyline(thumb_base, (.55, .42, .30), thumb_direction, thumb_curl, thumb_joints)
    fingers.insert(0, thumb)
    fiber_routes.insert(0, thumb - .055 * lateral)
    finger_builds.insert(0, {
        "base": thumb_base,
        "lengths": (.55, .42, .30),
        "start_direction_rad": float(thumb_direction),
        "commanded_curl_deg": float(thumb_curl),
        "joints": thumb_joints,
    })
    # 目标位于拇指与四根手指的闭合包络中心，闭合时由拇指与多根手指真实接触。
    target = palm_center + .05 * forward + .35 * lateral
    return {
        "arm_joints": arm_joints,
        "palm_center": palm_center,
        "palm_outline": palm_outline,
        "palm_fiber_route": palm_fiber_route,
        "fingers": fingers,
        "fiber_routes": fiber_routes,
        "finger_builds": finger_builds,
        "target": target,
        "finger_curls_deg": np.asarray(curls, dtype=float),
        "wrist_rotation_deg": float(wrist_rotation_deg),
    }


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


def _segment_rectangle_distance(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    half_width: float,
    half_height: float,
    samples: int = 21,
) -> float:
    """Approximate segment-to-axis-aligned-rectangle distance by axial samples."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    samples_axis = np.linspace(0.0, 1.0, samples)[:, None]
    points = start + samples_axis * (end - start)
    dx = np.maximum(np.abs(points[:, 0] - center[0]) - half_width, 0.0)
    dy = np.maximum(np.abs(points[:, 1] - center[1]) - half_height, 0.0)
    return float(np.min(np.hypot(dx, dy)))


def _planar_finger_clearance(build: dict, joints: np.ndarray, can_center: np.ndarray) -> float:
    """Signed clearance between a finger polyline and the can, minus the contact margin."""
    points = _planar_finger_polyline(
        np.asarray(build["base"], dtype=float),
        tuple(build["lengths"]),
        float(build["start_direction_rad"]),
        float(joints[0]),
        tuple(float(value) for value in joints),
    )
    return min(
        _segment_rectangle_distance(a, b, can_center, _PLANAR_CAN_HALF_WIDTH, _PLANAR_CAN_HALF_HEIGHT)
        - _PLANAR_FINGER_CONTACT_MARGIN
        for a, b in zip(points[:-1], points[1:])
    )


def _contact_limited_curl(build: dict, can_center: np.ndarray) -> float:
    """Stop a finger at first can contact instead of letting it penetrate the rectangle."""
    commanded_curl_deg = float(build["commanded_curl_deg"])
    joints = np.asarray(build["joints"], dtype=float)
    if _planar_finger_clearance(build, joints, can_center) >= 0.0:
        return commanded_curl_deg
    if _planar_finger_clearance(build, np.zeros_like(joints), can_center) < 0.0:
        return 0.0
    lower, upper = 0.0, 1.0
    for _ in range(28):
        middle = (lower + upper) / 2.0
        if _planar_finger_clearance(build, joints * middle, can_center) >= 0.0:
            lower = middle
        else:
            upper = middle
    return commanded_curl_deg * lower


def evaluate_can_grasp(pose: dict, can_center: np.ndarray) -> dict:
    """Estimate per-finger contact forces by stopping each finger at the can boundary."""
    can_center = np.asarray(can_center, dtype=float)
    builds = pose["finger_builds"]
    commanded_curls = np.asarray([float(build["commanded_curl_deg"]) for build in builds])
    limited_curls = np.asarray([
        _contact_limited_curl(build, can_center)
        for build in builds
    ])
    contact_force_n = np.maximum(0.0, commanded_curls - limited_curls) * GRASP_FORCE_PER_DEGREE
    contact_fingers = np.flatnonzero(contact_force_n >= _PLANAR_CONTACT_FORCE_THRESHOLD_N).astype(int).tolist()
    non_thumb_contacts = [index for index in contact_fingers if index != 0]
    is_grasped = bool(0 in contact_fingers and len(non_thumb_contacts) >= 2)
    stability = min(
        1.0,
        GRASP_STABILITY_CONTACT_WEIGHT * len(contact_fingers)
        + GRASP_STABILITY_FORCE_WEIGHT * float(min(contact_force_n.sum(), 1.0)),
    )
    tip_distances = np.asarray([np.linalg.norm(np.asarray(finger)[-1] - can_center) for finger in pose["fingers"]])
    return {
        "contact_fingers": contact_fingers,
        "contact_force_n": contact_force_n,
        "limited_curls_deg": limited_curls,
        "tip_distances": tip_distances,
        "stability": stability,
        "is_grasped": is_grasped,
    }


def can_offset_from_target(pose: dict, can_center: np.ndarray) -> np.ndarray:
    """Express the can displacement from the default target in hand-local axes."""
    joints = np.asarray(pose["arm_joints"], dtype=float)
    forward = joints[3] - joints[2]
    forward /= np.linalg.norm(forward)
    lateral = np.array([-forward[1], forward[0]])
    displacement = np.asarray(can_center, dtype=float) - np.asarray(pose["target"], dtype=float)
    return np.array([np.dot(displacement, forward), np.dot(displacement, lateral)])


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


FOOT_LOAD_STRAIN_PER_N = 4.0e-6
SOLE_NOMINAL_PRELOAD_UE = 240.0
SOLE_MEAN_RESIDUAL_LIMIT_UE = 45.0
SOLE_LEFT_RIGHT_DIFFERENCE_LIMIT_UE = 70.0

_FOOT_TERRAIN_WEIGHTS = {
    "平地": np.ones(6),
    "前倾坡面": np.array([1.3, 1.2, 1.1, .8, .7, .6]),
    "后倾坡面": np.array([.6, .7, .8, 1.1, 1.2, 1.3]),
    "柔软地面": np.array([.9, 1.0, .9, 1.0, .9, 1.0]),
}


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
    strains = loads * FOOT_LOAD_STRAIN_PER_N
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
    loads = np.maximum(0.0, _strain_from_shift(shifts, temperature_change_c) / FOOT_LOAD_STRAIN_PER_N)
    total = max(float(loads.sum()), 1e-9)
    zone_centers_x = (np.arange(6) % 3) + 0.45
    zone_centers_y = (1 - np.arange(6) // 3) + 0.42
    return {
        "zone_loads_n": loads,
        "cop_region": float(np.dot(np.arange(6), loads) / total),
        "cop_xy": np.array([
            float(np.dot(zone_centers_x, loads) / total),
            float(np.dot(zone_centers_y, loads) / total),
        ]),
    }


def simulate_foot_zone_loads(
    load_n: float, terrain: str, phase_percent: float, support: str
) -> np.ndarray:
    """Distribute the vertical load over six zones with terrain and phase envelopes.

    The stance-phase envelope moves weight from the heel row (zones 3-5) at 0%
    to the forefoot row (zones 0-2) at 100%, so dragging the phase control
    changes the loading pattern and the resulting CoP.
    """
    if terrain not in _FOOT_TERRAIN_WEIGHTS:
        raise ValueError("未知地形")
    if load_n < 0.0 or not 0.0 <= phase_percent <= 100.0:
        raise ValueError("垂直载荷不能为负，步态相位必须在 0 到 100 之间")
    weights = _FOOT_TERRAIN_WEIGHTS[terrain].copy()
    stance_rad = np.pi * float(phase_percent) / 100.0
    forefoot_envelope = 0.5 - 0.5 * np.cos(stance_rad)
    heel_envelope = 0.5 + 0.5 * np.cos(stance_rad)
    weights[:3] *= forefoot_envelope
    weights[3:] *= heel_envelope
    if weights.sum() <= 0.0:
        weights = _FOOT_TERRAIN_WEIGHTS[terrain].copy()
    zones = load_n * weights / weights.sum()
    if support == "摆动期":
        zones = zones * FOOT_SWING_LOAD_RATIO
    return zones


_REPLACEABLE_SOLE_CASES = {
    "正常装配": {"seating_ratio": 1.0, "lateral_offset_mm": 0.0, "insertion_deficit_mm": 0.0},
    "压入不足": {"seating_ratio": 0.60, "lateral_offset_mm": 0.0, "insertion_deficit_mm": 1.2},
    "单侧错位": {"seating_ratio": 1.0, "lateral_offset_mm": 4.0, "insertion_deficit_mm": 0.0},
}


def _classify_assembly(mean_baseline_residual_ue: float, left_right_difference_ue: float) -> str:
    if abs(mean_baseline_residual_ue) > SOLE_MEAN_RESIDUAL_LIMIT_UE:
        return "压入不足：预测不通过"
    if left_right_difference_ue > SOLE_LEFT_RIGHT_DIFFERENCE_LIMIT_UE:
        return "单侧错位：预测不通过"
    return "装配预测通过"


def _assembly_readout(
    seating_ratio: float,
    lateral_offset_mm: float,
    temperature_change_c: float,
    working_noise_ue: np.ndarray | None = None,
    reference_temperature_change_c: float | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Return one assembly readout where the reference FBG estimates temperature.

    The working temperature is not injected directly: the mechanically isolated
    reference FBG supplies the thermal estimate used to compensate the two
    working gratings, so a mismatched reference produces a baseline bias inside
    the same flow instead of in a separate diagnostic.
    """
    if reference_temperature_change_c is None:
        reference_temperature_change_c = temperature_change_c
    working_strain_ue = SOLE_NOMINAL_PRELOAD_UE * seating_ratio * np.array(
        [1.0 - lateral_offset_mm / 12.0, 1.0 + lateral_offset_mm / 12.0]
    )
    if working_noise_ue is not None:
        working_strain_ue = working_strain_ue + np.asarray(working_noise_ue, dtype=float)
    working_shifts_nm = fbg_wavelength_shift_nm(working_strain_ue * 1e-6, temperature_change_c)
    reference_shift_nm = float(fbg_wavelength_shift_nm(np.array([0.0]), reference_temperature_change_c)[0])
    estimated_temperature_change_c = float(reference_shift_nm / (WAVELENGTH_NM * THERMAL_SENSITIVITY_PER_C))
    compensated_working_ue = _strain_from_shift(working_shifts_nm, estimated_temperature_change_c) * 1e6
    compensated_reference_ue = float(
        _strain_from_shift(np.array([reference_shift_nm]), estimated_temperature_change_c)[0] * 1e6
    )
    baseline_residual_ue = compensated_working_ue - compensated_reference_ue - SOLE_NOMINAL_PRELOAD_UE
    mean_baseline_residual_ue = float(np.mean(baseline_residual_ue))
    left_right_difference_ue = float(abs(np.diff(compensated_working_ue)[0]))
    return {
        "working_wavelength_shifts_nm": working_shifts_nm,
        "reference_wavelength_shift_nm": reference_shift_nm,
        "estimated_temperature_change_c": estimated_temperature_change_c,
        "temperature_compensated_working_strain_ue": compensated_working_ue,
        "temperature_compensated_reference_strain_ue": compensated_reference_ue,
        "baseline_residual_ue": baseline_residual_ue,
        "mean_baseline_residual_ue": mean_baseline_residual_ue,
        "left_right_difference_ue": left_right_difference_ue,
        "assembly_prediction": _classify_assembly(
            mean_baseline_residual_ue, left_right_difference_ue
        ),
    }


def simulate_replaceable_sole_assembly(
    assembly_case: str, temperature_change_c: float
) -> dict[str, np.ndarray | float | str | dict[str, float]]:
    """Predict an empty-load assembly check for a replaceable sole concept.

    This is a two-dimensional, parameterised transmission-field model.  It
    compares two fixed-core working FBGs with one mechanically isolated
    reference FBG after common temperature compensation.  Its parameter values
    and screening limits are illustrative simulation inputs, not product
    specifications, material data, sealing ratings, or physical validation.
    """
    if assembly_case not in _REPLACEABLE_SOLE_CASES:
        raise ValueError("装配工况必须为：正常装配、压入不足或单侧错位")
    case = _REPLACEABLE_SOLE_CASES[assembly_case]
    lateral_offset_mm = float(case["lateral_offset_mm"])
    seating_ratio = float(case["seating_ratio"])
    readout = _assembly_readout(seating_ratio, lateral_offset_mm, temperature_change_c)

    transfer_field = solve_sole_transfer_sensitivity_field(seating_ratio, lateral_offset_mm)
    return {
        "assembly_case": assembly_case,
        **readout,
        "transfer_x_mm": transfer_field["x_mm"],
        "transfer_y_mm": transfer_field["y_mm"],
        "transfer_index": transfer_field["relative_transfer"],
        "transfer_centroid_x_mm": transfer_field["transfer_centroid_x_mm"],
        "case_parameters": {
            "nominal_preload_ue": SOLE_NOMINAL_PRELOAD_UE,
            "seating_ratio": seating_ratio,
            "lateral_offset_mm": lateral_offset_mm,
            "insertion_deficit_mm": float(case["insertion_deficit_mm"]),
            "mean_residual_limit_ue": SOLE_MEAN_RESIDUAL_LIMIT_UE,
            "left_right_difference_limit_ue": SOLE_LEFT_RIGHT_DIFFERENCE_LIMIT_UE,
        },
    }


def solve_sole_transfer_sensitivity_field(
    seating_ratio: float, lateral_offset_mm: float
) -> dict[str, np.ndarray | float | str]:
    """Solve a relative 2D transmission sensitivity field on a fixed grid.

    A screened finite-difference field is used to make the assumed boundary,
    source position and lateral response explicit without introducing an FEM
    package or claiming calibrated material mechanics.  It must be replaced or
    checked with a material-calibrated finite-element model before design use.
    """
    x_mm = np.linspace(-30.0, 30.0, 121)
    y_mm = np.linspace(-45.0, 45.0, 181)
    x_grid, y_grid = np.meshgrid(x_mm, y_mm)
    source = max(0.0, seating_ratio) * np.exp(
        -0.5 * ((x_grid - lateral_offset_mm) / 9.0) ** 2
        -0.5 * (y_grid / 18.0) ** 2
    )
    spacing_mm = float(x_mm[1] - x_mm[0])
    screening_per_mm2 = 0.018
    transfer = np.zeros_like(source)
    denominator = 4.0 + screening_per_mm2 * spacing_mm**2
    for _ in range(2000):
        next_transfer = transfer.copy()
        next_transfer[1:-1, 1:-1] = (
            transfer[:-2, 1:-1]
            + transfer[2:, 1:-1]
            + transfer[1:-1, :-2]
            + transfer[1:-1, 2:]
            + spacing_mm**2 * source[1:-1, 1:-1]
        ) / denominator
        change = float(np.max(np.abs(next_transfer - transfer)))
        transfer = next_transfer
        if change < 1e-8:
            break
    relative_transfer = transfer / max(float(transfer.max()), 1e-12)
    total_transfer = float(relative_transfer.sum())
    transfer_centroid_x_mm = float(
        np.sum(relative_transfer * x_grid) / max(total_transfer, 1e-12)
    )
    return {
        "x_mm": x_mm,
        "y_mm": y_mm,
        "relative_transfer": relative_transfer,
        "transfer_centroid_x_mm": transfer_centroid_x_mm,
        "validation_boundary": "需要材料参数标定与有限元复核",
    }


def simulate_replaceable_sole_tolerance_scan(
    samples_per_case: int,
    temperature_change_c: float,
    measurement_noise_nm: float,
    seed: int,
) -> dict[str, np.ndarray | tuple[str, str, str] | int]:
    """Run a reproducible tolerance-screening Monte Carlo using assumed spreads.

    The variation ranges are sensitivity-study inputs, not measured production
    tolerances.  The output shows only how the current decision rule behaves
    under those inputs.
    """
    if samples_per_case < 1 or measurement_noise_nm < 0.0:
        raise ValueError("每工况样本数必须为正，测量噪声不能为负")
    labels = ("正常装配", "压入不足", "单侧错位")
    expected = {
        "正常装配": (1.00, 0.03, 0.00, 0.35),
        "压入不足": (0.60, 0.05, 0.00, 0.35),
        "单侧错位": (1.00, 0.03, 4.00, 0.45),
    }
    prediction_index = {"装配预测通过": 0, "压入不足：预测不通过": 1, "单侧错位：预测不通过": 2}
    strain_noise_ue = measurement_noise_nm / (WAVELENGTH_NM * (1.0 - PHOTOELASTIC_COEFFICIENT)) * 1e6
    generator = np.random.default_rng(seed)
    confusion_matrix = np.zeros((3, 3), dtype=int)
    for true_index, case_name in enumerate(labels):
        seating_mean, seating_std, offset_mean, offset_std = expected[case_name]
        for _ in range(samples_per_case):
            seating_ratio = max(0.10, generator.normal(seating_mean, seating_std))
            lateral_offset_mm = generator.normal(offset_mean, offset_std)
            readout = _assembly_readout(
                seating_ratio,
                lateral_offset_mm,
                temperature_change_c,
                generator.normal(0.0, strain_noise_ue, 2),
            )
            confusion_matrix[true_index, prediction_index[str(readout["assembly_prediction"])]] += 1
    return {
        "labels": labels,
        "confusion_matrix": confusion_matrix,
        "samples_per_case": int(samples_per_case),
    }


def simulate_reference_temperature_mismatch(
    working_temperature_change_c: float, reference_temperature_change_c: float
) -> dict[str, float | str]:
    """Quantify the false baseline offset when reference and working FBGs differ in temperature."""
    readout = _assembly_readout(
        1.0,
        0.0,
        working_temperature_change_c,
        reference_temperature_change_c=reference_temperature_change_c,
    )
    return {
        "working_temperature_change_c": float(working_temperature_change_c),
        "reference_temperature_change_c": float(reference_temperature_change_c),
        "baseline_bias_ue": float(readout["mean_baseline_residual_ue"]),
        "validation_boundary": "需要温度梯度试验",
    }


def simulate_seal_compression_screen(
    nominal_compression_ratio: float, lateral_offset_mm: float
) -> dict[str, np.ndarray | float | str]:
    """Screen relative circumferential compression variation; it does not predict sealing performance."""
    if not 0.0 <= nominal_compression_ratio <= 1.0:
        raise ValueError("名义压缩率必须在 0 到 1 之间")
    angle_deg = np.linspace(0.0, 360.0, 181)
    compression_ratio = nominal_compression_ratio + lateral_offset_mm / 20.0 * np.cos(np.deg2rad(angle_deg))
    minimum_index = int(np.argmin(compression_ratio))
    return {
        "angle_deg": angle_deg,
        "compression_ratio": compression_ratio,
        "minimum_compression_ratio": float(compression_ratio[minimum_index]),
        "minimum_compression_angle_deg": float(angle_deg[minimum_index]),
        "validation_boundary": "需要密封实物试验",
    }


def simulate_preload_retention_sensitivity(
    maximum_cycles: int, assumed_retention_per_1000_cycles: float
) -> dict[str, np.ndarray | str]:
    """Generate an assumed preload-retention curve for test planning, not fatigue prediction."""
    if maximum_cycles < 0 or not 0.0 < assumed_retention_per_1000_cycles <= 1.0:
        raise ValueError("循环次数不能为负，千次保持率必须在 0 与 1 之间")
    cycle_count = np.linspace(0.0, float(maximum_cycles), 101)
    preload_ue = SOLE_NOMINAL_PRELOAD_UE * assumed_retention_per_1000_cycles ** (cycle_count / 1000.0)
    return {
        "cycle_count": cycle_count,
        "preload_ue": preload_ue,
        "validation_boundary": "需要循环装拆与载荷试验",
    }


def simulate_assembly_operational_load_interference(
    vertical_load_n: float, temperature_change_c: float
) -> dict[str, np.ndarray | str]:
    """Show why assembly self-check must be performed empty-load rather than during gait."""
    if vertical_load_n < 0.0:
        raise ValueError("垂直载荷不能为负")
    operational_signal_ue = vertical_load_n * 4.0 * np.array([0.9, 1.1])
    wavelength_shifts_nm = fbg_wavelength_shift_nm(
        operational_signal_ue * 1e-6, temperature_change_c
    )
    return {
        "operational_signal_ue": operational_signal_ue,
        "wavelength_shifts_nm": wavelength_shifts_nm,
        "assembly_check_condition": "仅空载",
        "validation_boundary": "需要步态载荷试验",
    }


def assess_replaceable_sole_sensing_readiness(
    assembly_case: str, current_vertical_load_n: float
) -> dict[str, str | bool]:
    """Gate gait-data interpretation with the empty-load assembly-screening condition.

    The return value is intentionally limited to the current teaching-model
    workflow: it neither certifies the assembly nor makes a safety claim.
    """
    if current_vertical_load_n < 0.0:
        raise ValueError("垂直载荷不能为负")
    assembly = simulate_replaceable_sole_assembly(assembly_case, 0.0)
    is_empty_load = bool(np.isclose(current_vertical_load_n, 0.0))
    prediction = str(assembly["assembly_prediction"])
    can_enter_flow = bool(is_empty_load and prediction == "装配预测通过")
    if not is_empty_load:
        status = "需空载复装校验"
    elif can_enter_flow:
        status = "可进入足底感知流程（仿真候选）"
    else:
        status = "需复装复核"
    return {
        "can_enter_foot_sensing_flow": can_enter_flow,
        "status": status,
        "assembly_prediction": prediction,
        "validation_boundary": "装配候选需由实物空载复装、温度梯度和重复装拆试验确认",
    }


def simulate_arm_health_fbg(
    load_n: float,
    damage_position_mm: float,
    damage_severity: float,
    temperature_change_c: float,
    noise_nm: float,
    seed: int,
    sensor_positions_mm: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Simulate an FBG array on an arm link with a local damage-sensitive peak."""
    positions = np.asarray([80.0, 200.0, 320.0, 440.0] if sensor_positions_mm is None else sensor_positions_mm, dtype=float)
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
    wavelength_shifts_nm: np.ndarray,
    temperature_change_c: float,
    sensor_positions_mm: np.ndarray | None = None,
) -> dict[str, float | str]:
    """Localise the damage peak with a sub-sensor parabola fit and report uncertainty."""
    shifts = np.asarray(wavelength_shifts_nm, dtype=float)
    positions = np.asarray([80.0, 200.0, 320.0, 440.0] if sensor_positions_mm is None else sensor_positions_mm, dtype=float)
    if shifts.shape != positions.shape:
        raise ValueError("机械臂健康诊断需要与传感器阵列等长的 FBG 读数")
    strain = _strain_from_shift(shifts, temperature_change_c)
    local_excess = np.maximum(0.0, strain - np.median(strain))
    index = int(np.argmax(local_excess))
    damage_index = float(local_excess[index] / 8.0e-4)
    spacing = float(np.diff(positions).min()) if len(positions) > 1 else 1.0
    if len(positions) >= 3 and 0 < index < len(positions) - 1:
        y0, y1, y2 = local_excess[index - 1], local_excess[index], local_excess[index + 1]
        denominator = y0 - 2.0 * y1 + y2
        offset_units = 0.5 * (y0 - y2) / denominator if abs(denominator) > 1e-12 else 0.0
        suspected = float(positions[index] + np.clip(offset_units, -0.5, 0.5) * spacing)
        uncertainty = spacing / 2.0
    else:
        suspected = float(positions[index])
        uncertainty = spacing
    return {
        "suspected_location_mm": suspected,
        "location_uncertainty_mm": uncertainty,
        "sensor_spacing_mm": spacing,
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
    core_temperature_gradient_c: float = 0.0,
) -> dict[str, np.ndarray | float]:
    """Simulate a three-core FBG shape sensor and reconstruct its centreline.

    A zero gradient gives the common-mode temperature that the differential
    estimator rejects; a nonzero gradient adds per-core temperature offsets and
    biases the estimate, which is exactly the model boundary to demonstrate.
    """
    direction_rad = np.deg2rad(bend_direction_deg)
    angles = _core_angles_rad()
    radius_m = core_radius_um * 1e-6
    strains = radius_m * curvature_per_m * np.cos(angles - direction_rad)
    core_temperature_c = temperature_change_c + core_temperature_gradient_c * (np.arange(3) - 1.0)
    clean_shifts = fbg_wavelength_shift_nm(strains, core_temperature_c)
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
        "core_temperature_change_c": core_temperature_c,
        "temperature_gradient_c": float(core_temperature_gradient_c),
    }


def simulate_shape_distributed_link(
    length_mm: float,
    core_strain: np.ndarray,
    curvature_per_m: float,
) -> dict[str, np.ndarray]:
    """Per-core strain along the fibre plus a Rayleigh-style local peak for comparison."""
    position = np.linspace(0.0, max(10.0, length_mm), 161)
    cores = np.asarray(core_strain, dtype=float) * 1e6
    core_profiles = np.repeat(cores[:, None], len(position), axis=1)
    local_peak = float(abs(curvature_per_m)) * 120.0 * np.exp(-0.5 * ((position - length_mm * 0.5) / 30.0) ** 2)
    return {
        "position_mm": position,
        "core_strain_ue": core_profiles,
        "rayleigh_strain_ue": local_peak,
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


_BRILLOUIN_TEMP_COEFFICIENT_GHZ_PER_C = 0.0011
_BRILLOUIN_STRAIN_COEFFICIENT_GHZ_PER_UE = 5e-5


def simulate_brillouin_raman_compensation(
    length_mm: float,
    event_position_mm: float,
    strain_peak_ue: float,
    ambient_temperature_change_c: float,
    local_temperature_rise_c: float,
    temperature_noise_c: float = 0.0,
    seed: int = 7,
) -> dict[str, np.ndarray | float]:
    """Show Brillouin/Raman temperature-strain decoupling on one fibre.

    Brillouin frequency shift responds to both strain and temperature, so a
    local temperature rise looks like apparent strain.  Raman measures the
    temperature profile; subtracting its contribution recovers the true strain.
    """
    if strain_peak_ue < 0.0 or local_temperature_rise_c < 0.0 or temperature_noise_c < 0.0:
        raise ValueError("应变峰值、局部温升与温度噪声不能为负")
    position = np.linspace(0.0, max(10.0, length_mm), 241)
    true_strain = strain_peak_ue * np.exp(-0.5 * ((position - event_position_mm) / 18.0) ** 2)
    temperature = ambient_temperature_change_c + local_temperature_rise_c * np.exp(-0.5 * ((position - event_position_mm) / 25.0) ** 2)
    frequency = 10.80 + _BRILLOUIN_TEMP_COEFFICIENT_GHZ_PER_C * temperature + _BRILLOUIN_STRAIN_COEFFICIENT_GHZ_PER_UE * true_strain
    raman_temperature = temperature + add_gaussian_noise(np.zeros_like(temperature), temperature_noise_c, seed)
    naive_strain = (frequency - 10.80) / _BRILLOUIN_STRAIN_COEFFICIENT_GHZ_PER_UE
    compensated_strain = (
        frequency - 10.80 - _BRILLOUIN_TEMP_COEFFICIENT_GHZ_PER_C * raman_temperature
    ) / _BRILLOUIN_STRAIN_COEFFICIENT_GHZ_PER_UE
    region = np.flatnonzero(true_strain > true_strain.max() * 0.1)

    def peak_error(estimate: np.ndarray) -> float:
        return float(np.max(np.abs(estimate[region] - true_strain[region])))

    return {
        "position_mm": position,
        "true_strain_ue": true_strain,
        "temperature_change_c": temperature,
        "raman_temperature_c": raman_temperature,
        "brillouin_frequency_ghz": frequency,
        "naive_strain_ue": naive_strain,
        "compensated_strain_ue": compensated_strain,
        "naive_peak_error_ue": peak_error(naive_strain),
        "compensated_peak_error_ue": peak_error(compensated_strain),
        "temperature_rms_error_c": float(np.sqrt(np.mean((raman_temperature - temperature) ** 2))),
        "validation_boundary": "系数为教学假设；真实 Brillouin 温/应变系数需要材料与波长标定",
    }


def simulate_raman_temperature(length_mm: float, heater_position_mm: float, peak_temperature_c: float) -> dict[str, np.ndarray]:
    """Teaching Raman DTS temperature profile from an anti-Stokes ratio proxy."""
    position = np.linspace(0.0, max(10.0, length_mm), 161)
    temperature = 20.0 + max(0.0, peak_temperature_c - 20.0) * np.exp(-.5 * ((position - heater_position_mm) / 25.0) ** 2)
    ratio = np.exp(-850.0 / (temperature + 273.15))
    return {"position_mm": position, "temperature_c": temperature, "anti_stokes_ratio": ratio}


_DISTRIBUTED_QUALITY = {
    "Rayleigh/OFDR": .93,
    "φ-OTDR / DAS": .88,
    "Brillouin": .90,
    "Raman": .90,
}


def simulate_distributed_mechanism(
    mode: str,
    fiber_length_mm: float,
    event_position_mm: float,
    event_strength: float,
    sample_rate_hz: int = 50,
) -> tuple[dict[str, np.ndarray], dict[str, str | np.ndarray | float]]:
    """Dispatch one distributed-sensing mechanism with a consistent quality frame."""
    if mode == "Rayleigh/OFDR":
        result = simulate_rayleigh_ofdr(fiber_length_mm, event_position_mm, event_strength, 2.0)
        frame = build_sensor_frame(mode, result["position_mm"], result["raw_strain_ue"], result["strain_ue"], _DISTRIBUTED_QUALITY[mode])
    elif mode == "φ-OTDR / DAS":
        result = simulate_das_event(fiber_length_mm, event_position_mm, 60.0, int(sample_rate_hz))
        frame = build_sensor_frame(mode, result["position_mm"], result["amplitude"], result["amplitude"], _DISTRIBUTED_QUALITY[mode])
    elif mode == "Brillouin":
        result = simulate_brillouin_distribution(fiber_length_mm, min(event_strength, 100.0), event_strength)
        frame = build_sensor_frame(mode, result["position_mm"], result["brillouin_frequency_ghz"], result["strain_ue"], _DISTRIBUTED_QUALITY[mode])
    elif mode == "Raman":
        result = simulate_raman_temperature(fiber_length_mm, event_position_mm, min(event_strength, 120.0))
        frame = build_sensor_frame(mode, result["position_mm"], result["anti_stokes_ratio"], result["temperature_c"], _DISTRIBUTED_QUALITY[mode])
    else:
        raise ValueError("未知分布式机制")
    return result, frame


def decimate_distributed_result(result: dict, spacing_mm: float) -> dict:
    """Decimate a distributed-sensing result to a coarser spatial sampling.

    Curves keep the same physics and only keep every ``step``-th spatial
    sample; the DAS heatmap keeps its time axis and decimates the distance
    axis.  This exposes the spatial-resolution point: coarse sampling
    underestimates or misses narrow event peaks.
    """
    spacing_mm = max(0.5, float(spacing_mm))
    sampled = dict(result)
    if "time_s" in result and "amplitude" in result:
        positions = np.asarray(result["position_mm"], dtype=float)
        step = max(1, int(round(spacing_mm / max(float(np.diff(positions)[0]), 1e-9))))
        sampled["position_mm"] = positions[::step]
        sampled["amplitude"] = np.asarray(result["amplitude"], dtype=float)[:, ::step]
        return sampled
    if "das_distance_mm" in result:
        positions = np.asarray(result["das_distance_mm"], dtype=float)
        step = max(1, int(round(spacing_mm / max(float(np.diff(positions)[0]), 1e-9))))
        sampled["das_distance_mm"] = positions[::step]
        sampled["das_amplitude"] = np.asarray(result["das_amplitude"], dtype=float)[:, ::step]
        return sampled
    positions = np.asarray(result["position_mm"], dtype=float)
    step = max(1, int(round(spacing_mm / max(float(np.diff(positions)[0]), 1e-9))))
    sampled["position_mm"] = positions[::step]
    for key in ("strain_ue", "raw_strain_ue", "temperature_c", "anti_stokes_ratio", "brillouin_frequency_ghz"):
        if key in result:
            sampled[key] = np.asarray(result[key], dtype=float)[::step]
    return sampled


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


def simulate_polarization_map(
    temperature_change_c: float = 0.0,
    grid: int = 21,
    stress_range_mpa: tuple[float, float] = (0.0, 250.0),
    twist_range_deg: tuple[float, float] = (-90.0, 90.0),
) -> dict[str, np.ndarray]:
    """Azimuth and ellipticity over a stress-twist grid for a joint-view map."""
    stress = np.linspace(stress_range_mpa[0], stress_range_mpa[1], int(grid))
    twist = np.linspace(twist_range_deg[0], twist_range_deg[1], int(grid))
    azimuth = np.zeros((int(grid), int(grid)))
    ellipticity = np.zeros((int(grid), int(grid)))
    for row, stress_value in enumerate(stress):
        for column, twist_value in enumerate(twist):
            state = simulate_polarization_sensing(stress_value, twist_value, temperature_change_c)
            azimuth[row, column] = state["azimuth_deg"]
            ellipticity[row, column] = state["ellipticity_deg"]
    return {"stress_mpa": stress, "twist_deg": twist, "azimuth_deg": azimuth, "ellipticity_deg": ellipticity}


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


@dataclass(frozen=True)
class ModuleQuality:
    """Per-module sensing quality consumed by the multimodal fusion layer."""

    score: float
    confidence: float
    note: str = ""


def assess_foot_quality(cop_region: float) -> ModuleQuality:
    """Foot-balance quality: how close the CoP is to the sole centre (zone 2.5 of 0-5)."""
    score = float(np.clip(1.0 - abs(float(cop_region) - 2.5) / 2.5, 0.0, 1.0))
    return ModuleQuality(score, 1.0, "CoP 距足底中心距离")


def assess_shape_quality(estimated_curvature_per_m: float, true_curvature_per_m: float) -> ModuleQuality:
    """Multicore shape quality: inverse curvature error within a ±5 1/m teaching window."""
    score = float(np.clip(1.0 - abs(float(estimated_curvature_per_m) - float(true_curvature_per_m)) / 5.0, 0.0, 1.0))
    return ModuleQuality(score, 1.0, "多芯曲率反演误差")


def assess_health_quality(status: str, damage_index: float) -> ModuleQuality:
    """Structural-health quality: a passing status reports full quality, otherwise low."""
    score = 1.0 if status == "正常" else 0.4
    return ModuleQuality(score, 1.0, f"健康状态 {status}（指数 {float(damage_index):.2f}）")


def fuse_robot_sensing(qualities: dict[str, ModuleQuality]) -> dict[str, float | str]:
    """Combine independent module qualities into one task-readiness summary.

    Grasp and health are hard gates; every other module must score at least
    0.75 for the task to be considered ready.  This is a teaching summary, not
    a safety controller.
    """
    scores = [float(np.clip(quality.score, 0.0, 1.0)) for quality in qualities.values()]
    soft_gates = [name for name in qualities if name not in {"grasp", "health"}]
    ready = (
        all(qualities[name].score >= 0.75 for name in soft_gates)
        and all(qualities[name].score >= 0.5 for name in ("grasp", "health"))
    )
    return {"status": "任务就绪" if ready else "需人工复核", "confidence": float(np.mean(scores))}


_HAND_FINGER_BASES = np.array([
    (.18, -.98, .04), (.92, -.56, .08), (1.04, -.18, .10), (.98, .23, .08), (.84, .59, .02),
])
_HAND_FINGER_LENGTHS = ((1.31, .85), (1.384, 1.0, .72), (1.512, 1.072, .768), (1.36, .976, .688), (1.104, .768, .56))
_HAND_FINGER_RADII = ((.19, .16), (.17, .145, .12), (.17, .145, .12), (.17, .145, .12), (.17, .145, .12))
_HAND_FINGER_SPREADS = (-.83, -.09, -.03, .05, .15)
_CAN_GRASP_CENTER = np.array((.30, -.20, .76))
_CAN_RADIUS = .48
_CAN_HALF_LENGTH = .86

THREE_D_GRASP_CALIBRATION = GraspCalibration()


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
            if finger_index == 0:
                # 拇指 IP 与渲染器一致：绕局部 Z 轴正向弯曲。
                transform = transform @ _rotation_z(np.deg2rad(joint_angles_deg[segment_index + 1]))
            else:
                transform = transform @ _rotation_y(-np.deg2rad(joint_angles_deg[segment_index + 1]))
    return capsules


def three_d_finger_capsules(
    finger_joint_angles_deg: tuple[tuple[float, ...], ...] | list[tuple[float, ...]],
) -> list[list[tuple[np.ndarray, np.ndarray, float]]]:
    """Return per-finger capsule geometry shared by collision and the renderer."""
    joint_counts = (2, 3, 3, 3, 3)
    if len(finger_joint_angles_deg) != 5 or any(
        len(angles) != count for angles, count in zip(finger_joint_angles_deg, joint_counts)
    ):
        raise ValueError("三维手指几何需要 14 个关节角")
    return [
        _finger_capsules(finger_index, tuple(map(float, angles)))
        for finger_index, angles in enumerate(finger_joint_angles_deg)
    ]


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
    """Limit each joint in base-to-tip order at first cylinder contact.

    A proximal joint bends as far as it can with the remaining joints straight,
    then the next joint folds around the can, so the rendered finger wraps the
    cylinder instead of being scaled down as a whole.
    """
    desired = np.asarray(desired_angles, dtype=float)
    if _finger_collision_clearance(finger_index, tuple(desired), center) >= 0.0:
        return desired, _finger_collision_clearance(finger_index, tuple(desired), center)
    if _finger_collision_clearance(finger_index, tuple(np.zeros_like(desired)), center) < 0.0:
        return np.zeros_like(desired), _finger_collision_clearance(finger_index, tuple(np.zeros_like(desired)), center)
    limited = np.zeros_like(desired)
    for joint_index in range(len(desired)):
        candidate = limited.copy()
        candidate[joint_index] = desired[joint_index]
        if _finger_collision_clearance(finger_index, tuple(candidate), center) >= 0.0:
            limited[joint_index] = desired[joint_index]
            continue
        lower, upper = 0.0, 1.0
        for _ in range(28):
            middle = (lower + upper) / 2.0
            candidate[joint_index] = desired[joint_index] * middle
            if _finger_collision_clearance(finger_index, tuple(candidate), center) >= 0.0:
                lower = middle
            else:
                upper = middle
        limited[joint_index] = desired[joint_index] * lower
    return limited, _finger_collision_clearance(finger_index, tuple(limited), center)


def classify_3d_grasp_from_fbg(
    sensing: dict[str, np.ndarray | list[int] | float | bool],
    temperature_change_c: float,
    calibration: GraspCalibration = THREE_D_GRASP_CALIBRATION,
) -> dict[str, np.ndarray | list[int] | float | bool]:
    """Classify holding state from compensated tactile FBG data, not geometry flags."""
    tactile_shift = np.asarray(sensing["tactile_fbg_shifts_nm"], dtype=float)
    arm_shift = np.asarray(sensing["arm_fbg_shifts_nm"], dtype=float)
    if tactile_shift.shape != (15,) or arm_shift.shape != (3,):
        raise ValueError("三维 FBG 判定需要 14 个指节、1 个掌心和 3 个臂部通道")
    tactile_strain = _strain_from_shift(tactile_shift, temperature_change_c)
    finger_touch = tactile_strain[:14] / calibration.finger_touch_strain_per_n
    palm_touch = max(0.0, float(tactile_strain[14] / calibration.palm_touch_strain_per_n))
    counts = (2, 3, 3, 3, 3)
    index = 0
    finger_force = []
    for count in counts:
        finger_force.append(float(np.sum(np.maximum(finger_touch[index : index + count], 0.0))))
        index += count
    finger_force_array = np.asarray(finger_force)
    contacts = np.flatnonzero(finger_force_array >= calibration.contact_force_threshold_n).astype(int).tolist()
    arm_strain_ue = _strain_from_shift(arm_shift, temperature_change_c) * 1.0e6
    is_grasped = bool(palm_touch >= calibration.palm_contact_threshold_n and 0 in contacts and len([item for item in contacts if item != 0]) >= 2)
    return {
        "is_grasped": is_grasped,
        "contact_fingers": contacts,
        "contact_force_n": finger_force_array,
        "palm_touch_n": palm_touch,
        "arm_bend_strain_ue": arm_strain_ue,
    }


def evaluate_3d_grasp_sensing(
    finger_curls_deg: tuple[float, float, float, float, float] | np.ndarray,
    can_offset_xyz: tuple[float, float, float] | np.ndarray,
    temperature_change_c: float = 0.0,
    arm_joint_angles_deg: tuple[float, float, float] | np.ndarray = (0.0, 0.0, 0.0),
    finger_joint_angles_deg: tuple[tuple[float, ...], ...] | None = None,
    calibration: GraspCalibration = THREE_D_GRASP_CALIBRATION,
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
    contact_force_n = np.maximum(0.0, commanded_bend - limited_bend) * GRASP_FORCE_PER_DEGREE
    contact_fingers = np.flatnonzero(contact_force_n >= calibration.contact_force_threshold_n).astype(int).tolist()
    non_thumb_contacts = [index for index in contact_fingers if index != 0]
    palm_support = bool(
        -1.12 <= can_center[0] <= 1.04
        and -.98 <= can_center[1] <= 1.00
        and .24 <= can_center[2] - _CAN_RADIUS <= .36
    )
    is_grasped = bool(palm_support and 0 in contact_fingers and len(non_thumb_contacts) >= 2)
    stability = min(
        1.0,
        GRASP_STABILITY_CONTACT_WEIGHT * len(contact_fingers)
        + GRASP_STABILITY_FORCE_WEIGHT_3D * float(np.minimum(contact_force_n, 1.0).sum())
        + (GRASP_STABILITY_PALM_BONUS if palm_support else 0.0),
    )
    tip_distances = collision_clearance
    # 五路读数保留为每根手指的综合弯曲/接触通道；分区通道在下方单独
    # 给出，避免把手臂弯曲误解释成触觉。
    strain = curls * calibration.bend_strain_per_deg + contact_force_n * calibration.contact_strain_per_n
    segment_angles = np.concatenate([np.asarray(angles, dtype=float) for angles in finger_joint_angles_deg])
    finger_segment_touch = np.repeat(contact_force_n, joint_counts) / np.repeat(joint_counts, joint_counts)
    palm_touch = float(contact_force_n.sum() * (GRASP_PALM_ACTIVE_FACTOR if is_grasped else GRASP_PALM_PASSIVE_FACTOR))
    arm_bend_strain = np.abs(arm_angles) * np.array([3.2, 4.1, 2.6])
    tactile_strain = np.r_[
        finger_segment_touch * calibration.finger_touch_strain_per_n,
        palm_touch * calibration.palm_touch_strain_per_n,
    ]
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
