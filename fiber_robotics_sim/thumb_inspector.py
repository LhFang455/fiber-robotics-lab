"""Thumb-local sensor registration and prescribed quasi-static demo replay."""
import json
from pathlib import Path

import numpy as np
import streamlit as st

from . import models


def registrations():
    """Local X follows the link; local Z is its demonstration mounting normal."""
    return [dict(id=f"FBG-{i+1:02d}", channel=i, part=f"TH-L{i+1}",
                 local=[length / 2, 0.0, radius + .045], direction=[1., 0., 0.])
            for i, (length, radius) in enumerate(zip(
                models._HAND_FINGER_LENGTHS[0], models._HAND_FINGER_RADII[0], strict=True))]


def thumb_transforms(angles):
    base = models._HAND_FINGER_BASES[0].copy()
    rotation = models._rotation_z(models._HAND_FINGER_SPREADS[0]) @ models._rotation_y(-np.deg2rad(angles[0]))
    result = []
    for i, length in enumerate(models._HAND_FINGER_LENGTHS[0]):
        result.append(dict(origin=base.tolist(), rotation=rotation.tolist(), length=length,
                           radius=models._HAND_FINGER_RADII[0][i]))
        base = base + rotation @ np.array([length, 0., 0.])
        if i == 0:
            rotation = rotation @ models._rotation_z(np.deg2rad(angles[1]))
    return result


def finite_or_none(value):
    return float(value) if value is not None and np.isfinite(value) else None


@st.cache_data(show_spinner=False, max_entries=4)
def sequence(joints, offset, temperature, arm_joints):
    """Two-second prescribed closing ramp; no dynamics or measured clock implied."""
    frames = []
    for t in np.linspace(0., 2., 21):
        command = tuple(tuple(float(v * t / 2) for v in angles) for angles in joints)
        sensing = models.evaluate_3d_grasp_sensing(
            tuple(float(np.mean(a)) for a in command), offset, temperature,
            arm_joint_angles_deg=arm_joints, finger_joint_angles_deg=command)
        angles = sensing["collision_limited_joint_angles_deg"][0]
        frames.append(dict(time=float(t), angles=list(angles), links=thumb_transforms(angles),
                           values=[finite_or_none(v) for v in sensing["tactile_fbg_shifts_nm"][:2]],
                           forces=[finite_or_none(v) for v in sensing["finger_segment_touch_n"][:2]]))
    values = [abs(v) for f in frames for v in f["values"] if v is not None]
    return dict(sensors=registrations(), frames=frames, color_max=max([.001, *values]),
                unit="nm", source="模拟数据 · 原三维抓取解析模型", temperature=float(temperature))


def inspector_html(payload):
    directory = Path(__file__).parent
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    return (directory / "thumb_inspector.html").read_text().replace(
        "__PAYLOAD__", data).replace("__THREE__", (directory / "vendor/three.min.js").read_text())
