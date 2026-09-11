import json
import numpy as np
from fiber_robotics_sim import models, thumb_inspector as inspector


def test_attachment_follows_rotated_link_and_preserves_thumb_axes():
    for angles in ((0, 0), (45, 0), (0, 60), (45, 60)):
        links = inspector.thumb_transforms(angles)
        capsules = models._finger_capsules(0, angles)
        for link, (start, end, _), sensor in zip(links, capsules, inspector.registrations(), strict=True):
            r = np.asarray(link['rotation'])
            np.testing.assert_allclose(link['origin'], start)
            np.testing.assert_allclose(start + r @ [link['length'], 0, 0], end)
            mounted = start + r @ sensor['local']
            np.testing.assert_allclose(r.T @ (mounted - start), sensor['local'], atol=1e-14)
            np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
    r = np.asarray(inspector.thumb_transforms((45, 0))[0]['rotation'])
    assert (r @ [1, 0, 0])[2] > 0
    first, second = inspector.thumb_transforms((0, 60))
    relative = np.asarray(first['rotation']).T @ np.asarray(second['rotation'])
    assert relative[1, 0] > 0


def test_sequence_matches_existing_channels_at_current_parameters():
    joints = ((60., 70.), (40., 50., 30.), (40., 50., 30.), (40., 50., 30.), (40., 50., 30.))
    payload = inspector.sequence(joints, (0., 0., 0.), 15., (30., -20., 10.))
    expected = models.evaluate_3d_grasp_sensing(tuple(np.mean(j) for j in joints), (0, 0, 0), 15,
        arm_joint_angles_deg=(30, -20, 10), finger_joint_angles_deg=joints)
    np.testing.assert_allclose(payload['frames'][-1]['values'], expected['tactile_fbg_shifts_nm'][:2])
    np.testing.assert_allclose(payload['frames'][-1]['angles'], expected['collision_limited_joint_angles_deg'][0])
    assert [s['channel'] for s in payload['sensors']] == [0, 1]
    assert payload['frames'][0]['time'] == 0
    assert payload['frames'][-1]['time'] == 2
    assert inspector.finite_or_none(np.nan) is None
    assert inspector.finite_or_none(None) is None
    assert inspector.finite_or_none(0) == 0
    json.dumps(payload, allow_nan=False)
