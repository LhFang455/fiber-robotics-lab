import numpy as np
from fiber_robotics_sim import demo_eskin, eskin


def test_channel_contributions_recover_total_and_current_readings():
    p = dict(fx_n=2, fy_n=-1.5, fz_n=8, curvature_per_m=4,
             strain_fraction=.01, temperature_c=38, noise_pf=.01,
             reference_match=.8, seed=7)
    r = eskin.simulate_triaxial_taxel(**p)
    demos = []
    for channel in range(5):
        args = {**p, 'inspection_channel': channel}
        d = demo_eskin.taxel_demo(args)
        demos.append(d)
        f = d['frames'][d['initial_index']]
        np.testing.assert_allclose(f['channel_values_pf'], [r[k][channel] for k in ('active_pf','reference_pf','corrected_pf')])
        np.testing.assert_allclose(f['contribution'], np.linalg.pinv(r['sensitivity_matrix'])[:, channel]*r['corrected_pf'][channel])
        assert len(f['signals']) == len(d['labels']) == 12
        assert args['inspection_channel'] == channel
        assert f'TAXEL-C{channel+1}' in d['subtitle']
    for frame in (0, 30, 60, 120):
        np.testing.assert_allclose(np.sum([d['frames'][frame]['contribution'] for d in demos],axis=0), demos[0]['frames'][frame]['corrected'],atol=1e-12)
    legacy = demo_eskin.taxel_demo(p)
    assert len(legacy['labels']) == 9
    assert demo_eskin.taxel_demo({**p,'inspection_channel':999})['inspection_channel'] == 4
    assert demo_eskin.taxel_demo({**p,'inspection_channel':-1})['inspection_channel'] == 0
