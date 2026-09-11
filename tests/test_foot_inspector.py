import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import experiments, foot_inspector as inspector, models

PARAMS = dict(load_n=180., terrain='平地', phase_percent=55., support='支撑期',
              temperature_c=10., noise_nm=0., seed=7, failed_zone=None, drift_nm=0.)


def test_registered_positions_recover_existing_model_cop():
    loads = np.array([3., 8., 13., 17., 21., 25.])
    shifts = models.simulate_foot_fbg(loads, 10., 0., 7)['wavelength_shifts_nm']
    expected = models.estimate_foot_load_distribution(shifts, 10.)['cop_xy']
    positions = np.array([z['local'][:2] for z in inspector.ZONES])
    np.testing.assert_allclose(np.average(positions, axis=0, weights=loads), expected)
    assert [z['channel'] for z in inspector.ZONES] == list(range(1, 7))


def test_phase_scan_matches_current_and_preserves_failure_semantics():
    params = dict(PARAMS, phase_percent=57)
    rows = inspector.phase_scan(params)
    current = next(r for r in rows if r['parameters']['phase_percent'] == 57)
    assert current == experiments.run_foot_experiment(params)
    assert rows[0]['results']['true_zone_loads_n'][0] == 0
    assert rows[-1]['results']['true_zone_loads_n'][3] == 0
    failed = experiments.run_foot_experiment(dict(params,failed_zone=1))
    assert inspector.observation(failed,0) is None
    assert inspector.observation(rows[0],0) == 0
    assert inspector.spatial_figure(failed,0).data[-1].name == '无有效观测'
    curve = inspector.phase_figure([failed],0,57)
    assert list(curve.data[1].y) == [None]
    assert inspector.selection_value({'selection':{'points':[{'customdata':4}]}}) == 4
    assert inspector.selection_value({'selection':{'points':[{}]}}) is None


def test_inspector_page_selection_and_failure():
    app = AppTest.from_file('app.py',default_timeout=30).run()
    assert not app.exception
    app.selectbox(key='foot_inspector_zone').set_value(4).run()
    assert not app.exception
    assert any('SOLE-05 → FBG 5' in c.value for c in app.caption)
    app.selectbox(key='foot_inspector_zone').set_value(0).run()
    app.selectbox(key='global_failed_channel').set_value('足底区域 1').run()
    assert not app.exception
    assert next(m.value for m in app.metric if m.label == '有效反演载荷') == '无有效观测'
