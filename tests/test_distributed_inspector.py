import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import experiments,models,distributed_inspector as inspector

PARAMS=dict(mode='Rayleigh/OFDR',fiber_length_mm=300.,event_position_mm=140.,event_strength=600.,spatial_spacing_mm=10,sample_rate_hz=50)

@pytest.mark.parametrize('mode',['Rayleigh/OFDR','φ-OTDR / DAS','Brillouin','Raman'])
def test_sample_profile_uses_original_positions_and_units(mode):
    record=experiments.run_distributed_experiment(dict(PARAMS,mode=mode))
    point=inspector.sample(record,3)
    assert point['value']==record['results']['profile_value'][3]
    assert point['position']==record['results']['profile_position_mm'][3]
    space,profile=inspector.figures(record,3)
    assert space.data[0].customdata[3]==3
    assert profile.data[0].y[3]==point['value']
    if mode=='Brillouin':
        assert not record['results']['localization_valid']
        assert len(space.layout.shapes)==2


def test_das_waveform_matches_selected_spatial_column():
    result,_=models.simulate_distributed_mechanism('φ-OTDR / DAS',300.,140.,600.,50)
    result=models.decimate_distributed_result(result,10)
    figure=inspector.waveform(result,4)
    np.testing.assert_allclose(figure.data[0].y,result['amplitude'][:,4])
    np.testing.assert_allclose(figure.data[0].x,result['time_s'])
    assert np.isclose(1/np.diff(result['time_s'])[0],100)


def test_missing_is_not_zero_and_page_switches_mechanism():
    record=experiments.run_distributed_experiment(PARAMS)
    record['results']['profile_value'][0]=float('nan')
    assert inspector.sample(record,0)['value'] is None
    assert inspector.figures(record,0)[1].data[0].y[0] is None
    app=AppTest.from_file('app.py',default_timeout=30).run()
    app.selectbox(key='distributed_mode').set_value('φ-OTDR / DAS').run()
    assert not app.exception
    assert any('生成采样率 100.0 Hz' in c.value for c in app.caption)
    app.select_slider(key='distributed_inspect_index').set_value(4).run()
    assert not app.exception
