import numpy as np
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import experiments, shape_inspector as inspector

PARAMS=dict(curvature_per_m=8.,direction_deg=35.,twist_per_m=0.,length_mm=150.,
            core_radius_um=125.,temperature_c=20.,noise_nm=0.,core_temperature_gradient_c=0.,seed=7)


def test_core_mapping_and_common_mode_rejection():
    record=experiments.run_shape_experiment(PARAMS)
    registry=inspector.core_registry(record)
    assert [c['channel'] for c in registry]==[1,2,3]
    np.testing.assert_allclose(np.linalg.norm([c['local_um'] for c in registry],axis=1),125.)
    np.testing.assert_allclose(inspector.differential_strain(record),np.array(record['results']['strain'])*1e6)
    record['results']['wavelength_shifts_nm'][1]=float('nan')
    assert inspector.differential_strain(record)==[None]*3


def test_error_node_and_3d_positions_match_original_solver():
    record=experiments.run_shape_experiment(dict(PARAMS,core_temperature_gradient_c=4.))
    i=120
    chart=inspector.error_figure(record,i)
    assert chart.data[0].customdata[i]==i
    assert chart.data[0].x[i]==112.5
    fig=inspector.position_figure(record,i)
    np.testing.assert_allclose([fig.data[3].x[0],fig.data[3].y[0],fig.data[3].z[0]],
                              record['results']['estimated_centerline_xyz_mm'][i])
    assert len(inspector.position_figure(record,i,False).data)==2
    assert record['results']['point_error_mm'][i]>0


def test_shape_inspector_page_selection():
    app=AppTest.from_file('app.py',default_timeout=30).run()
    assert not app.exception
    app.selectbox(key='shape_inspect_core').set_value(2).run()
    app.slider(key='shape_inspect_node').set_value(120).run()
    assert not app.exception
    assert any('CORE-03 → 通道 3' in c.value for c in app.caption)
    assert any('S-120 · 模型弧长 112.50 mm' in c.value for c in app.caption)


def test_error_background_does_not_intercept_sparse_node_selection():
    record = experiments.run_shape_experiment(PARAMS)
    figure = inspector.error_figure(record, 81)
    assert figure.data[0].hoverinfo == "skip"
    assert len(figure.data[0].x) == 161
    assert 81 in figure.data[1].customdata
    selected = list(figure.data[1].customdata).index(120)
    assert figure.data[1].x[selected] == 112.5
