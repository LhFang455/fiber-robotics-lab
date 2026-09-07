import json

import numpy as np
import pytest

from fiber_robotics_sim import demo_extensions as extra, experiments


@pytest.mark.parametrize('mode', ['Rayleigh/OFDR', 'φ-OTDR / DAS', 'Brillouin', 'Raman'])
def test_distributed_scan_preserves_sampled_profile_and_units(mode):
    params = dict(mode=mode, fiber_length_mm=500.0, event_position_mm=320.0,
                  event_strength=400.0, spatial_spacing_mm=11, sample_rate_hz=50)
    payload = extra.distributed_demo(params)
    result = experiments.run_distributed_experiment(params)['results']
    assert payload['positions'] == result['profile_position_mm']
    assert payload['profile'] == result['profile_value']
    assert payload['unit'] == result['unit']
    assert [f['signals'][0] for f in payload['frames']] == result['profile_value']
    assert payload['frames'][-1]['sample_index'] == len(payload['positions']) - 1
    assert payload['estimate'] == result['estimated_event_position_mm']
    json.dumps(payload, allow_nan=False)


def test_zero_strength_distributed_scan_does_not_invent_location():
    params = dict(mode='Rayleigh/OFDR', fiber_length_mm=100.0, event_position_mm=50.0,
                  event_strength=0.0, spatial_spacing_mm=40, sample_rate_hz=10)
    payload = extra.distributed_demo(params)
    assert payload['estimate'] is None and payload['truth'] is None


@pytest.mark.parametrize('mechanism', ['偏振态', 'Sagnac 环路', 'EFPI 微腔'])
def test_optical_end_frame_matches_existing_model_and_has_correct_units(mechanism):
    params = dict(stress_mpa=120.0, twist_deg=-35.0, temperature_c=20.0,
                  gyro_rate_deg_s=-90.0, pressure_mpa=1.0, cavity_um=10.0)
    payload = extra.optical_demo({**params, 'mechanism': mechanism})
    end = payload['frames'][-1]
    expected = experiments.run_optical_experiment(params)['results']
    assert end['stokes'] == pytest.approx(expected['stokes'])
    assert end['phase'] == pytest.approx(expected['sagnac_phase_shift_rad'])
    assert end['cavity'] == expected['effective_cavity_um']
    if mechanism == '偏振态':
        assert all(np.linalg.norm(f['stokes']) == pytest.approx(1) for f in payload['frames'])
    if mechanism == 'Sagnac 环路':
        assert end['phase'] < 0 and end['rate'] == -90
        assert 'rad' in payload['signal_title']
    if mechanism == 'EFPI 微腔':
        assert end['cavity'] < payload['frames'][0]['cavity']
        assert all(.05 - 1e-10 <= v <= .95 + 1e-10 for f in payload['frames'] for v in f['spectrum'])
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize('material', ['海绵', '硬块', '圆柱', '薄板'])
def test_tactile_comparison_current_frame_aligns_with_selected_experiment(material):
    params = dict(material=material, grip_force_n=5.0, contact_area_percent=35.0,
                  temperature_c=20.0, noise_nm=.002, pattern_noise=.1, seed=17)
    payload = extra.tactile_demo(params)
    current = payload['frames'][payload['initial_index']]
    expected = experiments.run_tactile_experiment(params)['results']
    assert current['signals'] == pytest.approx(expected['wavelength_shifts_nm'])
    assert current['diagnosis'] == expected['diagnosed_material']
    assert current['touches'][payload['materials'].index(material)] == pytest.approx(expected['estimated_touch_n'])
    assert payload['frames'][0]['diagnosis'] == '未接触'
    assert payload['frames'][-1]['diagnosis'] == '未接触'
    json.dumps(payload, allow_nan=False)


def test_brillouin_does_not_claim_unused_input_is_true_location():
    params = dict(mode='Brillouin', fiber_length_mm=500.0, event_position_mm=100.0,
                  event_strength=400.0, spatial_spacing_mm=11, sample_rate_hz=50)
    first = extra.distributed_demo(params)
    second = extra.distributed_demo({**params, 'event_position_mm': 450.0})
    assert first['profile'] == second['profile']
    assert first['truth'] is None and first['estimate'] is None
    record = experiments.run_distributed_experiment(params)
    assert record['results']['location_error_mm'] is None
    assert '0.35L' in experiments.distributed_report(record)


def test_das_ignores_amplitude_input_without_hiding_real_model_event():
    params = dict(mode='φ-OTDR / DAS', fiber_length_mm=500.0, event_position_mm=320.0,
                  event_strength=0.0, spatial_spacing_mm=11, sample_rate_hz=50)
    zero = extra.distributed_demo(params)
    nonzero = extra.distributed_demo({**params, 'event_strength': 400.0})
    assert zero['profile'] == nonzero['profile']
    assert zero['truth'] == 320.0 and zero['estimate'] == nonzero['estimate']
    assert '幅值滑块' in zero['frames'][0]['caption']
