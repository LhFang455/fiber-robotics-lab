import json

import numpy as np
import pytest

from fiber_robotics_sim import demos, experiments, models


@pytest.mark.parametrize('case', ['正常装配', '压入不足', '单侧错位'])
def test_assembly_only_reports_readings_after_seating_and_matches_existing_screen(case):
    sequence = demos.assembly_demo({'assembly_case': case, 'temperature_c': 20.0})
    first = sequence['frames'][0]
    assert first['explosion'] == 1
    assert first['signals'] == [None, None, None]
    assert not first['seated']
    current = sequence['frames'][sequence['initial_index']]
    expected = models.simulate_replaceable_sole_assembly(case, 20.0)
    assert current['seated'] and current['explosion'] == 0
    assert current['prediction'] == expected['assembly_prediction']
    assert current['signals'][:2] == pytest.approx(expected['working_wavelength_shifts_nm'])
    assert sequence['lateral_offset_mm'] == expected['case_parameters']['lateral_offset_mm']
    json.dumps(sequence, allow_nan=False)


def test_health_hides_invalid_location_and_current_frame_matches_experiment():
    params = experiments.HEALTH_PRESETS['健康基线']
    healthy = demos.health_demo(params)
    assert all(frame['suspected'] is None and frame['truth_location'] is None for frame in healthy['frames'])
    params = {**params, 'anomaly_severity': .8, 'anomaly_position_mm': 320.0}
    sequence = demos.health_demo(params)
    current = sequence['frames'][sequence['initial_index']]
    expected = experiments.run_health_experiment(params)['results']
    assert current['signals'] == pytest.approx(expected['wavelength_shifts_nm'])
    assert current['suspected'] == expected['suspected_location_mm']
    assert current['uncertainty'] == expected['location_uncertainty_mm']
    assert current['truth_location'] == 320.0
    assert sequence['frames'][0]['suspected'] is None
    json.dumps(sequence, allow_nan=False)


def test_fbg_second_stage_changes_temperature_not_mechanical_strain_and_compensation_removes_it():
    sequence = demos.fbg_demo(experiments.PRESETS['温漂对照'])
    middle, end = sequence['frames'][60], sequence['frames'][-1]
    assert middle['temperature'] == 0 and end['temperature'] == 20
    assert middle['strain'] == end['strain']
    assert middle['signals'][:3] != end['signals'][:3]
    assert middle['signals'][3:] == pytest.approx(end['signals'][3:])
    assert end['estimated_angle'] == pytest.approx(45)
    expected = experiments.run_calibration(experiments.PRESETS['温漂对照'])['results']
    assert end['signals'][:3] == pytest.approx(expected['raw_shifts_nm'])
    json.dumps(sequence, allow_nan=False)


def test_fbg_neutral_layer_is_unidentifiable_and_compression_geometry_stays_positive():
    neutral = demos.fbg_demo({**experiments.PRESETS['温漂对照'], 'fiber_offset_mm': 0.0})
    assert all(frame['estimated_angle'] is None for frame in neutral['frames'])
    compressed = demos.fbg_demo({**experiments.PRESETS['理想标定'], 'angle_deg': -100.0,
                                 'fiber_offset_mm': 2.0, 'length_mm': 40.0})
    scales = [1 + frame['strain'] * compressed['strain_amplification'] for frame in compressed['frames']]
    assert min(scales) >= .55 - 1e-12 and max(scales) == 1
    assert np.isfinite(scales).all()
    json.dumps(neutral, allow_nan=False)
