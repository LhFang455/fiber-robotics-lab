from copy import deepcopy
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / 'app.py'


def metric(app, label):
    return next(item.value for item in app.metric if item.label == label)


def test_preset_requires_explicit_load_and_preserves_unrelated_controls(caplog):
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    assert metric(app, '当前配置') == '50 Hz · 无'
    app.slider(key='calibration_contact_force').set_value(6.0)
    app.select_slider(key='global_sample_rate').set_value(100)
    app.selectbox(key='calibration_preset').set_value('温漂对照').run()
    assert app.slider(key='global_temperature').value == 0.0
    caplog.clear()
    app.button(key='load_calibration_preset').click().run()
    assert not app.exception
    assert 'was created with a default value' not in caplog.text
    assert app.slider(key='hand_bend_angle').value == 45.0
    assert app.slider(key='global_temperature').value == 20.0
    assert app.slider(key='global_noise').value == 0.0
    assert app.number_input(key='global_seed').value == 17
    assert app.slider(key='calibration_contact_force').value == 6.0
    assert app.select_slider(key='global_sample_rate').value == 100
    assert metric(app, 'FBG 融合角') == '45.00 °'
    assert metric(app, '未温补角') != '45.00 °'


def test_baseline_is_frozen_and_restore_recovers_conditions():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.button(key='load_calibration_preset').click().run()
    app.button(key='save_calibration_baseline').click().run()
    baseline = deepcopy(app.session_state['calibration_baseline'])
    app.slider(key='hand_bend_angle').set_value(70.0)
    app.slider(key='global_temperature').set_value(20.0)
    app.slider(key='global_noise').set_value(0.02).run()
    assert not app.exception
    assert app.session_state['calibration_baseline'] == baseline
    app.button(key='reset_calibration').click().run()
    assert app.session_state['calibration_baseline'] == baseline
    app.button(key='restore_calibration_baseline').click().run()
    assert not app.exception
    assert app.slider(key='hand_bend_angle').value == 45.0
    assert app.slider(key='global_temperature').value == 0.0
    assert app.slider(key='global_noise').value == 0.0
    assert metric(app, 'FBG 融合角') == '45.00 °'
    assert app.session_state['calibration_baseline'] == baseline


def test_zero_offset_does_not_present_zero_as_a_valid_inversion():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.slider(key='hand_bend_angle').set_value(45.0)
    app.slider(key='calibration_offset').set_value(0.0).run()
    assert not app.exception
    assert metric(app, 'FBG 融合角') == '不可反演'
    assert metric(app, '未温补角') == '不可反演'
    assert any('中性层' in item.value for item in app.warning)


def test_json_import_waits_for_click_then_applies_whitelist_and_recalculates(caplog):
    from fiber_robotics_sim import experiments

    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    app.button(key='load_calibration_preset').click().run()
    app.button(key='save_calibration_baseline').click().run()
    app.slider(key='calibration_contact_force').set_value(6.0)
    app.select_slider(key='global_sample_rate').set_value(100)
    app.selectbox(key='sole_assembly_case').set_value('压入不足')
    app.slider(key='three_d_shoulder').set_value(50.0).run()
    target = experiments.run_calibration({
        **experiments.PRESETS['噪声对照'], 'angle_deg': 60.0, 'attachment': '粘接式',
    })
    payload = __import__('json').loads(experiments.export_record(target).decode('utf-8'))
    payload['current']['results']['estimated_angle_deg'] = 999.0
    content = __import__('json').dumps(payload).encode('utf-8')

    next(item for item in app.get('file_uploader') if item.key == 'calibration_record_upload').upload(
        'calibration.json', content, 'application/json'
    ).run()
    assert not app.exception
    # 上传仅校验并显示按钮，不应立即改当前条件或 A。
    assert app.slider(key='hand_bend_angle').value == 45.0
    assert app.session_state['calibration_baseline'] is not None
    assert app.button(key='import_calibration_record')
    caplog.clear()
    app.button(key='import_calibration_record').click().run()

    assert not app.exception
    assert 'was created with a default value' not in caplog.text
    assert app.slider(key='hand_bend_angle').value == 60.0
    assert app.selectbox(key='calibration_attachment').value == '粘接式'
    assert app.slider(key='global_noise').value == 0.02
    assert app.number_input(key='global_seed').value == 17
    assert metric(app, 'FBG 融合角') != '999.00 °'
    assert app.session_state['calibration_baseline'] is None
    # 未列入记录白名单的局部/公共/三维状态保持不变。
    assert app.slider(key='calibration_contact_force').value == 6.0
    assert app.select_slider(key='global_sample_rate').value == 100
    assert app.selectbox(key='sole_assembly_case').value == '压入不足'
    assert app.slider(key='three_d_shoulder').value == 50.0
