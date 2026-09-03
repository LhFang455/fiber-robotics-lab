from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


@pytest.mark.parametrize("channel", [1, 2, 3])
def test_calibration_recomputes_angle_after_channel_failure(channel):
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.slider(key="hand_bend_angle").set_value(45.0)
    app.selectbox(key="global_failed_channel").set_value(f"手部 FBG {channel}").run()

    assert not app.exception
    angle = next(item for item in app.metric if item.label == "FBG 融合角")
    # 三路中一路置零，其余两路不变；普通平均融合应由 45° 降为 30°。
    assert angle.value == "30.00 °"
    assert angle.delta == "误差 -15.00 °"

    app.selectbox(key="global_failed_channel").set_value("无").run()
    assert not app.exception
    assert next(item for item in app.metric if item.label == "FBG 融合角").value == "45.00 °"


@pytest.mark.parametrize("temperature", [-10.0, 20.0])
def test_calibration_compensates_known_temperature(temperature):
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.slider(key="hand_bend_angle").set_value(45.0)
    app.slider(key="global_temperature").set_value(temperature).run()

    assert not app.exception
    assert next(item for item in app.metric if item.label == "FBG 融合角").value == "45.00 °"


def test_calibration_noise_is_reproducible():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.slider(key="hand_bend_angle").set_value(45.0)
    app.number_input(key="global_seed").set_value(17)
    app.slider(key="global_noise").set_value(0.02).run()

    assert not app.exception
    first = next(item for item in app.metric if item.label == "FBG 融合角").value
    assert first != "45.00 °"
    app.run()
    assert not app.exception
    assert next(item for item in app.metric if item.label == "FBG 融合角").value == first


def test_calibration_fault_fusion_uses_current_geometry_and_temperature():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.slider(key="hand_bend_angle").set_value(60.0)
    app.slider(key="calibration_length").set_value(100.0)
    app.slider(key="calibration_offset").set_value(1.5)
    app.selectbox(key="calibration_attachment").set_value("粘接式")
    app.slider(key="global_temperature").set_value(20.0)
    app.slider(key="global_drift").set_value(0.01)
    app.selectbox(key="global_failed_channel").set_value("手部 FBG 2").run()

    assert not app.exception
    # 独立代入模型公式：两路正常 + 0.01 nm 故障读数，温补后平均得 39.73307°。
    angle = next(item for item in app.metric if item.label == "FBG 融合角")
    assert angle.value == "39.73 °"
    assert angle.delta == "误差 -20.27 °"


def test_calibration_reset_preserves_public_and_other_page_parameters(caplog):
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    changed = {
        "真实弯曲角 (°)": (45.0, 0.0),
        "手指长度 (mm)": (100.0, 80.0),
        "光纤距中性层偏置 (mm)": (1.5, 1.0),
        "冗余通道真实弯曲角 (°)": (60.0, 45.0),
        "故障通道": (3, 2),
        "真实接触位置 (mm)": (25.0, 37.0),
        "真实法向力 (N)": (6.0, 4.0),
        "封装传力宽度 (mm)": (20.0, 12.0),
    }
    for label, (value, _) in changed.items():
        next(item for item in app.slider if item.label == label).set_value(value)
    next(item for item in app.selectbox if item.label == "光纤连接方式").set_value("粘接式")
    next(item for item in app.selectbox if item.label == "故障类型").set_value("漂移")
    app.slider(key="global_temperature").set_value(20.0)
    app.slider(key="global_noise").set_value(0.01)
    app.slider(key="global_drift").set_value(0.005)
    app.select_slider(key="global_sample_rate").set_value(100)
    app.selectbox(key="global_failed_channel").set_value("手部 FBG 2")
    app.selectbox(key="sole_assembly_case").set_value("压入不足")
    app.number_input(key="global_seed").set_value(17)
    app.slider(key="three_d_shoulder").set_value(50.0).run()
    assert not app.exception
    public_keys = ("global_temperature", "global_noise", "global_drift", "global_sample_rate",
                   "global_failed_channel", "sole_assembly_case", "global_seed", "three_d_shoulder")
    preserved = {key: app.session_state[key] for key in public_keys}

    caplog.clear()
    app.button(key="reset_calibration").click().run()

    assert not app.exception
    assert "was created with a default value" not in caplog.text
    for label, (_, default) in changed.items():
        assert next(item for item in app.slider if item.label == label).value == default
    assert next(item for item in app.selectbox if item.label == "光纤连接方式").value == "嵌入式"
    assert next(item for item in app.selectbox if item.label == "故障类型").value == "无"
    assert {key: app.session_state[key] for key in public_keys} == preserved


def test_public_preset_does_not_reset_calibration_controls():
    app = AppTest.from_file(APP_PATH, default_timeout=20).run()
    app.slider(key="hand_bend_angle").set_value(45.0)
    next(item for item in app.slider if item.label == "手指长度 (mm)").set_value(100.0)
    app.slider(key="global_temperature").set_value(20.0)
    app.selectbox(key="global_failed_channel").set_value("手部 FBG 1").run()
    app.button(key="demo_preset").click().run()

    assert not app.exception
    assert app.slider(key="global_temperature").value == 0.0
    assert app.selectbox(key="global_failed_channel").value == "无"
    assert app.slider(key="hand_bend_angle").value == 45.0
    assert next(item for item in app.slider if item.label == "手指长度 (mm)").value == 100.0
