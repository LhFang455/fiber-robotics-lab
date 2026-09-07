import base64
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def demo_payloads(app):
    result = {}
    for item in app.get("html"):
        source = re.search(r"data:text/html;charset=utf-8;base64,([A-Za-z0-9+/=]+)", item.proto.body)
        if not source:
            continue
        document = base64.b64decode(source.group(1)).decode()
        match = re.search(r'<script id="demo-config" type="application/json">(.*?)</script>', document, re.S)
        if match:
            payload = json.loads(match.group(1))
            result[payload["kind"]] = payload
    return result


def test_demo_frames_follow_existing_parameters_and_can_be_hidden_without_losing_experiments():
    app = AppTest.from_file(APP_PATH, default_timeout=35).run()
    assert not app.exception
    assert set(demo_payloads(app)) == {"foot", "skin", "shape", "assembly", "health", "fbg", "distributed", "tactile", "optical", "dynamic", "taxel", "pressure"}

    app.slider(key="foot_load").set_value(120.0).run()
    assert not app.exception
    foot = demo_payloads(app)["foot"]
    current = foot["frames"][foot["initial_index"]]
    assert abs(sum(current["true_loads"]) - 120.0) < 1e-8
    assert current["phase"] == app.slider(key="foot_phase").value

    app.toggle(key="show_foot_demo").set_value(False).run()
    assert not app.exception
    assert "foot" not in demo_payloads(app)
    assert app.slider(key="foot_load").value == 120.0
    assert "下载足底实验报告" in {item.label for item in app.get("download_button")}
    assert len(app.selectbox(key="sidebar_module_navigation").options) == 7


def test_second_batch_tracks_parameters_and_preserves_existing_downloads():
    app = AppTest.from_file(APP_PATH, default_timeout=35).run()
    app.selectbox(key="sole_assembly_case").set_value("单侧错位")
    app.slider(key="anomaly_severity").set_value(.8)
    app.slider(key="hand_bend_angle").set_value(45.0)
    app.slider(key="global_temperature").set_value(20.0)
    app.run()
    assert not app.exception
    payloads = demo_payloads(app)
    assert payloads["assembly"]["lateral_offset_mm"] == 4
    assert payloads["health"]["frames"][-1]["truth_location"] == 320
    assert abs(payloads["fbg"]["frames"][-1]["estimated_angle"] - 45) < 1e-8
    for kind in ("assembly", "health", "fbg"):
        app.toggle(key=f"show_{kind}_demo").set_value(False)
    app.run()
    assert not app.exception
    assert not {"assembly", "health", "fbg"}.intersection(demo_payloads(app))
    labels = {item.label for item in app.get("download_button")}
    assert {"下载装配验证参数摘要", "下载健康监测报告", "下载手部 FBG 读数 CSV"}.issubset(labels)


def test_third_batch_tracks_mechanisms_and_preserves_exports():
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    app.selectbox(key="distributed_mode").set_value("Raman")
    app.selectbox(key="tactile_material").set_value("硬块")
    app.selectbox(key="optical_demo_mechanism").set_value("Sagnac 环路")
    app.slider(key="optical_gyro_rate").set_value(-90.0)
    app.run()
    assert not app.exception
    payloads = demo_payloads(app)
    assert payloads["distributed"]["unit"] == "°C"
    assert payloads["tactile"]["selected"] == 1
    assert payloads["optical"]["frames"][-1]["phase"] < 0
    app.selectbox(key="optical_demo_mechanism").set_value("EFPI 微腔").run()
    assert not app.exception
    assert demo_payloads(app)["optical"]["mechanism"] == "EFPI 微腔"
    for kind in ("distributed", "tactile", "optical"):
        app.toggle(key=f"show_{kind}_demo").set_value(False)
    app.run()
    assert not app.exception
    assert not {"distributed", "tactile", "optical"}.intersection(demo_payloads(app))
    assert {"下载触觉实验报告", "下载分布式实验报告", "下载偏振与干涉报告"}.issubset({item.label for item in app.get("download_button")})
    assert len(app.selectbox(key="sidebar_module_navigation").options) == 7


def test_distributed_unsupported_truth_and_flat_profiles_render_without_false_metrics():
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    app.selectbox(key="distributed_mode").set_value("Brillouin").run()
    assert not app.exception
    payload = demo_payloads(app)["distributed"]
    assert payload["truth"] is None and payload["estimate"] is None
    app.selectbox(key="distributed_mode").set_value("Raman")
    app.slider(key="distributed_event_strength").set_value(0.0)
    app.run()
    assert not app.exception
    assert demo_payloads(app)["distributed"]["estimate"] is None


def test_dynamic_replay_tracks_event_and_preserves_original_records():
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    app.selectbox(key="eskin_dynamic_event").set_value("即将滑移")
    app.slider(key="eskin_dynamic_noise").set_value(0.0)
    app.run()
    assert not app.exception
    payload = demo_payloads(app)["dynamic"]
    assert len(payload["frames"]) == 400
    assert payload["frames"][-1]["alert"]
    app.selectbox(key="eskin_dynamic_event").set_value("热物体")
    app.slider(key="eskin_dynamic_duration").set_value(2.0)
    app.run()
    assert not app.exception
    payload = demo_payloads(app)["dynamic"]
    assert len(payload["frames"]) == 200
    assert not payload["frames"][-1]["alert"]
    app.toggle(key="show_dynamic_demo").set_value(False).run()
    assert not app.exception
    assert "dynamic" not in demo_payloads(app)
    assert app.slider(key="eskin_dynamic_duration").value == 2.0
    assert {"下载动态事件 CSV", "下载动态事件报告"}.issubset({item.label for item in app.get("download_button")})


def test_taxel_pressure_demos_track_inputs_and_keep_analysis_and_exports():
    app = AppTest.from_file(APP_PATH, default_timeout=40).run()
    app.slider(key="eskin_fx_n").set_value(-3.0)
    app.slider(key="eskin_taxel_temperature").set_value(50.0)
    app.selectbox(key="eskin_pressure_scenario").set_value("双点接触")
    app.select_slider(key="eskin_sparse_size").set_value(8)
    app.select_slider(key="eskin_output_size").set_value(32)
    app.run()
    assert not app.exception
    payloads = demo_payloads(app)
    assert payloads["taxel"]["frames"][-1]["truth"][0] == -3
    assert payloads["taxel"]["frames"][-1]["temperature"] == 50
    assert len(payloads["pressure"]["samples"]) == 8
    assert len(payloads["pressure"]["reconstruction"]) == 32
    assert payloads["pressure"]["frames"][-1]["read_count"] == 64
    for kind in ("taxel", "pressure"):
        app.toggle(key=f"show_{kind}_demo").set_value(False)
    app.run()
    assert not app.exception
    assert not {"taxel", "pressure"}.intersection(demo_payloads(app))
    assert {"下载三轴单元报告", "下载压力重建报告"}.issubset({item.label for item in app.get("download_button")})
    assert len(app.selectbox(key="sidebar_module_navigation").options) == 7
