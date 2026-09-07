import base64
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from fiber_robotics_sim import eskin, showcase


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def showcase_payloads(app):
    payloads = []
    for item in app.get("html"):
        source = re.search(r"data:text/html;charset=utf-8;base64,([A-Za-z0-9+/=]+)", item.proto.body)
        if not source:
            continue
        document = base64.b64decode(source.group(1)).decode()
        match = re.search(r'<script id="demo-config" type="application/json">(.*?)</script>', document, re.S)
        if match:
            payload = json.loads(match.group(1))
            if payload.get("presentation_context") == "showcase":
                payloads.append(payload)
    return payloads


def test_showcase_presets_are_reproducible_and_pressure_comparison_is_fair():
    assert [item["id"] for item in showcase.CHAPTERS] == ["foot", "fbg", "pressure"]
    first = showcase.demo_parameters("pressure", 4)
    second = showcase.demo_parameters("pressure", 8)
    assert {key: value for key, value in first.items() if key != "sparse_size"} == {
        key: value for key, value in second.items() if key != "sparse_size"
    }
    rows = showcase.pressure_comparison()
    assert [row["通道数"] for row in rows] == [16, 64]
    for size, row in zip((4, 8), rows, strict=True):
        expected = eskin.simulate_pressure_reconstruction(**showcase.demo_parameters("pressure", size))
        assert row["压力 RMSE (kPa)"] == expected["rmse_kpa"]
        assert row["峰值误差 (%)"] == expected["peak_error_pct"]
    assert [label for label, _, _ in showcase.chapter_metrics("pressure")] == [
        "采样通道", "压力 RMSE", "峰值误差", "质心误差",
    ]
    with pytest.raises(ValueError):
        showcase.demo_parameters("pressure", 6)


def test_showcase_renders_only_after_start_and_keeps_global_state_isolated():
    app = AppTest.from_file(APP_PATH, default_timeout=45).run()
    assert not app.exception
    assert showcase_payloads(app) == []
    app.slider(key="global_temperature").set_value(18.0)
    app.slider(key="global_noise").set_value(0.01)
    app.selectbox(key="global_failed_channel").set_value("手部 FBG 2")
    app.toggle(key="showcase_enabled").set_value(True).run()
    assert not app.exception
    payloads = showcase_payloads(app)
    assert len(payloads) == 1 and payloads[0]["kind"] == "foot"
    assert app.slider(key="global_temperature").value == 18.0
    assert app.slider(key="global_noise").value == 0.01
    assert app.selectbox(key="global_failed_channel").value == "手部 FBG 2"

    app.button(key="showcase_next").click().run()
    assert not app.exception
    payloads = showcase_payloads(app)
    assert len(payloads) == 1 and payloads[0]["kind"] == "fbg"
    assert payloads[0]["frames"][-1]["temperature"] == 20.0

    app.button(key="showcase_next").click().run()
    assert not app.exception
    payloads = showcase_payloads(app)
    assert len(payloads) == 1 and payloads[0]["kind"] == "pressure"
    assert len(payloads[0]["samples"]) == 4
    app.radio(key="showcase_pressure_density").set_value("8×8 · 64 通道").run()
    assert not app.exception
    payloads = showcase_payloads(app)
    assert len(payloads) == 1 and len(payloads[0]["samples"]) == 8
    app.button(key="tab_jump_6_进入完整实验").click().run()
    assert not app.exception
    assert app.selectbox(key="sidebar_module_navigation").value == "⑦ 电子皮肤与多模态感知"
    assert app.session_state["eskin_navigation"] == "稀疏压力重建"
