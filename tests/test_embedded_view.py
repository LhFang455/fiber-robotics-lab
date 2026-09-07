import base64
import json
import re

from fiber_robotics_sim import embedded_view


def config_from_markup(markup):
    return json.loads(re.search(r"const config = (.*);", markup).group(1))


def test_isolated_frame_preserves_unicode_and_script_content():
    document = '<html><body>三轴力<script>const x = "</script>";</script></body></html>'
    markup = embedded_view.frame_html(document, key="taxel", title="三轴演示")
    config = config_from_markup(markup)
    assert config["title"] == "三轴演示"
    assert config["source"].startswith("data:text/html;charset=utf-8;base64,")
    decoded = base64.b64decode(config["source"].split(",", 1)[1]).decode()
    assert decoded.startswith(document)
    assert "fiber-frame-height" in decoded
    assert config["autoHeight"] is True
    assert '<iframe srcdoc=' not in markup


def test_fixed_height_view_keeps_its_original_dimensions():
    markup = embedded_view.frame_html("<svg></svg>", key="chain", height=220, title="感知链")
    config = config_from_markup(markup)
    assert config["height"] == 220
    assert config["autoHeight"] is False
    decoded = base64.b64decode(config["source"].split(",", 1)[1]).decode()
    assert decoded.startswith("<svg></svg>")
    assert "html{background:#0e1117}" in decoded
    assert "fiber-frame-height" not in decoded


def test_frame_identity_is_stable_per_slot_and_distinct_between_slots():
    a = config_from_markup(embedded_view.frame_html("A", key="a"))
    changed = config_from_markup(embedded_view.frame_html("B", key="a"))
    other = config_from_markup(embedded_view.frame_html("A", key="b"))
    assert a["id"] == changed["id"]
    assert a["id"] != other["id"]
    assert a["source"] != changed["source"]
