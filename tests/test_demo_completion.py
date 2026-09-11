import base64
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import demo_views, demos

ROOT = Path(__file__).resolve().parents[1]


def payloads(app):
    result = {}
    for item in app.get('html'):
        match = re.search(r'data:text/html;charset=utf-8;base64,([A-Za-z0-9+/=]+)', item.proto.body)
        if not match:
            continue
        document = base64.b64decode(match[1]).decode()
        config = re.search(r'<script id="demo-config" type="application/json">(.*?)</script>', document, re.S)
        if config:
            value = json.loads(config[1])
            result[value['kind']] = value
    return result


def test_all_eight_observers_link_and_hidden_demo_preserves_selection():
    app = AppTest.from_file(ROOT / 'app.py', default_timeout=40).run()
    for key, selected in [('foot_inspector_zone', 2), ('skin_inspect_channel', 3), ('health_inspect_channel', 4), ('taxel_inspect_channel', 4), ('pressure_inspect_row', 3), ('pressure_inspect_col', 4)]:
        app.selectbox(key=key).set_value(selected)
    for key, selected in [('shape_inspect_node', 120), ('dynamic_inspect_index', 100)]:
        app.slider(key=key).set_value(selected)
    app.select_slider(key="distributed_inspect_index").set_value(40)
    app.run()
    assert not app.exception
    data = payloads(app)
    for kind, selected in [('foot', 2), ('skin', 3), ('health', 4), ('taxel', 4)]:
        assert data[kind]['inspection_channel'] == selected
    assert data['shape']['inspection_node'] == 120
    assert data['distributed']['initial_index'] == 40
    assert data['dynamic']['initial_index'] == 100
    assert data['pressure']['inspection_node']['row'] == 3
    assert data['pressure']['inspection_node']['col'] == 4
    markup = '\n'.join(x.value for x in app.markdown)
    for kind in demo_views.LINKED_INSPECTORS:
        assert f'id="demo-{kind}"' in markup
        assert f'id="inspector-{kind}"' in markup
        assert f'href="#demo-{kind}"' in markup
        assert f'href="#inspector-{kind}"' in markup
    captions = '\n'.join(x.value for x in app.caption)
    assert '点击‘观察器位置’' in captions
    assert '点击‘观察器时刻’' in captions
    app.toggle(key='show_taxel_demo').set_value(False).run()
    assert not app.exception
    assert app.selectbox(key='taxel_inspect_channel').value == 4
    assert 'taxel' not in payloads(app)
    app.toggle(key='show_taxel_demo').set_value(True).run()
    assert payloads(app)['taxel']['inspection_channel'] == 4


def test_javascript_missing_zero_negative_and_script_syntax():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is required for the readout runtime check')
    script = (ROOT/'fiber_robotics_sim/demo_readout.js').read_text()
    script += '''
const assert = require('node:assert/strict');
const origin = {yaw: .65, elevation: .65};
assert.deepEqual(demoViewKey('Home', 5, 1, origin), origin);
assert.equal(demoViewKey('ArrowLeft', 0, .65, origin).yaw, .12);
assert.equal(demoViewKey('ArrowRight', 0, .65, origin).yaw, -.12);
assert.equal(demoViewKey('ArrowUp', 0, 1.42, origin).elevation, 1.42);
assert.equal(demoViewKey('ArrowDown', 0, .12, origin).elevation, .12);
assert.equal(demoViewKey('Tab', 0, .65, origin), null);
const labels = [{visible:true},{visible:true}], sensor = {visible:true};
demoSetLayerVisibility(labels, false);
assert.ok(labels.every(label => !label.visible));assert.equal(sensor.visible, true);
demoSetLayerVisibility(labels, true);assert.ok(labels.every(label => label.visible));
for (const value of [null, undefined, NaN, Infinity]) {
  const r = demoSignalReadout(value, 10, false);
  assert.equal(r.width, 0); assert.match(r.text, /缺测/);
}
assert.equal(demoSignalReadout(0, 10, false).text, '0.0000');
assert.equal(demoSignalReadout(-5, 10, false).width, 50);
assert.equal(demoSignalReadout(-5, 10, false).text, '-5.0000');
assert.equal(demoSignalReadout(null, 10, false, '就位后读取').text, '就位后读取');
assert.equal(demoSignalReadout(.001, 10, true).text, '1.000e-3');
'''
    subprocess.run([node, '-e', script], check=True, capture_output=True, text=True)
    html = demos.demo_html({'kind': 'assembly'})
    for source in re.findall(r'<script>(.*?)</script>', html, re.S):
        subprocess.run([node, '--check'], input=source, text=True, capture_output=True, check=True)


def test_failed_foot_readout_is_missing_but_model_input_preserved():
    p = dict(load_n=180., terrain='平地', phase_percent=57., support='支撑期', temperature_c=10., noise_nm=0., seed=7, failed_zone=3, drift_nm=0.)
    d = demos.foot_demo(p)
    for frame in d['frames']:
        assert frame['signals'][2] is None
        assert frame['model_input_signals'][2] is not None
        assert not frame['valid_loads'][2]
    assert demos.assembly_demo({'assembly_case':'正常装配','temperature_c':0})['pending_signal_text'] == '就位后读取'
