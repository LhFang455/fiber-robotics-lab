import base64
from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest
from streamlit.testing.v1 import AppTest
from fiber_robotics_sim import demos, model_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenes():
    app = AppTest.from_file(ROOT / "app.py", default_timeout=40).run()
    assert not app.exception
    assert not any(x.key == "skin_layers_enabled" for x in app.toggle)
    result = {}
    for item in app.get("html"):
        match = re.search(
            r"data:text/html;charset=utf-8;base64,([A-Za-z0-9+/=]+)", item.proto.body
        )
        if not match:
            continue
        document = base64.b64decode(match[1]).decode()
        config = re.search(
            r'<script id="demo-config" type="application/json">(.*?)</script>',
            document,
            re.S,
        )
        if config:
            value = json.loads(config[1])
            result[value["kind"]] = value
    return result


def test_existing_scenes_use_explicit_sensor_contract_without_new_lab(scenes):
    for kind in model_assets.SUPPORTED:
        p = scenes[kind]
        assert p.get("asset_warning") is None
        assert p["asset"]["bindings"]
        model_assets.validate_bindings(p["asset"], p)
        untouched = {k: v for k, v in p.items() if k != "asset"}
        html = demos.demo_html(untouched)
        after = json.loads(
            re.search(
                r'<script id="demo-config" type="application/json">(.*?)</script>',
                html,
                re.S,
            )[1]
        )
        assert after["frames"] == p["frames"]
        assert after["initial_index"] == p["initial_index"]
    for kind in ("taxel", "pressure", "dynamic", "shape", "distributed"):
        assert scenes[kind]["asset"] is None


@pytest.mark.parametrize(
    "fault", ["duplicate", "cycle", "missing_parent", "nonfinite", "index"]
)
def test_asset_validator_rejects_invalid_geometry_or_hierarchy(fault):
    a = model_assets.load_asset("foot")
    if fault == "duplicate":
        a["parts"][1]["id"] = a["parts"][0]["id"]
    if fault == "cycle":
        a["parts"][0]["parent"] = a["parts"][1]["id"]
    if fault == "missing_parent":
        a["parts"][0]["parent"] = "missing"
    if fault == "nonfinite":
        a["parts"][0]["vertices"][0] = float("nan")
    if fault == "index":
        a["parts"][0]["indices"][0] = len(a["parts"][0]["vertices"])
    with pytest.raises(model_assets.AssetError):
        model_assets.validate_asset(a)


def test_bad_bindings_fail_closed_and_renderer_keeps_original_fallback(
    scenes, monkeypatch
):
    p = deepcopy(scenes["skin"])
    a = p["asset"]
    a["bindings"][0]["readings"][0]["index"] = 999
    with pytest.raises(model_assets.AssetError):
        model_assets.validate_bindings(a, p)

    def broken(_):
        raise model_assets.AssetError("bad source")

    monkeypatch.setattr(model_assets, "scene_asset", broken)
    html = demos.demo_html(scenes["skin"])
    result = json.loads(
        re.search(
            r'<script id="demo-config" type="application/json">(.*?)</script>',
            html,
            re.S,
        )[1]
    )
    assert result["asset"] is None
    assert "原程序模型" in result["asset_warning"]
    assert result["frames"] == scenes["skin"]["frames"]
    assert "if(surface)" in html


def test_node_runtime_world_positions_follow_parts_and_missing_is_not_zero(scenes):
    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const THREE=require('./fiber_robotics_sim/vendor/three.min.js');
vm.runInThisContext(fs.readFileSync('./fiber_robotics_sim/model_assets.js','utf8'));
const scenes=JSON.parse(fs.readFileSync(0,'utf8'));
function create(p){const a=createLabAsset(THREE,new THREE.Scene(),p.asset);p.asset.bindings.forEach((b,i)=>a.attach(i,new THREE.Mesh(new THREE.SphereGeometry(1),new THREE.MeshStandardMaterial())));return a;}
function near(a,b){assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-5,`${a} != ${b}`));}
const skin=scenes.skin,a=create(skin),b=skin.asset.bindings[0];
a.deformSkin(()=>-2);const start=a.world(b.id).toArray();
near(start,[b.local[0],-3+b.local[1]-2,b.local[2]]);
a.explosion=1;a.deformSkin(()=>-2);const end=a.world(b.id).toArray();near(end,[start[0],start[1]+14,start[2]]);
a.showStructure(false);near(a.world(b.id).toArray(),end);assert.equal(a.sensors.get(b.id).visible,true);
const foot=create(scenes.foot);
scenes.foot.asset.bindings.forEach((b,i)=>near(foot.world(b.id).toArray(),[(i%3-1)*27,11,i<3?-43:43]));
const health=create(scenes.health);scenes.health.asset.bindings.forEach((b,i)=>near(health.world(b.id).toArray(),[scenes.health.sensors[i]-260,19,0]));
const assembly=create(scenes.assembly),sensor=scenes.assembly.asset.bindings[0];
assembly.parts.get('ASSEMBLY-L6').position.y=27;const seated=assembly.world(sensor.id).toArray();
assembly.parts.get('ASSEMBLY-L1').position.set(20,-80,0);near(assembly.world(sensor.id).toArray(),seated);
assembly.parts.get('ASSEMBLY-L6').position.y=12;near(assembly.world(sensor.id).toArray(),[seated[0],seated[1]-15,seated[2]]);
const f=structuredClone(scenes.foot.frames[0]),binding=scenes.foot.asset.bindings[0];
f.valid_loads[0]=false;f.signals[0]=0;f.loads[0]=0;assert.ok(labBindingValues(binding,f).every(r=>r.value===null));
f.valid_loads[0]=true;assert.ok(labBindingValues(binding,f).every(r=>r.value===0));
f.signals[0]=null;assert.equal(labBindingValues(binding,f)[0].value,null);
const pending=scenes.assembly.frames.find(f=>!f.seated);assert.ok(pending);assert.equal(labBindingValues(sensor,pending)[0].value,null);
console.log('world transforms, follow, validity and missing values verified');
"""
    result = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        input=json.dumps(scenes),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "world transforms" in result.stdout


def test_glb_keeps_semantic_ids_and_parent_relationships():
    import struct

    raw = (ROOT / "fiber_robotics_sim/assets/lab_assets.glb").read_bytes()
    assert struct.unpack("<4sII", raw[:12]) == (b"glTF", 2, len(raw))
    length, kind = struct.unpack("<II", raw[12:20])
    assert kind == 0x4E4F534A
    gltf = json.loads(raw[20 : 20 + length])
    nodes = gltf["nodes"]
    by_id = {
        n["extras"]["part_id"]: i
        for i, n in enumerate(nodes)
        if "part_id" in n.get("extras", {})
    }
    for key in model_assets.SUPPORTED:
        for p in model_assets.load_asset(key)["parts"]:
            assert p["id"] in by_id
            if p["parent"]:
                assert by_id[p["id"]] in nodes[by_id[p["parent"]]]["children"]


def test_inspector_selection_and_scan_callbacks_without_browser_privileges(scenes):
    if not shutil.which("node"):
        pytest.skip("Node unavailable")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const THREE=require('./fiber_robotics_sim/vendor/three.min.js');
vm.runInThisContext(fs.readFileSync('./fiber_robotics_sim/model_assets.js','utf8'));
class Element {
 constructor(tag){this.tag=tag;this.children=[];this.attrs={};this.style={};this.value='';this.classList={add:()=>{},toggle:(k,v)=>this.attrs[k]=v};}
 append(...children){this.children.push(...children)} before(e){this.previous=e}
 setAttribute(k,v){this.attrs[k]=String(v)} getAttribute(k){return this.attrs[k]}
 get options(){return this.children.filter(e=>e.tag==='option')}
 getBoundingClientRect(){return {left:0,width:600}}
}
const elements=[];function make(tag){const e=new Element(tag);elements.push(e);return e}
const caption=make('caption');global.document={createElement:make,createElementNS:(_,tag)=>make(tag),getElementById:()=>caption};
const cfg=JSON.parse(fs.readFileSync(0,'utf8'));const asset=createLabAsset(THREE,new THREE.Scene(),cfg.asset);
cfg.asset.bindings.forEach((b,i)=>asset.attach(i,new THREE.Mesh(new THREE.SphereGeometry(1),new THREE.MeshStandardMaterial())));
const rows=cfg.labels.map(()=>({element:make('row')}));let changes=0,seeks=[];
const inspector=createLabAssetInspector(asset,THREE,cfg,make('view'),rows,i=>seeks.push(i),()=>changes++);
inspector.initialize();const select=elements.find(e=>e.tag==='select'),info=elements.find(e=>e.attrs['aria-live']==='polite'),svg=elements.find(e=>e.tag==='svg');
rows[1].element.onclick();assert.equal(select.value,cfg.asset.bindings[1].id);assert.ok(info.textContent.includes(cfg.asset.bindings[1].label));assert.ok(elements.find(e=>e.className==='asset-technical').textContent.includes(cfg.asset.bindings[1].id));assert.equal(rows[1].element.attrs['asset-selected'],true);
select.value='SKIN-COVER';select.onchange();assert.equal(svg.style.display,'none');assert.ok(info.textContent.includes('不是独立数据通道'));
select.value=cfg.asset.bindings[0].id;select.onchange();assert.equal(svg.style.display,'block');
svg.onclick({clientX:300});assert.equal(seeks[0],60);
const slider=elements.find(e=>e.attrs['aria-label']==='分层展开');slider.value='1';slider.oninput();assert.equal(asset.explosion,1);assert.ok(changes>=4);
const explode=elements.find(e=>e.tag==='button'&&e.textContent==='合起结构');explode.onclick();assert.equal(asset.explosion,0);assert.equal(explode.textContent,'拆开看结构');explode.onclick();assert.equal(asset.explosion,1);assert.equal(explode.textContent,'合起结构');
const frame=structuredClone(cfg.frames[0]);frame.signals[0]=null;inspector.update(frame,0);assert.ok(info.textContent.includes('缺测'));assert.ok(!info.textContent.includes('null nm'));
console.log('selection, data rows, part scope, seeking, explosion and missing UI verified');
"""
    result = subprocess.run(
        ["node", "-e", script],
        cwd=ROOT,
        input=json.dumps(scenes["skin"]),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
