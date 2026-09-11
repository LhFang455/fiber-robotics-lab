"""Validated Blender structures and explicit adapters to existing teaching data.

No solver or signal generation belongs here. Mesh order is never a channel map.
"""
from copy import deepcopy
from functools import lru_cache
import json
import math
from pathlib import Path

ROOT = Path(__file__).parent
ASSET_FILE = ROOT / 'assets/lab_assets.json'
SUPPORTED = frozenset({'skin', 'foot', 'assembly', 'health'})


class AssetError(ValueError):
    pass


def vector(value, size=3):
    return isinstance(value, list) and len(value) == size and all(
        isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in value)


def validate_asset(asset):
    parts = asset.get('parts', [])
    if not parts or len(parts) > 200:
        raise AssetError('Empty or oversized asset')
    ids = [p.get('id') for p in parts]
    if any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise AssetError('Part IDs must be non-empty and unique')
    by_id = {p['id']: p for p in parts}
    for p in parts:
        if not vector(p.get('position')) or not vector(p.get('color')):
            raise AssetError('Invalid part transform or color')
        if not isinstance(p.get('opacity'), (int, float)) or not 0 <= p['opacity'] <= 1:
            raise AssetError('Invalid opacity')
        vertices, indices = p.get('vertices'), p.get('indices')
        if not isinstance(vertices, list) or len(vertices) % 3 or len(vertices) > 300000:
            raise AssetError('Invalid vertex buffer')
        if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in vertices):
            raise AssetError('Non-finite geometry')
        if not isinstance(indices,list) or len(indices)%3 or any(type(i) is not int or i<0 or i>=len(vertices)//3 for i in indices):
            raise AssetError('Invalid triangle indices')
        normals=p.get('normals')
        if normals is not None and (not isinstance(normals,list) or len(normals)!=len(vertices) or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in normals)):
            raise AssetError('Invalid normal buffer')
        seen={p['id']}
        parent=p.get('parent')
        while parent is not None:
            if parent not in by_id or parent in seen:
                raise AssetError('Missing parent or hierarchy cycle')
            seen.add(parent)
            parent=by_id[parent].get('parent')
    return asset


@lru_cache(maxsize=2)
def _catalog(stamp):
    catalog=json.loads(ASSET_FILE.read_text())
    if catalog.get('schema')!=1 or catalog.get('coordinates')!='right-handed Y-up':
        raise AssetError('Unsupported asset contract')
    for asset in catalog['assets'].values():
        validate_asset(asset)
    return catalog


def load_asset(kind):
    return deepcopy(_catalog(ASSET_FILE.stat().st_mtime_ns)['assets'][kind])


def binding(pid, sid, label, local, index, *, readings=None, model_position=None):
    return dict(id=sid, part=pid, label=label, local=local, normal=[0.,1.,0.],
        signal_index=index, model_position=model_position,
        readings=readings or [dict(key='signals', index=index, label='波长变化', unit='nm')])


def validate_bindings(asset, payload):
    part_ids={p['id'] for p in asset['parts']}
    ids=set()
    for b in asset['bindings']:
        if b['id'] in ids or b['part'] not in part_ids or not vector(b['local']) or not vector(b['normal']):
            raise AssetError('Invalid sensor binding')
        if abs(sum(v*v for v in b['normal'])-1)>1e-6:
            raise AssetError('Mounting normal must have unit length')
        ids.add(b['id'])
        for r in b['readings']:
            if not r['unit'] or type(r['index']) is not int or r['index']<0:
                raise AssetError('Invalid signal reference')
            for f in payload['frames']:
                if r['key'] not in f or r['index']>=len(f[r['key']]):
                    raise AssetError('Signal channel is missing from a frame')
    return asset


def scene_asset(payload):
    kind=payload['kind']
    if kind not in SUPPORTED:
        return None
    asset=load_asset(kind)
    bindings=[]
    asset['scale']=[1.,1.,1.]
    if kind=='skin':
        w,h=payload['width'],payload['height']
        if not all(math.isfinite(v) and v>0 for v in (w,h)):
            raise AssetError('Invalid skin dimensions')
        asset['scale']=[w/80,1.,h/60]
        n=len(payload['sensors'])
        for i,(x,y) in enumerate(payload['sensors']):
            bindings.append(binding('SKIN-SENSE',f'SKIN-N{n:02d}-{i+1:02d}',payload['labels'][i],
                [x-w/2,1.4,h/2-y],i,model_position=[x,y],
                readings=[dict(key='signals',index=i,label='温补波长',unit='nm')]))
        asset['note']='XY 平面位置来自当前教学阵列；厚度和凹陷仅为示意。展开时测点随光纤承载层移动，读数不因展开改变。连线只表示示意连接顺序，不是已设计的真实走线。'
    elif kind=='foot':
        for i in range(6):
            b=binding(f'FOOT-Z{i+1}',f'SOLE-{i+1:02d}',payload['labels'][i],[0.,2.,0.],i,
                readings=[dict(key='signals',index=i,label='波长变化',unit='nm'),dict(key='loads',index=i,label='反演区域载荷',unit='N')])
            b['valid_key']='valid_loads'
            bindings.append(b)
        asset['note']='测点属于六个载荷区域；局部坐标为原演示尺度，非实物安装尺寸。CoP 保留原归一化坐标映射；失效区域不显示为零载荷。'
    elif kind=='health':
        n=len(payload['sensors'])
        for i,x in enumerate(payload['sensors']):
            bindings.append(binding('HEALTH-BEAM',f'ARM-N{n:02d}-{i+1:02d}',payload['labels'][i],[x-260,19.,0.],i,model_position=[x]))
        asset['note']='测点纵向毫米位置沿用当前梁模型；法向安装高度、支撑和截面仍为示意，不构成实物标定。'
    else:
        for i,(x,z) in enumerate([(-14,0),(14,0),(23,34)]):
            bindings.append(binding('ASSEMBLY-L6',f'ASSEMBLY-FBG-{i+1}',payload['labels'][i],[float(x),3.,float(z)],i))
        asset['note']='工作与参考 FBG 同属固定感知芯；可更换件的错位不迁移测点。就位前没有有效装配读数；播放横轴不是实际时间。'
    asset['bindings']=bindings
    asset['initial_binding']=bindings[payload.get('inspection_channel') or 0]['id'] if bindings else None
    asset['provenance']='Blender 原创结构 + 原实验数据适配；本页为教学模拟'
    return validate_bindings(asset,payload)
