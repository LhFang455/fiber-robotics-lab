"""Blender --background --python scripts/generate_lab_assets.py.
Original teaching assets derived from existing scene geometry, not a real CAD model.
Author meshes in existing web Y-up coordinates, store Blender Z-up coordinates.
"""
from pathlib import Path
import json
import math
import sys
import runpy
import bpy
import bmesh

OUT = Path(__file__).resolve().parents[1] / 'fiber_robotics_sim/assets'
if (OUT/'lab_assets.blend').exists() and '--regenerate' not in sys.argv:
    raise RuntimeError('Source exists. Use export_lab_assets.py to preserve manual edits; -- --regenerate explicitly rebuilds the generated source.')
CATALOG = {}
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)


def source(p):
    return (p[0], -p[2], p[1])


def web(p):
    return [float(p[0]), float(p[2]), -float(p[1])]


def box_vertices(size, grid=1):
    w, h, d = size
    vertices, faces = [], []
    for y in (-h/2, h/2):
        for row in range(grid+1):
            for col in range(grid+1):
                vertices.append((w*(col/grid-.5), y, d*(row/grid-.5)))
    n = (grid+1)**2
    for row in range(grid):
        for col in range(grid):
            a=row*(grid+1)+col
            faces.append((a,a+1,a+grid+2,a+grid+1))
            faces.append((a+n,a+grid+1+n,a+grid+2+n,a+1+n))
    border=list(range(grid+1))+[r*(grid+1)+grid for r in range(1,grid+1)]+[grid*(grid+1)+c for c in range(grid-1,-1,-1)]+[r*(grid+1) for r in range(grid-1,0,-1)]
    for a,b in zip(border,border[1:]+border[:1]):
        faces.append((a,b,b+n,a+n))
    return vertices, faces


def prism(points, height):
    n=len(points)
    vertices=[(x,y,z) for y in (0,height) for x,z in points]
    faces=[tuple(range(n-1,-1,-1)),tuple(range(n,2*n))]
    faces.extend((i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n))
    return vertices,faces


def part(asset, pid, label, vertices, faces, position=(0,0,0), parent=None, role='structure', color=(.35,.51,.61), note='', opacity=1):
    mesh=bpy.data.meshes.new(pid)
    mesh.from_pydata([source(v) for v in vertices], [], faces)
    mesh.update()
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces));bm.to_mesh(mesh);bm.free()
    obj=bpy.data.objects.new(pid,mesh)
    bpy.data.collections[asset].objects.link(obj)
    obj.location=source(position)
    obj['part_id']=pid
    obj['parent_id']=parent or ''
    obj['model_basis']='existing teaching geometry; not calibrated CAD'
    obj['role']=role
    obj['lab_label']=label
    obj['lab_note']=note
    mat=bpy.data.materials.new(pid)
    mat.diffuse_color=(*color,opacity)
    obj.data.materials.append(mat)
    if parent:
        obj.parent=bpy.data.objects[parent]
    obj['contract_matrix']=[float(v) for row in obj.matrix_basis for v in row]
    mesh.calc_loop_triangles()
    CATALOG[asset]['parts'].append(dict(id=pid,label=label,parent=parent,role=role,
        position=list(position),color=list(color),opacity=opacity,note=note,
        vertices=[v for vertex in mesh.vertices for v in web(vertex.co)],
        indices=[int(i) for tri in mesh.loop_triangles for i in tri.vertices]))
    return obj


def box(asset,pid,label,size,position=(0,0,0),**kw):
    grid=kw.pop('grid',1)
    return part(asset,pid,label,*box_vertices(size,grid),position,**kw)


def asset(key, name, coordinates):
    collection=bpy.data.collections.new(key);bpy.context.scene.collection.children.link(collection)
    collection['lab_asset_id']=key;collection['lab_label']=name;collection['lab_coordinates']=coordinates
    CATALOG[key]=dict(id=key,label=name,coordinates=coordinates,basis='教学结构，非实物 CAD',parts=[])

asset('skin','光学皮肤结构','平面 x/z 随实验宽高缩放；Y 为示意厚度方向')
for i,(pid,label,y,thickness,color,opacity) in enumerate([
    ('SKIN-BASE','支撑基底',-6.5,3,(.20,.30,.39),1),
    ('SKIN-BOND','粘接界面',-4.5,1,(.82,.57,.29),1),
    ('SKIN-SENSE','光纤承载层',-3,2,(.17,.58,.57),.65),
    ('SKIN-COVER','保护层',-1,2,(.52,.73,.82),.2),
]):
    box('skin',pid,label,(80,thickness,60),(0,y,0),color=color,opacity=opacity,grid=12,
        note='平面宽高来自当前实验；层厚、透明度及传力形变为展示假设。')

asset('foot','六区足底结构','保留原演示空间；CoP 到模型的仿射映射不变，非鞋底实测 mm')
# Sample the original quadratic outline rather than inventing a new sole shape.
outline=[(-38,82)]
segments=[((-49,80),(-47,25)),((-42,3),(-47,-40)),((-57,-92),(-3,-92)),((56,-90),(51,-42)),((44,2),(45,25)),((49,80),(38,82)),((0,90),(-38,82))]
for control,end in segments:
    start=outline[-1]
    for j in range(1,9):
        t=j/8
        outline.append(tuple((1-t)**2*start[k]+2*(1-t)*t*control[k]+t*t*end[k] for k in (0,1)))
outline=outline[:-1]
# Matches the old rotateX(-pi/2), then rotateY(pi) conversion.
outline=[(-x,z) for x,z in outline]
part('foot','FOOT-SOLE','足底承载体',*prism(outline,7),color=(.42,.53,.59),note='外形沿用旧演示轮廓；不代表真实鞋楦。')
for i in range(6):
    box('foot',f'FOOT-Z{i+1}',f'区域 {i+1}',(24,2,60),((i%3-1)*27,9,-43 if i<3 else 43),parent='FOOT-SOLE',role='region',color=(.2,.72,.74),note='六区位置沿用原模型映射；颜色来自当前反演载荷。')

asset('health','结构监测梁','X 与原梁模型毫米位置一致；截面和支架为示意')
box('health','HEALTH-BEAM','监测梁',(520,24,32),note='测点纵向位置来自原实验；无真实材料与截面标定。')
for i,x in enumerate([-230,230]):
    box('health',f'HEALTH-SUPPORT-{i+1}',f'支撑 {i+1}',(20,24,50),(x,-24,0),color=(.20,.28,.34),note='支撑位置沿用原演示，不计算边界接触。')

asset('assembly','可更换足底装配','沿用旧演示尺度；错位/压入不足仍按原逻辑放大 4 倍')
names=['耐磨外底','分区传力','隔离膜','密封圈','定位锁止','固定感知芯','基板限位']
colors=[(.34,.46,.55),(.91,.70,.42),(.48,.78,.80),(.20,.28,.35),(.65,.73,.79),(.30,.79,.70),(.51,.61,.68)]
for i,name in enumerate(names):
    pid=f'ASSEMBLY-L{i+1}'
    # Roots define semantic/motion groups; tiny meshes are unnecessary.
    part('assembly',pid,name,[],[],color=colors[i],note=('固定参考结构' if i>=3 else '可更换运动结构')+'；仅作装配层级解释。')
    if i in (0,2,5,6):
        pts=[(43*math.cos(t*math.tau/64),83*math.sin(t*math.tau/64)) for t in range(64)]
        part('assembly',pid+'-SHELL',name+'实体',*prism(pts,6 if i==0 else 2),parent=pid,color=colors[i],opacity=.42 if i==2 else .55 if i==6 else .9)
    if i==1:
        for k in range(6):
            box('assembly',f'{pid}-BLOCK-{k+1}',f'传力块 {k+1}',(23,5,46),((k%3-1)*25,2,-35 if k<3 else 35),parent=pid,color=colors[i])
    if i in (3,4):
        if i==3:
            # Elliptic seal, same major radii as the existing illustrative ring.
            vs,fs=[],[]
            for a in range(64):
                theta=a*math.tau/64
                for b in range(8):
                    phi=b*math.tau/8
                    vs.append((39*(1+.05*math.cos(phi))*math.cos(theta),.9*math.sin(phi),78*(1+.05*math.cos(phi))*math.sin(theta)))
            for a in range(64):
                for b in range(8):
                    fs.append((a*8+b,((a+1)%64)*8+b,((a+1)%64)*8+(b+1)%8,a*8+(b+1)%8))
            part('assembly',pid+'-RING','密封圈实体',vs,fs,parent=pid,color=colors[i])
        else:
            for k,(x,z) in enumerate([(x,z) for x in [-29,29] for z in [-53,53]]):
                pts=[(2.4*math.cos(t*math.tau/16),2.4*math.sin(t*math.tau/16)) for t in range(16)]
                part('assembly',f'{pid}-PIN-{k+1}',f'定位销 {k+1}',*prism(pts,10),(x,-3,z),parent=pid,color=colors[i])

OUT.mkdir(exist_ok=True)
for key,data in CATALOG.items():
    bpy.data.collections[key]['lab_part_order']=[p['id'] for p in data['parts']]
bpy.context.preferences.filepaths.save_version=0
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'lab_assets.blend'))
runpy.run_path(str(Path(__file__).with_name('export_lab_assets.py')),run_name='__main__')
