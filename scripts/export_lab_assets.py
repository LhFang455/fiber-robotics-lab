"""Export edited Blender source without rebuilding or overwriting its geometry.
Blender fiber_robotics_sim/assets/lab_assets.blend --background --python scripts/export_lab_assets.py
Edit meshes in Edit Mode; object origins/axes are protected mounting contracts.
"""
from pathlib import Path
import json
import math
import bpy

OUT=Path(__file__).resolve().parents[1]/'fiber_robotics_sim/assets'


def web(p):
    return [float(p[0]),float(p[2]),-float(p[1])]


def export_catalog():
    catalog={}
    depsgraph=bpy.context.evaluated_depsgraph_get()
    for collection in bpy.data.collections:
        if not collection.get('lab_asset_id'):continue
        asset=dict(id=collection['lab_asset_id'],label=collection['lab_label'],
            coordinates=collection['lab_coordinates'],basis='教学结构，非实物 CAD',parts=[])
        for obj in collection.objects:
            pid=obj.get('part_id')
            if not pid:continue
            matrix=[float(v) for row in obj.matrix_basis for v in row]
            expected=list(obj['contract_matrix'])
            if any(abs(a-b)>1e-6 for a,b in zip(matrix,expected)):
                raise ValueError(f'{pid}: object origin/rotation/scale changed. Edit geometry in Edit Mode; mounting transforms require a reviewed data-adapter change.')
            parent=obj.parent.get('part_id') if obj.parent else None
            if parent!=(obj.get('parent_id') or None):
                raise ValueError(f'{pid}: parent changed; update the mounting contract before export')
            evaluated=obj.evaluated_get(depsgraph)
            mesh=evaluated.to_mesh()
            try:
                mesh.calc_loop_triangles()
                vertices=[v for loop in mesh.loops for v in web(mesh.vertices[loop.vertex_index].co)]
                normals=[v for normal in mesh.corner_normals for v in web(normal.vector)]
                if not all(math.isfinite(v) for v in vertices):raise ValueError(f'{pid}: non-finite geometry')
                mat=obj.data.materials[0] if obj.data.materials else None
                rgba=list(mat.diffuse_color) if mat else [.35,.51,.61,1.]
                asset['parts'].append(dict(id=pid,label=obj['lab_label'],parent=parent,role=obj['role'],
                    position=web(obj.location),color=rgba[:3],opacity=rgba[3],note=obj.get('lab_note',''),
                    vertices=vertices,normals=normals,indices=[int(i) for tri in mesh.loop_triangles for i in tri.loops]))
            finally:evaluated.to_mesh_clear()
        # Save explicit display order, not mesh iteration order.
        asset['parts'].sort(key=lambda p:list(collection['lab_part_order']).index(p['id']))
        catalog[asset['id']]=asset
    if set(catalog)!={'skin','foot','assembly','health'}:raise ValueError('Required semantic collections are missing')
    manifest=dict(schema=1,authoring='Blender',source_file='lab_assets.blend',rebuild='scripts/generate_lab_assets.py',
        exporter='scripts/export_lab_assets.py',license='Original project-generated geometry; no third-party model assets',
        coordinates='right-handed Y-up',assets=catalog)
    # Validate through the same contract used by the website before replacing output.
    import importlib.util
    spec=importlib.util.spec_from_file_location('lab_asset_validation',OUT.parent/'model_assets.py')
    validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    for asset in catalog.values():validator.validate_asset(asset)
    bpy.ops.export_scene.gltf(filepath=str(OUT/'lab_assets.glb'),export_format='GLB',export_yup=True,export_extras=True)
    temporary=OUT/'lab_assets.json.tmp'
    temporary.write_text(json.dumps(manifest,ensure_ascii=False,separators=(',',':')))
    temporary.replace(OUT/'lab_assets.json')
    print('Exported and validated',len(catalog),'Blender assemblies; source file unchanged')


if __name__=='__main__':export_catalog()
