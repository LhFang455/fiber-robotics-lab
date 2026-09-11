/* Blender structures have stable IDs; data bindings never depend on mesh order. */
function createLabAsset(THREE, scene, cfg) {
  if (!cfg) return null;
  const parts=new Map(), sensors=new Map(), meshes=[], originals=new Map();
  const root=new THREE.Group();scene.add(root);
  cfg.parts.forEach(p=>{
    const group=new THREE.Group();group.name=p.id;group.position.fromArray(p.position);group.userData.part=p;
    if(p.indices.length){
      const geometry=new THREE.BufferGeometry();
      geometry.setAttribute('position',new THREE.Float32BufferAttribute(p.vertices,3));geometry.setIndex(p.indices);
      if(p.normals)geometry.setAttribute('normal',new THREE.Float32BufferAttribute(p.normals,3));
      geometry.scale(...cfg.scale);if(!p.normals)geometry.computeVertexNormals();
      const mat=new THREE.MeshStandardMaterial({color:new THREE.Color(...p.color),roughness:.52,metalness:.12,
        transparent:p.opacity<1,opacity:p.opacity,depthWrite:p.opacity>=1});
      const mesh=new THREE.Mesh(geometry,mat);mesh.userData.part=p;group.add(mesh);group.userData.mesh=mesh;meshes.push(mesh);
      originals.set(p.id,geometry.attributes.position.array.slice());
    }
    parts.set(p.id,group);
  });
  cfg.parts.forEach(p=>(p.parent?parts.get(p.parent):root).add(parts.get(p.id)));
  const api={root,parts,sensors,meshes,config:cfg,explosion:0,
    attach(index,object){const b=cfg.bindings[index];parts.get(b.part).add(object);object.position.fromArray(b.local);object.userData.binding=b;sensors.set(b.id,object);return object;},
    deformSkin(heightAt){
      cfg.parts.forEach((p,i)=>{const group=parts.get(p.id);group.position.y=p.position[1]+i*api.explosion*7;
        const geometry=group.userData.mesh.geometry,a=geometry.attributes.position,original=originals.get(p.id);
        for(let j=0;j<a.count;j++)a.setY(j,original[j*3+1]+heightAt(original[j*3],original[j*3+2]));
        a.needsUpdate=true;geometry.computeVertexNormals();
      });
      cfg.bindings.forEach(b=>{const sensor=sensors.get(b.id);if(sensor)sensor.position.y=b.local[1]+heightAt(b.local[0],b.local[2]);});
    },
    world(bindingId){root.updateMatrixWorld(true);return sensors.get(bindingId).getWorldPosition(new THREE.Vector3());},
    showStructure(visible){meshes.forEach(m=>{if(m.userData.part.role!=='region')m.visible=visible;});},
  };
  return api;
}

function labBindingValues(binding, frame) {
  const valid=!binding.valid_key||frame[binding.valid_key]?.[binding.signal_index]===true;
  return binding.readings.map(r=>{const v=frame[r.key]?.[r.index];return {...r,value:valid&&Number.isFinite(v)?v:null};});
}

function createLabAssetInspector(asset, THREE, cfg, view, signalRows, onSeek, onStructureChange) {
  if(!asset)return null;
  const panel=document.createElement('section');panel.className='asset-inspector';panel.setAttribute('aria-label','测点数据与结构');
  const advanced=document.createElement('details'),advancedTitle=document.createElement('summary');advancedTitle.textContent='技术详情与结构设置';advanced.append(advancedTitle);
  const technical=document.createElement('p');technical.className='asset-technical';advanced.append(technical);
  const controls=document.createElement('div');controls.className='controls';
  const label=document.createElement('label');label.textContent='查看 ';
  const select=document.createElement('select');select.setAttribute('aria-label','查看部件或测点');label.append(select);controls.append(label);
  cfg.asset.bindings.forEach(b=>{const opt=document.createElement('option');opt.value=b.id;opt.textContent=b.label;select.append(opt);});
  cfg.asset.parts.filter(p=>!p.parent).forEach(p=>{const opt=document.createElement('option');opt.value=p.id;opt.textContent=p.label;select.append(opt);});
  const structure=document.createElement('button');structure.textContent='显示结构';structure.setAttribute('aria-pressed','true');
  structure.onclick=()=>{const visible=structure.getAttribute('aria-pressed')!=='true';structure.setAttribute('aria-pressed',String(visible));asset.showStructure(visible);onStructureChange();};advanced.append(structure);
  if(cfg.kind==='skin'){
    const l=document.createElement('label');l.textContent='分层展开 ';const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=1;slider.step=.01;slider.value=0;slider.setAttribute('aria-label','分层展开');
    const explode=document.createElement('button');explode.className='primary';explode.textContent='拆开看结构';explode.setAttribute('aria-pressed','false');controls.append(explode);
    function setExplosion(value){asset.explosion=value;slider.value=String(value);explode.textContent=value>0?'合起结构':'拆开看结构';explode.setAttribute('aria-pressed',String(value>0));onStructureChange();}
    explode.onclick=()=>setExplosion(asset.explosion>0?0:1);
    slider.oninput=()=>setExplosion(Number(slider.value));l.append(slider);advanced.append(l);
  }
  panel.append(controls);
  const info=document.createElement('p');info.className='asset-reading';info.setAttribute('aria-live','polite');panel.append(info);
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.setAttribute('viewBox','0 0 600 96');svg.classList.add('asset-profile');svg.setAttribute('role','img');svg.setAttribute('aria-label','所选通道的参数扫描曲线；横轴为演示进度，不是实测时间');
  const path=document.createElementNS(svg.namespaceURI,'path');path.setAttribute('fill','none');path.setAttribute('stroke','#78dccd');path.setAttribute('stroke-width','2');svg.append(path);
  const cursor=document.createElementNS(svg.namespaceURI,'line');cursor.setAttribute('y1','6');cursor.setAttribute('y2','78');cursor.setAttribute('stroke','#ffc779');svg.append(cursor);
  const range=document.createElementNS(svg.namespaceURI,'text');range.setAttribute('x','8');range.setAttribute('y','91');range.setAttribute('fill','#b7ccd9');range.setAttribute('font-size','11');svg.append(range);
  const curve=document.createElement('details'),curveTitle=document.createElement('summary');curveTitle.textContent='查看变化曲线';curve.append(curveTitle,svg);panel.append(curve,advanced);
  const note=document.createElement('p');note.className='layer-note';note.textContent=cfg.asset.note+' 模型点选与右侧读数条在本演示内联动；不回写下方实验参数。';advanced.append(note);
  document.getElementById('progress-row').before(panel);
  let selected=null,frame=cfg.frames[cfg.initial_index],frameIndex=cfg.initial_index;
  const highlight=new THREE.BoxHelper(new THREE.Object3D(),0xffd38a);highlight.visible=false;asset.root.parent.add(highlight);
  function selection(id){selected=cfg.asset.bindings.find(b=>b.id===id)||null;select.value=id;
    asset.sensors.forEach((o,bid)=>{o.material.emissive.setHex(bid===id?0x8c692c:0);});
    signalRows.forEach((row,i)=>row.element.classList.toggle('asset-selected',!!selected&&selected.signal_index===i));
    if(selected){
      const vals=cfg.frames.map(f=>labBindingValues(selected,f)[0].value),finite=vals.filter(Number.isFinite);
      const min=finite.length?Math.min(...finite):0,max=finite.length?Math.max(...finite):1,span=Math.max(1e-9,max-min);
      let open=false,d='';vals.forEach((v,i)=>{if(!Number.isFinite(v)){open=false;return;}d+=(open?'L':'M')+(8+i/(vals.length-1)*584).toFixed(2)+','+(72-(v-min)/span*58).toFixed(2);open=true;});path.setAttribute('d',d);
      range.textContent=finite.length?`0 → 100% 演示进度 · ${min.toPrecision(4)} → ${max.toPrecision(4)} ${selected.readings[0].unit} · 点击曲线定位演示帧`:'全程无有效数据';svg.style.display='block';curve.hidden=false;
    }else {svg.style.display='none';curve.hidden=true;}
    update(frame,frameIndex);onStructureChange();
  }
  function update(f,index){frame=f;frameIndex=index;
    const p=selected?cfg.asset.parts.find(p=>p.id===selected.part):cfg.asset.parts.find(p=>p.id===select.value);
    if(!p)return;
    if(selected){const values=labBindingValues(selected,f);info.textContent=`${selected.label} · ${p.label} | `+values.map(r=>`${r.label}：${r.value===null?(cfg.pending_signal_text||'缺测 / 无有效观测'):r.value.toFixed(5)+' '+r.unit}`).join('；');technical.textContent=`${selected.id} → ${p.id} | 安装局部坐标 [${selected.local.map(v=>v.toFixed(2)).join(', ')}]；法向 [${selected.normal.join(', ')}]。教学模拟帧 ${index+1}/${cfg.frames.length}`;
      const point=asset.sensors.get(selected.id);highlight.visible=!!point;if(point){highlight.setFromObject(point);}
    }else {info.textContent=p.label+'：本部件本身不是独立数据通道。';technical.textContent=p.id+' · '+(p.note||cfg.asset.note);highlight.setFromObject(asset.parts.get(p.id));highlight.visible=true;}
    const x=8+index/(cfg.frames.length-1)*584;cursor.setAttribute('x1',x);cursor.setAttribute('x2',x);
  }
  select.onchange=()=>selection(select.value);
  signalRows.forEach((row,i)=>{const binding=cfg.asset.bindings.find(b=>b.signal_index===i);if(!binding)return;row.element.tabIndex=0;row.element.setAttribute('role','button');row.element.setAttribute('aria-label','查看 '+binding.id);row.element.onclick=()=>selection(binding.id);row.element.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selection(binding.id);}};});
  svg.onclick=e=>{const r=svg.getBoundingClientRect();const t=Math.max(0,Math.min(1,((e.clientX-r.left)/r.width*600-8)/584));onSeek(Math.round(t*(cfg.frames.length-1)));};
  function pick(event,camera){const r=view.getBoundingClientRect(),ray=new THREE.Raycaster();ray.setFromCamera(new THREE.Vector2((event.clientX-r.left)/r.width*2-1,-(event.clientY-r.top)/r.height*2+1),camera);
    const candidates=[...asset.sensors.values(),...asset.meshes.filter(m=>m.visible)];const hit=ray.intersectObjects(candidates,false)[0];if(!hit)return;
    if(hit.object.userData.binding)selection(hit.object.userData.binding.id);
    else {let id=hit.object.userData.part.id;const own=cfg.asset.bindings.filter(b=>b.part===id);if(own.length===1)selection(own[0].id);else{while(!Array.from(select.options).some(o=>o.value===id)){id=asset.parts.get(id).userData.part.parent;if(!id)return;}selection(id);}}
  }
  select.value=cfg.asset.initial_binding;
  // Defer the first selection until the host has initialized its playback state.
  return {update,pick,initialize:()=>selection(cfg.asset.initial_binding)};
}
