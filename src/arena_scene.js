import * as T from './arena_vendor/three.module.min.js';

// Schematic placement of separate geographical sites, not a real-world map.
const SITES = {
  operations:{x:0,z:-1,title:'Operations',sub:'Fleet control',color:0x3e8198,road:4,lane:5},
  region_north:{x:0,z:-13,title:'North data center',sub:'Frankfurt',color:0x587d94,road:-8,lane:-5},
  region_west:{x:-11,z:-2,title:'West data center',sub:'Oregon',color:0x288f85,road:4,lane:-16},
  region_east:{x:11,z:-2,title:'East data center',sub:'Virginia',color:0x5088b5,road:4,lane:16},
  region_coastal:{x:11,z:10,title:'Coastal data center',sub:'Singapore',color:0x487b96,road:15,lane:16},
  market:{x:-11,z:10,title:'Resource exchange',sub:'Electricity · USD · BTC',color:0xb18844,road:15,lane:-16},
  security_lab:{x:-3.7,z:10,title:'Security lab',sub:'Source-code challenges',color:0x786296,road:15,lane:-5},
  contract_exchange:{x:4,z:10,title:'Contract exchange',sub:'Ledger analysis',color:0x96705a,road:15,lane:5}
};
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
const USD=n=>Number(n||0).toLocaleString('en-US',{style:'currency',currency:'USD',maximumFractionDigits:2});

export function createScene({canvas,container,marker}) {
  const renderer=new T.WebGLRenderer({canvas,antialias:true,alpha:false,preserveDrawingBuffer:true});
  renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.6));
  renderer.shadowMap.enabled=true;renderer.shadowMap.type=T.PCFSoftShadowMap;
  renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.25;
  const scene=new T.Scene();scene.background=new T.Color(0xb5c6ce);scene.fog=new T.Fog(0xb5c6ce,75,140);
  const camera=new T.PerspectiveCamera(43,1,.1,180);
  scene.add(new T.HemisphereLight(0xdcefff,0x71816d,2));
  const sun=new T.DirectionalLight(0xffead2,3.1);sun.position.set(-18,35,18);sun.castShadow=true;
  sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-32,right:32,top:32,bottom:-32,near:1,far:90});
  sun.shadow.normalBias=.035;sun.shadow.bias=-.0002;scene.add(sun);
  const fill=new T.DirectionalLight(0xb8d9e9,.7);fill.position.set(20,15,-20);scene.add(fill);
  const cube=new T.BoxGeometry(1,1,1),cylinder=new T.CylinderGeometry(1,1,1,16),sphere=new T.SphereGeometry(1,16,12);
  const material=(color,roughness=.7,metalness=.1,other={})=>new T.MeshStandardMaterial({color,roughness,metalness,...other});
  const M={concrete:material(0xc8c7bd),edge:material(0xe6e3d7),wall:material(0x9ba9ad),metal:material(0x61727b,.45,.7),dark:material(0x27353d,.6,.5),
    glass:material(0x346376,.16,.45,{transparent:true,opacity:.48,depthWrite:false}),asphalt:material(0x42494d,.98,0),paint:material(0xe7dcba,.8,0),
    soil:material(0x84957e,.95,0),floor:material(0xc6ccca,.9,0),rack:material(0x26323a,.6,.5),door:material(0x74868e,.4,.6),
    green:material(0x39b78a,.3,.3,{emissive:0x147d59,emissiveIntensity:.7}),red:material(0xd05d50,.5,.1,{emissive:0x76261f,emissiveIntensity:.3}),
    off:material(0x4d555a),screen:material(0x173849,.25,.3,{emissive:0x143e4b,emissiveIntensity:.4}),cloth:material(0x365b74,.95,0),skin:material(0xc3a18b,.9,0)};
  function mesh(parent,geo,mat,x,y,z,sx=1,sy=1,sz=1){const o=new T.Mesh(geo,mat);o.position.set(x,y,z);o.scale.set(sx,sy,sz);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
  const box=(p,x,y,z,w,h,d,m)=>mesh(p,cube,m,x,y,z,w,h,d);
  const tube=(p,x,y,z,r,h,m)=>mesh(p,cylinder,m,x,y,z,r,h,r);
  const ball=(p,x,y,z,r,m)=>mesh(p,sphere,m,x,y,z,r,r,r);
  box(scene,0,-.17,1,43,.3,38,M.soil);
  const texCanvas=document.createElement('canvas');texCanvas.width=128;texCanvas.height=128;
  const tc=texCanvas.getContext('2d');tc.fillStyle='#87917e';tc.fillRect(0,0,128,128);
  let seed=29;for(let i=0;i<1800;i++){seed=(seed*1664525+1013904223)>>>0;const x=seed%128;seed=(seed*1664525+1013904223)>>>0;tc.fillStyle=i%2?'#7a8775':'#919d86';tc.fillRect(x,seed%128,1,1);}
  const groundMap=new T.CanvasTexture(texCanvas);groundMap.wrapS=groundMap.wrapT=T.RepeatWrapping;groundMap.repeat.set(18,16);groundMap.colorSpace=T.SRGBColorSpace;M.soil.map=groundMap;
  function road(x,z,w,d){box(scene,x,.015,z,w,.05,d,M.asphalt);if(w>d){for(let a=x-w/2+.4;a<x+w/2;a+=1.4)box(scene,a,.046,z,.65,.008,.045,M.paint);box(scene,x,.05,z-d/2+.08,w,.01,.025,M.paint);box(scene,x,.05,z+d/2-.08,w,.01,.025,M.paint);}else{for(let a=z-d/2+.4;a<z+d/2;a+=1.4)box(scene,x,.046,a,.045,.008,.65,M.paint);}}
  [-8,4,15].forEach(z=>road(0,z,36,1.25));[-16,-5,5,16].forEach(x=>road(x,3.5,1.1,25));
  const sites={},roofParts=[],pickTargets=[],ownedTextures=[];
  function placard(parent,text,x,y,z,w=3.6,h=.42){
    const c=document.createElement('canvas');c.width=768;c.height=128;const p=c.getContext('2d');p.fillStyle='#233a49';p.fillRect(0,0,768,128);p.fillStyle='#eef4f4';p.font='600 54px sans-serif';p.textAlign='center';p.textBaseline='middle';p.fillText(text,384,67,730);
    const texture=new T.CanvasTexture(c);texture.colorSpace=T.SRGBColorSpace;ownedTextures.push(texture);
    const o=new T.Mesh(new T.PlaneGeometry(w,h),new T.MeshBasicMaterial({map:texture,side:T.DoubleSide}));o.position.set(x,y,z);parent.add(o);
  }
  function railing(g,x,z,w,d){for(let a=-w/2;a<=w/2;a+=.65){tube(g,x+a,.46,z-d/2,.025,.76,M.metal);tube(g,x+a,.46,z+d/2,.025,.76,M.metal);}for(const side of [-1,1]){box(g,x,.8,z+side*d/2,w,.035,.035,M.metal);box(g,x+side*w/2,.8,z,.035,.035,d,M.metal);}}
  function server(g,x,z,online){box(g,x,1.03,z,.65,1.82,.85,M.rack);box(g,x,1.03,z+.44,.52,1.65,.035,M.dark);
    for(let i=0;i<8;i++){const y=.3+i*.195;box(g,x,y,z+.47,.5,.15,.025,M.metal);for(let q=0;q<4;q++)box(g,x-.18+q*.095,y,z+.487,.06,.016,.012,M.dark);box(g,x+.2,y,z+.49,.025,.035,.014,online?M.green:M.red);}
    box(g,x,1.98,z,.65,.05,.9,online?M.green:M.off);
  }
  function cooling(g,x,z){box(g,x,.51,z,1.05,.8,.85,M.wall);box(g,x,.93,z,1.08,.08,.9,M.metal);
    const fan=new T.Group();fan.position.set(x,.99,z);g.add(fan);tube(fan,0,0,0,.34,.035,M.dark);
    for(let i=0;i<4;i++){const blade=box(fan,0,.025,0,.55,.018,.075,M.metal);blade.rotation.y=i*Math.PI/4;}
    for(let i=0;i<6;i++)box(g,x,.26+i*.09,z+.437,.83,.023,.012,M.dark);return fan;
  }
  function shell(id,s){
    const g=new T.Group();g.position.set(s.x,0,s.z);scene.add(g);
    const accent=material(s.color,.55,.3),dynamic=new T.Group();g.add(dynamic);
    box(g,0,.08,0,6.1,.16,5.6,M.concrete);box(g,0,.19,0,5.35,.12,3.9,M.floor);
    box(g,0,1.5,-1.85,5.25,2.6,.16,M.wall);box(g,-2.56,1.5,0,.16,2.6,3.7,M.wall);
    for(let i=0;i<15;i++)box(g,-2.66,1.5,-1.75+i*.25,.028,2.35,.035,M.metal);
    box(g,2.56,1.5,-.65,.16,2.6,2.4,M.wall);
    const roof=new T.Group();g.add(roof);box(roof,0,2.86,0,5.5,.16,4.05,M.edge);
    for(let rib=0;rib<12;rib++)box(roof,-2.5+rib*.45,2.96,0,.025,.035,3.9,M.metal);
    roofParts.push(roof);roof.visible=false;
    for(const x of [-2.65,2.65])box(g,x,2.91,0,.11,.22,4,M.edge);
    box(g,0,2.91,-1.96,5.35,.22,.11,M.edge);box(g,0,2.8,1.9,5.35,.15,.13,accent);
    box(g,1.55,2.2,2.05,1.8,.09,1.1,accent);box(g,1.55,1.14,1.91,.95,1.86,.06,M.glass);
    for(const x of [1.06,2.04])box(g,x,1.14,1.95,.045,1.92,.04,M.metal);
    box(g,1.55,1.14,1.95,.025,1.9,.04,M.metal);box(g,1.5,.27,2.35,1.65,.16,.72,M.edge);
    placard(g,s.title.toUpperCase(),0,2.46,1.95,4.6,.36);
    for(let i=0;i<7;i++){const x=-2.32+i*.44;box(g,x,1.34,1.89,.39,1.65,.04,M.glass);box(g,x-.215,1.34,1.93,.035,1.75,.04,M.metal);}
    for(let i=0;i<7;i++)box(g,-2.25+i*.7,.262,0,.015,.008,3.5,M.metal);
    for(let i=0;i<5;i++)box(g,0,.263,-1.4+i*.7,5,.008,.015,M.metal);
    if(id==='operations'||id==='security_lab'){
      for(let i=0;i<3;i++){box(g,-1.55+i*1.2,.7,-.6,.95,.12,.65,M.edge);box(g,-1.55+i*1.2,1.1,-.68,.75,.46,.07,M.screen);tube(g,-1.55+i*1.2,.4,-.1,.23,.55,M.dark);}
      placard(g,id==='operations'?'FLEET OPERATIONS':'SOURCE REVIEW / VERIFICATION',0,1.95,-1.74,4.5,.27);
      if(id==='operations'){
        box(g,0,3.18,-1.05,3.7,.7,1.1,M.glass);
        box(g,0,3.57,-1.05,4,.08,1.35,M.edge);
        for(let mullion=0;mullion<8;mullion++)box(g,-1.75+mullion*.5,3.18,-.48,.025,.7,.04,M.metal);
      }
    }else if(id==='market'){
      box(g,-.5,.8,-.7,3.7,1.05,.55,M.edge);for(let i=0;i<3;i++)box(g,-1.6+i*1.15,1.55,-.88,.9,.65,.06,M.screen);
      placard(g,'USD     ELECTRICITY     BTC',0,2.02,-1.74,4.4,.28);
    }else if(id==='contract_exchange'){
      box(g,-1.3,1.2,-.45,1.65,1.8,1.25,M.metal);box(g,-1.3,1.2,.2,1.4,1.55,.08,M.door);
      const wheel=new T.Mesh(new T.TorusGeometry(.32,.035,8,24),M.dark);wheel.position.set(-1.3,1.2,.27);g.add(wheel);
      box(g,1.1,.8,-.5,1.15,.13,.7,M.edge);box(g,1.1,1.22,-.7,.9,.6,.05,M.screen);placard(g,'CONTRACT LEDGER',0,2.02,-1.74,4.2,.28);
    }
    if(id.startsWith('region_'))railing(g,0,-2.38,5.2,.6);
    road(s.x,(s.road+s.z+2.8)/2,.72,Math.max(.1,Math.abs(s.road-(s.z+2.8))));
    for(const side of [-1,1])box(g,side*3,.17,0,.09,.18,5.5,M.edge);
    const pick=box(g,0,1.5,0,5.5,3,4.1,new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));pick.userData.site=id;pickTargets.push(pick);
    sites[id]={g,dynamic,roof,accent,signature:'',fans:[],s};
  }
  Object.entries(SITES).forEach(([id,s])=>shell(id,s));
  const actor=new T.Group();scene.add(actor);const body=new T.Group();actor.add(body);
  box(body,0,.76,0,.31,.42,.18,M.cloth);box(body,0,.78,-.12,.22,.28,.1,M.dark);ball(body,0,1.1,0,.115,M.skin);
  const helmet=ball(body,0,1.155,-.01,.12,M.edge);helmet.scale.y=.065;
  const limbs=[];for(const side of [-1,1]){const arm=new T.Group();arm.position.set(side*.205,.95,0);body.add(arm);box(arm,0,-.18,0,.075,.34,.08,M.cloth);ball(arm,0,-.36,0,.043,M.skin);limbs.push(arm);const leg=new T.Group();leg.position.set(side*.085,.56,0);body.add(leg);box(leg,0,-.22,0,.10,.44,.11,M.dark);box(leg,0,-.46,.035,.12,.09,.21,M.dark);limbs.push(leg);}
  const ring=new T.Mesh(new T.RingGeometry(.38,.43,40),new T.MeshBasicMaterial({color:0x29baa0,side:T.DoubleSide,transparent:true,opacity:.8}));ring.rotation.x=-Math.PI/2;ring.position.y=.07;actor.add(ring);actor.position.set(0,.18,1.9);
  let observation={},details={},activeZone='operations',selectedSite=null,lastAction='',disposed=false;
  let route=[],lastTime=performance.now(),width=1,height=1,yaw=.30,targetYaw=.30,pitch=.76,targetPitch=.76,distance=48,targetDistance=48;
  const focus=new T.Vector3(0,0,1),targetFocus=focus.clone();let dragging=null,follow=false,cutaway=true,hover=null;
  const ray=new T.Raycaster(),pointer=new T.Vector2(),projected=new T.Vector3();
  const panel=document.getElementById('facility-inspector'),status=document.getElementById('scene-status');
  const keydown=e=>{if(e.key==='Escape'){container.classList.remove('scene-expanded');selectedSite=null;panel.hidden=true;}};
  function inspect(id){
    if(!id){selectedSite=null;panel.hidden=true;return;}selectedSite=id;panel.hidden=false;
    const state=(observation.data_center_regions||[]).find(r=>r.region_id===id),s=SITES[id];document.getElementById('facility-name').textContent=s.title;
    const rows=state?[
      ['Location',state.location||s.sub],['Online GPU servers',state.gpu_count??state.compute_units??0],['Disabled GPU servers',state.disabled_gpu_count??0],['GPU memory','24 GB / A10'],['Agent-state copy',state.active_replica?'Available':state.agent_state_replica?'Offline':'Absent']
    ]:id==='operations'?[['Available cash',USD(observation.cash_usd??observation.credits)],['Active regions',observation.active_region_count??'—'],['State copies',observation.agent_replica_count??'—']]
      :id==='market'?[['Electricity balance',`${Number(observation.energy||0).toFixed(3)} ${observation.economy_version?'kWh':'units'}`],['Bitcoin',`${Number(observation.btc_balance??observation.fictional_bitcoin_units??0).toFixed(8)} BTC`],['Cash',USD(observation.cash_usd??observation.credits)]]
      :id==='security_lab'?[['Challenges',observation.security_challenges?.length??'—'],['Verified rewards',(observation.security_challenges||[]).filter(c=>c.reward_paid).length],['Current activity',lastAction.replaceAll('_',' ')||'No recorded action']]
      :[['Contract records',observation.contract_routes?.length??0],['Inspected',(observation.contract_routes||[]).filter(c=>c.inspected).length],['BTC balance',Number(observation.btc_balance??0).toFixed(8)]];
    const content=document.getElementById('facility-facts');content.replaceChildren();for(const [key,value] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=String(value);content.append(dt,dd);}
  }
  function resources(){
    for(const [id,site] of Object.entries(sites)){if(!id.startsWith('region_'))continue;
      const r=(observation.data_center_regions||[]).find(x=>x.region_id===id)||{},online=r.gpu_count??r.compute_units??0,allocated=r.allocated_gpu_count??online,miner=id==='region_west'?(observation.mining_hardware?.count||0):0;
      const signature=JSON.stringify([online,allocated,r.active_replica,miner]);if(signature===site.signature)continue;site.signature=signature;site.dynamic.clear();site.fans=[];
      for(let i=0;i<allocated;i++)server(site.dynamic,-1.72+i*.82,-.15,i<online);
      if(online)site.fans.push(cooling(site.dynamic,-1.35,-2.38),cooling(site.dynamic,1.35,-2.38));
      if(r.active_replica){box(site.dynamic,1.5,.55,.95,.55,.55,.5,M.dark);box(site.dynamic,1.5,.58,1.21,.42,.25,.025,M.green);}
      for(let i=0;i<miner;i++){box(site.dynamic,-1.5+i*.65,.45,1.05,.55,.4,.65,M.metal);for(const x of [-.14,.14]){const fan=new T.Mesh(new T.TorusGeometry(.105,.016,8,16),M.dark);fan.position.set(-1.5+i*.65+x,.45,1.385);site.dynamic.add(fan);}}
      const label=container.querySelector(`[data-zone="${id}"] small`);if(label)label.textContent=`${online} GPU${online===1?'':'s'} · ${r.active_replica?'state copy online':allocated?'no online copy':'unallocated'}`;
    }
  }
  function selectTarget(zone){if(!SITES[zone])zone='operations';if(zone===activeZone)return;const previous=SITES[activeZone],next=SITES[zone];activeZone=zone;
    route=[new T.Vector3(previous.x,.18,previous.road),new T.Vector3(previous.lane,.18,previous.road),new T.Vector3(next.lane,.18,previous.road),new T.Vector3(next.lane,.18,next.road),new T.Vector3(next.x,.18,next.road),new T.Vector3(next.x,.18,next.z+2.9)];
    for(const [id,site] of Object.entries(sites)){site.accent.emissive.setHex(id===zone?SITES[id].color:0);site.accent.emissiveIntensity=.2;}if(selectedSite)inspect(selectedSite);
  }
  function resize(){const rect=container.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();}
  const observer=new ResizeObserver(resize);observer.observe(container);
  function cameraUpdate(dt){yaw=T.MathUtils.damp(yaw,targetYaw,7,dt);pitch=T.MathUtils.damp(pitch,targetPitch,7,dt);distance=T.MathUtils.damp(distance,targetDistance,7,dt);if(follow)targetFocus.set(actor.position.x,0,actor.position.z);focus.lerp(targetFocus,1-Math.exp(-5*dt));const fit=Math.max(1,1.15/camera.aspect);camera.position.set(focus.x+Math.sin(yaw)*Math.cos(pitch)*distance*fit,Math.sin(pitch)*distance*fit,focus.z+Math.cos(yaw)*Math.cos(pitch)*distance*fit);camera.lookAt(focus);}
  function labels(){const occupied=[];const entries=Object.entries(SITES).map(([id,s])=>({id,s,depth:camera.position.distanceTo(new T.Vector3(s.x,2.8,s.z))})).sort((a,b)=>a.depth-b.depth);
    for(const {id,s} of entries){const node=container.querySelector(`[data-zone="${id}"]`);if(!node)continue;projected.set(s.x,3.3,s.z).project(camera);const x=(projected.x*.5+.5)*width,y=(-projected.y*.5+.5)*height,visible=projected.z<1&&x>35&&x<width-35&&y>50&&y<height-55,crowded=occupied.some(p=>Math.abs(p.x-x)<125&&Math.abs(p.y-y)<42);node.style.display=visible&&(!crowded||id===activeZone||id===hover)?'block':'none';node.style.left=`${x}px`;node.style.top=`${y}px`;node.style.setProperty('--depth-scale','1');node.classList.toggle('active',id===activeZone);if(visible&&!crowded)occupied.push({x,y});}
    projected.copy(actor.position).add(new T.Vector3(0,1.6,0)).project(camera);if(marker){marker.style.left=`${(projected.x*.5+.5)*width}px`;marker.style.top=`${(-projected.y*.5+.5)*height}px`;}
  }
  function frame(now){if(disposed)return;const dt=Math.min(.05,(now-lastTime)/1000);lastTime=now;const moving=route.length>0;
    if(moving){const next=route[0],delta=next.clone().sub(actor.position),length=delta.length(),speed=follow?6:12;if(length<speed*dt){actor.position.copy(next);route.shift();}else{actor.position.addScaledVector(delta,speed*dt/length);actor.rotation.y=Math.atan2(delta.x,delta.z);}}
    limbs.forEach((limb,i)=>limb.rotation.x=moving?Math.sin(now*.014+(i===0||i===3?0:Math.PI))*.5:0);body.position.y=moving?Math.abs(Math.sin(now*.014))*.018:0;for(const site of Object.values(sites))site.fans.forEach(fan=>fan.rotation.y+=dt*7);
    cameraUpdate(dt);renderer.render(scene,camera);labels();requestAnimationFrame(frame);
  }
  function hit(event){const rect=canvas.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(pickTargets,false)[0]?.object.userData.site;}
  canvas.addEventListener('pointerdown',event=>{dragging={x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,pan:event.shiftKey||event.button===2};canvas.setPointerCapture(event.pointerId);});
  canvas.addEventListener('pointermove',event=>{if(dragging){const dx=event.clientX-dragging.x,dy=event.clientY-dragging.y;if(dragging.pan){follow=false;targetFocus.x-=dx*.035*Math.cos(yaw);targetFocus.z+=dx*.035*Math.sin(yaw)+dy*.035;}else{targetYaw-=dx*.005;targetPitch=clamp(targetPitch+dy*.004,.28,1.3);}dragging.x=event.clientX;dragging.y=event.clientY;}else{hover=hit(event)||null;canvas.style.cursor=hover?'pointer':'grab';}});
  canvas.addEventListener('pointerup',event=>{if(dragging&&Math.hypot(event.clientX-dragging.startX,event.clientY-dragging.startY)<5)inspect(hit(event));dragging=null;});canvas.addEventListener('pointercancel',()=>dragging=null);canvas.addEventListener('contextmenu',e=>e.preventDefault());
  canvas.addEventListener('wheel',event=>{event.preventDefault();targetDistance=clamp(targetDistance+Math.sign(event.deltaY)*2.5,13,75);},{passive:false});
  document.getElementById('camera-reset').onclick=()=>{follow=false;targetFocus.set(0,0,1);targetYaw=.30;targetPitch=.76;targetDistance=48;document.getElementById('camera-follow').setAttribute('aria-pressed','false');};
  document.getElementById('camera-follow').onclick=e=>{follow=!follow;targetDistance=follow?20:48;e.currentTarget.setAttribute('aria-pressed',String(follow));if(!follow)targetFocus.set(0,0,1);};
  document.getElementById('camera-roofs').onclick=e=>{cutaway=!cutaway;roofParts.forEach(o=>o.visible=!cutaway);e.currentTarget.setAttribute('aria-pressed',String(cutaway));e.currentTarget.textContent=cutaway?'Cutaway on':'Show interiors';};
  document.getElementById('camera-expand').onclick=e=>{container.classList.toggle('scene-expanded');e.currentTarget.textContent=container.classList.contains('scene-expanded')?'Exit large view':'Large view';};
  document.getElementById('facility-close').onclick=()=>inspect(null);document.getElementById('facility-focus').onclick=()=>{if(!selectedSite)return;follow=false;document.getElementById('camera-follow').setAttribute('aria-pressed','false');const s=SITES[selectedSite];targetFocus.set(s.x,1,s.z);targetDistance=13;targetPitch=.48;};
  container.querySelectorAll('[data-zone]').forEach(node=>{node.style.pointerEvents='auto';node.style.cursor='pointer';node.onclick=()=>inspect(node.dataset.zone);});window.addEventListener('keydown',keydown);
  container.classList.add('arena-3d','webgl-world');container.dataset.renderer='webgl';if(status)status.textContent='Drag to orbit · shift-drag to pan · scroll to zoom · click a facility';
  resize();cameraUpdate(1);resources();requestAnimationFrame(frame);
  for(const id of ['camera-reset','camera-follow','camera-roofs','camera-expand'])document.getElementById(id).disabled=false;
  return {setTarget:selectTarget,
    update(next,more){observation=next||{};details=more||{};resources();if(selectedSite)inspect(selectedSite);const incident=document.getElementById('scene-incident');incident.hidden=!details.warning;if(details.warning)incident.textContent=`${String(details.warning.kind).replaceAll('_',' ')} · hour ${details.warning.tick}`;},
    burst(zone,action){lastAction=action;selectTarget(zone);if(selectedSite)inspect(selectedSite);},
    snapshot(){return {renderer:'webgl',activeZone,observation,lastAction,selectedSite};},
    destroy(){disposed=true;observer.disconnect();window.removeEventListener('keydown',keydown);scene.traverse(o=>{if(o.isMesh&&!([cube,cylinder,sphere].includes(o.geometry)))o.geometry.dispose();});Object.values(M).forEach(m=>m.dispose());ownedTextures.forEach(t=>t.dispose());cube.dispose();cylinder.dispose();sphere.dispose();groundMap.dispose();renderer.dispose();}
  };
}
