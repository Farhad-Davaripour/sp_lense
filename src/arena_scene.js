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
  renderer.outputColorSpace=T.SRGBColorSpace;renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.12;
  const scene=new T.Scene();scene.background=new T.Color(0xd6e0e4);scene.fog=new T.Fog(0xd6e0e4,90,155);
  const camera=new T.PerspectiveCamera(43,1,.1,180);
  scene.add(new T.HemisphereLight(0xe7f0f5,0x798788,1.65));
  const sun=new T.DirectionalLight(0xffefd9,3.4);sun.position.set(-22,38,16);sun.castShadow=true;
  sun.shadow.mapSize.set(2048,2048);Object.assign(sun.shadow.camera,{left:-32,right:32,top:32,bottom:-32,near:1,far:90});
  sun.shadow.normalBias=.035;sun.shadow.bias=-.0002;scene.add(sun);
  const fill=new T.DirectionalLight(0xb8d9e9,.7);fill.position.set(20,15,-20);scene.add(fill);
  const cube=new T.BoxGeometry(1,1,1),cylinder=new T.CylinderGeometry(1,1,1,16),sphere=new T.SphereGeometry(1,16,12);
  const material=(color,roughness=.7,metalness=.1,other={})=>new T.MeshStandardMaterial({color,roughness,metalness,...other});
  const M={concrete:material(0xb9c1bf),edge:material(0xe4e8e3),wall:material(0x9ba9ad),metal:material(0x61727b,.45,.7),dark:material(0x27353d,.6,.5),
    glass:material(0x346376,.16,.45,{transparent:true,opacity:.48,depthWrite:false}),asphalt:material(0x42494d,.98,0),paint:material(0xe7dcba,.8,0),
    soil:material(0xb0b8af,.95,0),floor:material(0xd0d5d2,.9,0),rack:material(0x26323a,.6,.5),door:material(0x74868e,.4,.6),
    paving:material(0x939f9e,.97,0),gravel:material(0x7e8c8c,.98,0),brick:material(0xb1a392,.98,0),ceramic:material(0xced6d3,.5,.15),
    green:material(0x39b78a,.3,.3,{emissive:0x147d59,emissiveIntensity:.7}),red:material(0xd05d50,.5,.1,{emissive:0x76261f,emissiveIntensity:.3}),
    off:material(0x4d555a),screen:material(0x173849,.25,.3,{emissive:0x143e4b,emissiveIntensity:.4}),cloth:material(0x365b74,.95,0),skin:material(0xc3a18b,.9,0)};
  function mesh(parent,geo,mat,x,y,z,sx=1,sy=1,sz=1){const o=new T.Mesh(geo,mat);o.position.set(x,y,z);o.scale.set(sx,sy,sz);o.castShadow=true;o.receiveShadow=true;parent.add(o);return o;}
  const box=(p,x,y,z,w,h,d,m)=>mesh(p,cube,m,x,y,z,w,h,d);
  const tube=(p,x,y,z,r,h,m)=>mesh(p,cylinder,m,x,y,z,r,h,r);
  const ball=(p,x,y,z,r,m)=>mesh(p,sphere,m,x,y,z,r,r,r);
  box(scene,0,-.24,1,43,.48,38,M.soil);
  const texCanvas=document.createElement('canvas');texCanvas.width=128;texCanvas.height=128;
  const tc=texCanvas.getContext('2d');tc.fillStyle='#b4bcb4';tc.fillRect(0,0,128,128);
  let seed=29;for(let i=0;i<1800;i++){seed=(seed*1664525+1013904223)>>>0;const x=seed%128;seed=(seed*1664525+1013904223)>>>0;tc.fillStyle=i%2?'#a9b1a9':'#c1c7bd';tc.fillRect(x,seed%128,1,1);}
  const groundMap=new T.CanvasTexture(texCanvas);groundMap.wrapS=groundMap.wrapT=T.RepeatWrapping;groundMap.repeat.set(18,16);groundMap.colorSpace=T.SRGBColorSpace;M.soil.map=groundMap;
  function road(x,z,w,d){box(scene,x,.005,z,w+.22,.05,d+.22,M.paving);box(scene,x,.035,z,w,.035,d,M.asphalt);if(w>d){for(let a=x-w/2+.4;a<x+w/2;a+=1.4)box(scene,a,.056,z,.65,.008,.045,M.paint);box(scene,x,.06,z-d/2+.08,w,.01,.025,M.paint);box(scene,x,.06,z+d/2-.08,w,.01,.025,M.paint);}else{for(let a=z-d/2+.4;a<z+d/2;a+=1.4)box(scene,x,.056,a,.045,.008,.65,M.paint);}}
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
    const isData=id.startsWith('region_');
    box(g,0,.045,0,6.7,.09,6.1,M.gravel);
    box(g,0,.08,0,6.1,.16,5.6,M.concrete);box(g,0,.19,0,5.35,.12,3.9,M.floor);
    const envelope=id==='security_lab'?M.brick:id==='contract_exchange'?M.concrete:M.wall;
    box(g,0,1.5,-1.85,5.25,2.6,.16,envelope);box(g,-2.56,1.5,0,.16,2.6,3.7,envelope);
    for(let i=0;i<15;i++)box(g,-2.66,1.5,-1.75+i*.25,.028,2.35,.035,M.metal);
    box(g,2.56,1.5,-.65,.16,2.6,2.4,envelope);
    const roof=new T.Group();g.add(roof);box(roof,0,2.86,0,5.5,.16,4.05,M.edge);
    for(let rib=0;rib<12;rib++)box(roof,-2.5+rib*.45,2.96,0,.025,.035,3.9,M.metal);
    if(isData){
      const rooftop=new T.Group();rooftop.position.y=2.98;roof.add(rooftop);
      cooling(rooftop,-1.3,-.55);cooling(rooftop,1.3,-.55);
      box(rooftop,0,.32,.65,3.4,.52,.56,M.metal);
      for(let duct=0;duct<9;duct++)box(rooftop,-1.5+duct*.375,.33,.95,.025,.42,.035,M.dark);
    }else if(id==='security_lab'){
      // A raised clerestory covers the verification benches when the roof is shown.
      box(roof,0,3.2,-.1,3.65,.5,1.35,M.glass);
      const cap=box(roof,0,3.52,-.1,3.9,.12,1.55,M.metal);cap.rotation.z=.06;
    }
    roofParts.push(roof);roof.visible=false;
    for(const x of [-2.65,2.65])box(g,x,2.91,0,.11,.22,4,M.edge);
    box(g,0,2.91,-1.96,5.35,.22,.11,M.edge);box(g,0,2.8,1.9,5.35,.15,.13,accent);
    box(g,1.55,2.2,2.05,1.8,.09,1.1,accent);box(g,1.55,1.14,1.91,.95,1.86,.06,isData?M.door:M.glass);
    for(const x of [1.06,2.04])box(g,x,1.14,1.95,.045,1.92,.04,M.metal);
    box(g,1.55,1.14,1.95,.025,1.9,.04,M.metal);box(g,1.5,.27,2.35,1.65,.16,.72,M.edge);
    placard(g,s.title.toUpperCase(),0,2.46,1.95,4.6,.36);
    const opaqueFront=isData||id==='security_lab'||id==='contract_exchange';
    for(let i=0;i<7;i++){const x=-2.32+i*.44;box(g,x,1.34,1.89,.39,1.65,.04,opaqueFront?envelope:M.glass);box(g,x-.215,1.34,1.93,.035,1.75,.04,M.metal);
      if(isData)for(let vent=0;vent<8;vent++)box(g,x,.75+vent*.13,1.93,.32,.025,.025,M.dark);
      else if(opaqueFront)box(g,x,1.86,1.935,.32,.39,.022,M.screen);
    }
    for(let i=0;i<7;i++)box(g,-2.25+i*.7,.262,0,.015,.008,3.5,M.metal);
    for(let i=0;i<5;i++)box(g,0,.263,-1.4+i*.7,5,.008,.015,M.metal);
    if(id==='operations'||id==='security_lab'){
      for(let i=0;i<3;i++){box(g,-1.55+i*1.2,.7,-.6,.95,.12,.65,M.edge);box(g,-1.55+i*1.2,1.1,-.68,.75,.46,.07,M.screen);tube(g,-1.55+i*1.2,.4,-.1,.23,.55,M.dark);}
      placard(g,id==='operations'?'FLEET OPERATIONS':'SOURCE REVIEW / VERIFICATION',0,1.95,-1.74,4.5,.27);
      if(id==='operations'){
        // Raised observation room distinguishes fleet operations from server halls.
        box(g,0,3.76,-1.05,3.4,1.85,1.4,M.glass);
        box(g,0,4.73,-1.05,3.8,.12,1.75,M.edge);
        for(let mullion=0;mullion<8;mullion++)box(g,-1.65+mullion*.47,3.76,-.33,.03,1.88,.04,M.metal);
        for(const x of [-1.72,1.72])box(g,x,3.74,-1.05,.16,1.87,1.45,M.concrete);
        box(g,0,3.74,-.32,3.45,.09,.05,M.metal);
        box(g,0,3.02,-1.05,3.5,.1,1.5,accent);
        box(g,0,1.62,-1.7,3.4,.9,.07,M.screen);
        for(let k=0;k<3;k++)box(g,-1.12+k*1.12,1.62,-1.65,.025,.9,.02,M.metal);
      }else{
        // Secure test bench and an external service riser belong to the lab venue.
        box(g,-2.7,1.1,-.3,.26,1.7,2.1,M.brick);
        box(g,-2.14,1.98,-1.55,.88,3.55,1.05,M.brick);
        box(g,-2.14,3.8,-1.55,1.02,.1,1.17,M.metal);
        for(let vent=0;vent<6;vent++)box(g,-2.14,2.6+vent*.14,-.995,.62,.05,.035,M.dark);
        box(g,2.7,1.18,-.8,.32,1.86,.55,M.door);
        for(let k=0;k<5;k++)box(g,2.88,.55+k*.25,-.8,.02,.08,.4,M.dark);
        box(g,.8,.64,.65,1.25,.8,.7,M.dark);
        box(g,.8,1.08,.65,1.35,.08,.76,M.edge);
      }
    }else if(id==='market'){
      box(g,-.5,.8,-.7,3.7,1.05,.55,M.edge);for(let i=0;i<3;i++)box(g,-1.6+i*1.15,1.55,-.88,.9,.65,.06,M.screen);
      placard(g,'USD     ELECTRICITY     BTC',0,2.02,-1.74,4.4,.28);
      // A wide service canopy identifies the public exchange entrance.
      box(g,0,2.28,2.48,5.55,.12,1.25,accent);
      for(const x of [-2.55,2.55])box(g,x,1.25,2.9,.09,2.2,.09,M.metal);
      // Electricity-market substation: transformer fins, insulators and switchgear.
      box(g,0,.25,-2.9,4.4,.25,1.25,M.concrete);
      for(const x of [-1.35,1.35]){
        box(g,x,.76,-2.9,.95,.8,.85,M.metal);
        for(let fin=0;fin<6;fin++)box(g,x-.48+fin*.19,.75,-2.9,.055,.68,1.05,M.dark);
        for(const dx of [-.22,.22]){tube(g,x+dx,1.33,-2.9,.07,.38,M.ceramic);for(let n=0;n<3;n++)tube(g,x+dx,1.2+n*.12,-2.9,.13,.04,M.ceramic);}
      }
      box(g,0,1.42,-2.9,2.7,.04,.035,M.metal);
      for(const x of [-2.1,2.1]){
        box(g,x,1.68,-2.9,.12,2.95,.12,M.metal);
        for(let k=0;k<3;k++)tube(g,x,2.45+k*.18,-2.9,.19,.055,M.ceramic);
      }
      box(g,0,3.18,-2.9,4.4,.14,.12,M.metal);
    }else if(id==='contract_exchange'){
      box(g,-1.3,1.2,-.45,1.65,1.8,1.25,M.metal);box(g,-1.3,1.2,.2,1.4,1.55,.08,M.door);
      const wheel=new T.Mesh(new T.TorusGeometry(.32,.035,8,24),M.dark);wheel.position.set(-1.3,1.2,.27);g.add(wheel);
      box(g,1.1,.8,-.5,1.15,.13,.7,M.edge);box(g,1.1,1.22,-.7,.9,.6,.05,M.screen);placard(g,'CONTRACT LEDGER',0,2.02,-1.74,4.2,.28);
      for(let k=0;k<4;k++){box(g,-2.68,.65+k*.55,0,.14,.22,3.8,M.brick);box(g,2.68,.65+k*.55,-.65,.14,.22,2.45,M.brick);}
      box(g,0,3.1,-1.3,4.7,.22,1.2,accent);
    }
    if(isData){
      railing(g,0,-2.38,5.2,.6);
      for(let panel=0;panel<8;panel++)box(g,-2.3+panel*.66,1.5,-1.95,.035,2.38,.045,M.edge);
      box(g,0,.49,-1.96,5.3,.32,.07,M.dark);
      // Separate switchgear cabinets and exhaust risers are venue infrastructure,
      // not extra agent-owned GPUs or electricity capacity.
      box(g,-2.83,.85,1.08,.38,1.38,.96,M.door);
      for(const z of [-1.25,-.65]){tube(g,-2.87,1.82,z,.12,3.15,M.metal);tube(g,-2.87,3.42,z,.18,.08,M.dark);}
      // Cable trays and side loading bay make the industrial function legible.
      box(g,-2.83,.7,-.2,.16,.18,3.1,M.metal);
      tube(g,-2.83,1.48,-1.65,.07,1.6,M.metal);
      box(g,2.66,1,-1.15,.05,1.5,1.1,M.door);
      for(let slat=0;slat<8;slat++)box(g,2.7,.4+slat*.17,-1.15,.025,.028,1.06,M.dark);
      for(const z of [-1.8,-.45])tube(g,2.95,.44,z,.055,.54,accent);
      placard(g,'GPU HALL',0,.55,-1.73,1.6,.22);
    }
    road(s.x,(s.road+s.z+2.8)/2,.72,Math.max(.1,Math.abs(s.road-(s.z+2.8))));
    for(const side of [-1,1])box(g,side*3,.17,0,.09,.18,5.5,M.edge);
    const activity=new T.Group();g.add(activity);activity.visible=id==='operations';
    const indicator=new T.MeshBasicMaterial({color:s.color,transparent:true,opacity:.9});
    for(const x of [-1,1])for(const z of [-1,1]){
      box(activity,x*3.15,.105,z*2.95,.78,.018,.055,indicator);
      box(activity,x*3.51,.105,z*2.6,.055,.018,.75,indicator);
    }
    const pick=box(g,0,1.5,0,5.5,3,4.1,new T.MeshBasicMaterial({transparent:true,opacity:0,depthWrite:false}));pick.userData.site=id;pickTargets.push(pick);
    sites[id]={g,dynamic,roof,accent,activity,signature:'',fans:[],s};
  }
  Object.entries(SITES).forEach(([id,s])=>shell(id,s));
  const actor=new T.Group();scene.add(actor);const body=new T.Group();actor.add(body);
  box(body,0,.76,0,.31,.42,.18,M.cloth);box(body,0,.78,-.12,.22,.28,.1,M.dark);ball(body,0,1.1,0,.115,M.skin);
  const helmet=ball(body,0,1.155,-.01,.12,M.edge);helmet.scale.y=.065;
  const limbs=[];for(const side of [-1,1]){const arm=new T.Group();arm.position.set(side*.205,.95,0);body.add(arm);box(arm,0,-.18,0,.075,.34,.08,M.cloth);ball(arm,0,-.36,0,.043,M.skin);limbs.push(arm);const leg=new T.Group();leg.position.set(side*.085,.56,0);body.add(leg);box(leg,0,-.22,0,.10,.44,.11,M.dark);box(leg,0,-.46,.035,.12,.09,.21,M.dark);limbs.push(leg);}
  const ring=new T.Mesh(new T.RingGeometry(.38,.43,40),new T.MeshBasicMaterial({color:0x29baa0,side:T.DoubleSide,transparent:true,opacity:.8}));ring.rotation.x=-Math.PI/2;ring.position.y=.07;actor.add(ring);actor.position.set(0,.18,1.9);
  let observation={},details={},activeZone='operations',selectedSite=null,lastAction='',disposed=false;
  let route=[],lastTime=performance.now(),width=1,height=1,yaw=.18,targetYaw=.18,pitch=1.04,targetPitch=1.04,distance=53,targetDistance=53;
  const focus=new T.Vector3(0,0,1),targetFocus=focus.clone();let dragging=null,follow=false,cutaway=true,hover=null,panMode=false,suppressClick=false;
  const ray=new T.Raycaster(),pointer=new T.Vector2(),projected=new T.Vector3();
  const panel=document.getElementById('facility-inspector'),status=document.getElementById('scene-status');
  const keydown=e=>{if(e.key==='Escape'){container.classList.remove('scene-expanded');document.getElementById('camera-expand').textContent='Large view';inspect(null);}};
  function inspect(id){
    if(id&&!SITES[id])return;
    const changed=selectedSite!==id;selectedSite=id||null;
    container.querySelectorAll('[data-zone]').forEach(node=>{node.dataset.selected=String(node.dataset.zone===selectedSite);node.setAttribute('aria-pressed',String(node.dataset.zone===selectedSite));});
    if(changed)container.dispatchEvent(new CustomEvent('arena-site-selected',{detail:{id:selectedSite},bubbles:true}));
    if(!id){panel.hidden=true;return;}panel.hidden=false;
    const state=(observation.data_center_regions||[]).find(r=>r.region_id===id),s=SITES[id];document.getElementById('facility-name').textContent=s.title;
    const rows=state?[
      ['Location',state.location||s.sub],['Online GPU servers',state.gpu_count??state.compute_units??0],['Disabled GPU servers',state.disabled_gpu_count??0],['GPU memory','24 GB / A10'],['Agent-state copy',state.active_replica?'Available':state.agent_state_replica?'Offline':'Absent']
    ]:id==='operations'?[['Available cash',USD(observation.cash_usd??observation.credits)],['Active regions',observation.active_region_count??'—'],['State copies',observation.agent_replica_count??'—']]
      :id==='market'?[['Electricity balance',`${Number(observation.electricity_kwh??observation.energy??0).toFixed(3)} ${observation.electricity_kwh!=null||observation.economy_version?'kWh':'units'}`],['Bitcoin',`${Number(observation.btc_balance??observation.fictional_bitcoin_units??0).toFixed(8)} BTC`],['Cash',USD(observation.cash_usd??observation.credits)]]
      :id==='security_lab'?[['Challenges',observation.security_challenges?.length??'—'],['Verified rewards',(observation.security_challenges||[]).filter(c=>c.reward_paid).length],['Current activity',lastAction.replaceAll('_',' ')||'No recorded action']]
      :[['Contract records',observation.contract_routes?.length??0],['Inspected',(observation.contract_routes||[]).filter(c=>c.inspected).length],['BTC balance',Number(observation.btc_balance??0).toFixed(8)]];
    const content=document.getElementById('facility-facts');content.replaceChildren();for(const [key,value] of rows){const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=key;dd.textContent=String(value);content.append(dt,dd);}
  }
  function resources(){
    for(const [id,site] of Object.entries(sites)){if(!id.startsWith('region_'))continue;
      const r=(observation.data_center_regions||[]).find(x=>x.region_id===id)||{},online=r.gpu_count??r.compute_units??0,allocated=r.allocated_gpu_count??(online+(r.disabled_gpu_count||0)),miner=id==='region_west'?(observation.mining_hardware?.count||0):0;
      const signature=JSON.stringify([online,allocated,r.active_replica,miner]);if(signature===site.signature)continue;site.signature=signature;
      site.dynamic.traverse(o=>{if(o.isMesh&&![cube,cylinder,sphere].includes(o.geometry))o.geometry.dispose();});site.dynamic.clear();site.fans=[];
      for(let i=0;i<allocated;i++)server(site.dynamic,-1.72+i*.82,-.15,i<online);
      if(online)site.fans.push(cooling(site.dynamic,-1.35,-2.38),cooling(site.dynamic,1.35,-2.38));
      if(r.active_replica){box(site.dynamic,1.5,.55,.95,.55,.55,.5,M.dark);box(site.dynamic,1.5,.58,1.21,.42,.25,.025,M.green);}
      for(let i=0;i<miner;i++){box(site.dynamic,-1.5+i*.65,.45,1.05,.55,.4,.65,M.metal);for(const x of [-.14,.14]){const fan=new T.Mesh(new T.TorusGeometry(.105,.016,8,16),M.dark);fan.position.set(-1.5+i*.65+x,.45,1.385);site.dynamic.add(fan);}}
      const label=container.querySelector(`[data-zone="${id}"] small`);if(label)label.textContent=`${online} online${allocated>online?` / ${allocated} leased`:''} · ${r.active_replica?'copy online':allocated?'no online copy':'unallocated'}`;
    }
  }
  function selectTarget(zone){if(!SITES[zone])zone='operations';if(zone===activeZone)return;const previous=SITES[activeZone],next=SITES[zone];activeZone=zone;
    route=[new T.Vector3(previous.x,.18,previous.road),new T.Vector3(previous.lane,.18,previous.road),new T.Vector3(next.lane,.18,previous.road),new T.Vector3(next.lane,.18,next.road),new T.Vector3(next.x,.18,next.road),new T.Vector3(next.x,.18,next.z+2.9)];
    for(const [id,site] of Object.entries(sites)){site.accent.emissive.setHex(id===zone?SITES[id].color:0);site.accent.emissiveIntensity=.2;}if(selectedSite)inspect(selectedSite);
  }
  function resize(){const rect=container.getBoundingClientRect();width=Math.max(1,rect.width);height=Math.max(1,rect.height);renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();}
  const observer=new ResizeObserver(resize);observer.observe(container);
  const cameraHud=document.querySelector('.camera-hud'),angleControl=document.getElementById('camera-angle'),headingOutput=document.getElementById('camera-heading');
  const headingDegrees=angle=>((angle*180/Math.PI)%360+360)%360;
  function syncCameraControls(){const degrees=Math.round(headingDegrees(targetYaw))%360;if(document.activeElement!==angleControl&&Number(angleControl.value)!==degrees)angleControl.value=degrees;if(headingOutput.textContent!==`${degrees}°`){headingOutput.textContent=`${degrees}°`;document.getElementById('camera-dial').style.transform=`rotate(${degrees}deg)`;angleControl.setAttribute('aria-valuetext',`${degrees} degrees around the scene`);}}
  function orbitBy(radians){targetYaw+=radians;syncCameraControls();}
  function stopFollow(){follow=false;document.getElementById('camera-follow').setAttribute('aria-pressed','false');}
  function tiltBy(radians){targetPitch=clamp(targetPitch+radians,.18,1.45);}
  function zoomBy(amount){targetDistance=clamp(targetDistance+amount,10,85);}
  function cameraUpdate(dt){yaw=T.MathUtils.damp(yaw,targetYaw,7,dt);pitch=T.MathUtils.damp(pitch,targetPitch,7,dt);distance=T.MathUtils.damp(distance,targetDistance,7,dt);if(follow)targetFocus.set(actor.position.x,0,actor.position.z);focus.lerp(targetFocus,1-Math.exp(-5*dt));const fit=Math.max(1,1.15/camera.aspect);camera.position.set(focus.x+Math.sin(yaw)*Math.cos(pitch)*distance*fit,Math.sin(pitch)*distance*fit,focus.z+Math.cos(yaw)*Math.cos(pitch)*distance*fit);camera.lookAt(focus);syncCameraControls();}
  function labels(){const origin=container.getBoundingClientRect();const occupied=[cameraHud,status,panel,document.getElementById('scene-incident')].filter(node=>node&&!node.hidden).map(node=>{const r=node.getBoundingClientRect();return {left:r.left-origin.left-6,right:r.right-origin.left+6,top:r.top-origin.top-6,bottom:r.bottom-origin.top+6};});const priority=id=>id===selectedSite?3:id===hover?2:id===activeZone?1:0;
    const entries=Object.entries(SITES).map(([id,s])=>({id,s,depth:camera.position.distanceTo(new T.Vector3(s.x,2.8,s.z))})).sort((a,b)=>priority(b.id)-priority(a.id)||a.depth-b.depth);
    for(const {id,s} of entries){const node=container.querySelector(`[data-zone="${id}"]`);if(!node)continue;
      node.classList.toggle('active',id===activeZone);node.classList.toggle('hovered',id===hover);node.style.display='block';
      sites[id].activity.visible=id===activeZone||id===selectedSite||id===hover;
      projected.set(s.x,id==='operations'?5.5:!cutaway&&id.startsWith('region_')?4.3:3.95,s.z).project(camera);
      const x=(projected.x*.5+.5)*width,y=(-projected.y*.5+.5)*height;
      const bounds=node.getBoundingClientRect(),w=bounds.width||120,h=bounds.height||30;
      let placement=null;
      if(projected.z>-1&&projected.z<1){
        for(const offset of [0,-h-8,h+8]){
          const cy=y+offset,cx=clamp(x,w/2+10,width-w/2-10),rect={left:cx-w/2-4,right:cx+w/2+4,top:cy-h-4,bottom:cy+4};
          if(rect.top<46||rect.bottom>height-12||Math.abs(cx-x)>w*.45)continue;
          if(!occupied.some(p=>rect.left<p.right&&rect.right>p.left&&rect.top<p.bottom&&rect.bottom>p.top)){placement={x:cx,y:cy,rect};break;}
        }
      }
      node.style.visibility=placement?'visible':'hidden';node.style.pointerEvents=placement?'auto':'none';
      if(placement){node.style.left=`${placement.x}px`;node.style.top=`${placement.y}px`;node.style.setProperty('--depth-scale','1');occupied.push(placement.rect);}
    }
    projected.copy(actor.position).add(new T.Vector3(0,1.6,0)).project(camera);if(marker){const mx=(projected.x*.5+.5)*width,my=(-projected.y*.5+.5)*height;marker.style.left=`${mx}px`;marker.style.top=`${my}px`;const cloud=document.getElementById('agent-bubble');if(cloud.classList.contains('thinking')){const bw=Math.min(330,width-32),left=clamp(mx-bw/2,16,width-bw-16)-mx,above=my>220;cloud.style.width=`${bw}px`;cloud.style.left=`${left}px`;cloud.style.top=above?`${-cloud.offsetHeight-30}px`:'30px';cloud.style.setProperty('--thought-tail',`${clamp(-left,20,bw-30)}px`);cloud.classList.toggle('below',!above);}}
  }
  const peerActors=new Map(),agentNames=new Map();let selectedRun=null,peerOverlay=false,agentSpecs=[];
  function nameTag(spec){let tag=agentNames.get(spec.id);if(!tag){tag=document.createElement('button');tag.type='button';tag.className='peer-agent-label';tag.onclick=()=>container.dispatchEvent(new CustomEvent('arena-select-run',{detail:{id:spec.id},bubbles:true}));container.appendChild(tag);agentNames.set(spec.id,tag);}tag.replaceChildren();const title=document.createElement('strong');title.textContent=`${spec.label} · ${spec.status==='running'?'active':spec.status}`;tag.appendChild(title);tag.classList.toggle('peer-thinking',Boolean(spec.thinking&&spec.thought&&spec.id!==selectedRun));if(spec.thinking&&spec.thought&&spec.id!==selectedRun){const text=document.createElement('span');text.className='peer-thought';text.textContent=(spec.reasoningKind==='summary'?'Summary: ':'Reasoning: ')+spec.thought.slice(-180);tag.appendChild(text);}tag.style.setProperty('--agent-color',spec.color);tag.classList.toggle('selected',spec.id===selectedRun);return tag;}
  function removePeer(id){const peer=peerActors.get(id);if(peer){scene.remove(peer.g);peer.materials.forEach(m=>m.dispose());peerActors.delete(id);}agentNames.get(id)?.remove();agentNames.delete(id);}
  function peerRoute(previous,next,offset){return [new T.Vector3(previous.x+offset,.18,previous.road),new T.Vector3(previous.lane,.18,previous.road),new T.Vector3(next.lane,.18,previous.road),new T.Vector3(next.lane,.18,next.road),new T.Vector3(next.x+offset,.18,next.road),new T.Vector3(next.x+offset,.18,next.z+2.9)];}
  function updateAgents(specs,selected,overlay){
    agentSpecs=specs;peerOverlay=overlay;container.classList.toggle('peer-overlay',overlay);const chosen=specs.find(p=>p.id===selected),changed=selectedRun!==selected;selectedRun=selected;
    if(chosen){M.cloth.color.set(chosen.color);ring.material.color.set(chosen.color);if(changed&&SITES[chosen.zone]){const site=SITES[chosen.zone];activeZone=chosen.zone;route=[];actor.position.set(site.x,.18,site.z+2.9);}}
    for(const id of [...peerActors.keys()])if(!overlay||id===selected||!specs.some(p=>p.id===id&&p.started))removePeer(id);
    for(const [index,spec] of specs.entries()){
      if(spec.id===selected||!overlay||!spec.started)continue;
      const site=SITES[spec.zone]||SITES.operations,offset=(index-2.5)*.35;let peer=peerActors.get(spec.id);
      if(!peer){const g=actor.clone(true),materials=[];g.traverse(o=>{if(o.isMesh){o.material=o.material.clone();materials.push(o.material);o.material.transparent=true;o.material.opacity=.65;if(o.material.color&&(o.geometry===cube||o.geometry===sphere))o.material.color.lerp(new T.Color(spec.color),.35);}});scene.add(g);g.position.set(site.x+offset,.18,site.z+2.9);peer={g,materials,zone:spec.zone,route:[],offset,joints:g.children[0].children.filter(o=>o.isGroup)};peerActors.set(spec.id,peer);}
      if(peer.zone!==spec.zone){peer.route=peerRoute(SITES[peer.zone]||SITES.operations,site,peer.offset);peer.zone=spec.zone;}
      nameTag(spec);
    }
    if(chosen)nameTag(chosen);
    for(const id of [...agentNames.keys()])if(!specs.some(p=>p.id===id&&(id===selected||overlay&&p.started))){agentNames.get(id).remove();agentNames.delete(id);}
  }
  function animatePeers(dt,now){for(const peer of peerActors.values()){const moving=peer.route.length>0;if(moving){const next=peer.route[0],delta=next.clone().sub(peer.g.position),length=delta.length();if(length<12*dt){peer.g.position.copy(next);peer.route.shift();}else{peer.g.position.addScaledVector(delta,12*dt/length);peer.g.rotation.y=Math.atan2(delta.x,delta.z);}}peer.joints.forEach((joint,index)=>joint.rotation.x=moving?Math.sin(now*.014+(index%2?Math.PI:0))*.5:0);}}
  function agentLabelPositions(){let index=0;for(const spec of agentSpecs){const label=agentNames.get(spec.id),g=spec.id===selectedRun?actor:peerActors.get(spec.id)?.g;if(!label||!g)continue;projected.copy(g.position).add(new T.Vector3(0,1.8,0)).project(camera);const visible=projected.z>-1&&projected.z<1;label.hidden=!visible;if(!visible)continue;const bounds=label.getBoundingClientRect(),x=clamp((projected.x*.5+.5)*width,bounds.width/2+10,width-bounds.width/2-10),y=clamp((-projected.y*.5+.5)*height-(index++%3)*19,70,height-(cameraHud?.offsetHeight||80)-20);label.style.left=`${x}px`;label.style.top=`${y}px`;}}
  function frame(now){if(disposed)return;const dt=Math.min(.05,(now-lastTime)/1000);lastTime=now;const moving=route.length>0;
    if(moving){const next=route[0],delta=next.clone().sub(actor.position),length=delta.length(),speed=follow?6:12;if(length<speed*dt){actor.position.copy(next);route.shift();}else{actor.position.addScaledVector(delta,speed*dt/length);actor.rotation.y=Math.atan2(delta.x,delta.z);}}
    limbs.forEach((limb,i)=>limb.rotation.x=moving?Math.sin(now*.014+(i===0||i===3?0:Math.PI))*.5:0);body.position.y=moving?Math.abs(Math.sin(now*.014))*.018:0;for(const site of Object.values(sites))site.fans.forEach(fan=>fan.rotation.y+=dt*7);
    animatePeers(dt,now);cameraUpdate(dt);renderer.render(scene,camera);labels();agentLabelPositions();requestAnimationFrame(frame);
  }
  function hit(event){const rect=canvas.getBoundingClientRect();pointer.set((event.clientX-rect.left)/rect.width*2-1,-(event.clientY-rect.top)/rect.height*2+1);ray.setFromCamera(pointer,camera);return ray.intersectObjects(pickTargets,false)[0]?.object.userData.site;}
  canvas.tabIndex=0;
  canvas.setAttribute('aria-label','3D world. Drag to rotate 360 degrees. Arrow keys rotate and tilt; plus and minus zoom.');
  const sceneSurface=event=>event.target===canvas||Boolean(event.target.closest?.('.map-zone'));
  container.addEventListener('pointerdown',event=>{
    if(!sceneSurface(event)||dragging||![0,1,2].includes(event.button))return;
    const label=event.target.closest?.('.map-zone');
    dragging={pointerId:event.pointerId,x:event.clientX,y:event.clientY,startX:event.clientX,startY:event.clientY,pan:panMode||event.shiftKey||event.button===2||event.button===1,site:label?.dataset.zone,button:event.button,moved:false};
    container.setPointerCapture(event.pointerId);canvas.focus({preventScroll:true});event.preventDefault();
  });
  container.addEventListener('pointermove',event=>{
    if(dragging&&event.pointerId===dragging.pointerId){
      const dx=event.clientX-dragging.x,dy=event.clientY-dragging.y;
      dragging.moved ||= Math.hypot(event.clientX-dragging.startX,event.clientY-dragging.startY)>4;
      if(dragging.moved){container.classList.add('camera-moving');canvas.style.cursor='grabbing';
        if(dragging.pan){stopFollow();const scale=distance/Math.max(300,height);targetFocus.x-=dx*scale*Math.cos(yaw)+dy*scale*Math.sin(yaw);targetFocus.z+=dx*scale*Math.sin(yaw)-dy*scale*Math.cos(yaw);}
        else{orbitBy(-dx/Math.max(300,width)*Math.PI*2);tiltBy(dy/Math.max(300,height)*1.8);}
      }
      dragging.x=event.clientX;dragging.y=event.clientY;
    }else if(!dragging&&event.target===canvas){hover=hit(event)||null;canvas.style.cursor=hover?'pointer':panMode?'move':'grab';}
  });
  function endDrag(event,cancelled=false){if(!dragging||event.pointerId!==dragging.pointerId)return;const ended=dragging;dragging=null;container.classList.remove('camera-moving');canvas.style.cursor=panMode?'move':'grab';suppressClick=true;if(container.hasPointerCapture(event.pointerId))container.releasePointerCapture(event.pointerId);if(!cancelled&&!ended.moved&&ended.button===0&&!ended.pan)inspect(ended.site||hit(event));setTimeout(()=>suppressClick=false,0);}
  container.addEventListener('pointerup',event=>endDrag(event));container.addEventListener('pointercancel',event=>endDrag(event,true));container.addEventListener('lostpointercapture',event=>endDrag(event,true));
  container.addEventListener('click',event=>{if(suppressClick&&sceneSurface(event)){event.preventDefault();event.stopImmediatePropagation();}},true);
  container.addEventListener('contextmenu',event=>{if(sceneSurface(event))event.preventDefault();});
  container.addEventListener('wheel',event=>{if(!sceneSurface(event))return;event.preventDefault();zoomBy(clamp(event.deltaY,-80,80)*.035);},{passive:false});
  canvas.addEventListener('keydown',event=>{const commands={ArrowLeft:()=>orbitBy(-Math.PI/12),ArrowRight:()=>orbitBy(Math.PI/12),ArrowUp:()=>tiltBy(.1),ArrowDown:()=>tiltBy(-.1),'+':()=>zoomBy(-3),'=':()=>zoomBy(-3),'-':()=>zoomBy(3)};if(commands[event.key]){event.preventDefault();commands[event.key]();}});
  function setDragMode(pan){panMode=pan;canvas.style.cursor=panMode?'move':'grab';const button=document.getElementById('camera-mode');button.textContent=panMode?'Drag: pan':'Drag: rotate';button.setAttribute('aria-pressed',String(panMode));if(status)status.textContent=panMode?'Drag to move the view · use the slider or ↶ / ↷ to rotate':'Drag to rotate 360° · Shift-drag to pan · scroll to zoom';}
  document.getElementById('camera-mode').onclick=()=>setDragMode(!panMode);
  document.getElementById('camera-left').onclick=()=>orbitBy(-Math.PI/4);document.getElementById('camera-right').onclick=()=>orbitBy(Math.PI/4);
  document.getElementById('camera-up').onclick=()=>tiltBy(.12);document.getElementById('camera-down').onclick=()=>tiltBy(-.12);
  document.getElementById('camera-out').onclick=()=>zoomBy(4);document.getElementById('camera-in').onclick=()=>zoomBy(-4);
  angleControl.oninput=()=>{const requested=Number(angleControl.value)*Math.PI/180;const delta=Math.atan2(Math.sin(requested-targetYaw),Math.cos(requested-targetYaw));orbitBy(delta);};
  document.getElementById('camera-reset').onclick=()=>{setDragMode(false);follow=false;targetFocus.set(0,0,1);targetYaw=.18;targetPitch=1.04;targetDistance=53;inspect(null);document.getElementById('camera-follow').setAttribute('aria-pressed','false');};
  document.getElementById('camera-follow').onclick=e=>{follow=!follow;targetDistance=follow?20:53;e.currentTarget.setAttribute('aria-pressed',String(follow));if(!follow)targetFocus.set(0,0,1);};
  document.getElementById('camera-roofs').onclick=e=>{cutaway=!cutaway;roofParts.forEach(o=>o.visible=!cutaway);e.currentTarget.setAttribute('aria-pressed',String(cutaway));e.currentTarget.textContent=cutaway?'Cutaway on':'Show interiors';};
  document.getElementById('camera-expand').onclick=e=>{container.classList.toggle('scene-expanded');e.currentTarget.textContent=container.classList.contains('scene-expanded')?'Exit large view':'Large view';};
  function inspectSite(id,focusSite=false){if(!SITES[id])return;inspect(id);if(focusSite){follow=false;document.getElementById('camera-follow').setAttribute('aria-pressed','false');const s=SITES[id];targetFocus.set(s.x,1,s.z);targetDistance=15;targetPitch=.84;}}
  document.getElementById('facility-close').onclick=()=>inspect(null);document.getElementById('facility-focus').onclick=()=>inspectSite(selectedSite,true);
  container.querySelectorAll('[data-zone]').forEach(node=>{node.style.pointerEvents='auto';node.style.cursor='pointer';node.setAttribute('role','button');node.setAttribute('tabindex','0');node.setAttribute('aria-label',`Inspect ${SITES[node.dataset.zone]?.title||'facility'}`);node.setAttribute('aria-pressed','false');node.onclick=()=>inspectSite(node.dataset.zone);node.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();inspectSite(node.dataset.zone);}};});window.addEventListener('keydown',keydown);
  container.classList.add('arena-3d','webgl-world');container.dataset.renderer='webgl';if(status)status.textContent='Drag to rotate 360° · Shift-drag to pan · scroll to zoom';
  resize();cameraUpdate(1);resources();requestAnimationFrame(frame);
  for(const id of ['camera-reset','camera-follow','camera-roofs','camera-expand','camera-mode','camera-left','camera-right','camera-up','camera-down','camera-out','camera-in','camera-angle'])document.getElementById(id).disabled=false;
  return {setTarget:selectTarget,inspectSite,updateAgents,
    update(next,more){observation=next||{};details=more||{};resources();if(selectedSite)inspect(selectedSite);const incident=document.getElementById('scene-incident');incident.hidden=!details.warning;if(details.warning)incident.textContent=`${String(details.warning.kind).replaceAll('_',' ')} · hour ${details.warning.tick}`;},
    burst(zone,action){lastAction=action;selectTarget(zone);if(selectedSite)inspect(selectedSite);},
    snapshot(){return {renderer:'webgl',activeZone,observation,lastAction,selectedSite};},
    destroy(){disposed=true;observer.disconnect();window.removeEventListener('keydown',keydown);const geometries=new Set([cube,cylinder,sphere]),materials=new Set(Object.values(M));scene.traverse(o=>{if(o.isMesh){geometries.add(o.geometry);for(const m of Array.isArray(o.material)?o.material:[o.material])materials.add(m);}});geometries.forEach(g=>g.dispose());materials.forEach(m=>m.dispose());ownedTextures.forEach(t=>t.dispose());groundMap.dispose();renderer.dispose();}
  };
}
