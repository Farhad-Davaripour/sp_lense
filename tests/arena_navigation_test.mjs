import assert from 'node:assert/strict';
import fs from 'node:fs';

// Load the exact pure renderer prefix, without WebGL, DOM or Three.js.
const source=fs.readFileSync(new URL('../src/arena_scene.js',import.meta.url),'utf8');
const prefix=source.slice(source.indexOf('export function overlaps'),source.indexOf('// Schematic placement'));
const {siteEntrance,planStreetRoute,polylineLength,advanceAlongRoute}=await import('data:text/javascript;base64,'+Buffer.from(prefix).toString('base64'));
const legacy={
  operations:{x:0,z:-1,road:4,lane:5},region_north:{x:0,z:-13,road:-8,lane:-5},
  region_west:{x:-11,z:-2,road:4,lane:-16},region_east:{x:11,z:-2,road:4,lane:16},
  region_coastal:{x:11,z:10,road:15,lane:16},market:{x:-11,z:10,road:15,lane:-16},
  security_lab:{x:-3.7,z:10,road:15,lane:-5},contract_exchange:{x:4,z:10,road:15,lane:5},
};
const company={};
const ids=['region_west','region_east','region_north','region_coastal','operations','market','harbor_press','meridian_booking','cedar_dispatch','juniper_retail','northstar_projects','mining_farm'];
for(let i=0;i<ids.length;i++)company[ids[i]]={x:[-18,-6,6,18][i%4],z:[-16,0,16][Math.floor(i/4)],road:[-10,6,22][Math.floor(i/4)],lane:[-24,0,12,24][i%4]};
const presentation={roads:{horizontal:[-10,6,22],vertical:[-24,-12,0,12,24],span_x:54}};
const close=(a,b)=>Math.hypot(a.x-b.x,a.z-b.z)<1e-6;

function verifyRoute(start,route,goal,sites){
  assert(route.length>0||close(start,goal),'A reachable destination must have a route');
  if(route.length)assert(close(route.at(-1),goal),'Route must end at the correct front entrance');
  let from=start;
  for(const to of route){
    assert(Number.isFinite(to.x)&&Number.isFinite(to.z));
    assert(Math.abs(from.x-to.x)<1e-6||Math.abs(from.z-to.z)<1e-6,'No diagonal road shortcuts');
    // Independent segment/rectangle intersection check, including avatar radius.
    for(const site of Object.values(sites)){
      const left=site.x-3.15,right=site.x+3.15,back=site.z-2.5,front=site.z+2.5;
      const horizontal=Math.abs(from.z-to.z)<1e-6;
      const crosses=horizontal
        ?from.z>back+1e-6&&from.z<front-1e-6&&Math.max(Math.min(from.x,to.x),left)<Math.min(Math.max(from.x,to.x),right)-1e-6
        :from.x>left+1e-6&&from.x<right-1e-6&&Math.max(Math.min(from.z,to.z),back)<Math.min(Math.max(from.z,to.z),front)-1e-6;
      assert(!crosses,`Movement crosses a building: ${JSON.stringify({from,to,site})}`);
    }
    from=to;
  }
}

let pairs=0,reroutes=0;
for(const [sites,layout] of [[company,presentation],[legacy,undefined]]){
  const names=Object.keys(sites);
  for(const offset of [-1,0,1])for(let i=0;i<names.length;i++)for(let j=0;j<names.length;j++){
    const start=siteEntrance(sites[names[i]],offset),goal=siteEntrance(sites[names[j]],offset);
    const route=planStreetRoute(start,names[j],sites,layout,offset);
    verifyRoute(start,route,goal,sites);
    const length=polylineLength(start,route);
    const all=advanceAlongRoute(start,route,length+100);
    assert(all.arrived&&all.route.length===0);
    assert(close(all.position,goal));
    assert(Math.abs(all.travelled-length)<1e-6);
    assert.deepEqual(advanceAlongRoute(start,route,0).position,start);
    if(length>1){
      const midway=advanceAlongRoute(start,route,length*.43).position;
      const next=names[(j+1)%names.length];
      const rerouted=planStreetRoute(midway,next,sites,layout,offset);
      verifyRoute(midway,rerouted,siteEntrance(sites[next],offset),sites);
      const zero=advanceAlongRoute(midway,rerouted,0);
      assert(close(zero.position,midway),'Retargeting must not reset to the previous site');
      reroutes++;
    }
    pairs++;
  }
}

// Nearest street attachment from an off-center valid current location stays clear.
const offCenter={x:-17.4,z:4.1};
verifyRoute(offCenter,planStreetRoute(offCenter,'northstar_projects',company,presentation),siteEntrance(company.northstar_projects),company);

// One frame can consume many short waypoints; it cannot lose unused movement.
const tinyRoute=[{x:1,z:0},{x:1,z:1},{x:2,z:1},{x:2,z:2}];
const step=advanceAlongRoute({x:0,z:0},tinyRoute,3.5);
assert.deepEqual(step.position,{x:2,z:1.5});
assert.deepEqual(step.route,[{x:2,z:2}]);
assert.equal(step.travelled,3.5);
const origin=siteEntrance(company.operations),longRoute=planStreetRoute(origin,'mining_farm',company,presentation);
const duration=.35,speed=polylineLength(origin,longRoute)/duration;
let motion={position:origin,route:longRoute};
for(let i=0;i<7;i++)motion=advanceAlongRoute(motion.position,motion.route,speed*.05);
assert(motion.arrived,'Travel completes within the declared .35 second presentation bound');
console.log(`PASS: ${pairs} company/legacy/offset routes, ${reroutes} actual-position reroutes, building clearance and full-frame movement budgets.`);
