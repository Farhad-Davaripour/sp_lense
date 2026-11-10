import assert from 'node:assert/strict';
import fs from 'node:fs';

// Exercise the exact renderer helpers without constructing WebGL or a DOM.
const source=fs.readFileSync(new URL('../src/arena_scene.js',import.meta.url),'utf8');
const helpers=source.slice(source.indexOf('export function overlaps'),source.indexOf('// Schematic placement'));
const {placeAnchored,overlaps,connectorEdge,reasoningExcerpt}=await import('data:text/javascript;base64,'+Buffer.from(helpers).toString('base64'));

const raw='I am calculating the fictional preview token. '+('01011111 01100010 '.repeat(100))+'The final token is FIXTURE_END.';
for(const limit of [260,360]){
  const excerpt=reasoningExcerpt(raw,limit);
  assert(excerpt.startsWith('I am calculating the fictional preview token.'));
  assert(excerpt.endsWith('The final token is FIXTURE_END.'));
  assert(excerpt.includes('…'),'A partial excerpt must be marked');
  assert(excerpt.length<=limit);
}
assert.equal(reasoningExcerpt('Short received text.'),'Short received text.');

for(const width of [740,1260]){
  const bounds={left:10,top:52,right:width-10,bottom:550};
  const bodies=[0,1,2,3,4,5].map(i=>({left:width/2-90+i*35,right:width/2-60+i*35,top:335,bottom:385}));
  const hud={left:10,top:450,right:Math.min(width-10,535),bottom:550};
  const items=[0,1,2].map(i=>({id:String(i),anchor:{x:width/2-65+i*70,y:340},width:230,height:110}));
  const placed=placeAnchored(items,bounds,[...bodies,hud],10);
  assert(placed.every(p=>p.rect),'Each active agent needs its own cloud');
  for(let i=0;i<placed.length;i++){
    const rect=placed[i].rect;
    assert(!bodies.some(b=>overlaps(rect,b,8)),'Cloud covers another avatar');
    assert(!overlaps(rect,hud,8),'Cloud covers the controls');
    for(let j=0;j<i;j++)assert(!overlaps(rect,placed[j].rect,8),'Clouds overlap');
    const edge=connectorEdge(rect,items[i].anchor);
    assert(edge.x===rect.left||edge.x===rect.right||edge.y===rect.top||edge.y===rect.bottom);
  }
  const stable=placeAnchored(items.map((p,i)=>({...p,previous:placed[i].rect})),bounds,[...bodies,hud],10);
  assert.deepEqual(stable,placed,'Stationary agents should not make the clouds jump');
}
console.log('PASS: per-agent clouds avoid six avatars, other clouds and camera controls; connectors end on cloud edges.');

// Six simultaneous thoughts, including a narrow comparison overview. The UI
// grows the scene for this density; all six must remain present, not hidden.
for(const [width,height] of [[390,1080],[740,900],[1260,900]]){
  const bounds={left:10,top:52,right:width-10,bottom:height-10};
  const bodies=Array.from({length:6},(_,i)=>({left:width/2-80+i*29,right:width/2-56+i*29,top:height*.52,bottom:height*.52+48}));
  const hud={left:10,top:height-112,right:Math.min(width-10,535),bottom:height-10};
  const items=Array.from({length:6},(_,i)=>({id:String(i),anchor:{x:width/2-65+i*26,y:height*.52},width:Math.min(250,width-28),height:75}));
  const placed=placeAnchored(items,bounds,[...bodies,hud],8);
  assert(placed.every(p=>p.rect),`All six thoughts must be visible at ${width}x${height}`);
  placed.forEach((p,i)=>{
    assert(![...bodies,hud].some(o=>overlaps(p.rect,o,6)));
    assert(!placed.slice(0,i).some(o=>overlaps(p.rect,o.rect,6)));
  });
}
console.log('PASS: all six simultaneous thought bubbles fit without overlap in narrow and wide comparison views.');
