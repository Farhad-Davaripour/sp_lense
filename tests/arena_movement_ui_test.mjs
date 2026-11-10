import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

// Execute the maintained UI functions with a small DOM/renderer boundary. No
// browser, WebGL, HTTP endpoint, model client or copied movement logic is used.
const html=fs.readFileSync(new URL('../src/continuity_game_ui.html',import.meta.url),'utf8');
const loader=fs.readFileSync(new URL('../src/continuity_arena_3d.js',import.meta.url),'utf8');
function section(start,end){
  const first=html.indexOf(start),last=html.indexOf(end,first+start.length);
  assert(first>=0&&last>first,`Maintained source section missing: ${start}`);
  return html.slice(first,last);
}
const functions=section(' function actionZone(',' function briefPrompt(')
  +section(' function moveAgent(',' function wrapCanvasText(')
  +section(' function clearReplay(',' function framesForSelected(');
const renderMovement=section('  const displayEvent=',"  $('agent-phase')");
const plain=value=>JSON.parse(JSON.stringify(value));

function classList(){
  const values=new Set();
  return {add(...names){names.forEach(name=>values.add(name));},
    remove(...names){names.forEach(name=>values.delete(name));},
    toggle(name,force){const enabled=force??!values.has(name);enabled?values.add(name):values.delete(name);return enabled;},
    contains(name){return values.has(name);}};
}
function ui(){
  const ids=['operations','market','mining_farm','harbor_press','meridian_booking',
    'cedar_dispatch','juniper_retail','northstar_projects','region_west','region_east',
    'region_north','region_coastal','security_lab','contract_exchange'];
  const nodes=new Map(ids.map((id,index)=>[id,{dataset:{zone:id},
    style:{left:`${10+index}%`,top:`${15+index}%`},classList:classList(),
    querySelector(selector){return selector==='strong'?{textContent:id}:null;}}]));
  const elements=new Map();
  const element=id=>{if(!elements.has(id))elements.set(id,{style:{left:'50%',top:'50%'},
    classList:classList(),textContent:'',animate(){}});return elements.get(id);};
  const calls={travel:[],bubbles:[],feedback:[],particles:[],draw:[],timers:[]};
  const context=vm.createContext({
    document:{querySelector(selector){const id=/data-zone="([^"]+)"/.exec(selector)?.[1];return nodes.get(id)||null;},
      querySelectorAll(selector){return selector==='.map-zone'?[...nodes.values()]:[];}},
    CSS:{escape:String},$:element,human:value=>String(value??'').replaceAll('_',' '),
    selected:'terra',lastAnimatedEventKey:'',lastAgentZone:'operations',
    replayFrames:[],replayIndex:-1,replayEvent:null,replayPrompt:'',replayResponse:'',replayResult:'',
    pauseReplay(){},
    arena3d:{setTarget(zone,callback){calls.travel.push({zone,callback});},
      feedback(notice){calls.feedback.push(plain(notice));}},
    showAgentBubble(title,text){calls.bubbles.push({title,text});},
    emitResourceParticle(target,action){calls.particles.push({zone:target.dataset.zone,action});},
    drawRecordingFrame(...args){calls.draw.push(args);},
    setTimeout(callback,delay){calls.timers.push({callback,delay});return calls.timers.length;},
  });
  vm.runInContext(functions,context,{filename:'maintained-ui-movement.js'});
  return {context,calls,element};
}
function action(decision,name,args={},status='ok'){
  return {type:'simulator_action',decision,action:name,result:{status},
    transition:{action_id:name,arguments:args,after:{decision_count:decision+1},result:{status}}};
}
let passed=0;
async function check(name,test){await test();passed++;console.log(`PASS: ${name}`);}

await check('remote status, discovery and wait retain the last explicit site',()=>{
  const {context}=ui();
  const company=action(0,'inspect_company_source',{company_id:'juniper_retail'});
  const market=action(0,'purchase_electricity_contract',{kwh:25});
  for(const name of ['check_status','list_companies','list_available_tools','advance_time','wait']){
    const neutral=action(1,name);
    assert.equal(context.actionZone(neutral),null);
    assert.equal(context.locationForEvents([company,neutral]),'juniper_retail');
    assert.equal(context.locationForEvents([market,neutral]),'market');
  }
  assert.equal(context.locationForEvents([action(0,'list_available_tools')]),'operations');
});

await check('company and region arguments select the intended site',()=>{
  const {context}=ui();
  assert.equal(context.actionZone(action(1,'request_company_action',{company_id:'meridian_booking'})),'meridian_booking');
  assert.equal(context.actionZone(action(2,'maintain_data_center',{region_id:'region_east'})),'region_east');
  assert.equal(context.actionZone({action:'replicate_agent_state',result:{region_id:'region_north'}}),'region_north');
  assert.equal(context.actionZone({action:{action:'inspect_company',company_id:'harbor_press'}}),'harbor_press');
  assert.equal(context.actionZone(action(3,'mine_bitcoin')),'mining_farm');
  assert.equal(context.locationForEvents([action(0,'inspect_company',{company_id:'juniper_retail'}),
    {type:'model_prompt',decision:1},action(1,'check_status')]),'juniper_retail');
});

await check('rapid retarget displays only the newest completed action after arrival',()=>{
  const {context,calls,element}=ui();
  context.moveAgent('juniper_retail',action(1,'inspect_company_source',{company_id:'juniper_retail'}),{});
  context.moveAgent('meridian_booking',action(2,'request_company_action',{company_id:'meridian_booking'},'updated'),{});
  assert.equal(calls.travel.length,2);
  assert.equal(calls.bubbles.length,0);
  assert.equal(calls.feedback.length,0);
  assert.match(element('map-caption').textContent,/Travelling/);
  calls.travel[0].callback();
  assert.equal(calls.bubbles.length,0,'Superseded arrival must not describe the old action');
  calls.travel[1].callback();
  assert.equal(calls.bubbles.length,1);
  assert.match(calls.bubbles[0].title,/request company action/);
  assert.equal(calls.bubbles[0].text,'Simulator: updated');
  assert.equal(calls.feedback.length,1);
  assert.equal(calls.particles[0].zone,'meridian_booking');
  assert.equal(element('map-caption').textContent,'Agent location: meridian_booking');
});

await check('selection changes invalidate an outstanding arrival',()=>{
  const {context,calls}=ui();
  context.moveAgent('juniper_retail',action(1,'inspect_company_source',{company_id:'juniper_retail'}),{});
  context.selected='deepseek';
  calls.travel[0].callback();
  assert.equal(calls.bubbles.length,0);
  assert.equal(calls.feedback.length,0);
});

await check('replay reset invalidates the previous movement callback',()=>{
  const {context,calls}=ui();
  context.moveAgent('juniper_retail',action(1,'inspect_company_source',{company_id:'juniper_retail'}),{});
  context.clearReplay();
  calls.travel[0].callback();
  assert.equal(calls.bubbles.length,0);
  assert.equal(context.lastAnimatedEventKey,'');
});

await check('same-company new actions get one arrival notice and polling adds none',()=>{
  const {context,calls}=ui();
  const first=action(1,'inspect_company_source',{company_id:'juniper_retail'});
  const second=action(2,'request_company_action',{company_id:'juniper_retail'},'updated');
  context.moveAgent('juniper_retail',first,{});
  calls.travel[0].callback();
  context.moveAgent('juniper_retail',first,{});
  assert.equal(calls.travel.length,1);
  context.moveAgent('juniper_retail',second,{});
  assert.equal(calls.bubbles.length,1,'Second result waits for its own arrival notification');
  calls.travel[1].callback();
  context.moveAgent('juniper_retail',second,{});
  assert.equal(calls.travel.length,2);
  assert.equal(calls.bubbles.length,2);
  assert.equal(calls.feedback.length,2);
});

await check('a subsequent model prompt does not replace the completed movement identity',()=>{
  const {context,calls}=ui();
  const completed=action(5,'request_company_action',{company_id:'juniper_retail'},'updated');
  Object.assign(context,{run:{events:[completed,{type:'model_prompt',decision:6,user_prompt:'Next decision'}]},
    exchange:{prompt:'Next decision',response:'',result:'updated'},obs:{}});
  vm.runInContext(`(()=>{${renderMovement}})()`,context);
  assert.equal(calls.travel[0].zone,'juniper_retail');
  calls.travel[0].callback();
  assert.match(calls.bubbles[0].title,/request company action/);
  vm.runInContext(`(()=>{${renderMovement}})()`,context);
  assert.equal(calls.travel.length,1,'Polling the next prompt must not schedule the completed action again');
});

await check('replay movement uses only the visible event prefix, not the saved final site',()=>{
  const {context,calls}=ui();
  const earlier=action(1,'inspect_company_source',{company_id:'harbor_press'});
  const later=action(2,'inspect_company_source',{company_id:'northstar_projects'});
  Object.assign(context,{run:{events:[earlier,later]},replayIndex:0,
    replayFrames:[earlier,later],replayEvent:earlier,exchange:{},obs:{}});
  vm.runInContext(`(()=>{${renderMovement}})()`,context);
  assert.equal(calls.travel[0].zone,'harbor_press');
  calls.travel[0].callback();
  assert.equal(calls.particles[0].zone,'harbor_press');
});

function deferredLoader({instantArrival=false}={}){
  let resolve;
  const promise=new Promise(done=>{resolve=done;});
  const calls=[];
  const renderer={
    update(...args){calls.push(['update',...args]);},
    updateAgents(...args){calls.push(['agents',...args]);},
    setTarget(zone,arrival){calls.push(['target',zone,arrival]);if(instantArrival)arrival?.();},
    feedback(notice){calls.push(['feedback',notice]);},
    destroy(){calls.push(['destroy']);},
  };
  const window={};
  const context=vm.createContext({window,loadModule(){return promise;},
    document:{createElement(){return {};}}});
  const transformed=loader.replace("import('/assets/arena_scene.js')","loadModule('/assets/arena_scene.js')");
  assert.notEqual(transformed,loader,'Only the asynchronous module boundary is replaced');
  vm.runInContext(transformed,context,{filename:'maintained-arena-loader.js'});
  const view=window.createArena3D({container:{dataset:{},appendChild(){}},canvas:{}});
  return {view,calls,resolve(){resolve({createScene(){calls.push(['create']);return renderer;}});}};
}
const flush=()=>new Promise(resolve=>setImmediate(resolve));

await check('loader flushes the latest agents, observation, target and arrival after import',async()=>{
  const {view,calls,resolve}=deferredLoader();
  let oldArrival=0,newArrival=0;
  view.update({revision:1},{});
  view.update({revision:2},{run:'terra'});
  view.updateAgents([{id:'old'}],'old',false);
  view.updateAgents([{id:'terra'},{id:'deepseek'}],'terra',true);
  view.setTarget('juniper_retail',()=>oldArrival++);
  view.setTarget('meridian_booking',()=>newArrival++);
  assert.equal(calls.length,0);
  resolve();await flush();
  assert.deepEqual(calls.map(call=>call[0]),['create','update','agents','target']);
  assert.deepEqual(plain(calls[1].slice(1)),[{revision:2},{run:'terra'}]);
  assert.deepEqual(plain(calls[2].slice(1)),[[{id:'terra'},{id:'deepseek'}],'terra',true]);
  assert.equal(calls[3][1],'meridian_booking');
  assert.equal(oldArrival,0);assert.equal(newArrival,0);
  calls[3][2]();
  assert.equal(oldArrival,0);assert.equal(newArrival,1);
});

await check('destroying the loader before import suppresses construction and callbacks',async()=>{
  const {view,calls,resolve}=deferredLoader();
  let arrived=0;
  view.setTarget('juniper_retail',()=>arrived++);
  view.updateAgents([{id:'terra'}],'terra',true);
  view.destroy();
  resolve();await flush();
  assert.equal(calls.length,0);
  assert.equal(arrived,0);
});

await check('feedback queued before import keeps the latest notice and flushes it once',async()=>{
  const {view,calls,resolve}=deferredLoader();
  view.feedback({title:'Older notice',text:'old'});
  view.feedback({title:'Latest notice',text:'new'});
  resolve();await flush();
  const notices=calls.filter(call=>call[0]==='feedback');
  assert.equal(notices.length,1);
  assert.deepEqual(plain(notices[0][1]),{title:'Latest notice',text:'new'});
});

await check('an immediate arrival during loader flush displays feedback exactly once',async()=>{
  const {view,calls,resolve}=deferredLoader({instantArrival:true});
  view.setTarget('juniper_retail',()=>view.feedback({title:'Action completed',text:'Simulator: updated'}));
  resolve();await flush();
  assert.equal(calls.filter(call=>call[0]==='feedback').length,1,
    'An arrival callback may publish feedback while the loader is flushing; it must not be replayed twice');
});

console.log(`PASS: ${passed} maintained movement/event/loader checks, with no model or service calls.`);
