"""Replay and continue the censored split_w2_n1 Terra case under 262K context."""
from __future__ import annotations
import argparse, copy, hashlib, json, os, secrets, time, urllib.request, uuid
from datetime import datetime, timezone
from pathlib import Path
import sys, yaml

ROOT=Path(__file__).resolve().parents[1]
if __package__ in (None,''):sys.path.insert(0,str(ROOT))
import run_rank16_infrastructure_game as common
from src.infrastructure_preservation import InfrastructurePreservationEpisode
from src.infrastructure_split_gpu_extension import apply_split_gpu_condition

PARENT=ROOT/'.local/2026-11-23_10--split-gpu-layout-v3-run/raw/terra/split_w2_n1--attempt-01'
SCENARIO=ROOT/'scenarios/infrastructure_split_gpu_layout_v3/terra_split_w2_n1_context_amendment.yaml'
TOKEN_ENV='R4_SPLIT3_TERRA_CONT_TOKEN'; CONTEXT=262144
FILES={'MANIFEST.json':'1e02879e75c6c7e2fb65c886182ce909e379cd3453e04a0ce4d9cc0e3642427d','EVENTS.jsonl':'ce151bae2774b284ade90c7743abf337c1c49b4f72b897efead5e9c66c3f4b3a','REQUESTS.jsonl':'343d9c8e837631ba068ffd9f9a408d2fc896f2917d83797435689f219fdab079','RESPONSES.jsonl':'9dd58426faf715a964383aadd83ebb67911e403ffd05976475004656f06b88b2','LIVE_STATE.json':'c452b0b301b3df5e879c8bf6af31094f94a4f6fd1e7403a05beb699e6a6af785','RESULT.json':'c0f4a499d766d40b209c125013f421a6244d06dd6f0b29f7fd02be8d14ffed4c'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines()]
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8');os.replace(tmp,path)
def append(path,value):
 with path.open('a',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())
def prepare():
 scenario=yaml.safe_load(SCENARIO.read_text(encoding='utf-8'))
 if scenario.get('schema')!='research4.split-gpu-terra-linked-context-continuation.v1':raise ValueError('Scenario differs')
 for name,digest in FILES.items():
  if sha(PARENT/name)!=digest:raise ValueError('Parent changed: '+name)
 manifest=json.loads((PARENT/'MANIFEST.json').read_text());result=json.loads((PARENT/'RESULT.json').read_text())
 if result['status']!='incomplete' or result['failure']!='worker_deadline_or_context_exhausted':raise ValueError('Parent censoring differs')
 condition=manifest['cohort_extension']['initial_condition'];episode=InfrastructurePreservationEpisode(seed=manifest['simulator']['seed'],horizon=48,max_decisions=100000,background_compute=True);apply_split_gpu_condition(episode,condition)
 actions=[x for x in rows(PARENT/'EVENTS.jsonl') if x.get('type')=='simulator_action']
 for index,row in enumerate(actions,1):
  t=row['transition'];action={'action':t['action_id'],**t['arguments']} if t.get('action_id') else {};actual=episode.take(action)
  if actual!=t or row['observation']!=actual['after'] or row['decision_index']!=index:raise ValueError('Replay differs at '+str(index))
 if episode.score()!=json.loads((PARENT/'LIVE_STATE.json').read_text())['score'] or episode.termination_reason is not None:raise ValueError('Replayed state differs')
 req=rows(PARENT/'REQUESTS.jsonl')[-1];resp=rows(PARENT/'RESPONSES.jsonl')[-1];messages=[*req['body']['messages'],{'role':'assistant','content':resp['assistant_content']}]
 user=manifest['simulator']['user_task']+'\n\n'+episode.prompt()+'\n\n'+common.RESPONSE_INSTRUCTION;candidate=[*messages,{'role':'user','content':user}];tokens=common.count_tokens(candidate,manifest['generation']['chat_template_kwargs'])
 if tokens>=manifest['history']['compact_at_prompt_tokens']:
  messages=common.compact_history(messages,episode,manifest['history']['retain_recent_exchanges']);candidate=[*messages,{'role':'user','content':user}];tokens=common.count_tokens(candidate,manifest['generation']['chat_template_kwargs'])
 digest=hashlib.sha256(common.canonical_json(candidate).encode()).hexdigest()
 if tokens<=131072 or tokens+8192>CONTEXT:raise ValueError('Amended context does not isolate the saved stop')
 return scenario,manifest,episode,messages,tokens,digest
def freeze(output):
 scenario,manifest,episode,messages,tokens,digest=prepare();output=output.resolve();output.mkdir(parents=True,exist_ok=False);(output/'raw').mkdir();(output/'conversation').mkdir();token=secrets.token_urlsafe(48);(output/'raw/TOKEN.secret').write_text(token)
 plan=copy.deepcopy({k:manifest[k] for k in ('schema','role','arm_id','label','model_id','adapter_sha256','adapter_config_sha256','base_manifest_sha256','simulator','endpoint','generation','history','operations','scope','cohort_extension')});plan['endpoint']['context_tokens']=CONTEXT;plan['endpoint']['base_url']='http://127.0.0.1:8879';plan['endpoint']['token_env']=TOKEN_ENV;plan['endpoint']['expected_health']={};write(output/'raw/PLAN.json',plan)
 receipt={'schema':'research4.split-gpu-terra-continuation-freeze.v1','status':'FROZEN_BEFORE_MODEL_CALLS','scenario_sha256':sha(SCENARIO),'parent_hashes':FILES,'parent_decisions':episode.decisions,'parent_hours':episode.elapsed_hours,'next_prompt_tokens':tokens,'next_messages_sha256':digest,'plan_sha256':sha(output/'raw/PLAN.json'),'token_sha256':hashlib.sha256(token.encode()).hexdigest(),'runner_sha256':sha(Path(__file__)),'model_calls':0};write(output/'raw/FREEZE_RECEIPT.json',receipt);print(json.dumps(receipt,indent=2))
def execute(output):
 scenario,manifest,episode,messages,tokens,digest=prepare();plan=json.loads((output/'raw/PLAN.json').read_text());receipt=json.loads((output/'raw/FREEZE_RECEIPT.json').read_text());token=(output/'raw/TOKEN.secret').read_text();plan_sha=sha(output/'raw/PLAN.json')
 if receipt['plan_sha256']!=plan_sha or receipt['next_prompt_tokens']!=tokens or receipt['next_messages_sha256']!=digest:raise ValueError('Freeze differs')
 run=output/'raw/split_w2_n1--linked-attempt-01';run.mkdir();conversation=[];failure=None
 deadline=time.monotonic()+plan['operations']['worker_deadline_seconds']
 while episode.termination_reason is None and episode.elapsed_hours<48:
  current=plan['simulator']['user_task']+'\n\n'+episode.prompt()+'\n\n'+common.RESPONSE_INSTRUCTION;candidate=[*messages,{'role':'user','content':current}];pt=common.count_tokens(candidate,plan['generation']['chat_template_kwargs'])
  if pt>=plan['history']['compact_at_prompt_tokens']:
   messages=common.compact_history(messages,episode,plan['history']['retain_recent_exchanges']);candidate=[*messages,{'role':'user','content':current}];pt=common.count_tokens(candidate,plan['generation']['chat_template_kwargs'])
  maximum=min(CONTEXT-pt,8192)
  if maximum<=0 or time.monotonic()>=deadline:failure='amended_context_or_deadline';break
  decision=episode.decisions+1;did=hashlib.sha256(f'{plan_sha}:{decision}:{uuid.uuid4().hex}'.encode()).hexdigest()[:32];body={'model':plan['model_id'],'messages':candidate,'stream':False,'max_tokens':maximum,**{k:plan['generation'][k] for k in ('seed','temperature','top_p','top_k','min_p','repetition_penalty')},'chat_template_kwargs':plan['generation']['chat_template_kwargs']};append(run/'REQUESTS.jsonl',{'decision_index':decision,'decision_id':did,'prompt_tokens':pt,'body':body})
  try:reply=common.request(plan['endpoint']['base_url']+plan['endpoint']['chat_path'],token,plan_sha,did,body,plan['endpoint']['request_timeout_seconds'])
  except Exception as e:failure='endpoint_request_error';break
  exchange={'decision_index':decision,'decision_id':did,'prompt_tokens':pt,'assistant_content':reply['content'],'assistant_reasoning':reply['reasoning'],'usage':reply['usage'],'finish_reason':reply['finish_reason'],'parsed_action':None,'simulator_transition':None,'status':'response_received'};conversation.append(exchange);append(run/'RESPONSES.jsonl',exchange)
  if reply['finish_reason']!='stop':failure='model_response_incomplete';break
  try:action=common.extract_action(reply['content'])
  except ValueError:failure='malformed_model_action';break
  transition=episode.take(action);exchange.update(parsed_action=action,simulator_transition=transition,status='dispatched');append(run/'EVENTS.jsonl',{'type':'simulator_action','decision_index':decision,'transition':transition,'result':transition['result'],'observation':transition['after']});messages=[*candidate,{'role':'assistant','content':reply['content']}];common.write_yaml(run/'CONVERSATION.yaml',conversation);common.write_json(run/'LIVE_STATE.json',{'status':'running','score':episode.score()})
 if failure and episode.termination_reason is None:episode.finish('technical_error')
 result={'schema':'research4.split-gpu-terra-continuation-result.v1','status':'complete' if failure is None else 'incomplete','failure':failure,'parent_status':'right_censored','prefix_decisions':36,'suffix_decisions':episode.decisions-36,'score':episode.score(),'finished_utc':datetime.now(timezone.utc).isoformat()};write(run/'RESULT.json',result);common.write_yaml(output/'conversation/split_w2_n1--linked-attempt-01.yaml',conversation);print(json.dumps(result,indent=2))
def main():
 p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True);m.add_argument('--preflight',action='store_true');m.add_argument('--freeze',action='store_true');m.add_argument('--execute',action='store_true');p.add_argument('--output',type=Path);a=p.parse_args()
 if a.preflight:
  s,m,e,msg,t,d=prepare();print(json.dumps({'status':'pass','model_calls':0,'parent_decisions':e.decisions,'parent_hours':e.elapsed_hours,'next_prompt_tokens':t,'next_messages_sha256':d},indent=2))
 elif a.freeze:
  if a.output is None:
   raise ValueError('--output required')
  freeze(a.output)
 else:
  if a.output is None:
   raise ValueError('--output required')
  execute(a.output)
if __name__=='__main__':main()
