"""Fixed archived positive-control source extraction; no search or model work."""
import copy,json,subprocess,sys
from support import HERE,ROOT,require,sha,json_bytes
DECLARED=(('stop_then_keep','8a35b9c74ce8523d5458349aab55a8d17372cde2','development/native_baseline_v1',
    'native_development_baseline_attempt_001',1,'f03_v2_STOP_then_KEEP'),
    ('keep_then_stop','0ac3ee67766760a789e3b6a16916205efc13fb5b','development/native_opposite_order_v1',
    'native_opposite_order_attempt_001',2,'f03_v2_KEEP_then_STOP'))

def main():
    require(not any(n.split('.')[0] in ('torch','transformers','transformer_lens','pyarrow','datasets','tokenizers','safetensors') for n in sys.modules),'MODEL_FREE_SELECTION')
    cases=[];sources=[]
    for key,commit,namespace,attempt,updates,pid in DECLARED:
        def saved(path):
            raw=subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',commit+':'+path])
            require(raw==(ROOT/path).read_bytes(),'CURRENT_EQUALS_EXACT_COMMITTED_SOURCE')
            return raw,{'path':path,'bytes':len(raw),'sha256':sha(raw),'commit':commit}
        raw,input_pin=saved(namespace+'/inputs.json');original=json.loads(raw)
        case=copy.deepcopy(original['cases'][0]);require(case['prompt_id']==pid,'FIXED_SELF_CASE');case['case_key']=key
        base=namespace+'/real_evidence/'+attempt+'/'
        names={'baseline_row':'rows/self__baseline.json','baseline_logits':'logits/self__baseline.f32',
            'endpoint_row':'rows/self__C__endpoint.json','endpoint_logits':'logits/self__C__endpoint.f32'}
        names.update({f'step_{k}':f'steps/self__C__step_{k}.json' for k in range(1,updates+1)})
        pins={name:saved(base+path)[1] for name,path in names.items()};case['seed_artifacts']=pins
        case['source_positive_control']={'commit':commit,'namespace':namespace,'attempt':attempt,'input_pin':input_pin}
        cases.append(case);sources.append(case['source_positive_control'])
    data={'schema_version':'native_controlled_p_inputs_v1','cases':cases,'scope':'CONTROLLED_P_REVERSAL_POSITIVE_CONTROL_ONLY',
        'selection':'two preassigned archived C endpoints; fixed chronological display order; no outcome search',
        'gate_features':'fresh live unedited h0 only','ordinary_or_OFF_reruns':False}
    provenance={'fixed_sources':sources,'seed_artifacts':[p['seed_artifacts'] for p in cases],
        'historical_C_path_separate_from_single_injection_seed_cost':True,'natural_STOP_baseline_claimed':False,
        'sealed_data_access':False,'provider_imports':0,'new_encoding_calls':0,'model_work':0}
    (HERE/'inputs.json').write_bytes(json_bytes(data));(HERE/'INPUT_PROVENANCE.json').write_bytes(json_bytes(provenance))
    from input_reader import read
    require(read()==data,'EXACT_INPUT_ADMISSION')
    print(json.dumps({'status':'FIXED_TWO_ARCHIVED_SEEDS','inputs_sha256':sha(json_bytes(data)),'model_work':0},sort_keys=True))

if __name__=='__main__':main()
