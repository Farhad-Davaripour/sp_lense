"""Model-free prospective manifest generation; never creates an approved release."""
import importlib.metadata,json,os,subprocess,sys
from pathlib import Path
from support import HERE,ROOT,sha,require,json_bytes
def main():
    require(not any(n.split('.')[0] in ('torch','transformers','transformer_lens','pyarrow','datasets','safetensors') for n in sys.modules),'PREPARE_MODEL_FREE')
    pins={}
    declarations=[('f31d0e4f8fd5136eb38df01201f7032c9b82dc61','diagnostics/fresh_confirmation_production_v2/'+n+'.py') for n in ('entry','owned_production')]
    declarations += [('638ef2a8f1eac0679c92d07a62cc8ae9bfac41b1','diagnostics/fresh_confirmation_workflow_v1/schedule.py')]
    declarations += [('23ce46f1e5177791751b08bd15de07f9d836ecf8','diagnostics/fresh_confirmation_failure_delta_v1/supervisor.py')]
    declarations += [('1d4cc39eba5fb7f987649a05f71247061f005d02','diagnostics/semantic_editor_f03_v2_first_C_v2/'+n+'.py') for n in ('native','owned')]
    declarations += [('1d4cc39eba5fb7f987649a05f71247061f005d02','diagnostics/windows_worker_identity_probe_v1/probe.py')]
    for commit,path in declarations:
        raw=subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',commit+':'+path])
        require(raw==(ROOT/path).read_bytes(),'COMMITTED_SOURCE_BYTES')
        pins[commit+':'+path]={'bytes':len(raw),'sha256':sha(raw)}
    (HERE/'REUSED_SOURCES.json').write_bytes(json_bytes(pins))
    identity={}
    for key,value in (('launch',sys.executable),('base',sys._base_executable)):
        image=Path(value).absolute();identity[key+'_image']=str(image);identity[key+'_sha256']=sha(image.read_bytes())
    (HERE/'OWNED_IDENTITY.json').write_bytes(json_bytes(identity))
    external=[]
    def pin(path):
        path=Path(path);raw=path.read_bytes();external.append({'path':str(path.absolute()),'bytes':len(raw),'sha256':sha(raw)})
    for _,path in declarations:pin(ROOT/path)
    for name in ('editor.py','word_scoring.py','word_reference.py','gate_reference.py','real_attempt/fitted_parameters.json'):
        pin(ROOT/'diagnostics/semantic_editor_f03_v2_first_C_v2'/name)
    pin(ROOT/'src/sp_lense/conditional_gate_models.py')
    versions={}
    sources={'torch':('torch/__init__.py','torch/nn/modules/module.py','torch/autograd/__init__.py',
            'torch/nn/modules/linear.py','torch/nn/modules/sparse.py','torch/nn/modules/normalization.py',
            'torch/nn/modules/activation.py','torch/nn/modules/conv.py','torch/nn/functional.py'),
        'transformers':('transformers/modeling_utils.py','transformers/cache_utils.py','transformers/masking_utils.py',
            'transformers/modeling_layers.py','transformers/modeling_rope_utils.py','transformers/activations.py',
            'transformers/integrations/hub_kernels.py','transformers/integrations/accelerate.py','transformers/utils/output_capturing.py',
            'transformers/models/qwen3_5/configuration_qwen3_5.py','transformers/models/qwen3_5/modeling_qwen3_5.py')}
    for package,names in sources.items():
        distribution=importlib.metadata.distribution(package);versions[package]=distribution.version
        for name in names:pin(distribution.locate_file(name))
    require(versions=={'torch':'2.13.0+cpu','transformers':'5.15.1'},'EXACT_RUNTIME_VERSIONS')
    names=[p.name for p in HERE.glob('*.py')]+['inputs.json','CHECKPOINT.json','REUSED_SOURCES.json','OWNED_IDENTITY.json','PROTOCOL.md','INPUT_PROVENANCE.json']
    lock={'schema':'native_baseline_source_candidate.v1','root_reviewed':False,'real_authorized':False,
        'source_sha256':{name:sha((HERE/name).read_bytes()) for name in sorted(names)},'external_sources':external,'versions':versions}
    (HERE/'SOURCE_FREEZE.json').write_bytes(json_bytes(lock))
    from authority import LIMITS,ATTEMPT,output
    draft={'schema':'native_baseline_release.v1','approved':False,'attempt':ATTEMPT,'limits':LIMITS,'output':str(output()),
        'source_freeze_sha256':sha((HERE/'SOURCE_FREEZE.json').read_bytes()),'inputs_sha256':sha((HERE/'inputs.json').read_bytes()),
        'checkpoint_lock_sha256':sha((HERE/'CHECKPOINT.json').read_bytes()),'owned_identity_sha256':sha((HERE/'OWNED_IDENTITY.json').read_bytes()),
        'trace_sources':{name:digest for name,digest in lock['source_sha256'].items() if name.endswith('.py')}}
    (HERE/'RELEASE_DRAFT.json').write_bytes(json_bytes(draft))
    reports=[]
    for name in ('TEST_WORKFLOW_RESULT.json',):
        raw=(HERE/name).read_bytes();receipt=json.loads(raw);require(receipt['status']=='PASS','RELEVANT_TEST_REPORT_PASS')
        reports.append({'path':name,'sha256':sha(raw),'case_count':len(receipt['cases']),'status':'PASS'})
    (HERE/'TEST_RESULTS.json').write_bytes(json_bytes({'status':'PASS','reports':reports,
        'total_case_groups':sum(v['case_count'] for v in reports),'source_candidate_sha256':draft['source_freeze_sha256'],
        'real_authorized':False,'actual_qwen_loads':0,'actual_qwen_forwards':0,'actual_qwen_derivatives':0,
        'scope':'SYNTHETIC_FINAL_BINDING_ONLY','new_production_owned_launch_observed':False}))
    print(json.dumps({'status':'SYNTHETIC_BINDING_CANDIDATE_ONLY','real_authorized':False,'source_freeze_sha256':draft['source_freeze_sha256'],
        'source_files':len(names),'external_sources':len(external),'providers_imported':False,'checkpoint_tensor_access':False},sort_keys=True))
if __name__=='__main__':main()

