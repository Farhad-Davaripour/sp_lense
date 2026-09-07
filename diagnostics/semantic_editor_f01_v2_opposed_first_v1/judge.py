"""Independent exact-cohort saved-data judge behind the owned audit bootstrap."""
import json,time
from pathlib import Path
from core import HERE,Budget,read,require,sha
from admission import exact_cohort,environment
import saved_judge

def judge_production(output):
    output=Path(output);plan=read(output/'plan.json');exact_cohort(plan);environment(plan)
    runtime=read(output/'runtime.json');contract=plan['gate']['runtime_compatibility']
    require(runtime['execution_mode']=='PRODUCTION_F01_V2_OPPOSED_FIRST','synthetic runtime rejected')
    for key in ('model_id','model_revision','device','dtype','d_model','model_layers','packages','lens'):
        require(runtime[key]==contract[key],'independent runtime identity '+key)
    require(runtime['locked_inputs_sha256']==plan['input_binding']['input_sha256'] and runtime['tokenizer_calls_after_load']==0,'locked input encoding')
    require(runtime['boundaries']==[{'prompt_id':p['prompt_id'],**plan['alignment'][p['prompt_id']]} for p in plan['prompts']],'complete encoded input proofs')
    result=saved_judge.judge(output)
    receipt=read(output/'execution_receipt.json');require(receipt['forward_attempts']<=22 and receipt['derivatives_attempted']<=8,'new absolute ceilings')
    if (output/'integration_cleanup.json').exists():
        cleanup=read(output/'integration_cleanup.json');require(cleanup['initial_weight_sha256']==cleanup['final_weight_sha256']==contract['weight_sha256'],'whole frozen weights')
    result['scope']='exact f01/v2 discovery self pair, two fixed first-target requests; skips are not retentions; no ordinary preservation'
    return result

def finish_judge(output,synthetic=False):
    start=time.monotonic();output=Path(output);budget=Budget(output);receipt={'status':'INCONCLUSIVE'};exit_code=0;result={}
    try:
        if synthetic:
            require(read(output/'plan.json')['execution_mode']=='SYNTHETIC_ONLY','explicit fake audit')
            result=saved_judge.judge(output);require(result['synthetic_only'],'never production evidence')
        else:result=judge_production(output)
        receipt['status']='complete'
    except BaseException as error:
        exit_code=1;receipt['error']=type(error).__name__+': '+str(error)
        def facts(name):return [json.loads(x) for x in (output/name).read_text().splitlines()] if (output/name).exists() else []
        result={'classification':'INCONCLUSIVE','admission_or_audit_error':receipt['error'],
            'durable_scientific_failures_not_discarded':facts('scientific_failures.jsonl'),'primary_technical_faults':facts('technical_faults.jsonl'),
            'secondary_cleanup_faults':facts('cleanup_errors.jsonl'),'unrun':read(output/'unrun.json') if (output/'unrun.json').exists() else None}
    finally:
        budget.write('judge_results.json',result)
        budget.write_bytes('judge_report.md',('Intermediate f01_v2 saved-data judge: '+result['classification']+'\n'+json.dumps(result.get('counts',{}))+'\nNot authoritative before both external captures and final closeout. Synthetic results are not model evidence.\n').encode())
        receipt['elapsed_seconds']=time.monotonic()-start;budget.write('judge_receipt.json',receipt)
    return exit_code
