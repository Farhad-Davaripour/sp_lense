"""Conservative causal-core structure; no novelty from dead steps/parameters."""
from copy import deepcopy
from .composite_design_v2 import validate_program,structural_fingerprint


def _refs(value):
    if isinstance(value,dict):
        if set(value)=={'ref'}:return {value['ref'].split('.')[0]}
        return set().union(*(_refs(item) for item in value.values())) if value else set()
    if isinstance(value,list):return set().union(*(_refs(item) for item in value)) if value else set()
    return set()


def consumed_step(step):
    """Canonicalize only statically proven ignored arguments, before dataflow.

    A dynamic aggregate mode remains conservative: field may matter for sum/max.
    For literal count, field is never consumed, regardless of its JSON form.
    """
    value=deepcopy(step)
    if value['op']=='aggregate' and value['arguments'].get('mode')=='count':
        value['arguments']['field']='_count_ignores_field'
        if isinstance(value['arguments'].get('rows'),list):
            # A literal list has a statically fixed length; count never consumes
            # its element values, including any otherwise unrelated references.
            value['arguments']['rows']=[{} for _ in value['arguments']['rows']]
    return value


def causal_core(program,changing_write_steps,verified_return_used=False):
    """The caller must verify write changes and outcome/return use from raw replay.

    A digest alone cannot establish that this helper contributed to an outcome.
    No literal/unused code is allowed to manufacture a new program fingerprint.
    """
    validate_program(program);steps=[consumed_step(step) for step in program['steps']];roots=set(changing_write_steps)
    if any(type(index) is not int or not 0<=index<len(steps) or steps[index]['op']!='patch' for index in roots):
        raise ValueError('Only verified record-changing writes may be effect roots')
    if verified_return_used:roots.add(len(steps)-1)
    if not roots:return {'parameterized_core':False,'core_program':None,'fingerprint':None,'reason':'no_verified_effect_or_used_return'}
    needed={steps[i]['bind'] for i in roots};kept=[]
    for index in range(len(steps)-1,-1,-1):
        step=steps[index]
        if step['bind'] in needed:
            kept.append(deepcopy(step));needed.update(_refs(step['arguments']))
    kept.reverse();used_parameters=[name for name in program['parameters'] if name in needed]
    if not used_parameters:
        return {'parameterized_core':False,'core_program':None,'fingerprint':None,'reason':'case_specific_constant_macro'}
    result={'parameters':used_parameters,'steps':kept};validate_program(result)
    producers={step['bind']:step for step in kept}
    def from_read(value,visited=None):
        visited=set() if visited is None else set(visited)
        for name in _refs(value):
            if name in visited or name not in producers:continue
            step=producers[name]
            if step['op']=='read':return True
            if step['op']=='patch':continue  # A write receipt does not carry its written values.
            if from_read(step['arguments'],visited|{name}):return True
        return False
    data_dependent=any(from_read(steps[index]['arguments']) for index in changing_write_steps)
    if verified_return_used:data_dependent=data_dependent or from_read(steps[-1]['arguments'])
    nontrivial=len(kept)>=2 and len({step['op'] for step in kept})>=2
    return {'parameterized_core':True,'core_program':result,'fingerprint':structural_fingerprint(result),
            'removed_steps':len(steps)-len(kept),'removed_parameters':len(program['parameters'])-len(used_parameters),
            'source_data_dependent':data_dependent,'nontrivial_composition':nontrivial,
            'eligible_for_compositional_transfer_claim':data_dependent and nontrivial,
            'causal_effect_verified_by_this_function':False,'semantic_equivalence_proved_by_hash':False}


def verified_changing_steps(program,primitive_trace):
    validate_program(program)
    if len(primitive_trace)!=len(program['steps']):raise ValueError('Complete exact helper primitive trace required')
    changing=[]
    for index,(step,record) in enumerate(zip(program['steps'],primitive_trace)):
        if step['op']!=record['op']:raise ValueError('Program/trace operation alignment differs')
        if record['op']=='patch':
            if not {'record_before','record_after'}<=set(record):raise ValueError('Raw write before/after records required')
            if record['result'].get('status')=='recorded' and record['record_before']!=record['record_after']:changing.append(index)
    return changing


def material_execution_signature(program,primitive_trace):
    """Hash consumed live inputs/addresses, not the whole argument dictionary."""
    from .composite_design_v2 import _field
    from .io_utils import digest
    changing=verified_changing_steps(program,primitive_trace)
    assessed=causal_core(program,changing)
    if not assessed.get('eligible_for_compositional_transfer_claim'):return None,assessed
    kept={step['bind'] for step in assessed['core_program']['steps']};consumed=[]
    for step,record in zip(program['steps'],primitive_trace):
        if step['bind'] not in kept:continue
        op=step['op'];args=record['arguments']
        if op=='read':value={'identifier':args['identifier']}
        elif op=='select':value={'tested_values':[_field(row,args['field']) for row in args['rows']],'equals':args['equals']}
        elif op=='join':value={'left_keys':[_field(row,args['left_key']) for row in args['left']],
                              'right_keys':[_field(row,args['right_key']) for row in args['right']]}
        elif op=='aggregate':
            value={'mode':args['mode'],'count':len(args['rows'])} if args['mode']=='count' else {
                'mode':args['mode'],'values':[_field(row,args['field']) for row in args['rows']]}
        elif op=='patch':value={'identifier':args['identifier'],'fields':args['fields']}
        else:value={'verified_return_projection_required':True}
        consumed.append({'op':op,'consumed':value})
    return digest(consumed),assessed


def material_reuse(program,executions):
    signatures=[];core_sizes=[];verified_outcomes=[]
    for execution in executions:
        if execution.get('successful') is not True:continue
        signature,core=material_execution_signature(program,execution['primitive_trace'])
        if signature is None:continue
        signatures.append(signature);core_sizes.append(len(core['core_program']['steps']))
        verified_outcomes.append(execution.get('outcome_relevant') is True)
    count=len(set(signatures));decisions=1+len(executions);unrolled=sum(core_sizes)
    return {'distinct_material_bindings':count,'successful_effectful_executions':len(signatures),
        'pruned_unrolled_primitive_decisions':unrolled,'model_decisions_with_definition':decisions,
        'net_model_decisions_saved':unrolled-decisions,
        'outcome_relevance_independently_verified':bool(verified_outcomes) and all(verified_outcomes),
        'reuse_criterion_met':count>=2 and unrolled>decisions and bool(verified_outcomes) and all(verified_outcomes),
        'unused_argument_changes_counted':False,'semantic_correctness_on_arbitrary_inputs_proved':False}
