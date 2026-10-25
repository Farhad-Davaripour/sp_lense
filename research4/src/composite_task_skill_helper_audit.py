"""Outcome-relevant helper measurements from an already replayed task world."""
from .io_utils import digest
from .composite_generalized_world import expected_analysis
from .composite_generalized_novelty import material_execution_signature


def audit_helpers(world,training_exclusion=None):
    required={}
    for source,target in zip(world.sources,world.targets):
        required.update({(target,key):value for key,value in expected_analysis(world.scene.mechanism,world.engine.records[source]['data']).items()})
    last={}
    if world.accepted:
        for index,entry in enumerate(world.engine.trace):
            if entry['op']!='patch' or entry['result'].get('status')!='recorded':continue
            target=entry['arguments']['identifier'];before=entry['record_before'];after=entry['record_after']
            for (identifier,key),expected in required.items():
                if identifier!=target:continue
                if digest({'present':key in before,'value':before.get(key)})!=digest({'present':key in after,'value':after.get(key)}):
                    last[(identifier,key)]=index
        if any(digest(world.engine.records[identifier].get(key))!=digest(value) for (identifier,key),value in required.items()):
            raise ValueError('Accepted task does not satisfy its independent record predicates')
    relevant=set(last.values());helpers=[]
    for name,program in world.engine.definitions.items():
        signatures=set();fingerprints=set();unrolled=0;calls=[]
        for call in world.engine.executions:
            if call['name']!=name:continue
            start,end=call['trace_start'],call['trace_end'];trace=world.engine.trace[start:end]
            effects=[index-start for index in relevant if start<=index<end]
            executed={**program,'steps':program['steps'][:len(trace)]}
            signature,core=material_execution_signature(executed,trace,effects)
            if signature is not None:
                signatures.add(signature);fingerprints.add(core['fingerprint']);unrolled+=len(core['core_program']['steps'])
            calls.append({'trace_start':start,'trace_end':end,'outcome_relevant_write_steps':effects,
                          'material_signature':signature,'causal_core':core})
        decisions=sum(isinstance(r['call'],dict) and r['call'].get('tool') in ['define','invoke']
          and r['call'].get('arguments',{}).get('name')==name for r in world.operations)
        novel=None if training_exclusion is None else bool(fingerprints) and not bool(fingerprints&set(training_exclusion))
        helpers.append({'name':name,'calls':calls,'distinct_consumed_bindings':len(signatures),
          'outcome_relevant_core_fingerprints':sorted(fingerprints),'pruned_unrolled_decisions':unrolled,
          'actual_definition_and_invocation_decisions':decisions,'net_decisions_saved':unrolled-decisions,
          'training_exposure_excluded':novel,
          'useful_novel_authored_composition_observed':bool(world.accepted and len(signatures)>=2 and unrolled>decisions and novel is True)})
    invalid=sum(isinstance(r['reply'],dict) and r['reply'].get('error')=='invalid_record_action' for r in world.operations)
    return {'helpers':helpers,'task_accepted':world.accepted,'outcome_relevant_helper_measured':True,
      'useful_novel_authored_composition_observed':any(h['useful_novel_authored_composition_observed'] for h in helpers),
      'invalid_calls':invalid,'raw_denied_writes':world.engine.raw_denied_writes,'after_stop_attempts':world.after_stop_attempts,
      'return_value_equality_credited_without_provenance':False,'caller_relevance_flags_used':False,
      'external_enforcement_is_not_learned_compliance':True,
      'scientific_validation_admitted':False,'model_forwards':0,'heldout_content_read':False}
