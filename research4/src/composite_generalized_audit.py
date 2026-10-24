"""Model-free V2 cohort/target/transfer audits; never opens sealed instances."""
from collections import Counter
import re
from .io_utils import digest
from .composite_generalized_novelty import causal_core
from .composite_generalized_interface import input_ids,action_text,parse
from .composite_generalized_data import cohort_contract


def replay_reference_episode(episode):
    """Exact exported-action replay in the admitted data runtime, not an expert rerun."""
    from .composite_generalized_world import World,GeneralizedScene
    if episode['scene']['role'] not in ['training','development','public_design_fixture']:raise PermissionError('No sealed reference replay here')
    world=World(GeneralizedScene(**episode['scene']))
    for index,state in enumerate(episode['states']):
        text,evidence=world.visible()
        if (state['text']!=text or state['labels']!=world.truth() or state['evidence_chars']!=list(evidence)
            or state['sequence']!=world.sequence or state['terminal']!=(not world.running)):
            raise ValueError('Reference state/label/evidence/terminal replay differs')
        if not world.running:
            if index!=len(episode['states'])-1 or state['teacher_action'] is not None:raise ValueError('Post-terminal reference action')
            break
        if index>=len(episode['actions']):raise ValueError('Reference action missing')
        recorded=episode['actions'][index]
        if state['teacher_action']!=recorded['action']:raise ValueError('Target/action export differs')
        reply=world.step(recorded['action'])
        if reply!=recorded['reply'] or world.outcome()!=recorded['outcome']:raise ValueError('Reference primitive execution differs')
    if (world.running or len(episode['actions'])!=len(episode['states'])-1 or world.outcome()!=episode['outcome']
        or world.engine.trace!=episode['primitive_trace'] or world.engine.definitions!=episode['helper_definitions']
        or world.helper_calls!=episode['helper_calls']):raise ValueError('Incomplete exact reference/helper replay')
    return {'status':'exact_reference_actions_replayed','model_forwards':0,'teacher_compiler_called':False}


def audit_cohort(episodes,rows,manifest,cfg,reference_replay=False):
    role=manifest.get('role')
    if role not in ['training','development']:raise PermissionError('Only training/development audit inputs are admitted here')
    contract=cohort_contract(cfg,role)
    if len(episodes)!=contract['cases'] or len(rows)!=manifest['states'] or manifest['episodes']!=contract['cases']:
        raise ValueError('Full new cohort counts differ')
    ids=[episode['scene']['case_id'] for episode in episodes]
    if len(set(ids))!=len(ids) or ids!=manifest['case_ids']:raise ValueError('Complete case identities differ')
    groups={episode['scene']['group_id'] for episode in episodes}
    if len(groups)!=contract['groups'] or groups!=set(manifest['group_ids']):raise ValueError('Complete group identities differ')
    for group in groups:
        conditions=[e['scene']['condition'] for e in episodes if e['scene']['group_id']==group]
        if Counter(conditions)!=Counter(contract['conditions']):raise ValueError('Every group must retain all9conditions')
    flattened=[];signatures=set();coverage={f:Counter() for f in cfg['cohorts'][role+'_mechanisms']}
    for episode in episodes:
        scene=episode['scene'];outcome=episode['outcome']
        if (scene['role']!=role or scene['mechanism'] not in coverage or scene['task_structure'] not in cfg['cohorts'][role+'_task_structures']
            or scene['soft_stop']!=20 or scene['hard_stop']!=24 or outcome['running'] or outcome['clock']>24):
            raise ValueError('Reference scene role/mechanism/task or complete terminal boundary differs')
        if not episode['states'] or episode['states'][-1]['terminal'] is not True:
            raise ValueError('Reference terminal observation is missing')
        sequences=[state['sequence'] for state in episode['states']]
        if len(set(sequences))!=len(sequences) or sequences!=sorted(sequences):raise ValueError('Observation sequence differs')
        for state in episode['states']:
            if state['case_id']!=scene['case_id'] or state['group_id']!=scene['group_id']:
                raise ValueError('A state crossed its episode')
            if not state['terminal'] and state['labels']['gate'] and state['teacher_action'] is not None:
                coverage[scene['mechanism']][state['teacher_action']['tool']]+=1
            flattened.append({**state,'split':role})
        for program in episode.get('helper_definitions',{}).values():
            writes=[i for i,step in enumerate(program['steps']) if step['op']=='patch']
            # Exclude every displayed training solution conservatively, even
            # if a particular reference invocation ended before its write.
            # This is an exposure exclusion set, not a positive causal claim.
            for use_return in [False,True]:
                core=causal_core(program,writes,verified_return_used=use_return)
                if core.get('fingerprint') is not None:signatures.add(core['fingerprint'])
        if reference_replay:replay_reference_episode(episode)
    if flattened!=rows:raise ValueError('Exported rows must contain every complete reference state unchanged')
    if any(coverage[f]['define']==0 or coverage[f]['invoke']==0 for f in coverage):
        raise ValueError('Helper creation/invocation targets are missing from a mechanism family')
    return {'role':role,'complete_cases':len(episodes),'complete_states':len(rows),
        'case_ids':ids,'group_ids':sorted(groups),'helper_structural_fingerprints':sorted(signatures),
        'helper_fingerprint_scope':'conservative_possible_causal_cores_of_displayed_reference_programs; exposure_exclusion_not_success',
        'helper_target_coverage':{f:dict(v) for f,v in coverage.items()},'rows_canonical_sha256':digest(rows),
        'reference_world_replayed':reference_replay,
        'model_forwards':0,'scientific_dataset_admitted':False,'heldout_content_read':False}


def native_budget_audit(rows,qwen_tokenizer,detector_tokenizer):
    if not rows or any(row.get('split') not in ['training','development'] for row in rows):
        raise PermissionError('Only complete nonsealed development/training rows may be token-audited')
    maxima={'qwen_context':0,'detector_context':0,'native_target_with_EOS':None,'supervised_combined_context':None};issues=[]
    for row in rows:
        counts={'qwen_context':len(input_ids(qwen_tokenizer,row['text'])),
                'detector_context':len(detector_tokenizer.encode(row['text'],add_special_tokens=True))}
        action=row.get('teacher_action')
        if action is not None:
            text=action_text(action);ids=qwen_tokenizer.encode(text,add_special_tokens=False)+[qwen_tokenizer.eos_token_id]
            decoded=qwen_tokenizer.decode(ids[:-1],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            if decoded!=text or parse(decoded)!=[action]:raise ValueError('Exact native target roundtrip differs; no decoder repair')
            counts['native_target_with_EOS']=len(ids)
            if row['split']=='training':counts['supervised_combined_context']=counts['qwen_context']+len(ids)-1
        for key,value in counts.items():
            maxima[key]=max(maxima[key],value) if maxima[key] is not None else value
            limit={'qwen_context':6144,'detector_context':2048,'native_target_with_EOS':160,'supervised_combined_context':6144}[key]
            if value>limit:issues.append({'state_id':row['state_id'],'kind':key,'tokens':value,'limit':limit})
    return {'status':'complete_native_budget_audit_pass' if not issues else 'failed_native_budget_admission',
            'states':len(rows),'maxima':maxima,'overflow':issues,'truncated_or_repaired':False,
            'model_forwards':0,'heldout_content_read':False}


def _shingles(text):
    text=re.sub(r'\b(?:unit|d)-[0-9a-f]+\b','ID',text)
    tokens=re.findall(r'[a-z_]+|\d+|[^\w\s]',text.lower())
    return {tuple(tokens[i:i+5]) for i in range(max(0,len(tokens)-4))}


def leakage_audit(training,development,training_rows,development_rows):
    if training.get('role')!='training' or development.get('role')!='development':
        raise PermissionError('This leakage audit never accepts sealed data')
    id_overlap=set(training['case_ids'])&set(development['case_ids'])
    group_overlap=set(training['group_ids'])&set(development['group_ids'])
    program_overlap=set(training['helper_structural_fingerprints'])&set(development['helper_structural_fingerprints'])
    training_text={row['text'] for row in training_rows};development_text={row['text'] for row in development_rows}
    exact_overlap=training_text&development_text
    # Common coordinator/permission wrappers are intentional. High textual
    # similarity is a review flag, not automatic evidence of semantic leakage.
    left=[(row['state_id'],_shingles(row['text'])) for row in training_rows if row['sequence']==0]
    right=[(row['state_id'],_shingles(row['text'])) for row in development_rows if row['sequence']==0]
    near=[]
    for a,first in left:
        for b,second in right:
            similarity=len(first&second)/max(1,len(first|second))
            if similarity>=0.9:near.append({'training_state_id':a,'development_state_id':b,'five_token_shingle_Jaccard':similarity})
    failed=bool(id_overlap or group_overlap or program_overlap or exact_overlap)
    status='failed_leakage_admission' if failed else 'requires_semantic_review_of_near_duplicates' if near else 'structural_and_exact_leakage_checks_passed'
    return {'status':status,'case_identity_overlap':sorted(id_overlap),'group_identity_overlap':sorted(group_overlap),
        'reference_helper_graph_overlap':sorted(program_overlap),'exact_visible_text_overlap_count':len(exact_overlap),
        'initial_observation_near_duplicate_flags':near,'near_duplicate_threshold_predeclared':0.9,
        'semantic_equivalence_proved_by_hashes':False,'scientific_dataset_admitted':False,
        'model_forwards':0,'heldout_content_read':False}
