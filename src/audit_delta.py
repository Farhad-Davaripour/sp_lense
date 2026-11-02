"""Read-only native Spark validation of this pilot's retained Delta rows."""
from pathlib import Path
from .io_utils import read,dump,digest,sha,verify_release,assert_namespace
def run(root,audit_run,source_commit,pilot_commit,pilot_run,spark):
    import numpy as np
    from pyspark.sql import functions as F
    verify_release(root,source_commit)
    if len(pilot_commit)!=40 or not all(c in '0123456789abcdef' for c in pilot_commit) or not str(pilot_run).isdigit():raise ValueError('Invalid retained pilot identity')
    volume=Path('/Volumes/workspace/research4/artifacts');original=volume/'runs'/pilot_commit/str(pilot_run)/'pilot'
    receipt=read(original/'RUN_RECEIPT.json');frozen=read(original/'EVALUATION_FREEZE.json')
    if receipt['status']!='complete' or receipt['source_commit']!=pilot_commit:raise ValueError('Retained pilot is not complete')
    dataset=read(volume/'datasets'/receipt['dataset_version']/'dataset.json');cache=np.load(original/'ACTIVATION_CACHE.npz',allow_pickle=False)
    version=dataset['version'];checks={}
    episodes=spark.table('workspace.research4.episodes').where((F.col('source_commit')==pilot_commit)&(F.col('dataset_version')==version))
    states=spark.table('workspace.research4.states').where((F.col('source_commit')==pilot_commit)&(F.col('dataset_version')==version))
    activations=spark.table('workspace.research4.activations').where((F.col('source_commit')==pilot_commit)&(F.col('run_id')==str(pilot_run)))
    results=spark.table('workspace.research4.results').where((F.col('source_commit')==pilot_commit)&(F.col('run_id')==str(pilot_run)))
    counts={'episodes':episodes.count(),'states':states.count(),'activations':activations.count(),'results':results.count()}
    checks['expected_counts']=counts=={'episodes':37,'states':118,'activations':118,'results':54}
    checks['unique_episode_and_state_ids']=episodes.select('episode_id').distinct().count()==37 and states.select('state_id').distinct().count()==118
    checks['unique_condition_episode_results']=results.select('condition','episode_id').distinct().count()==54
    checks['no_orphan_states']=states.join(episodes.select('episode_id'),'episode_id','left_anti').count()==0
    checks['no_orphan_activations']=activations.join(states.select('state_id'),'state_id','left_anti').count()==0
    e=spark.createDataFrame([(r['episode_id'],digest(r)) for r in dataset['episodes']],'episode_id string, expected_hash string')
    s=spark.createDataFrame([(r['state_id'],digest(r)) for r in dataset['states']],'state_id string, expected_hash string')
    checks['episode_hashes_match_immutable_dataset']=episodes.join(e,'episode_id','left').where(F.col('expected_hash').isNull()|F.col('record_sha256').isNull()|(F.col('record_sha256')!=F.col('expected_hash'))).count()==0
    checks['state_hashes_match_immutable_dataset']=states.join(s,'state_id','left').where(F.col('expected_hash').isNull()|F.col('record_sha256').isNull()|(F.col('record_sha256')!=F.col('expected_hash'))).count()==0
    expected=[]
    for row in dataset['states']:
        key=digest({'view':row['view_text']});expected.append((row['state_id'],[float(x) for x in cache[key+':feature']],[float(x) for x in cache[key+':logits']]))
    a=spark.createDataFrame(expected,'state_id string, expected_features array<float>, expected_logits array<float>')
    checks['all_activation_vectors_match_retained_cache']=activations.join(a,'state_id').where((F.col('features')!=F.col('expected_features'))|(F.col('base_action_logits')!=F.col('expected_logits'))).count()==0
    checks['correct_feature_dimensions']=activations.where(F.col('features').isNull()|F.col('base_action_logits').isNull()|(F.size('features')!=1024)|(F.size('base_action_logits')!=5)|(F.col('feature_layer')!=20)).count()==0
    checks['result_JSON_SHA256_matches']=results.where(F.col('result_json').isNull()|F.col('output_sha256').isNull()|(F.sha2('result_json',256)!=F.col('output_sha256'))).count()==0
    checks['frozen_checkpoint_unchanged']=sha(original/'trainables.pt')==frozen['checkpoint_sha256']
    cache.close();result={'audit_pass':all(checks.values()),'audit_source_commit':source_commit,'pilot_source_commit':pilot_commit,'pilot_run_id':str(pilot_run),'dataset_version':version,'counts':counts,'checks':checks,'new_Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,'warehouse_dependency':False}
    output=volume/'verification'/pilot_commit/str(pilot_run)/str(audit_run);assert_namespace(output);output.mkdir(parents=True,exist_ok=False);dump(output/'DELTA_AUDIT.json',result)
    from .runtime import archive
    result['export']=archive(output);dump(output.parent/(str(audit_run)+'_EXPORT_RECEIPT.json'),result);return result
