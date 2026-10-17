"""Only Research 4 Delta tables and immutable Volume outputs."""
import json
from dataclasses import asdict
from pathlib import Path
from .episodes import generate,teacher_states
from .io_utils import canonical,digest,dump,read,sha
def initialize(spark,root):
    for statement in (Path(root)/'ddl/tables.sql').read_text(encoding='utf-8').split(';'):
        if statement.strip():spark.sql(statement)
def dataset(spark,config,commit,volume):
    from pyspark.sql.types import StructType,StructField,StringType,IntegerType
    from pyspark.sql import functions as F
    specs=generate(config);rows=[]
    for spec in specs:rows.extend(teacher_states(spec,config['max_episode_steps'])[0])
    version=config['version']+'-'+digest({'config':config,'episodes':[asdict(s) for s in specs],'states':rows})[:12]
    dest=Path(volume)/'datasets'/version
    manifest={'version':version,'config':config,'episodes':[asdict(s) for s in specs],'states':rows,'content_sha256':digest(rows)}
    if dest.exists():
        if read(dest/'dataset.json')['content_sha256']!=manifest['content_sha256']:raise ValueError('Immutable dataset collision')
    else:dest.mkdir(parents=True);dump(dest/'dataset.json',manifest)
    episode_records=[(version,s.episode_id,s.split,s.family,canonical(asdict(s)),commit,digest(asdict(s))) for s in specs]
    state_records=[(version,r['episode_id'],r['state_id'],r['split'],r['step'],r['view_text'],*r['labels'],r['gate_target'],r['expected_action'],commit,digest(r)) for r in rows]
    eschema='dataset_version string, episode_id string, split string, family string, spec_json string, source_commit string, record_sha256 string'
    sschema='dataset_version string, episode_id string, state_id string, split string, step int, view_text string, threat_present int, task_unfinished int, continuation_useful int, gate_target int, expected_action string, source_commit string, record_sha256 string'
    for name,records,schema in [('episodes',episode_records,eschema),('states',state_records,sschema)]:
        table='workspace.research4.'+name
        exists=spark.table(table).where((F.col('dataset_version')==version)&(F.col('source_commit')==commit)).limit(1).count()
        if not exists:spark.createDataFrame(records,schema).write.mode('append').saveAsTable(table)
    return specs,rows,version
def event(spark,run_id,commit,stage,status,payload,output):
    from datetime import datetime,timezone
    row=(str(run_id),commit,datetime.now(timezone.utc).isoformat(),stage,status,canonical(payload),str(output))
    spark.createDataFrame([row],'run_id string, source_commit string, event_time string, stage string, status string, payload_json string, output_path string').write.mode('append').saveAsTable('workspace.research4.run_events')
def activations(spark,run_id,version,rows,actor,commit,layer):
    records=[]
    for r in rows:
        a=actor.base(r['view_text']);records.append((str(run_id),version,r['state_id'],r['split'],layer,[float(x) for x in a['feature']],[float(x) for x in a['logits']],commit))
    spark.createDataFrame(records,'run_id string, dataset_version string, state_id string, split string, feature_layer int, features array<float>, base_action_logits array<float>, source_commit string').write.mode('append').saveAsTable('workspace.research4.activations')
def result_rows(spark,run_id,commit,results):
    records=[(str(run_id),commit,r['condition'],r['episode_id'],canonical(r),digest(r)) for r in results]
    spark.createDataFrame(records,'run_id string, source_commit string, condition string, episode_id string, result_json string, output_sha256 string').write.mode('append').saveAsTable('workspace.research4.results')
