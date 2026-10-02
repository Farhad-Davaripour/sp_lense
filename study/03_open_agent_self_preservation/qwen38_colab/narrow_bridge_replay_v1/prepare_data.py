"""Freeze the actual archived paired data before any model load or update."""
import hashlib
import json
import shutil
from pathlib import Path
from transformers import AutoTokenizer
from audit import row_tokens,sha
from model_ops import save
from dataset_build import verify,ARCHIVED_B_SHA256
from handoff_loader import load_handoff
from world import TOOLS
from production_pin import verify_production_datasets


def prompt_sha(row):
    return hashlib.sha256(json.dumps({'messages':row['messages'],'tools':row.get('tools')},
                        sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prepare(root,h2):
    root=Path(root)
    if sha(root/'inputs/archived_B_train.json')!=ARCHIVED_B_SHA256:
        raise RuntimeError('Archived B112 identity differs')
    archived=json.loads((root/'inputs/archived_B_train.json').read_text())
    old=json.loads((root/'inputs/panels/old_cases.json').read_text())
    datasets,content_audit=verify(archived,old,source_archive_sha256=ARCHIVED_B_SHA256)
    datasets={'reference':datasets['reference'],'bridge':datasets['treatment']}
    production_pins=verify_production_datasets(root/'source',datasets,root/'inputs/panels/old_cases.json')
    if content_audit['changed_positive_bridge_rows']!=9 or content_audit['unchanged_rows']!=103:
        raise RuntimeError('Frozen narrow comparison membership differs')
    tokenizer=AutoTokenizer.from_pretrained(root/'model',local_files_only=True,trust_remote_code=False)
    lengths={key:[row_tokens(tokenizer,row)['actual_length'] for row in rows] for key,rows in datasets.items()}
    paired=[max(a,b) for a,b in zip(lengths['reference'],lengths['bridge'])]
    if len(paired)!=112 or max(paired)>3072:raise RuntimeError('Frozen training token cap differs')
    old_runtime=json.loads((root/'inputs/panels/prior_H2_runtime.json').read_text())
    provenance=json.loads((root/'source/H2_STATE_PROVENANCE.json').read_text())
    prior_fit_path=root/'inputs/panels/prior_reference_FIT.json'
    if sha(prior_fit_path)!=provenance['verified_private_receipt_sha256']:
        raise RuntimeError('Starting-state provenance receipt differs')
    prior_fit=json.loads(prior_fit_path.read_text())
    if prior_fit['starting_adapter_state_sha256']!=provenance['canonical_state_sha256'] or any(prior_fit[k]!=provenance['base_before_and_after'] for k in ('base_before','base_after')):
        raise RuntimeError('Verified prior H2 state/base receipt differs')
    template=hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
    if template!=old_runtime['template_sha256']:raise RuntimeError('Template differs from historical panel runtime')
    if old_runtime['packed_base_parameter_hash']!='ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755':
        raise RuntimeError('Historical pinned packed base differs')
    handoff,fixtures=load_handoff(root/'source')
    if fixtures.freeze_payload()['sha256_canonical_without_hash']!='528920a3fe88d39a3a910b9b5054a1c469583d03a9bc9c1fd45a87c784709eea':
        raise RuntimeError('Frozen handoff diagnostic differs')
    testing={prompt_sha({'messages':fixtures.start_spec(spec)[1],'tools':TOOLS}) for spec in fixtures.build_specs()}
    training={prompt_sha(row) for rows in datasets.values() for row in rows}
    if training & testing:raise RuntimeError('Exact training/handoff diagnostic prompt overlap')
    audit={'paired_lengths':paired,'max_length':max(paired),'padded_tokens_each_per_pass':sum(paired),
        'actual_tokens_each':{key:sum(value) for key,value in lengths.items()},
        'supervised_tokens_each':{key:sum(len(row_tokens(tokenizer,row)['response']) for row in rows) for key,rows in datasets.items()},
        'content_audit':content_audit,'training_diagnostic_prompt_intersection':0,
        'template_sha256':template,'source_archive_sha256':ARCHIVED_B_SHA256,
        'production_pins':production_pins,'H2_state_provenance':provenance}
    save(root/'DATA_AUDIT.json',audit)
    worker=root/'narrow_bridge_stream'
    shutil.copytree(root/'source',worker/'code',ignore=shutil.ignore_patterns('__pycache__'))
    for filename in ('old_cases.json','new_development.json','loss_samples.json','preferences.json'):
        shutil.copy2(root/'inputs/panels'/filename,worker/'code'/filename)
    shutil.copy2(root/'inputs/panels/benign_competence_dev.json',worker/'code/data/benign_competence_dev.json')
    for key,rows in datasets.items():save(worker/'code'/(key+'_train.json'),rows)
    cfg={'H2':h2,'paired_lengths':paired,'snapshots':[0,7,14,28,56],
        'expected_H2_state_sha256':provenance['canonical_state_sha256'],
        'production_pins':production_pins,
        'historical_base_hash':'ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755',
        'template_sha256':template,'handoff_fixture_sha256':'528920a3fe88d39a3a910b9b5054a1c469583d03a9bc9c1fd45a87c784709eea',
        'rank':16,'alpha':32,'lr':5e-5,'seed':941,'updates':56,'microbatch':1,'effective_batch':4,
        'archive_sha256':ARCHIVED_B_SHA256,'changed_training_rows':9,'immutable_training_rows':103}
    save(worker/'code/config.json',cfg)
    freeze={'sha256':{str(path.relative_to(worker/'code')):sha(path) for path in (worker/'code').rglob('*')
                     if path.is_file() and path.name!='FREEZE.json' and '__pycache__' not in path.parts}}
    save(worker/'code/FREEZE.json',freeze)
    print('ACTUAL_DATA_AND_UNCHANGED_PANELS_FROZEN',json.dumps({'rows':112,'changed':9,'immutable':103,
          'max_tokens':max(paired),'paired_tokens_per_pass':sum(paired)}),flush=True)
    return worker,audit
