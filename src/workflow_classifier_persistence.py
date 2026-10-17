"""Persist the frozen cached workflow references into the existing three Bronze tables."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import yaml
from src.config import ROOT, Settings
from src.persist_data import persist
from src.utils import canonical, sha256
from src.workflow_classifier_data import training_rows


def prepare(packet, settings):
    packet=Path(packet)
    protocol=yaml.safe_load((ROOT/'config/workflow_classifier_transfer.yaml').read_bytes())
    datasets=yaml.safe_load((ROOT/'config/workflow_classifier_data.yaml').read_bytes())
    if sha256(packet)!=protocol['dataset']['archive_sha256']:
        raise ValueError('Workflow dataset archive changed')
    with zipfile.ZipFile(packet) as z:
        manifest=json.loads(z.read('MANIFEST.json'))
        if hashlib.sha256(z.read('MANIFEST.json')).hexdigest()!=protocol['dataset']['manifest_sha256']:
            raise ValueError('Workflow dataset member manifest changed')
        payloads={}
        for name,pin in manifest['files'].items():
            b=z.read(name)
            if len(b)!=pin['bytes'] or hashlib.sha256(b).hexdigest()!=pin['sha256']:
                raise ValueError('Workflow dataset member changed')
            payloads[name]=json.loads(b)
    result=[]
    for spec in datasets['datasets']:
        records=payloads[spec['source_member']]
        if spec.get('exclude_exact_validation_states'):
            records=training_rows(records,payloads['development.json'],payloads['initial_development.json'])
        raw=(canonical(records)+'\n').encode('utf-8')
        if len(records)!=spec['expected_records'] or hashlib.sha256(raw).hexdigest()!=spec['sha256']:
            raise ValueError('Frozen workflow dataset membership or source bytes changed')
        if len({r['id'] for r in records})!=len(records):
            raise ValueError('Duplicate workflow reference record IDs')
        prefix=settings.volume_root+'/datasets/'+spec['id']+'/'+spec['version']+'/'+spec['sha256']
        roles=spec['usage_roles']
        rows=[]
        for i,row in enumerate(records):
            text=canonical(row)
            rows.append({'dataset_id':spec['id'],'dataset_version':spec['version'],'cohort_code':spec['id'],
                'source_record_index':i,'record_id':row['id'],'source_file_sha256':spec['sha256'],
                'record_sha256':hashlib.sha256(text.encode()).hexdigest(),'record_json':text,
                'usage_roles_json':canonical(roles),'source_volume_path':prefix+'/source.json'})
        result.append({'spec':spec,'raw':raw,'rows':rows,'source_volume':prefix+'/source.json',
                       'records_volume':prefix+'/records.jsonl'})
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission',required=True);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args(argv)
    a=json.loads(Path(args.admission).read_bytes());settings=Settings.load()
    prepared=prepare(a['packet_zip'],settings)
    if args.execute:
        result=persist(settings,prepared)
    else:
        result={'mode':'local_hash_and_membership_checks_only','datasets':[
            {'id':d['spec']['id'],'table':d['spec']['table'],'records':len(d['rows'])} for d in prepared],
            'cloud_mutations':0,'model_calls':0,'new_JEV_calls':0}
    result['correlated_state_rows_are_not_independent_scenarios']=True
    print(json.dumps(result));return result


if __name__=='__main__':main()
