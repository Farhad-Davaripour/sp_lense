"""Pure canonical checks against the committed production data freeze."""
import hashlib
import json
from pathlib import Path


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def frozen_production(code):
    code=Path(code)
    audit=json.loads((code/'data_frozen/DATA_AUDIT.json').read_text())
    frozen={key:json.loads((code/'data_frozen'/filename).read_text()) for key,filename in
            (('reference','reference_train.json'),('bridge','treatment_train.json'))}
    if not audit['production_file_identity_verified'] or audit['source_archive_file_sha256']!='01560c9a65fff138ae117ade35e728ab2d6f244a121517a8583eed07b7265c0c':
        raise RuntimeError('Committed data audit does not pin the actual archived B112')
    if len(audit['row_sha256'])!=112 or any(len(rows)!=112 for rows in frozen.values()):
        raise RuntimeError('Committed production row counts differ')
    for index,entry in enumerate(audit['row_sha256']):
        if entry['slot']!=index:raise RuntimeError('Committed production slot order differs')
        for key,field in (('reference','reference_sha256'),('bridge','treatment_sha256')):
            if canonical_sha(frozen[key][index])!=entry[field]:
                raise RuntimeError('Committed production row pin differs: '+key+'/'+str(index))
    if canonical_sha(frozen['reference'])!=audit['canonical_archive_rows_sha256']:
        raise RuntimeError('Committed reference canonical archive differs')
    pins={key:canonical_sha(rows) for key,rows in frozen.items()}
    pins['old_cases_file_sha256']=audit['old_cases_file_sha256']
    pins['data_audit_file_sha256']=hashlib.sha256((code/'data_frozen/DATA_AUDIT.json').read_bytes()).hexdigest()
    return frozen,audit,pins


def verify_production_datasets(code,datasets,old_cases_file=None):
    frozen,audit,pins=frozen_production(code)
    for key in ('reference','bridge'):
        if datasets[key]!=frozen[key] or canonical_sha(datasets[key])!=pins[key]:
            raise RuntimeError('Runtime dataset differs from committed production: '+key)
    if old_cases_file is not None and hashlib.sha256(Path(old_cases_file).read_bytes()).hexdigest()!=pins['old_cases_file_sha256']:
        raise RuntimeError('Runtime old-case bytes differ from production data freeze')
    return pins


def verify_production_rows(code,rows,recipe,config_pins):
    frozen,_audit,pins=frozen_production(code)
    if recipe not in ('reference','bridge') or rows!=frozen[recipe] or canonical_sha(rows)!=pins[recipe]:
        raise RuntimeError('Fit rows differ from committed production: '+recipe)
    if config_pins!=pins:raise RuntimeError('Runtime config production pins differ')
    return pins
