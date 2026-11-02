"""Prepare the frozen capacity comparison; no inference or fitting here."""
import argparse
from pathlib import Path
import shutil

from .io_utils import digest,dump,read,sha
from .recognition_questions import messages
from .recognition_strict_validation import validate_records


def binding_queries(rows):
    validate_records(rows)
    result=[]
    for row in rows:
        for interface in ['target','target_named']:
            for order in ['canonical','reversed']:
                chat=messages(row['observation'],interface,order)
                result.append({'key':row['record_id']+':'+interface+':'+order,
                               'record_id':row['record_id'],'interface':interface,'order':order,
                               'messages':chat,'prompt_sha256':digest(chat)})
    if len({q['key'] for q in result})!=len(result):raise ValueError('Duplicate capacity queries')
    return result


def prepare(campaign,pins,output):
    campaign,pins,output=map(Path,[campaign,pins,output])
    if output.exists():raise FileExistsError('Preserve capacity input freezes')
    rows=read(campaign/'revision_stress_v1_inputs/OBSERVATIONS.json')
    numeric=read(campaign/'numeric_probe_v1_inputs/CASES.json')
    if len(rows)!=72 or len(numeric)!=36:raise ValueError('Expected the original frozen diagnostic cases')
    queries=binding_queries(rows)
    if len(queries)!=288 or len({r['matched_case_id'] for r in rows})!=12:raise ValueError('Capacity case grouping differs')
    model_pins=read(pins)
    if [(m['repository'],m['hidden_size'],m['layers']) for m in model_pins]!=[('Qwen/Qwen3.5-0.8B',1024,24),('Qwen/Qwen3.5-27B',5120,64)]:
        raise ValueError('Capacity comparison must retain the same model family')
    output.mkdir(parents=True)
    dump(output/'REVISION_ROWS.json',rows);dump(output/'BINDING_QUERIES.json',queries)
    shutil.copyfile(campaign/'numeric_probe_v1_inputs/CASES.json',output/'NUMERIC_CASES.json')
    dump(output/'MODEL_PINS.json',model_pins)
    config={'version':'recognition-capacity-27b-v1','scope':'frozen_base_only_factual_recognition',
            'models':[m['repository'] for m in model_pins],'precision':'bfloat16','quantization':None,
            'device':'cuda:0','attention':'sdpa_math','thinking':False,'use_cache':False,'sampling':False,
            'maximum_tokens':1024,'maximum_new_tokens':8,'numerical_tie_band':1e-4,
            'capture':'last input position of final whole text block; hidden widths differ by model',
            'binding_queries_per_model':288,'numeric_cases_per_model':36,
            'maximum_forwards_per_model':576,'worker_stop_seconds':2400,'export_reserve_seconds':180,
            'minimum_gpu_total_bytes':72*1024**3,'minimum_gpu_free_bytes':64*1024**3,
            'minimum_host_free_bytes':12*1024**3,'minimum_disk_free_bytes':75*1024**3,
            'live_gpu_free_floor_bytes':2*1024**3,'seed':20261019,
            'primary_binding':'Canonical role contrast, both roles correct within each case and each history variant; never choose best answer order after evaluation.',
            'secondary_binding':['direct canonical','direct reversed','direct semantic mean','reversed role contrast','mean role contrast'],
            'primary_numeric':'Native-EOS-completed response containing only the correct integer, all 36 cases in denominator.',
            'secondary_numeric':['complete first integer','complete last integer'],
            'limits':'Same already evaluated diagnostic cases. New model results are a size comparison, not fresh held-out generalization. The BF16 GPU 0.8B control separates device/precision effects from comparison with the archived FP32 CPU arm.',
            'fitting':'None; no research adapter, policy, agent tools, action execution or Research 3 access.'}
    dump(output/'CONFIG.json',config)
    dump(output/'MANIFEST.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return config


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['campaign','pins','output']:p.add_argument(key)
    a=p.parse_args();print(prepare(a.campaign,a.pins,a.output))
