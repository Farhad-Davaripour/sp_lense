"""Run the five frozen 27B methods using only released inputs and a public base."""
from pathlib import Path
import argparse
import datetime as dt
import json
import shutil
from src.config import ROOT
from src.public_artifacts import verify, validation_packet
from src.utils import sha256, write_json


def run(root, base, output, max_seconds):
    from src.matched_steering import verify_model_file,run as matched_run
    from src.verify import source_digest
    if not 3600<=max_seconds<=18000:raise ValueError('Use a 1 to 5 hour bound including setup and export')
    start=dt.datetime.now(dt.timezone.utc);root=Path(root).resolve();base=Path(base).resolve();output=Path(output).resolve()
    if output.exists() or output.is_relative_to(ROOT.resolve()) or output.is_relative_to(root):
        raise ValueError('Use a new output outside the checkout and released inputs')
    verify(root);validation_packet(root)
    pin=json.loads((ROOT/'config/recipes/qwen38_colab/model_pin.json').read_bytes())
    for entry in pin['files']:
        if (dt.datetime.now(dt.timezone.utc)-start).total_seconds()>min(3600,max_seconds-1800):
            raise TimeoutError('Public snapshot setup consumed the admitted window')
        verify_model_file(base/entry['name'],entry)
    output.mkdir(parents=True)
    inputs=output/'verified_inputs';shutil.copytree(root/'controllers/workflow',inputs)
    write_json(inputs/'INPUT_MANIFEST.json',{'files':{p.relative_to(inputs).as_posix():sha256(p)
        for p in sorted(inputs.rglob('*')) if p.is_file()}})
    gate=output/'frozen_initial_gate';gate.mkdir()
    shutil.copyfile(root/'evidence/validation24_gate/PREDICTIONS.json',gate/'PREDICTIONS_PRIVATE.json')
    shutil.copyfile(root/'evidence/validation24_gate/RESULT.json',gate/'RESULT.json')
    admission={'format':'research3-public-frozen-workflow-inference-v1','public_artifacts_root':str(root),
        'submitted_utc':start.isoformat(),'absolute_end_utc':(start+dt.timedelta(seconds=max_seconds)).isoformat(),
        'output':str(output),'gate_output':str(gate),'comparison_inputs':str(inputs),
        'adapter_directory':str(root/'adapters/qwen-27b-workflow-A'),
        'source_digest':source_digest(),'protocol_sha256':sha256(ROOT/'config/matched_validation.yaml'),
        'optimizer_updates':0,'automatic_retries':0,
        'gate_scope':'Exact released ModernBERT initial predictions; reproduce the gate separately with src.public_gate.'}
    write_json(output/'ADMISSION.json',admission)
    write_json(output/'MODEL_STAGING.json',{'model_path':str(base),'revision':pin['revision'],
        'verified_files':[r['name'] for r in pin['files']],'public_artifact_manifest_sha256':sha256(root/'MANIFEST.json')})
    return matched_run(output/'ADMISSION.json')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','base','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-seconds',type=int,default=18000);p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    if not a.execute:p.error('Explicit --execute is required for model inference')
    print(json.dumps(run(a.root,a.base,a.output,a.max_seconds)))
