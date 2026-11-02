"""Freeze a matched NF4 amendment and export an owned Databricks AIR run."""
import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

from .io_utils import dump,read,sha,verify_release


def prepare(original,output):
    original,output=map(Path,[original,output])
    if output.exists():raise FileExistsError('Preserve capacity amendments')
    manifest=read(original/'MANIFEST.json')
    for name,value in manifest.items():
        if sha(original/name)!=value:raise ValueError('Original capacity freeze changed')
    output.mkdir(parents=True)
    for name in manifest:
        if name!='CONFIG.json':shutil.copyfile(original/name,output/name)
    cfg=read(original/'CONFIG.json')
    cfg.update(version='recognition-capacity-27b-databricks-nf4-v1',platform='databricks',
               quantization={'load_in_4bit':True,'bnb_4bit_quant_type':'nf4','bnb_4bit_use_double_quant':True,'bnb_4bit_compute_dtype':'bfloat16'},
               worker_stop_seconds=3600,export_reserve_seconds=180,
               minimum_gpu_total_bytes=20*1024**3,minimum_gpu_free_bytes=18*1024**3,
               limits='Same diagnostic cases; both model sizes use the same declared NF4 profile with BF16 computation on one Databricks A10. Precision differs from the original unrun BF16 plan and archived FP32 CPU control. Quantization error can differ with size; this is not a pure causal size-only or held-out generalization result.')
    dump(output/'CONFIG.json',cfg)
    dump(output/'AMENDMENT.json',dict(original_manifest_sha256=sha(original/'MANIFEST.json'),original_config_sha256=sha(original/'CONFIG.json'),reason='Databricks-only migration and declared matched single-A10 memory profile before any capacity model call.',unchanged=[n for n in manifest if n!='CONFIG.json'],models_executed_before_amendment=0,scientific_retries=0))
    dump(output/'MANIFEST.json',{p.name:sha(p) for p in output.iterdir() if p.is_file()})
    return cfg


def launch(snapshot,commit,destination):
    from databricks.sdk import WorkspaceClient
    snapshot=Path(snapshot).resolve();verify_release(snapshot,commit)
    if snapshot!=Path(os.environ['CODE_SOURCE_PATH']).resolve() or not os.environ.get('MLFLOW_RUN_ID'):raise ValueError('Databricks AIR execution required')
    prefix='/Volumes/workspace/research4/artifacts/recognition_capacity_databricks_nf4_v1/'
    if not destination.startswith(prefix) or '..' in destination.split('/'):raise ValueError('Dedicated Research 4 output required')
    client=WorkspaceClient();client.files.create_directory(destination)
    # Files API exports work on AIR even when Volume FUSE is unavailable.
    def upload(name,data):
        client.files.upload(destination+'/'+name,io.BytesIO(data),overwrite=False)
        with client.files.download(destination+'/'+name).contents as f:back=f.read()
        if back!=data:raise ValueError('Databricks Volume readback differs')
    receipt={'status':'starting','source_commit':commit,'MLFLOW_run_id':os.environ['MLFLOW_RUN_ID'],'platform':'databricks','scientific_retries':0}
    upload('ALLOCATION_RECEIPT.json',json.dumps(receipt,indent=2).encode())
    with tempfile.TemporaryDirectory(prefix='research4-capacity-') as temporary:
        root=Path(temporary);output=root/'output';cache=root/'cache'
        try:
            from .recognition_capacity_worker import run
            receipt=run(snapshot/'inputs',output,cache,allocation_deadline=time.time()+3900)
        except Exception as error:
            receipt={'status':'failed','error_type':type(error).__name__,'error':str(error),'backbone_forwards':None,'model_objects_released':None}
            upload('WRAPPER_FAILURE.json',json.dumps({'traceback':traceback.format_exc()}).encode())
        finally:
            if output.with_suffix('.zip').exists():
                payload=output.with_suffix('.zip').read_bytes();upload('evidence.zip',payload)
                receipt.update(archive_sha256=sha(output.with_suffix('.zip')),archive_bytes=len(payload))
            for name in ['RUN_RECEIPT.json','PROGRESS.json','FAILURE.json','ADMISSION.json','SOFTWARE.json','PRECALL_FREEZE.json']:
                if (output/name).is_file():upload(name,(output/name).read_bytes())
            receipt.update(source_commit=commit,destination=destination,worker_process_exiting=True,temporary_owned_cache_cleanup=True)
            upload('EXPORT_RECEIPT.json',json.dumps(receipt,indent=2).encode())
    print(json.dumps(receipt),flush=True)
    if receipt['status']!='complete':raise RuntimeError('Capacity run failed; retained its export')
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('original');a.add_argument('output')
    a=sub.add_parser('run');a.add_argument('snapshot');a.add_argument('commit');a.add_argument('destination')
    args=p.parse_args()
    if args.command=='prepare':print(prepare(args.original,args.output))
    else:launch(args.snapshot,args.commit,args.destination)
