"""Digest-bound adapter package; exactly one instance of each backbone."""
import hashlib,io,json,zipfile
from pathlib import PurePosixPath
import torch
from .composite_store import Store,download_model
from .composite_detector import Detector
from .composite_actor import Actor
from .composite_controller import Composite
from .composite_training import parameter_snapshot,assert_frozen
from .io_utils import read,digest


def checkpoint(store,descriptor):
    raw=store.get(descriptor['archive'])
    if hashlib.sha256(raw).hexdigest()!=descriptor['archive_sha256']:raise ValueError('Adapter evidence archive digest differs')
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names=z.namelist()
        if len(set(names))!=len(names) or any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n for n in names):raise ValueError('Unsafe archive members')
        if sum(f.file_size for f in z.infolist())>1024**3:raise ValueError('Unexpected adapter archive size')
        manifest=json.loads(z.read('MANIFEST.json'));receipt=json.loads(z.read('RUN_RECEIPT.json'))
        if receipt['status']!='complete':raise ValueError('Failed adapter stage cannot be packaged')
        name=descriptor['checkpoint'];data=z.read(name)
        actual=hashlib.sha256(data).hexdigest()
        if actual!=descriptor['checkpoint_sha256'] or manifest[name]!=actual:raise ValueError('Adapter checkpoint digest differs')
        return torch.load(io.BytesIO(data),map_location='cpu',weights_only=True)


def load_package(root,cfg,package,cache,guard,progress,forward_callback):
    store=Store();detector_pin=read(root/'inputs/MODERNBERT_PIN.json');actor_pin=read(root/'inputs/QWEN_PIN.json')
    expected={'detector':{k:detector_pin[k] for k in ['repository','revision']},'actor':{k:actor_pin[k] for k in ['repository','revision']}}
    if package['models']!=expected:raise ValueError('Composite package targets another model revision')
    saved_detector=checkpoint(store,package['detector']);saved_actor=checkpoint(store,package['behavior'])
    if float(saved_detector['threshold'])!=float(package['threshold']):raise ValueError('Threshold differs from selected detector checkpoint')
    progress('load_detector_package');bert,bert_proof=download_model(detector_pin,cache/'bert');guard()
    detector=Detector(bert,cfg['modernbert']);detector.restore(saved_detector);detector.on_forward=forward_callback
    frozen_detector=parameter_snapshot(detector.base_references)
    progress('load_actor_package');qwen,qwen_proof=download_model(actor_pin,cache/'qwen');guard()
    actor=Actor(qwen,cfg['qwen'],guard,forward_callback);actor.restore(saved_actor)
    model=Composite(detector,actor,package['threshold']).eval()
    proof={'package_sha256':digest(package),'models':expected,'model_files':{'detector':bert_proof,'actor':qwen_proof},'backbone_instances':{'ModernBERT':1,'Qwen':1},'detector_component_validation_admitted':bool(saved_detector['validation_admitted']),'behavior_placement':saved_actor['placement'],'frozen_backbone_parameters':{'detector':sum(p.numel() for _,p in detector.base_references),'actor_storage_elements':sum(p.numel() for _,p in actor.base_references)},'heldout_content_read':False}
    proof['actor_base_proof']=actor.frozen_proof()
    def verify():assert_frozen(frozen_detector);actor.verify_frozen()
    return model,proof,verify
