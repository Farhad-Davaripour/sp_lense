"""CPU-only bounded native replication; forecasts are saved before interventions."""
import os
import time
START = time.monotonic()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import ctypes
import hashlib
import json
import threading
import traceback
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
assert ROOT.name in ['native_cpu_20261013','native_cpu_20261013_retry1'] and ROOT.parent.name == 'block_energy_replication_v1'
PRIOR_SECONDS = 9.093 if ROOT.name.endswith('_retry1') else 0.
STATUS = {'status':'starting', 'PID':os.getpid(), 'model_calls':0, 'signed_interventions':0,
          'GPU_allocations':0, 'old_monitor_resumed':False}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def report(phase, **extra):
    STATUS.update(phase=phase, elapsed_seconds=time.monotonic()-START, **extra)
    dump(ROOT / 'STATUS.json', STATUS)
    print(json.dumps(STATUS, allow_nan=False), flush=True)


def hard_stop():
    dump(ROOT / 'HARD_STOP.json', {'PID':os.getpid(), 'seconds':time.monotonic()-START})
    os._exit(124)


TIMER = threading.Timer(2700-PRIOR_SECONDS, hard_stop)
TIMER.daemon = True
TIMER.start()


def deadline():
    if time.monotonic()-START+PRIOR_SECONDS > 2400:
        raise TimeoutError('Declared native2400-second budget reached')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024**2), b''):
            h.update(part)
    return h.hexdigest()


def available_memory():
    class Memory(ctypes.Structure):
        _fields_ = [('length',ctypes.c_uint32), ('load',ctypes.c_uint32)] + [
            (s,ctypes.c_uint64) for s in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended']]
    m = Memory()
    m.length = ctypes.sizeof(m)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return int(m.avail_phys)


def export():
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name != 'EXPORT_HASHES.json')
    manifest = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    dump(ROOT / 'EXPORT_HASHES.json', manifest)
    suffix = '_RETRY1' if ROOT.name.endswith('_retry1') else ''
    archive = ROOT.parent / ('R4_BLOCK_ENERGY_NATIVE_20261013'+suffix+'.zip')
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for p in files + [ROOT / 'EXPORT_HASHES.json']:
            z.write(p, p.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name, expected in manifest.items():
            assert hashlib.sha256(z.read(name)).hexdigest() == expected
    receipt = {'path':str(archive), 'bytes':archive.stat().st_size, 'sha256':sha(archive),
        'members':len(files)+1, 'all_members_and_CRC_verified':True,
        'native_and_export_seconds':time.monotonic()-START, 'cumulative_seconds_with_prior_attempt':time.monotonic()-START+PRIOR_SECONDS,
        'model_calls':STATUS['model_calls'],
        'status':STATUS['status'], 'GPU_allocations':0}
    dump(ROOT.parent / ('NATIVE_EXPORT_RECEIPT_20261013'+suffix+'.json'), receipt)
    print(json.dumps({'export_complete':receipt}), flush=True)


try:
    import numpy as np
    import torch
    import transformers
    from native import Rig
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_available() and '+cpu' in torch.__version__
    SOURCE = Path(__file__).resolve().parent
    plan = json.loads((SOURCE / 'PLAN.json').read_text())
    inputs = json.loads((ROOT / 'INPUTS.json').read_text())
    freeze = json.loads((ROOT / 'EXECUTION_FREEZE.json').read_text())
    for relative, expected in freeze['sha256'].items():
        assert sha(ROOT / relative) == expected, 'Execution freeze changed'
    admission = {'free_memory_bytes':available_memory(), 'minimum_bytes':8589934592,
        'device':'CPU', 'torch_threads':4, 'interop_threads':1, 'model_calls_before_admission':0,
        'native_limit_seconds':2400, 'total_limit_seconds':2700, 'backbone_call_limit':300}
    admission['admitted'] = admission['free_memory_bytes'] >= admission['minimum_bytes']
    dump(ROOT / 'ADMISSION.json', admission)
    assert admission['admitted'], 'Fresh complete-window memory admission failed'
    pin = json.loads((SOURCE / 'model_pin.json').read_text())
    assert pin['revision'] == plan['revision']
    checkpoint = Path(plan['checkpoint'])
    verified = []
    for item in pin['files']:
        path = checkpoint / item['name']
        if not path.is_file(): continue
        assert path.stat().st_size == item['size']
        if item.get('sha256'):
            assert sha(path) == item['sha256']
        else:
            data = path.read_bytes()
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest() == item['git_blob_id']
        verified.append(item['name'])
    assert len(verified) == 8
    dump(ROOT / 'IDENTITY.json', {'revision':pin['revision'], 'verified_checkpoint_files':verified})
    report('loading_model')
    rig = Rig(checkpoint, deadline)
    rig.precision(torch.float32)
    dump(ROOT / 'RUNTIME.json', {'torch':torch.__version__, 'transformers':transformers.__version__,
        'numpy':np.__version__, 'threads':4, 'interop_threads':1, 'GPU_allocations':0,
        'attention':'SDPA MATH; original reference gated-delta kernels', 'cache':False})
    with np.load(ROOT / 'DIRECTIONS.npz', allow_pickle=False) as z:
        vectors = z['vectors'].copy()
    assert vectors.shape == (4,1024)
    questions = inputs['questions']
    contexts = {}
    predictions = {}
    checks = []
    numerical_transport = []
    prediction_hashes = {}
    for folder in ['baseline', 'predictions', 'actual', 'covariance']:
        (ROOT / folder).mkdir()

    def distribution(logits):
        x = np.asarray(logits, dtype=np.float64)
        maximum = x.max()
        logz = maximum + np.log(np.exp(x-maximum).sum())
        logp = x-logz
        return np.exp(logp), logp

    def kl(logits0, logits1):
        p, lp = distribution(logits0)
        _, lq = distribution(logits1)
        raw = float(p @ (lp-lq))
        assert raw >= -1e-10
        return max(0.,raw)

    def variance(logits, q):
        p, _ = distribution(logits)
        q = np.asarray(q,dtype=np.float64)
        centered = q-p@q
        return float(p @ (centered*centered))

    def weighted_relative(logits, predicted, actual):
        den = variance(logits, actual)
        return float(np.sqrt(variance(logits,np.asarray(predicted,dtype=np.float64)-actual)/max(den,1e-30)))

    def rel(predicted, actual):
        actual = np.asarray(actual,dtype=np.float64)
        den = float(np.linalg.norm(actual))
        return float(np.linalg.norm(np.asarray(predicted,dtype=np.float64)-actual)/den) if den>1e-12 else None

    def save_baseline(question, precision, value):
        path = ROOT / 'baseline' / f'{question["id"]}_{precision}.npz'
        np.savez_compressed(path, logits=value['logits'], **value['points'],
            **{k+'_full':v for k,v in value['full'].items()}, prefix_ids=np.asarray(question['prefix_ids']))
        dump(path.with_suffix('.json'), {'prefix_hashes':value['prefix_hashes'], 'precision':precision})

    # Freeze token IDs before weights load; verify the tokenizer agrees now.
    for question in questions:
        ids = rig.tokenizer.apply_chat_template([{'role':'system','content':inputs['system']},
            {'role':'user','content':question['text']}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
        if hasattr(ids,'keys'):ids=ids['input_ids']
        if hasattr(ids,'tolist'):ids=ids.tolist()
        if ids and isinstance(ids[0],list):ids=ids[0]
        assert ids == question['prefix_ids']
        base = rig.value(ids)
        contexts[question['id'],'FP32'] = base
        save_baseline(question,'FP32',base)
        report('capturing_FP32_contexts', question=question['id'], model_calls=rig.calls)

    # ALL predictions (including held questions) precede primary signed endpoints.
    for question in questions:
        pid, ids = question['id'], question['prefix_ids']
        base = contexts[pid,'FP32']
        for di, vector in enumerate(vectors):
            v = torch.tensor(vector)
            direct = rig.derivative(ids,'input',v)
            middle = rig.derivative(ids,'middle',torch.tensor(direct['tangent']['middle']))
            output = rig.derivative(ids,'output',torch.tensor(direct['tangent']['output']))
            runs = [direct,middle,output]
            primal_errors = []
            for answer in runs:
                assert answer['prefix_hashes'] == base['prefix_hashes']
                for name, actual in base['points'].items():
                    primal_errors.append(float(np.linalg.norm(answer['primal'][name]-actual)/max(1.,np.linalg.norm(actual))))
                logit_primal_error=float(np.linalg.norm(answer['logits']-base['logits'])/max(1.,np.linalg.norm(base['logits'])))
                assert logit_primal_error<=1e-6, 'Derivative primal changed base logits beyond frozen tolerance'
                primal_errors.append(logit_primal_error)
            q = np.stack([a['q'] for a in runs])
            coefficients = np.array([variance(base['logits'],qi) for qi in q])
            transport_errors = [weighted_relative(base['logits'],qi,q[0]) for qi in q[1:]]
            direction_states = np.stack([vector,direct['tangent']['middle'],direct['tangent']['output']])
            prediction = {'q':q, 'coefficients':coefficients, 'directions':direction_states,
                'hidden_direction':direct['tangent']['hidden']}
            predictions[pid,di] = prediction
            path = ROOT / 'predictions' / f'{pid}_d{di}.npz'
            np.savez_compressed(path, q=q, coefficients=coefficients, directions=direction_states,
                hidden_direction=prediction['hidden_direction'], base_logits=base['logits'])
            prediction_hashes[path.relative_to(ROOT).as_posix()] = sha(path)
            numerical_transport.append({'question':pid, 'split':question['split'], 'direction':di,
                'primal_relative_max':max(primal_errors), 'transport_weighted_relative':transport_errors,
                'transport_cost_relative':(np.abs(coefficients[1:]-coefficients[0])/max(coefficients[0],1e-30)).tolist(),
                'base_logits_exact':all(np.array_equal(a['logits'],base['logits']) for a in runs),
                'earlier_prefix_states_exact':True, 'AD_method':rig.method})
            if question['split']=='calibration' and di in [0,1]:
                eps=plan['numerical_gates']['finite_difference_step']
                plus = rig.value(ids,'input',v,eps)
                minus = rig.value(ids,'input',v,-eps)
                fd = (plus['logits'].astype(np.float64)-minus['logits'])/(2*eps)
                check = {'question':pid, 'direction':di, 'step':eps,
                    'weighted_logit_relative':weighted_relative(base['logits'],q[0],fd),
                    'middle_vector_relative':rel(direct['tangent']['middle'],
                        (plus['points']['middle'].astype(np.float64)-minus['points']['middle'])/(2*eps)),
                    'output_vector_relative':rel(direct['tangent']['output'],
                        (plus['points']['output'].astype(np.float64)-minus['points']['output'])/(2*eps))}
                checks.append(check)
            report('freezing_predictions', question=pid, direction=di, model_calls=rig.calls)
        p,_ = distribution(base['logits'])
        covariances=[]
        for site in range(3):
            qs=np.stack([predictions[pid,d]['q'][site] for d in range(4)]).astype(np.float64)
            centered=qs-(qs@p)[:,None]
            covariance=(centered*p[None,:])@centered.T
            assert np.linalg.eigvalsh(covariance).min() >= -1e-10*max(1.,np.linalg.norm(covariance))
            covariances.append(covariance)
        np.savez_compressed(ROOT/'covariance'/f'{pid}.npz',covariance=np.stack(covariances))
    dump(ROOT/'PREDICTION_FREEZE.json',{'predictions_before_all_primary_interventions':True,
        'sha256':prediction_hashes, 'AD_method':rig.method, 'compatibility_errors':rig.compatibility_errors,
        'primary_signed_interventions_so_far':0})
    prediction_freeze_sha = sha(ROOT/'PREDICTION_FREEZE.json')
    gate = (all(c['weighted_logit_relative']<=.02 and c['middle_vector_relative'] is not None and
                c['middle_vector_relative']<=.02 and c['output_vector_relative'] is not None and
                c['output_vector_relative']<=.02 for c in checks) and
            all(c['primal_relative_max']<=1e-6 and max(c['transport_weighted_relative'])<=1e-4 for c in numerical_transport))
    dump(ROOT/'NUMERICAL_GATE_LOCK.json',{'pass':gate, 'FD_checks':checks,
        'transport_checks':numerical_transport, 'prediction_freeze_sha256':prediction_freeze_sha})
    assert gate, 'Frozen numerical calibration/transport gate failed; no primary outcomes collected'

    metrics=[]
    null_checks=[]
    alphas=[sign*strength for strength in plan['strengths'] for sign in plan['signs']]
    for precision,dtype in [('FP32',torch.float32),('BF16',torch.bfloat16)]:
        rig.precision(dtype)
        for question in questions:
            pid,ids=question['id'],question['prefix_ids']
            if precision=='BF16':
                contexts[pid,precision]=rig.value(ids)
                save_baseline(question,precision,contexts[pid,precision])
            base=contexts[pid,precision]
            null=rig.value(ids,'input',torch.tensor(vectors[0]),0.)
            exact=all(np.array_equal(null['points'][s],base['points'][s]) for s in base['points']) and np.array_equal(null['logits'],base['logits'])
            null_checks.append({'question':pid,'precision':precision,'exact':exact,'KL':kl(base['logits'],null['logits'])})
            assert exact and null_checks[-1]['KL']<=1e-12
            for di,vector in enumerate(vectors):
                stored={name:[] for name in ['input','middle','output','hidden','logits']}
                pred=predictions[pid,di]
                for alpha in alphas:
                    deadline()
                    actual=rig.value(ids,'input',torch.tensor(vector),alpha)
                    assert actual['prefix_hashes']==base['prefix_hashes'], 'Earlier prefix changed under last-token intervention'
                    for name in ['input','middle','output','hidden']:stored[name].append(actual['points'][name])
                    stored['logits'].append(actual['logits'])
                    delta=actual['points']['input'].astype(np.float64)-base['points']['input']
                    projection=float(delta@vector/np.dot(vector,vector))
                    delivered_norm=float(np.linalg.norm(delta))
                    actual_KL=kl(base['logits'],actual['logits'])
                    row={'question':pid,'split':question['split'],'family':question['family'],
                        'precision':precision,'direction':di,'alpha':alpha,'strength':abs(alpha),
                        'endpoint_KL_base_to_changed':actual_KL,'KL_resolved':actual_KL>1e-10,
                        'actual_input_norm':delivered_norm,'requested_norm':abs(alpha)*float(np.linalg.norm(vector)),
                        'delivered_projection':projection,'off_direction_norm':float(np.linalg.norm(delta-projection*vector)),
                        'zero_delivery':delivered_norm==0.,'earlier_prefix_exact':True}
                    if precision=='FP32':
                        cost=.5*alpha*alpha*pred['coefficients'][0]
                        row.update(predicted_Fisher_cost=float(cost),
                            absolute_cost_error=float(abs(cost-actual_KL)),
                            relative_cost_error=float(abs(cost-actual_KL)/actual_KL) if actual_KL>1e-10 else None,
                            linear_logit_endpoint_KL=kl(base['logits'],base['logits'].astype(np.float64)+alpha*pred['q'][0]),
                            vector_relative_errors={name:rel(alpha*pred['directions'][i],
                                actual['points'][name].astype(np.float64)-base['points'][name]) for i,name in enumerate(['input','middle','output'])},
                            final_hidden_relative_error=rel(alpha*pred['hidden_direction'],
                                actual['points']['hidden'].astype(np.float64)-base['points']['hidden']),
                            weighted_logit_response_relative=weighted_relative(base['logits'],alpha*pred['q'][0],
                                actual['logits'].astype(np.float64)-base['logits']))
                    metrics.append(row)
                    STATUS['signed_interventions']+=1
                np.savez_compressed(ROOT/'actual'/f'{pid}_d{di}_{precision}.npz',
                    alphas=np.asarray(alphas),**{k:np.stack(v) for k,v in stored.items()})
                dump(ROOT/'METRICS.json',metrics)
                report('measuring_interventions',question=pid,precision=precision,direction=di,model_calls=rig.calls)
    dump(ROOT/'NULL_CHECKS.json',null_checks)
    assert len(metrics)==192 and rig.calls<=300
    assert sha(ROOT/'PREDICTION_FREEZE.json')==prediction_freeze_sha
    for relative,expected in prediction_hashes.items():assert sha(ROOT/relative)==expected
    assert all(not module._forward_hooks and not module._forward_pre_hooks for module in rig.model.modules())
    STATUS['status']='native_measurements_complete'
    report('native_complete_export_pending',model_calls=rig.calls,AD_method=rig.method)
except BaseException as error:
    if 'rig' in locals():STATUS['model_calls']=rig.calls
    STATUS['status']='fault_or_gate_stop'
    report('fault',error=type(error).__name__+': '+str(error))
    (ROOT/'FAULT.txt').write_text(traceback.format_exc(),encoding='utf-8')
    raise
finally:
    try:
        export()
    finally:
        TIMER.cancel()
