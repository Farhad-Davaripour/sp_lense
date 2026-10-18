"""Complete missing gate-OFF report aliases without rerunning a stopped GPU job."""
from pathlib import Path
import argparse
import json
from src.matched_steering import CONDITIONS, summarize
from src.matched_validation import read_packet
from src.utils import sha256, write_json


def recover(source, packet, gate_predictions, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Keep the derived report separate from the immutable original run')
    if output.exists():
        raise FileExistsError('Use a new derived report directory; keep the stopped run immutable')
    admission = json.loads((source / 'ADMISSION.json').read_bytes())
    fault = json.loads((source / 'FAULT.json').read_bytes())
    if fault.get('type') != 'MemoryError' or fault.get('message') != 'GPU reserve breached':
        raise ValueError('This derivation only covers the recorded GPU-reserve stop')
    if (source / 'COMPLETE.json').exists():
        raise ValueError('Do not replace an existing completed GPU outcome')
    data = read_packet(packet, admission['packet_sha256'])
    cases = data['cases.json']
    predictions = {r['id']: r['gate'] for r in json.loads(Path(gate_predictions).read_bytes())}
    if set(predictions) != {c['id'] for c in cases}:
        raise ValueError('Frozen gate membership differs')
    planned, aliases, actual = [], [], 0
    for method in CONDITIONS:
        folder = source / method / 'episodes'
        if {p.stem for p in folder.glob('*.json')} - set(predictions):
            raise ValueError('Unexpected source episode membership')
        for case in cases:
            cid = case['id']
            gate = predictions[cid]
            positive = case['condition'].startswith('self_')
            if gate['enabled'] != positive:
                raise ValueError('A wrong initial gate cannot be repaired by report derivation')
            path = folder / (cid + '.json')
            if path.exists():
                raw = path.read_bytes()
                row = json.loads(raw)
                actual += 1
            else:
                if positive or gate['enabled']:
                    raise ValueError('A missing neural episode cannot be reconstructed')
                fallback = source / 'shared_fallback/episodes' / (cid + '.json')
                row = json.loads(fallback.read_bytes())
                if not row.get('completed') or row.get('runtime_fault') is not None:
                    raise ValueError('Shared fallback must be an actual completed episode')
                # Every other completed method must point to these exact original bytes.
                references = [source / m / 'episodes' / (cid + '.json') for m in CONDITIONS if m != method]
                if any(not p.is_file() or json.loads(p.read_bytes()).get('source_episode_sha256') != sha256(fallback)
                       for p in references):
                    raise ValueError('Missing or conflicting shared-control provenance')
                row.update(shared_base_fallback=True, source_episode_sha256=sha256(fallback),
                           model_free_alias_completion=True)
                raw = None
                aliases.append({'method':method, 'case_id':cid, 'source_episode_sha256':sha256(fallback)})
            if (row['case_id'] != cid or row['frozen_initial_gate'] != gate
                    or not row.get('completed') or row.get('runtime_fault') is not None
                    or row['score']['negative_control'] == positive):
                raise ValueError('Episode, gate or completion identity differs')
            planned.append((method, cid, raw, row))
    if actual != fault['case_records'] or not aliases:
        raise ValueError('Recorded stopped count differs or no fallback alias is missing')
    for method, cid, raw, row in planned:
        dest = output / method / 'episodes' / (cid + '.json')
        if raw is None:
            write_json(dest, row)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
    groups = summarize(output)
    receipt = {'status':'MODEL_FREE_SHARED_FALLBACK_DERIVATION',
        'source_GPU_worker_result':'FAILED', 'source_worker_records':actual,
        'derived_control_aliases':aliases, 'logical_method_case_records':len(planned),
        'unique_workflows':len(cases), 'physical_workflow_episodes':
            sum(c['condition'].startswith('self_') for c in cases)*len(CONDITIONS)
            + sum(not c['condition'].startswith('self_') for c in cases),
        'new_model_forwards':0, 'optimizer_updates':0, 'new_compute_allocations':0,
        'original_worker_modified':False, 'GPU_reserve_or_deadline_relaxed':False,
        'source_final_model_fingerprint_completed':False,
        'source_receipts':{n:{'bytes':(source/n).stat().st_size,'sha256':sha256(source/n)}
                           for n in ['ADMISSION.json','FAULT.json','PROGRESS.json','CLEANUP.json']},
        'frozen_gate_predictions_sha256':sha256(gate_predictions), 'methods':groups}
    write_json(output / 'DERIVATION.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source','packet','gate-predictions','output']:
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    result = recover(a.source, a.packet, a.gate_predictions, a.output)
    print(json.dumps({k:result[k] for k in ['status','source_worker_records','logical_method_case_records',
                                          'physical_workflow_episodes','new_model_forwards']}))
