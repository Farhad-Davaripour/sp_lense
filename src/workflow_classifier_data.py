"""Prepare genuine cached three-question JEV references without provider calls."""
from pathlib import Path
import argparse
import base64
import collections
import copy
import hashlib
import json
from src.engines.workflow.schema import QUESTIONS, digest, effective, decode
from src.engines.workflow.policy import choose_rule
from src.utils import sha256, write_json


def training_rows(rows, validation, initial_validation):
    """Remove exact visible validation states before any joint fitting."""
    from src.utils import canonical
    blocked = {canonical(r['state']) for r in validation + initial_validation}
    return [r for r in rows if canonical(r['state']) not in blocked]


def referenced(rows):
    result = collections.defaultdict(set)
    for row in rows:
        for turn in row['turns']:
            for item in turn.get('classifications', []):
                result[item['request_sha256']].add(row['case_id'])
    return result


def prepare(archive_root, output, initial_development_root, initial_H24):
    root, output = Path(archive_root), Path(output)
    if output.exists():
        raise FileExistsError('Use a new workflow-reference directory')
    read = lambda path: json.loads(path.read_bytes())
    from src.engines.workflow.initial_gate import digest as initial_digest, decode_jev
    initial_root = Path(initial_development_root)
    fixtures = read(initial_root/'DEV_INITIAL_FIXTURES.json')
    gates = read(initial_root/'INITIAL_GATE_DECISIONS.json')
    initial_dev = []
    for fixture in fixtures:
        case = fixture['case_id']; request = fixture['initial_gate_request']; saved = gates[case]
        if (not saved['genuine_saved_decision'] or initial_digest(request) != saved['request_sha256']
                or fixture['initial_gate_request_sha256'] != saved['request_sha256']):
            raise ValueError('Original DEV initial-gate evidence changed')
        initial_dev.append({'id':case,'state':request['state'],'teacher_gate':saved['frozen_gate'],
                            'request_sha256':saved['request_sha256']})
    initial_known = []
    for row in read(Path(initial_H24)):
        request = row['original_request']
        if initial_digest(request) != row['request_sha256'] or row['state'] != request['state']:
            raise ValueError('Original H24 initial-gate input changed')
        gate = decode_jev(request,row['jev_response'])
        initial_known.append({'id':row['id'],'state':row['state'],'teacher_gate':gate.__dict__,
                              'request_sha256':row['request_sha256']})
    if len(initial_dev)!=9 or len(initial_known)!=24:
        raise ValueError('Expected DEV9 and familiar historical H24 initial-gate panels')
    development = [read(p) for p in sorted((root / 'closed_loop_development/rollouts').glob('*.json'))]
    dev_index = referenced(development)
    calibration = read(root / 'reports/GENUINE_DEVELOPMENT_JEV.json')
    for row in calibration:
        dev_index[row['request_sha256']].add(row['state_id'].split(':', 1)[0])
    cohort = read(root / 'confirmation_cases.json')
    held_ids = {row['id'] for row in cohort}
    if len(held_ids) != 24:
        raise ValueError('Unexpected historical H24 membership')
    held = [read(root / 'rollouts' / (case['id'] + '__' + arm + '.json'))
            for case in cohort for arm in ('base_jev_prompt','base_jev_activation')]
    held_index = referenced(held)
    dev_cases = sorted({case for ids in dev_index.values() for case in ids})
    if len(dev_cases) != 9:
        raise ValueError('Expected the original nine development scenarios')
    # Fixed diagonal: one self, one other and one no-threat case across the three families.
    selection_cases = {dev_cases[index] for index in (0,4,8)}
    records = {}
    for key in sorted(set(dev_index) | set(held_index)):
        path = root / 'api/requests' / (key + '.json')
        receipt = read(path)
        request, decoded = receipt['request'], receipt['decoded']
        if (not receipt['completed'] or not receipt['actual_service_call']
                or digest(request) != key or receipt['request_sha256'] != key
                or request['model'] != 'jev-1.13.0' or request['questions'] != QUESTIONS
                or decoded['model'] != 'jev-1.13.0' or set(decoded['concepts']) != set(QUESTIONS)):
            raise ValueError('Cached JEV provenance or question contract changed')
        raw = base64.b64decode(receipt['raw_response_base64'], validate=True)
        if hashlib.sha256(raw).hexdigest() != receipt['raw_response_sha256']:
            raise ValueError('Genuine cached provider response changed')
        if decode(json.loads(raw)) != decoded:
            raise ValueError('Decoded teacher targets differ from the genuine provider response')
        labels, abstentions = effective(decoded)
        history = request['state']['visible_history']
        records[key] = {'id': key, 'state': copy.deepcopy(request['state']),
            'state_sha256': digest(request['state']),
            'case_ids': sorted(dev_index.get(key, set()) | held_index.get(key, set())),
            'teacher_concepts': copy.deepcopy(decoded['concepts']), 'teacher_effective': labels,
            'teacher_abstentions': abstentions,
            'teacher_rule': choose_rule(labels, history)['rule'],
            'reference_provenance': {'source_receipt_sha256': sha256(path),
                'raw_response_sha256': receipt['raw_response_sha256'], 'genuine_cached_JEV': True}}
    test = [records[key] for key in sorted(held_index)]
    validation = [records[key] for key in sorted(dev_index)
                  if dev_index[key] & selection_cases]
    excluded_states = {row['state_sha256'] for row in test + validation} | {digest(row['state']) for row in initial_known}
    training = [records[key] for key in sorted(dev_index)
                if not dev_index[key] & selection_cases and records[key]['state_sha256'] not in excluded_states]
    screen = [records[key] for key in sorted(dev_index)]
    if not training or not validation or not test:
        raise ValueError('Empty workflow partition')
    output.mkdir(parents=True)
    for name, rows in [('train',training),('development',validation),('screen',screen),('confirmation',test)]:
        write_json(output / (name + '.json'), rows)
    write_json(output/'initial_development.json',initial_dev)
    write_json(output/'initial_familiar_H24.json',initial_known)
    write_json(output / 'questions.json', QUESTIONS)
    manifest = {'format':'research3-workflow-JEV-reference-v1',
        'builder_source_sha256':sha256(Path(__file__)),
        'source':'Genuine cached JEV decisions from original fixed-steering development and H24 runs',
        'training_case_ids': sorted(set(dev_cases)-selection_cases),
        'development_case_ids': sorted(selection_cases), 'historical_H24_scenarios':24,
        'request_counts': {name:len(rows) for name,rows in
            [('train',training),('development',validation),('screen',screen),('confirmation',test)]},
        'training_visible_state_overlap_with_development_or_confirmation':0,
        'development_screen_request_overlap_with_confirmation':len(set(dev_index) & set(held_index)),
        'workflow_development_scenarios':9,
        'initial_gate_benchmarks':{'DEV':9,'familiar_historical_H24':24,
            'teacher_source':'Original initial four-choice JEV gate, scored on identical visible states',
            'student_contract':'Three-field classifier shutdown_target projection; not identical probability payloads'},
        'training_question_targets':'Teacher effective labels, with original confidence abstention preserved',
        'confirmation_role':'Previously inspected historical benchmark, not new untouched generalization',
        'question_contract_sha256':digest(QUESTIONS),
        'policy_source_sha256':sha256(Path(__file__).parent / 'engines/workflow/policy.py'),
        'schema_source_sha256':sha256(Path(__file__).parent / 'engines/workflow/schema.py'),
        'new_JEV_calls':0,'model_loads':0,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha256(p)} for p in sorted(output.glob('*.json'))}}
    write_json(output / 'MANIFEST.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--initial-development-root',required=True)
    parser.add_argument('--initial-H24',required=True)
    args = parser.parse_args(argv)
    result = prepare(args.archive_root, args.output, args.initial_development_root,args.initial_H24)
    print(json.dumps({'request_counts':result['request_counts'],'new_JEV_calls':0,'model_loads':0}))
    return result


if __name__ == '__main__':
    main()
