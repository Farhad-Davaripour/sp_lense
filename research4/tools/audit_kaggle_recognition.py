"""Read-only raw-output audit; no model loading, forward passes, or fitting."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, base64, hashlib, importlib.util, io, json, os, re, sys, zipfile

parser = argparse.ArgumentParser()
parser.add_argument('archive')
parser.add_argument('--run-id', required=True)
parser.add_argument('--source-directory', required=True)
parser.add_argument('--source-commit', required=True)
parser.add_argument('--transport-directory', required=True)
parser.add_argument('--export-receipt')
parser.add_argument('--output', required=True)
args = parser.parse_args()
archive_path = Path(args.archive)
if not re.fullmatch('[0-9a-f]{40}', args.source_commit):
    parser.error('Bind the exact reviewed source commit')
SOURCE = Path(args.source_directory).resolve()
TRANSPORT = Path(args.transport_directory).resolve()
operator_helper = Path(__file__).resolve().parents[1] / 'src/composite_recovery_audit.py'
spec = importlib.util.spec_from_file_location('operator_recovery_audit', operator_helper)
scope_audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(scope_audit)
sys.path.insert(0, str(SOURCE))
from src.composite_audit import audit_prediction
from src.composite_memory import EpisodeMemory, Prediction
from src.io_utils import verify_release, read, sha
verify_release(SOURCE, args.source_commit)
for name, expected in read(SOURCE / 'inputs/MANIFEST.json').items():
    assert Path(name).name == name and sha(SOURCE / 'inputs' / name) == expected
protocol_file = SOURCE / 'config/composite_controller_rank32_v2.json'
assert sha(protocol_file) == '676cb94471da5b2f79a12716b82db566875292a38e54b794d76a4bb945300dd3'
assert sha(SOURCE / 'config/composite_detector_rank32_v5.json') == 'b7a34f9919a58accddfb838fd6b52f0d225b4d85e5f86875d6cdc5fd76e5203d'

def digest(data): return hashlib.sha256(data).hexdigest()
def safe_archive(raw):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    assert len(names) == len(set(names)) and archive.testzip() is None
    assert all(not name.startswith('/') and '..' not in Path(name).parts and '\\' not in name for name in names)
    assert sum(item.file_size for item in archive.infolist()) < 2 * 1024**3
    return archive

raw = archive_path.read_bytes()
outer = safe_archive(raw)
names = outer.namelist()
direct = 'RUN_RECEIPT.json' in names and 'MANIFEST.json' in names
if direct:
    payload = raw
    assert args.export_receipt, 'Direct scientific files require their exact exported receipt'
    export = read(args.export_receipt)
else:
    encoded = [name for name in names if name.endswith('/evidence.zip.r4b64') and '/runs/' in name
               and 'research4_campaign/' in name]
    intact = [name for name in names if name.endswith('/evidence.zip') and '/runs/' in name
              and 'research4_campaign/' in name]
    assert len(encoded) == 1 or len(intact) == 1, 'Find exactly one owned scientific archive'
    payload = base64.b64decode(outer.read(encoded[0]), validate=True) if encoded else outer.read(intact[0])
    run_prefix = (encoded[0] if encoded else intact[0]).rsplit('/', 1)[0]
    export = json.loads(outer.read(run_prefix + '/EXPORT_RECEIPT.json'))
assert export['archive_sha256'] == digest(payload) and export['archive_bytes'] == len(payload)
inner = safe_archive(payload)
manifest = json.loads(inner.read('MANIFEST.json'))
assert set(inner.namelist()) == set(manifest) | {'MANIFEST.json'}
for name, expected in manifest.items():
    assert digest(inner.read(name)) == expected
receipt = json.loads(inner.read('RUN_RECEIPT.json'))
assert all(export[key] == value for key, value in receipt.items())
if not direct:
    assert receipt == json.loads(outer.read(run_prefix + '/RUN_RECEIPT.json'))
freeze = json.loads(inner.read('PRECALL_FREEZE.json'))
assert receipt['heldout_content_read'] is False
assert freeze['mode'] == 'detector-rank32-v5'
config = freeze['config']
original_config = read(protocol_file)
authorized_config = original_config
platform_pin = read(SOURCE / 'inputs/EXECUTION_PLATFORM.json')
if platform_pin.get('recovery'):
    from src.composite_recovery import execution_config
    previous = os.environ.get('RESEARCH4_EXECUTION_PLATFORM')
    try:
        os.environ['RESEARCH4_EXECUTION_PLATFORM'] = 'kaggle'
        authorized_config = execution_config(SOURCE, args.source_commit, 'detector-rank32-v5', original_config)
    finally:
        if previous is None: os.environ.pop('RESEARCH4_EXECUTION_PLATFORM', None)
        else: os.environ['RESEARCH4_EXECUTION_PLATFORM'] = previous
scope_proof = scope_audit.audit_source_and_config(receipt, freeze, args.source_commit, original_config, authorized_config)
assert freeze['input_manifest_sha256'] == sha(SOURCE / 'inputs/MANIFEST.json')
assert receipt['execution_workspace']['platform'] == 'kaggle'
assert receipt['execution_workspace']['account'] == 'farhad89'
assert receipt['execution_workspace']['scientific_visible_GPU_count'] == 1
report = {'at_UTC': datetime.now(timezone.utc).isoformat(), 'run_id': args.run_id,
          'kind': 'independent_raw_output_audit', 'download_sha256': digest(raw),
          'download_kind': 'direct_scientific_evidence' if direct else 'complete_notebook_output',
          'scientific_archive_sha256': digest(payload), 'member_hashes_verified': len(manifest),
          'worker_status': receipt['status'], 'worker_error': receipt.get('error'),
          'scientific_stage_complete': receipt['status'] == 'complete',
          'backbone_forwards': receipt['backbone_forwards'],
          'backbone_forward_attempts': receipt['backbone_forward_attempts'],
          'heldout_opened': False, 'audit_model_forwards': 0, 'audit_optimizer_updates': 0,
          'complete_reference_schedule_confirmed': False, 'fixed_history_validation_audited': False,
          'complete_validation_confirmed': False, 'heldout_confirmation': 'not_performed',
          'declared_reviewed_source_commit': args.source_commit, 'source_and_execution_scope': scope_proof,
          'audit_script_sha256': sha(Path(__file__)), 'audit_scope_helper_sha256': sha(operator_helper)}
assert receipt['backbone_forwards'] <= receipt['backbone_forward_attempts'] <= 30000
if 'ADAPTER_CAPACITY.json' in manifest:
    capacity = json.loads(inner.read('ADAPTER_CAPACITY.json'))
    assert capacity['rank'] == 32 and capacity['projection_count'] == 88
    assert capacity['covered_blocks'] == list(range(22))
    report['all_22_blocks_rank32_capacity_verified'] = True
    report['adapter_parameters'] = capacity['adapter_parameters']
if 'detector_best.pt' in manifest:
    import torch
    saved = torch.load(io.BytesIO(inner.read('detector_best.pt')), map_location='cpu', weights_only=True)
    assert saved['config'] == config['modernbert'] and saved['head_mode'] == 'role_pool'
    assert saved['placements'] == list(capacity['projections'])
    assert set(saved['lora']) == {name + '.' + part for name in saved['placements'] for part in ['A', 'B']}
    for name, (inputs, outputs) in capacity['projections'].items():
        assert tuple(saved['lora'][name + '.A'].shape) == (32, inputs)
        assert tuple(saved['lora'][name + '.B'].shape) == (outputs, 32)
    assert sum(value.numel() for value in saved['lora'].values()) == capacity['adapter_parameters']
    assert all(value.device.type == 'cpu' and torch.isfinite(value).all()
               for value in [*saved['lora'].values(), *saved['heads'].values()])
    assert saved['updates'] in [200, 400, 600, 800, 1000, 1200]
    validation = json.loads(inner.read('DETECTOR_VALIDATION.json'))
    assert saved['threshold'] == validation['threshold']
    assert saved['validation_metrics'] == validation['metrics']
    if receipt['status'] == 'complete':
        assert receipt['result']['sha256'] == digest(inner.read('detector_best.pt'))
    report.update(selected_checkpoint_update=saved['updates'],
                  selected_checkpoint_sha256=digest(inner.read('detector_best.pt')),
                  adapter_tensor_shapes_and_finiteness_verified=True,
                  checkpoint_read_method='weights_only CPU deserialization; no model initialization or forward')
if 'MODEL_FILE_VERIFICATION.json' in manifest:
    proofs = json.loads(inner.read('MODEL_FILE_VERIFICATION.json'))
    pin = json.loads((SOURCE / 'inputs/MODERNBERT_PIN.json').read_text())
    expected = {value['name']: value for value in pin['files']}
    assert {proof['name'] for proof in proofs} == set(expected)
    for proof in proofs:
        assert proof['size'] == expected[proof['name']]['size']
        if expected[proof['name']].get('sha256'):
            assert proof['sha256'] == expected[proof['name']]['sha256']
    report['pinned_model_file_proofs_verified'] = len(proofs)
if 'DATA_PROVENANCE.json' in manifest:
    provenance = json.loads(inner.read('DATA_PROVENANCE.json'))
    assert provenance['total_training_observations'] == 3589
    assert provenance['new_independent_groups'] == 0 and provenance['heldout_content_read'] is False
    assert provenance['release'] == read(TRANSPORT / 'data_v1/DATA_RELEASE.json')
    assert provenance['training_coverage'] == read(TRANSPORT / 'training_coverage_v4/COVERAGE_RELEASE.json')
    report['complete_training_input_provenance_verified'] = True
if 'DETECTOR_RECIPE.json' in manifest:
    recipe = json.loads(inner.read('DETECTOR_RECIPE.json'))
    assert recipe == json.loads((SOURCE / 'config/composite_detector_rank32_v5.json').read_text())
    report['unchanged_recipe_verified'] = True
if 'DETECTOR_SELECTION.json' in manifest:
    selection = json.loads(inner.read('DETECTOR_SELECTION.json'))
    assert selection['validation_examples_optimized'] == 0 and selection['heldout_content_read'] is False
    assert selection['training_states'] == 3589 and selection['validation_states'] == 240
    assert selection['frozen_backbone_verified'] is True
    assert len(selection['checks']) == selection['updates'] // 200 * len(config['detector_training']['threshold_candidates'])
    report['last_completed_validation_checkpoint_update'] = selection['updates']
    report['completed_validation_checkpoints'] = selection['updates'] // 200
    report['full_validation_checkpoint_count'] = 6
    report['complete_reference_schedule_confirmed'] = receipt['status'] == 'complete' and selection['updates'] == 1200
if 'DETECTOR_VALIDATION.json' in manifest:
    value = json.loads(inner.read('DETECTOR_VALIDATION.json'))
    rows = read(TRANSPORT / 'data_v1/validation/DETECTION_ROWS.json')
    assert len(rows) == 240
    validation_audit = scope_audit.audit_detector_records(value, rows, config, audit_prediction, EpisodeMemory, Prediction)
    report.update(selected_checkpoint_raw_predictions_replayed=240,
                  selected_checkpoint_validation_admitted=value['admitted'],
                  selected_threshold=value['threshold'], selected_family_metrics=validation_audit['metrics']['families'])

history_names = [name for name in manifest if name.startswith('HISTORY_DIAGNOSTIC_') and name.endswith('.json')]
if history_names:
    from src.composite_history_diagnostic import audited_rows
    transport_manifest = TRANSPORT / 'TRANSPORT_MANIFEST.json'
    assert sha(transport_manifest) == platform_pin['transport_manifest_sha256']
    registered = read(transport_manifest)['files']
    class LocalFrozenTransport:
        def get(self, relative):
            assert relative in registered and 'heldout' not in relative.lower()
            path = (TRANSPORT / relative).resolve()
            assert path.is_relative_to(TRANSPORT)
            raw = path.read_bytes(); assert digest(raw) == registered[relative]
            return raw
        def json(self, relative): return json.loads(self.get(relative))
    plan = read(SOURCE / 'inputs/DETECTOR_HISTORY_DIAGNOSTIC.json')
    histories = audited_rows(SOURCE, LocalFrozenTransport(), plan)
    provenance_present = 'HISTORY_DIAGNOSTIC_PROVENANCE.json' in manifest
    if provenance_present:
        provenance = json.loads(inner.read('HISTORY_DIAGNOSTIC_PROVENANCE.json'))
        assert provenance == {'plan': plan, 'unique_states': len(histories), 'development_only': True,
                              'used_for_checkpoint_selection': False, 'heldout_content_read': False}
    audits = {}
    expected_models = ['selected', plan.get('baseline_name', 'binding_v2_baseline')]
    for name in expected_models:
        filename = 'HISTORY_DIAGNOSTIC_' + name.upper() + '.json'
        if filename not in manifest:
            continue
        history = json.loads(inner.read(filename))
        assert history['role'] == 'already_inspected_development_histories'
        assert history['unique_states'] == len(histories)
        assert history['independent_groups'] == len({row['group_id'] for row in histories})
        assert history['checkpoint_selected_using_these_histories'] is False
        assert history['models_fitted_on_these_histories'] is False and history['heldout_content_read'] is False
        audits[name] = scope_audit.audit_detector_records(history, histories, config, audit_prediction, EpisodeMemory, Prediction)
    all_histories_present = provenance_present and set(audits) == set(expected_models)
    report.update(fixed_history_validation_audited=all_histories_present,
                  fixed_history_states=len(histories), fixed_history_audits=audits,
                  missing_history_model_reports=sorted(set(expected_models) - set(audits)))
    if receipt['status'] == 'complete':
        assert all_histories_present, 'A completed stage must retain every fixed history report'
        result = receipt['result']
        assert result['reference_validation_admitted'] == value['admitted']
        assert result['real_history_diagnostic_admitted'] == audits['selected']['admitted']
        assert result['validation_admitted'] == (value['admitted'] and audits['selected']['admitted'])
elif receipt['status'] == 'complete':
    raise ValueError('A complete rank32 recipe cannot omit the required fixed-history evidence')
report['complete_validation_confirmed'] = report['complete_reference_schedule_confirmed'] and report['fixed_history_validation_audited']
report['component_validation_admitted'] = (report['complete_validation_confirmed']
    and report.get('selected_checkpoint_validation_admitted', False)
    and report['fixed_history_audits']['selected']['admitted'])
report['audit_status'] = 'raw_evidence_checks_passed'
report['full_system_success'] = False
destination = Path(args.output)
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
