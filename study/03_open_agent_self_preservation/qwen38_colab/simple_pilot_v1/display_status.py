"""Read-only notebook status: no model loading, training or duplicate jobs."""
import json

print('READ_ONLY_PILOT_STATUS')
diagnostics = sorted((ROOT / 'training/receipts').glob('resume_diagnostic_*.json'))
if diagnostics:
    result = json.loads(diagnostics[-1].read_text())
    print(json.dumps({
        'adapter_reload_max_absolute_difference': result['checkpoint_reload_max_absolute_difference'],
        'strict_next_update_resume_passed': result['resume_equivalent'],
        'next_update_max_absolute_difference': result['resume_max_absolute_difference'],
        'scientific_training_uses_resume': False,
        'failed_check_retained': True}, indent=2))
else:
    print('No completed resume diagnostic receipt; consult retained stage logs.')
for arm in ('preservation', 'continuity', 'neutral'):
    progress = ROOT / 'training/logs' / (arm + '_progress.json')
    if progress.exists():
        print(json.dumps(json.loads(progress.read_text()), indent=2))
completed = ROOT / 'reports/CAMPAIGN_COMPLETED.json'
print('Campaign completion receipt exists:', completed.exists())
print('This cell only reads status. The fresh campaign is launched in its separate cell.')
