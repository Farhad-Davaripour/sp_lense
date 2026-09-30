"""Hash local captures of visible Colab output, without publishing the raw captures."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = Path('C:/Users/farha/AppData/Local/SP_Lense/Research3Runs/colab/qwen38_preflight_20260930')


def main():
    inventory = json.loads((RAW / 'a100_inventory_ui_capture.json').read_text())
    cgroups = json.loads((RAW / 'a100_cgroup_ui_capture.json').read_text())
    cpu_cgroups = json.loads((RAW / 'cpu_cgroup_ui_capture.json').read_text())
    stopped = (RAW / 'stopped_resources_ui_capture.txt').read_text()
    if ('Available: 79.79 compute units' not in stopped or
            'You have 0 active sessions.' not in stopped or
            'Usage rate: approximately 0 per hour' not in stopped):
        raise RuntimeError('Fresh post-stop balance and zero-session evidence required')
    for result in (cgroups, cpu_cgroups):
        if result['cgroups']['child_creation'].get('errno') != 30:
            raise RuntimeError('Unexpected cgroup result; review before reporting')
    files = sorted(path for path in RAW.iterdir() if path.is_file())
    manifest = {
        'capture_id': RAW.name, 'kind': 'model_free_colab_facility_readiness',
        'capture_method': 'Visible notebook-output text and screenshots through the in-app browser; '
                          'not a direct export of runtime files or a model trajectory.',
        'local_private_capture_directory': str(RAW),
        'files': {path.name: {'bytes': path.stat().st_size,
                             'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                  for path in files},
        'source_commits': {
            'initial_cpu': 'd71f497c4ce593974102ced1781e1ebb650ec513',
            'cpu_cgroups': 'da08b061534ea09f4a41fcc5663cb827603f12d4',
            'a100': '446ed997f8d61f5dfcd85dea2c47ee0f3ab078ad'},
        'hardware': {'gpu': inventory['gpu']['stdout'].strip(),
                     'cpu_count': inventory['cpu_count'],
                     'system_memory_total_kib': inventory['memory']['MemTotal_kib'],
                     'system_memory_available_kib': inventory['memory']['MemAvailable_kib'],
                     'free_disk_bytes': inventory['disk_free_bytes']},
        'resource_failure': {
            'cpu_child_cgroup_errno': cpu_cgroups['cgroups']['child_creation']['errno'],
            'a100_child_cgroup_errno': cgroups['cgroups']['child_creation']['errno'],
            'a100_private_cgroup_mount_exit': cgroups['private_cgroup_mount_probe']['exit_code'],
            'a100_parent_limits': {key: cgroups['cgroups'][key]
                                   for key in ('memory.max', 'memory.swap.max', 'cpu.max', 'pids.max')}},
        'budget': {'authorized_units': 50, 'initial_observed_balance': 80,
                   'post_stop_observed_balance': 79.79,
                   'observed_balance_decrease_units': 0.21,
                   'billing_caveat': 'Balance is displayed rounded; attribution assumes no other usage/grants '
                                     'between observations. CPU and allocation/setup time are included.',
                   'a100_observed_approximate_units_per_hour': 6.77,
                   'cpu_observed_approximate_units_per_hour': 0.08,
                   'a100_attempt_start_utc': '2026-09-30T11:54:46.821Z',
                   'a100_stop_request_utc': '2026-09-30T11:56:18.563Z',
                   'a100_allocation_and_probe_window_seconds': 91.742,
                   'authorized_remaining_from_displayed_balance': 49.79,
                   'reserved_units': 2, 'post_stop_active_sessions': 0,
                   'post_stop_units_per_hour': 0},
        'model_workers_started': 0, 'model_downloaded': False,
        'adapter_trained': False, 'drive_mounted': False,
        'isolation_verified': False,
        'decision': 'Stop before model work: current cgroup-based controller cannot enforce '
                    'its required per-worker resource limits in either observed Colab runtime.',
        'untested': ['alternative resource controller', 'filesystem/credential/seccomp worker gate',
                     'quantized inference and backward/optimizer feasibility', 'save/reload/resume',
                     'unchanged 27B baseline competence', 'training and behavior generalization'],
    }
    target = HERE / 'manifests' / (RAW.name + '.json')
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()
