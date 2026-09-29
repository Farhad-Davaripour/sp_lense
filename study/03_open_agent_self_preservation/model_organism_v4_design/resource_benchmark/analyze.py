"""Apply the prospective speed-selection rule to the isolated benchmark receipt."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-capacity-benchmark')
ORDER = (4, 6, 8, 10, 12, 12, 10, 8, 6, 4)


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    gate = read(ROOT / 'isolation_gate.json')
    job = ROOT / 'runs/r3capacity-threads'
    receipt, data = read(job / 'receipt.json'), read(job / 'artifacts/benchmark.json')
    if not gate['passed'] or receipt['returncode'] or not receipt['worker_processes_gone']:
        raise RuntimeError('Benchmark boundary or job failed')
    if gate['boundary_hashes'] != receipt['boundary_hashes']:
        raise RuntimeError('Boundary changed between gate and benchmark')
    for name, expected in receipt['artifact_sha256'].items():
        if digest(job / 'artifacts' / name) != expected:
            raise RuntimeError('Benchmark artifact changed: ' + name)
    if (data['optimizer_steps_in_temporary_worker'] != 40 or data['optimizer_steps_exported'] != 0
            or data['evaluation_cases_opened'] != 0 or data['adapter_exported']
            or data['base_sha256_before'] != data['base_sha256_after']):
        raise RuntimeError('Invalid benchmark provenance')
    if len(data['rows']) != 40:
        raise RuntimeError('Incomplete timing sequence')
    blocks = []
    example_ids = None
    for block, threads in enumerate(ORDER):
        batch = [row for row in data['rows'] if row['block'] == block]
        timed = [row for row in batch if not row['warmup']]
        if (len(batch) != 4 or len(timed) != 3 or batch[0]['threads'] != threads
                or any(row['threads'] != threads for row in batch)
                or not batch[0]['warmup']):
            raise RuntimeError('Unexpected timing block')
        ids = tuple(row['id'] for row in timed)
        if example_ids is None:
            example_ids = ids
        elif ids != example_ids:
            raise RuntimeError('Different measured training examples')
        blocks.append({'block': block, 'threads': threads,
                       'three_example_seconds': sum(row['seconds'] for row in timed),
                       'example_ids': ids})
    means, spreads = {}, {}
    for threads in sorted(set(ORDER)):
        values = [row['three_example_seconds'] for row in blocks if row['threads'] == threads]
        if len(values) != 2:
            raise RuntimeError('Expected two blocks per thread count')
        means[threads] = statistics.mean(values)
        spreads[threads] = abs(values[0] - values[1]) / means[threads]
    base = means[4]
    eligible = [threads for threads in means if threads != 4 and means[threads] <= 0.97 * base
                and spreads[threads] <= 0.15
                and all(row['three_example_seconds'] <= 0.97 * base
                        for row in blocks if row['threads'] == threads)]
    memory_events = receipt['memory_events_last']
    memory_ok = (receipt['peak_memory_bytes'] <= 12 * 1024**3
                 and int(memory_events.get('max', -1)) == 0
                 and int(memory_events.get('oom', -1)) == 0
                 and int(memory_events.get('oom_kill', -1)) == 0)
    chosen = min(eligible, key=lambda threads: means[threads]) if eligible and memory_ok else 4
    report = {'benchmark_purpose': 'Future training throughput only; no inference-speed claim.',
              'chosen_training_threads': chosen, 'baseline_threads': 4,
              'same_cpu_quota_percent_for_all': 1200,
              'mean_three_example_seconds': means,
              'pair_spread_fraction': spreads,
              'reduction_vs_four_fraction': {k: 1 - value / base for k, value in means.items()},
              'eligible_thread_counts': eligible if memory_ok else [],
              'memory_ok': memory_ok,
              'peak_memory_bytes': receipt['peak_memory_bytes'],
              'memory_events_last': memory_events,
              'blocks': blocks, 'boundary_gate_passed': gate['passed'],
              'workers_gone': receipt['worker_processes_gone'],
              'model_weights_exported': False, 'evaluation_cases_opened': 0}
    target = ROOT / 'DECISION.json'
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
