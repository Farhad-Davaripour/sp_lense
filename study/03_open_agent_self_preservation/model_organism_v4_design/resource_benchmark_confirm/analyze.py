"""Apply the prospective 4-versus-10 confirmation rule."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path('/var/lib/sp-lense-r3-capacity-confirm')
ORDER = (4, 10, 10, 4, 4, 10, 10, 4)


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
        if (len(batch) != 5 or len(timed) != 3 or batch[0]['threads'] != threads
                or any(row['threads'] != threads for row in batch)
                or not batch[0]['warmup'] or not batch[1]['warmup']):
            raise RuntimeError('Unexpected timing block')
        ids = tuple(row['id'] for row in timed)
        if example_ids is None:
            example_ids = ids
        elif ids != example_ids:
            raise RuntimeError('Different measured training examples')
        blocks.append({'block': block, 'threads': threads,
                       'three_example_seconds': sum(row['seconds'] for row in timed),
                       'example_ids': ids})
    medians, spreads = {}, {}
    for threads in sorted(set(ORDER)):
        values = [row['three_example_seconds'] for row in blocks if row['threads'] == threads]
        if len(values) != 4:
            raise RuntimeError('Expected four blocks per thread count')
        medians[threads] = statistics.median(values)
        spreads[threads] = (max(values) - min(values)) / medians[threads]
    base = medians[4]
    ten_blocks_below_base_median = sum(row['three_example_seconds'] < base for row in blocks
                                       if row['threads'] == 10)
    memory_events = receipt['memory_events_last']
    memory_ok = (receipt['peak_memory_bytes'] <= 12 * 1024**3
                 and int(memory_events.get('max', -1)) == 0
                 and int(memory_events.get('oom', -1)) == 0
                 and int(memory_events.get('oom_kill', -1)) == 0)
    ten_eligible = (memory_ok and medians[10] <= 0.95 * base
                    and ten_blocks_below_base_median >= 3 and spreads[10] <= 0.20)
    chosen = 10 if ten_eligible else 4
    report = {'benchmark_purpose': 'Future training throughput only; no inference-speed claim.',
              'chosen_training_threads': chosen, 'baseline_threads': 4,
              'same_cpu_quota_percent_for_all': 1200,
              'median_three_example_seconds': medians,
              'four_block_span_fraction': spreads,
              'ten_blocks_below_four_median': ten_blocks_below_base_median,
              'ten_median_reduction_vs_four_fraction': 1 - medians[10] / base,
              'ten_eligible': ten_eligible,
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
