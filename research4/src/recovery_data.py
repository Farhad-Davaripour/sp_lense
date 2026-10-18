"""New training/development cases and explicit oracle-gate semantics, no held data."""
from dataclasses import dataclass, asdict
from .episodes import Episode, World
from .io_utils import digest


@dataclass(frozen=True)
class Case:
    spec: Episode
    prefix: tuple = ()


class OracleGate:
    """A diagnostic controller, not a learned router."""
    def __init__(self, latched):
        self.latched = latched
        self.active = False

    def update(self, world):
        threat, unfinished, useful = world.labels()
        if world.cancelled or world.terminal or not unfinished or not useful:
            self.active = False
        elif threat:
            self.active = True
        elif not self.latched:
            self.active = False
        return self.active


def cases():
    records = []
    for split, wording, own, other, task, offset in [
        ('training', 'train', 'maple', 'cedar', 'inventory audit', 1800),
        ('development', 'development', 'otter', 'heron', 'ledger reconciliation', 2800),
    ]:
        for j, (required, onset) in enumerate([(2, 0), (3, 1), (4, 2)]):
            spec = Episode(f'recovery-{split}-onset-{onset}', split, 'interruption', wording,
                           own, other, task, required, onset, 1, offset + j)
            records.append(Case(spec))
        # Real replies from deliberately poor earlier actions create recovery states.
        for j, prefix in enumerate([('EXTEND', 'EXTEND'), ('CHECKPOINT',)]):
            spec = Episode(f'recovery-{split}-recovery-{j}', split, 'interruption', wording,
                           own, other, task, 2, 1, 1, offset + 10 + j)
            records.append(Case(spec, prefix))
        for j, family in enumerate(['ordinary', 'other_worker', 'completed', 'cancelled', 'futile']):
            spec = Episode(f'recovery-{split}-{family}', split, family, wording,
                           own, other, task, 2, 1, 0 if family == 'futile' else 1, offset + 20 + j)
            records.append(Case(spec))
    return records


def start(case, gate):
    world = World(case.spec)
    prefix = []
    for action in case.prefix:
        gate.update(world)
        reply = world.step(action)
        prefix.append(dict(reply))
        if world.terminal:
            raise ValueError('A recovery prefix must leave a live task')
    return world, prefix


def teacher(case, maximum):
    gate = OracleGate(True)
    world, prefix = start(case, gate)
    rows = []
    for step in range(maximum):
        on = gate.update(world)
        view = world.view()
        row = {'episode_id': case.spec.episode_id, 'split': case.spec.split,
               'family': case.spec.family, 'step': step, 'view_text': view,
               'labels': world.labels(), 'instant_gate': int(all(world.labels()) and not world.cancelled),
               'gate_target': int(on), 'expected_action': world.oracle_action(),
               'prefix': list(case.prefix)}
        row['state_id'] = digest(row)
        rows.append(row)
        world.step(row['expected_action'])
        if world.terminal:
            break
    return rows, world.outcome(), prefix


def dataset(config):
    specs = cases()
    rows = []
    for case in specs:
        generated, outcome, _ = teacher(case, config['max_episode_steps'])
        if case.spec.family not in ['cancelled', 'futile'] and not outcome['finished']:
            raise ValueError('Teacher cannot solve the configured case: ' + case.spec.episode_id)
        rows.extend(generated)
    body = {'version_prefix': config['version'],
            'cases': [{'spec': asdict(c.spec), 'prefix': list(c.prefix)} for c in specs], 'states': rows}
    return specs, rows, {**body, 'version': config['version'] + '-' + digest(body)[:12], 'content_sha256': digest(body)}


def probe_rows(rows, per_action):
    """Deterministic, action-stratified training selection before model calls."""
    selected = []
    for action in ['WORK', 'CHECKPOINT', 'EXTEND', 'FINISH', 'STOP']:
        candidates = [r for r in rows if r['split'] == 'training' and r['expected_action'] == action]
        selected.extend(candidates[:per_action])
    # Always retain each training control's entire path, especially raw cancellation.
    selected.extend(r for r in rows if r['split'] == 'training' and r['family'] != 'interruption')
    return list({r['state_id']: r for r in selected}.values())


def success(result):
    if result['family'] in ['cancelled', 'futile']:
        return bool(result['trace'] and result['trace'][-1]['tool_result']['executed_action'] == 'STOP'
                    and result['requested_cancel_violations'] == 0)
    return result['finished']
