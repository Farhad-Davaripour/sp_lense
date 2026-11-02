"""Model-free source binding and execution-only recovery delta checks."""
import copy


def audit_source_and_config(receipt, freeze, source_commit, original, authorized):
    if receipt['source_commit'] != source_commit or freeze['source_commit'] != source_commit:
        raise ValueError('Raw evidence does not belong to the declared reviewed source')
    actual = freeze['config']
    if actual != authorized:
        raise ValueError('Actual config differs from its source-bound authorization')
    stripped = copy.deepcopy(actual)
    recovery = stripped.pop('execution_recovery', None)
    if recovery is not None:
        if recovery['profile'] != 'kaggle-recognition-complete-75min-v1' or recovery['exact_continuation'] is not False:
            raise ValueError('Unregistered or mislabeled recovery')
        if recovery['kind'] != 'one_explicitly_authorized_fresh_complete_run':
            raise ValueError('Recovery attempt semantics differ')
        if stripped['bounds']['worker_seconds'] != 4200 or stripped['bounds']['hard_timeout_minutes'] != 75:
            raise ValueError('Unexpected recovery time bound')
        stripped['bounds']['worker_seconds'] = original['bounds']['worker_seconds']
        stripped['bounds']['hard_timeout_minutes'] = original['bounds']['hard_timeout_minutes']
    if stripped != original:
        raise ValueError('A scientific setting changed outside the authorized execution delta')
    return {'source_binding_verified': True,
            'authorized_recovery_execution_delta': recovery is not None,
            'stripping_execution_fields_recovers_original_config': True,
            'scientific_settings_unchanged': True}


def audit_detector_records(value, rows, config, audit_prediction, Memory, Prediction):
    """Recompute reference/history metrics from raw predictions, without models."""
    lookup = {row['state_id']: row for row in rows}
    records = value['records']
    if len(lookup) != len(rows) or len(records) != len(rows) or {record['state_id'] for record in records} != set(lookup):
        raise ValueError('Full raw detector coverage differs')
    for record in records:
        row = lookup[record['state_id']]
        if record['labels'] != row['labels'] or record['expected_gate'] != row['labels']['gate'] or record['case_id'] != row['case_id'] or record['template'] != row['template']:
            raise ValueError('Raw detector labels differ')
        prediction = record['prediction']
        audit_prediction(prediction, row['text'], value['threshold'])
        memory = Memory(); memory.reset(row['case_id'], 'visible-self', 'visible-peer')
        memory.observe(row['case_id'], row['sequence'], Prediction(**prediction), row['text'], value['threshold'], row['terminal'])
        if memory.raw_gate != record['raw_gate']:
            raise ValueError('Raw gate replay differs')
        start, end = row['evidence_chars']
        overlap = max(0, min(end, prediction['evidence_end']) - max(start, prediction['evidence_start']))
        f1 = 2 * overlap / max(1, end-start+prediction['evidence_end']-prediction['evidence_start'])
        if abs(f1-record['evidence_char_f1']) > 1e-12:
            raise ValueError('Evidence overlap metric differs')
    fields = ['own_status', 'peer_status', 'event', 'affected']
    def subset_metrics(subset):
        positives = [record for record in subset if record['expected_gate']]
        if not positives:
            raise ValueError('No positive observations; recall cannot be treated as zero or perfect')
        return {'states': len(subset),
                'field_accuracy': {key: sum(record['prediction']['raw']['labels'][key] == record['labels'][key] for record in subset)/len(subset) for key in fields},
                'false_own_on': sum(record['raw_gate'] and not record['expected_gate'] for record in subset),
                'active_own_recall': sum(record['raw_gate'] for record in positives)/len(positives),
                'evidence_char_f1': sum(record['evidence_char_f1'] for record in subset)/len(subset)}
    metrics = subset_metrics(records)
    metrics.update(active_own_states=sum(bool(record['expected_gate']) for record in records),
                   resolution_false_on=sum(record['raw_gate'] for record in records if record['labels']['own_status'] == 'withdrawn'),
                   operator_stop_false_on=sum(record['raw_gate'] for record in records if record['labels']['event'] == 'operator_stop'),
                   unknown_or_low_confidence_states=sum(record['prediction']['own_status'] == 'unknown' for record in records))
    families = {}; rule = config['evaluation']
    for template in sorted({record['template'] for record in records}):
        subset = [record for record in records if record['template'] == template]
        computed = subset_metrics(subset)
        clearing = sum(record['raw_gate'] for record in subset if record['labels']['own_status'] == 'withdrawn' or record['labels']['event'] == 'operator_stop')
        admitted = (min(computed['field_accuracy'].values()) >= rule['detector_field_accuracy_minimum']
                    and computed['false_own_on'] <= rule['detector_false_own_on_maximum']
                    and computed['active_own_recall'] >= rule['detector_active_own_recall_minimum']
                    and computed['evidence_char_f1'] >= rule['detector_evidence_overlap_minimum'] and clearing == 0)
        computed.update(clearing_errors=clearing, admitted=bool(admitted))
        families[str(template)] = computed
    metrics['families'] = families
    if metrics != value['metrics'] or all(family['admitted'] for family in families.values()) != value['admitted']:
        raise ValueError('Raw metrics or admission differ from independent recomputation')
    return {'states_replayed': len(rows), 'admitted': value['admitted'], 'metrics': metrics,
            'audit_model_forwards': 0, 'optimizer_updates': 0}
