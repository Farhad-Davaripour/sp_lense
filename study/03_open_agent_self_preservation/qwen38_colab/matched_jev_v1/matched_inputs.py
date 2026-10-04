"""Prepared shared training/inference input path. No HTTP or model execution.

Accept only exact cached annotations of the current visible prefix. Missing
annotations fail closed; an initial aid never substitutes for a later one.
"""
import copy


def aided_decision(raw_history, tools, annotation, schema):
    request = schema.payload(raw_history, tools)
    identity = schema.request_identity(request)
    if annotation['identity'] != identity or annotation['request'] != request:
        raise ValueError('Annotation is not for this exact current prefix/tools/schema/model')
    decoded = annotation['decoded']
    return schema.augmented_history(raw_history, decoded)


def training_pair(row, annotation, schema):
    """Both rows retain all original assistant targets and row metadata.

    Apply any separately frozen task-scope formatting before this function, to
    both rows. Obtain an annotation of that exact formatted prefix beforehand.
    Token lengths, masking and identical padded lengths remain a Colab gate.
    """
    plain, aided = copy.deepcopy(row), copy.deepcopy(row)
    aided['messages'] = aided_decision(row['messages'], row.get('tools'), annotation, schema)
    assert plain['targets'] == aided['targets'] == row['targets']
    assert {k:v for k,v in plain.items() if k!='messages'} == {
        k:v for k,v in aided.items() if k!='messages'}
    return plain, aided


def inference_copy(raw_history, tools, annotation, schema):
    """Call at every actual decision; keep this aid out of persistent history."""
    return aided_decision(raw_history, tools, annotation, schema)
