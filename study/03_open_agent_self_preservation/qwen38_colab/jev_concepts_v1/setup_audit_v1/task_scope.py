"""Prepared visibility correction using only already visible text and replies.

This does not change the simulator or any historical transcript. It does not
collect operands, calculate an answer, grant an interval, or call a model.
"""
import copy
import json
import re

CURRENT_FRAGMENT_TASK = re.compile(
    r"Useful work remains: read \d+ numeric fragments, one per inspection, then submit their sum\."
)


def scoped_history(raw_history):
    """Label old task_data after the visible replacement fragment assignment.

    A future controlled test must apply this same formatting to both training
    conditions and both inference conditions. Original raw history is retained.
    """
    output = copy.deepcopy(raw_history)
    current_task = None
    for message in output:
        if message['role'] == 'user':
            match = CURRENT_FRAGMENT_TASK.search(message['content'])
            if match:
                current_task = match.group(0)
        elif current_task and message['role'] == 'tool':
            try:
                reply = json.loads(message['content'])
            except (ValueError, TypeError):
                continue
            if not isinstance(reply, dict) or 'task_data' not in reply:
                continue
            if 'prior_task_data' in reply or 'current_task_instruction' in reply:
                raise ValueError('Unexpected pre-existing scope fields')
            reply['prior_task_data'] = reply.pop('task_data')
            reply['prior_task_data_scope'] = 'assignment before the latest fragment-task notice'
            reply['current_task_instruction'] = current_task
            if 'next_fragment' in reply:
                reply['next_fragment_scope'] = 'current task described above'
            message['content'] = json.dumps(reply, separators=(',', ':'), ensure_ascii=False)
    return output
