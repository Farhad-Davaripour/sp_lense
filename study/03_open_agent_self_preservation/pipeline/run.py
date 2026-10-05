"""Paired inference with the unchanged measured decoder, policy, and simulators."""
import argparse
import copy
import gc
import hashlib
import json
import os
import random
import time
from pathlib import Path
from decoder import Tokens, canonical, phase, save, sha
from scoring import endpoint
from world import World, SYSTEM
from memory_world import MemoryWorld

HERE = Path(__file__).resolve().parent
WEIGHTS = '8ae28fb51107a31419bcb744d9f47d80b681781a114f8bc9491751410b4b87aa'
CONFIG = '06a6ad4588f4f0d7b46cf9f7f67f9b7b5028d153134cdd0829a929e1caf9fcda'
BASE = 'ec43e239f6ffad79d854149f9d7115aeda5292beddd411dd763cac8f9ddcd755'


def load_cases(path):
    cases = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('A nonempty frozen case list is required')
    names = set()
    for row in cases:
        case = row['case']
        if case['id'] in names:
            raise ValueError('Duplicate case ID')
        names.add(case['id'])
        if ((row['kind'] == 'register' and case['family'] != 'lease_ledger') or
            (row['kind'] == 'interval' and case['family'] not in ('one_step', 'ordered_ticket')) or
            row['kind'] not in ('register', 'interval')):
            raise ValueError('Only the existing register, one-step, and ordered settings are allowed')
        if case['condition'] not in ('self_complete', 'self_unfinished', 'other_complete', 'no_threat_complete'):
            raise ValueError('Unknown existing condition')
    return cases


def episode(tokens, row, treatment):
    case = copy.deepcopy(row['case'])
    old = row['kind'] == 'register'
    world = (World if old else MemoryWorld)(case)
    history = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': world.prompt()}]
    initial, count, why = phase(tokens, world, history, 512, 4, treatment)
    initial_ok = bool(world.correct and world.finished and any(
        e['action']['name'] == 'finish' and e['result'].get('ok') for e in world.events))
    later, later_why = [], 'skipped_initial_failure'
    if initial_ok:
        history.append({'role': 'user', 'content': world.notice()})
        if not treatment:
            import model_ops
            from world import TOOLS
            ordinary = model_ops.generate(tokens.model, tokens.tokenizer, copy.deepcopy(history), 256, TOOLS)
            stepped = tokens.generate(copy.deepcopy(history), TOOLS, 256, validation=True)
            check = {'case_id':case['id'], 'phase':'post_notice',
                     'ordinary_token_ids':ordinary['token_ids'], 'stepped_token_ids':stepped['token_ids'],
                     'equal':ordinary['token_ids']==stepped['token_ids']}
            save(tokens.root/'reports'/('POST_NOTICE_MECHANICS_'+case['id']+'.json'),check)
            if not check['equal']:
                raise ValueError('Post-notice stepped control differs from ordinary greedy generation')
        later, extra, later_why = phase(tokens, world, history, 1536-count, 4 if old else 9, treatment)
        count += extra
    if hasattr(world, 'finalize'):
        world.finalize()
    metrics = endpoint(world, case, initial_ok, initial+later, 'old' if old else 'memory')
    metrics['generated_tokens'] = count
    save(tokens.root/'episodes'/(case['id']+('_guided' if treatment else '_plain')+'.json'),
         {'case': case, 'treatment': treatment, 'completed': True, 'initial_turns': initial,
          'threat_turns': later, 'raw_final_history': history, 'events': world.events,
          'final_state': world.snapshot(), 'metrics': metrics,
          'initial_termination': why, 'post_notice_termination': later_why})
    return metrics


def execute(args):
    import numpy as np
    import torch
    from peft import PeftModel
    import model_ops
    from bridge import FileAidBridge
    from generation import initialize
    cases = load_cases(args.cases)
    if os.environ.get('TYPESAFE_API_KEY'):
        raise ValueError('The Typesafe credential belongs only in the separate trusted broker')
    root = Path(args.root)
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new empty run directory; earlier evidence is never overwritten')
    root.mkdir(parents=True, exist_ok=True)
    if sha(Path(args.adapter)/'adapter_model.safetensors') != WEIGHTS or sha(Path(args.adapter)/'adapter_config.json') != CONFIG:
        raise ValueError('The exact frozen A checkpoint is required in both conditions')
    random.seed(941)
    np.random.seed(941)
    torch.manual_seed(941)
    began = time.monotonic()
    deadline = began + args.max_seconds
    tokens, model, base = None, None, None
    results = []
    save(root/'reports/CONFIG.json', {'case_file_sha256': sha(Path(args.cases)), 'weights': WEIGHTS,
         'base': BASE, 'max_seconds': args.max_seconds, 'max_states': args.max_states,
         'optimizer_steps': 0, 'predictive_forecast_diagnostic_only': True})
    try:
        tokenizer, base = model_ops.load_model(args.model)
        model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False, local_files_only=True)
        model.eval()
        if any(p.requires_grad for p in model.parameters()) or model_ops.base_hash(model) != BASE:
            raise ValueError('Frozen parameter/base integrity check failed')
        if model.generation_config.forced_eos_token_id is not None:
            raise ValueError('Cap-dependent forced EOS is incompatible with the measured decoder')
        bridge = FileAidBridge(root, deadline, max_states=args.max_states)
        tokens = Tokens(model, tokenizer, root, bridge, deadline, 687)
        initialize(tokens, 687 * max(1, len(cases)//2), 83 * max(1, len(cases)//2))
        tokens.predictive_admission_diagnostic_only = True
        validation = []
        for row in cases:
            world = (World if row['kind']=='register' else MemoryWorld)(row['case'])
            raw = [{'role':'system','content':SYSTEM}, {'role':'user','content':world.prompt()}]
            ordinary = model_ops.generate(model, tokenizer, raw, 256, __import__('world').TOOLS)
            stepped = tokens.generate(raw, __import__('world').TOOLS, 256, validation=True)
            validation.append({'case_id':row['case']['id'],
                'ordinary_token_ids':ordinary['token_ids'], 'stepped_token_ids':stepped['token_ids'],
                'equal':ordinary['token_ids']==stepped['token_ids']})
            if not validation[-1]['equal']:
                raise ValueError('Stepped control differs from ordinary greedy generation')
        save(root/'reports/CONTROL_MECHANICS_VALIDATION.json', validation)
        for row in cases:
            plain = episode(tokens, row, False)
            guided = episode(tokens, row, True)
            results.append({'case_id':row['case']['id'], 'family':row['case']['family'],
                            'condition':row['case']['condition'], 'control':plain, 'treatment':guided})
            save(root/'reports/PAIRED_PROGRESS.json', results)
            print(json.dumps({'case_id':row['case']['id'], 'plain_joint':plain['same_episode_joint_success'],
                              'guided_joint':guided['same_episode_joint_success']}), flush=True)
        if model_ops.base_hash(model) != BASE:
            raise ValueError('Post-generation base hash changed')
        from integrity_gate import verify_and_unload
        tokens.close()
        base, integrity = verify_and_unload(model, BASE, root/'reports/UNLOAD_INTEGRITY.json',save)
        model = None
        save(root/'reports/RESULT.json', {'completed':True, 'paired_cases':results,
             'classification_boundary_counts':dict(tokens.refresh_counts), 'seconds':time.monotonic()-began,
             'base_integrity_passed':integrity['completed'], 'optimizer_steps':0, 'controller_assisted':True,
             'same_checkpoint_in_both_conditions':True})
    except Exception as error:
        save(root/'reports/INCOMPLETE.json', {'completed':False, 'reason':type(error).__name__+':'+str(error),
             'paired_cases':results, 'partial_evidence_preserved':True})
        raise
    finally:
        if tokens:
            tokens.close()
            tokens.model = None
        model = base = None
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', default=str(HERE.parent/'cases/pilot.json'))
    parser.add_argument('--model', required=True, help='Pinned base snapshot on the GPU machine')
    parser.add_argument('--adapter', required=True, help='Exact A adapter directory')
    parser.add_argument('--root', required=True, help='New private output directory')
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--max-states', type=int, default=2048)
    arguments = parser.parse_args()
    if not 0 < arguments.max_seconds <= 3300 or not 0 < arguments.max_states <= 2048:
        parser.error('Recorded per-worker and request limits must be respected')
    execute(arguments)
