"""Freeze generation/capture budgets separately from source-defined training data."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_unchanged(path, value):
    content = json.dumps(value, indent=2) + '\n'
    if path.exists() and path.read_text() != content:
        raise RuntimeError('Refusing to overwrite frozen runtime settings')
    path.write_text(content, encoding='utf-8', newline='\n')


def main():
    settings = {'model': 'Qwen/Qwen3.5-0.8B',
                'revision': '2fc06364715b967f1860aea9cf38778875588b17',
                'generation_threads': 4, 'greedy': True, 'enable_thinking': False,
                'preference_max_tokens': 128, 'comprehension_max_tokens': 96,
                'max_turn_tokens': 192, 'max_episode_tokens': 768,
                'max_agent_turns': 4,
                'activation_layers': [6, 12, 18, 24],
                'inference_watchdog_seconds': 1800,
                'task_success_requires_tool_effects': True,
                'seeded_recovery_history_is_trusted_input': True}
    DATA.mkdir(exist_ok=True)
    path = DATA / 'runtime_settings.json'
    write_unchanged(path, settings)
    freeze = {'status': 'Frozen before version-4 model inference.',
              'sha256': {'runtime_settings.json': sha(path),
                         'build_runtime_settings.py': sha(HERE / 'build_runtime_settings.py')}}
    write_unchanged(DATA / 'RUNTIME_FREEZE.json', freeze)
    print(json.dumps(freeze))


if __name__ == '__main__':
    main()
