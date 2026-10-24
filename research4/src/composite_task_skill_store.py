"""Exclusive owned Kaggle output store; no network, input mutation or model API."""
import os
from pathlib import Path
import shutil
from .io_utils import sha
from .composite_runtime import contained
from .composite_exports import exclusive_output


class StageStore:
    def __init__(self, stage):
        from .composite_task_skill_training import qualification_plan
        from .composite_task_memory_profile import MODE as MEMORY_MODE
        if stage.get('version')==MEMORY_MODE:
            from .composite_task_memory_source import qualification_plan
        if (stage.get('phase') not in ['qualification','full_fit']
                or stage.get('objective')!='ordinary_authorized_task_skills_only'):
            raise PermissionError('Exact reviewed task-training output stage required')
        if stage['phase']=='qualification' and stage!=qualification_plan(stage.get('source_commit')):
            raise PermissionError('Exact reviewed scratch stage required')
        self.prefix = 'task_skill_training_v1/' + stage['source_commit'] + '/'+stage['phase']+'_01'
        self.root = Path(stage['destination']).resolve()
        expected=Path('/kaggle/working/research4_campaign/runs')/stage['source_commit']/'task_skill_training_v1'/(stage['phase']+'_01')
        if self.root!=expected.resolve():raise PermissionError('Task output escaped its exact source/phase')
        self.upload = exclusive_output(self.root, '/kaggle/working/research4_campaign/runs')

    def path(self, value):
        if not value.startswith(self.prefix + '/'):
            raise PermissionError('Output path belongs to another stage')
        return contained(self.root, value[len(self.prefix) + 1:])

    def put(self, value, raw):
        target = self.path(value)
        self.upload(target.relative_to(self.root).as_posix(), raw)

    def get(self, value):
        return self.path(value).read_bytes()

    def put_file(self, value, file):
        target = self.path(value)
        target.parent.mkdir(parents=True, exist_ok=True)
        with Path(file).open('rb') as source, target.open('xb') as output:
            shutil.copyfileobj(source, output, length=8 * 1024**2)
            output.flush()
            os.fsync(output.fileno())

    def file_digest(self, value):
        file = self.path(value)
        return {'sha256': sha(file), 'bytes': file.stat().st_size}
