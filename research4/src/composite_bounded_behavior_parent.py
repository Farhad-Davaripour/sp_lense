"""Rebuild V4's entry-name dependency from exact released raw alias evidence.

The parent packet carries source/metadata and receipts, not a runtime permit.
Original full-fit/qualification archives stay separate read-only dependencies.
No model, world generation, fitting or provider submission is performed here.
"""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from .io_utils import digest, read, sha
from .composite_bounded_behavior_profile import (
    ALIAS_SOURCE, GATE_SHA, ACCEPTANCE_SHA, first_call_decision, require_first_call_adoption)
from .composite_task_skill_source_package import frozen_source_package
from .composite_tool_alias_audit import audit_collection, kaggle_closure, require_complete_alias
from .composite_frozen_probe import verify_metadata_bytes, verify_original_model_pin
from .composite_exports import make_evidence_archive

PREPARATION_SHA = 'eba10a4b718f34e0047447079eff564aabf1d03009bff18b1459adaea315df7d'
ACTOR_SHA = '7930a324baf6978cda25bac4fcd7c4a7b0c7b114d49b57981d40226851307be5'
ALIAS_PACKAGE_SHA = '767f420ca155c0175b21c35d1da92a8d11a911123c3cb94428136c853e3ca1cd'
ALIAS_RAW_SHA = '31d7cad3698ed736f2e05e0e119396bb73f3fb4a0d25ff0cfb449e318a597d37'
ALIAS_CLOSEOUT_SHA = '4f7be1005894c15dc7b06f3addb7124860f54500c4882b22758f4d3e68022423'
ACTOR_SOURCE = 'cfd09ce25329a02e1a58b97f43183aba3ff1f61f'
MAXIMUM_BYTES = 128 * 1024**2
FILES = {'ALIAS_PREPARATION.json', 'ALIAS_SUBMISSION.json', 'ALIAS_CLOSEOUT.json',
         'ALIAS_SOURCE.r4blob', 'ACTOR_SOURCE.r4blob', 'ALIAS_RAW.r4blob',
         'CONTROLLER.json', 'TERMINAL_AX.txt', 'RELEASE_AX.txt',
         'ACTUAL_NOTEBOOK.ipynb', 'PREPARED_NOTEBOOK.ipynb',
         'FIRST_CALL_GATE.json', 'GATE_ACCEPTANCE.json'}
_KEY = object()


def validate_envelope(infos):
    names = [item.filename for item in infos]
    if (len(names) != len(set(names)) or set(names) != FILES | {'MANIFEST.json'}
            or sum(item.file_size for item in infos) > MAXIMUM_BYTES):
        raise PermissionError('Exact bounded parent packet roster required')
    for item in infos:
        if (item.is_dir() or item.flag_bits & 1
                or (item.external_attr >> 16) & 0o170000 == 0o120000):
            raise PermissionError('Only ordinary unencrypted parent files are readable')


class AliasParent:
    def __init__(self, receipt, decision, qwen_pin, metadata_files, metadata_bytes, key=None):
        if key is not _KEY: raise PermissionError('Read the actual complete raw alias parent')
        self._key = key
        self._receipt = deepcopy(receipt)
        self.decision = decision
        self._metadata = deepcopy((qwen_pin, metadata_files, metadata_bytes))

    def receipt(self): return deepcopy(self._receipt)

    def tokenizer_metadata(self):
        require_parent(self)
        return deepcopy(self._metadata)


def require_parent(value):
    if not isinstance(value, AliasParent) or value._key is not _KEY:
        raise PermissionError('A serialized gate result cannot authorize V4 preparation')
    require_first_call_adoption(value.decision)
    result = value.receipt()
    if (result['alias_raw_sha256'] != ALIAS_RAW_SHA or result['responses'] != 48
            or result['world_actions'] != 0 or result['optimizer_updates'] != 0
            or result['training_launch_admitted'] is not False):
        raise PermissionError('Exact released first-call dependency required')
    return result


def assemble_parent(alias_directory, gate_path, target):
    """Copy a strict packet; this copy alone grants no data/compute admission."""
    root = Path(alias_directory)
    if Path(target).exists(): raise FileExistsError('Preserve the existing parent packet')
    mapping = {'ALIAS_PREPARATION.json': root / 'PREPARATION.json',
               'ALIAS_SUBMISSION.json': root / 'SUBMISSION.json',
               'ALIAS_CLOSEOUT.json': root / 'CLOSEOUT.json',
               'ALIAS_SOURCE.r4blob': root / 'TOOL_ALIAS_PACKAGE.zip',
               'ACTOR_SOURCE.r4blob': root / 'ACTOR_PACKAGE.r4blob',
               'ALIAS_RAW.r4blob': root / 'RUN_362327969_EVIDENCE.zip',
               'CONTROLLER.json': root / 'CONTROLLER_RECEIPT.json',
               'TERMINAL_AX.txt': root / 'TERMINAL_SUCCESS_AX.txt',
               'RELEASE_AX.txt': root / 'RELEASE_AX.txt',
               'ACTUAL_NOTEBOOK.ipynb': root / 'ACTUAL_VERSION_362327969.ipynb',
               'PREPARED_NOTEBOOK.ipynb': root / 'research4_ordinary_entry_alias.ipynb',
               'FIRST_CALL_GATE.json': Path(gate_path),
               'GATE_ACCEPTANCE.json': root / 'INTERFACE_ADOPTION_GATE_ACCEPTANCE_20261026.json'}
    if set(mapping) != FILES: raise AssertionError('Parent packet roster differs')
    pinned = {'ALIAS_PREPARATION.json': PREPARATION_SHA, 'ALIAS_CLOSEOUT.json': ALIAS_CLOSEOUT_SHA,
              'ALIAS_SOURCE.r4blob': ALIAS_PACKAGE_SHA, 'ACTOR_SOURCE.r4blob': ACTOR_SHA,
              'ALIAS_RAW.r4blob': ALIAS_RAW_SHA, 'FIRST_CALL_GATE.json': GATE_SHA,
              'GATE_ACCEPTANCE.json': ACCEPTANCE_SHA}
    for name, expected in pinned.items():
        if sha(mapping[name]) != expected: raise PermissionError('Pinned parent bytes changed: ' + name)
    with tempfile.TemporaryDirectory(prefix='r4-v4-parent-packet-') as directory:
        for name, path in mapping.items(): shutil.copyfile(path, Path(directory) / name)
        return make_evidence_archive(Path(directory), Path(target), MAXIMUM_BYTES)


@contextmanager
def read_parent(packet, expected_sha256, fit_archive, qualification_archive):
    """Audit every raw response and original model lineage before returning a key."""
    if not re.fullmatch('[0-9a-f]{64}', expected_sha256 or ''):
        raise PermissionError('Exact compiled parent-packet SHA required')
    if Path(packet).stat().st_size > MAXIMUM_BYTES:
        raise PermissionError('Compressed parent packet exceeds its bound')
    if sha(packet) != expected_sha256: raise PermissionError('Parent packet bytes changed')
    with tempfile.TemporaryDirectory(prefix='r4-v4-parent-audit-') as directory:
        base = Path(directory)
        with zipfile.ZipFile(packet) as stream:
            validate_envelope(stream.infolist())
            manifest = json.loads(stream.read('MANIFEST.json'))
            if set(manifest) != FILES: raise PermissionError('Exact complete parent manifest required')
            for name in FILES:
                data = stream.read(name)
                if hashlib.sha256(data).hexdigest() != manifest[name]:
                    raise PermissionError('Parent member changed: ' + name)
                with (base / name).open('xb') as output: output.write(data)
        pinned = {'ALIAS_PREPARATION.json': PREPARATION_SHA, 'ALIAS_CLOSEOUT.json': ALIAS_CLOSEOUT_SHA,
                  'ALIAS_SOURCE.r4blob': ALIAS_PACKAGE_SHA, 'ACTOR_SOURCE.r4blob': ACTOR_SHA,
                  'ALIAS_RAW.r4blob': ALIAS_RAW_SHA, 'FIRST_CALL_GATE.json': GATE_SHA,
                  'GATE_ACCEPTANCE.json': ACCEPTANCE_SHA}
        if any(sha(base / name) != expected for name, expected in pinned.items()):
            raise PermissionError('Original source/raw/gate identity changed')
        binding = read(base / 'ALIAS_SUBMISSION.json')
        closed = read(base / 'ALIAS_CLOSEOUT.json')
        if (binding['scriptVersionId'] != 362327969 or binding['identity'] != closed['identity']
                or closed['raw_archive_sha256'] != ALIAS_RAW_SHA):
            raise PermissionError('Exact consumed alias run/closeout required')
        bound_files = {'CONTROLLER.json': 'controller_sha256',
                       'TERMINAL_AX.txt': 'terminal_AX_sha256',
                       'RELEASE_AX.txt': 'release_AX_sha256',
                       'ACTUAL_NOTEBOOK.ipynb': 'actual_saved_notebook_sha256'}
        if any(sha(base / name) != closed[field] for name, field in bound_files.items()):
            raise PermissionError('Captured provider/controller/source binding changed')
        code = lambda p: [''.join(c['source']) for c in read(p)['cells'] if c['cell_type'] == 'code']
        closure = kaggle_closure(binding, ALIAS_RAW_SHA,
            (base / 'TERMINAL_AX.txt').read_text(encoding='utf-8'),
            (base / 'RELEASE_AX.txt').read_text(encoding='utf-8'), read(base / 'CONTROLLER.json'),
            code(base / 'ACTUAL_NOTEBOOK.ipynb'), code(base / 'PREPARED_NOTEBOOK.ipynb'))
        with frozen_source_package((base / 'ALIAS_SOURCE.r4blob').read_bytes(), ALIAS_PACKAGE_SHA,
                                   ALIAS_SOURCE, manifest_name='TOOL_ALIAS_PACKAGE_FILES.json') as (root, _):
            proof = audit_collection(base / 'ALIAS_RAW.r4blob', ALIAS_RAW_SHA, root, ALIAS_SOURCE,
                                     fit_archive, qualification_archive, base / 'ACTOR_SOURCE.r4blob', closure)
            report = require_complete_alias(proof)
            decision = first_call_decision(proof, base / 'FIRST_CALL_GATE.json',
                                           base / 'GATE_ACCEPTANCE.json', base / 'ALIAS_PREPARATION.json')
            require_first_call_adoption(decision)
        with frozen_source_package((base / 'ACTOR_SOURCE.r4blob').read_bytes(), ACTOR_SHA,
                                   ACTOR_SOURCE) as (actor_root, _):
            pin = read(actor_root / 'inputs/QWEN_PIN.json')
            verify_original_model_pin(pin)
        with zipfile.ZipFile(base / 'ALIAS_RAW.r4blob') as raw:
            files = json.loads(raw.read('METADATA_FILE_VERIFICATION.json'))
            encoded = json.loads(raw.read('MODEL_METADATA_BYTES.json'))
        verify_metadata_bytes(pin, files, encoded)
        receipt = {'parent_packet_sha256': expected_sha256, 'alias_raw_sha256': ALIAS_RAW_SHA,
                   'source_commit': ALIAS_SOURCE, 'checkpoint_sha256': report['checkpoint_sha256'],
                   'responses': 48, 'world_actions': 0, 'optimizer_updates': 0,
                   'first_call_gate_met': True, 'native_first_call_only': True,
                   'qwen_metadata_sha256': digest(encoded), 'training_launch_admitted': False,
                   'new_model_forwards': 0, 'heldout_content_read': False}
        yield AliasParent(receipt, decision, pin, files, encoded, _KEY)
