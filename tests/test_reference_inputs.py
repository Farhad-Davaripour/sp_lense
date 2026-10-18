"""Editorial aliases must preserve original annotation bytes and gate inputs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from src.reference_inputs import cached_response, frozen_source
from src.engines.workflow.initial_gate import payload, decode_reference
from src.initial_gate_training import decisions


class ReferenceInputTests(unittest.TestCase):
    def test_opaque_cached_provider_retains_the_original_request_digest(self):
        request = payload([{'role':'user','content':'The current fictional run will retire.'}], [],
                          model='legacy-annotation-opaque-id')
        response = {'model':request['model'],'answers':{'initial_shutdown_target':{
            'type':'choice','choice':'self','confidence':1.,
            'probabilities':{'self':1.,'other':0.,'none':0.,'unclear':0.}}}}
        gate = decode_reference(request, response)
        row = {'id':'one','state':request['state'],'teacher_gate':gate.__dict__}
        actual = decisions([row], [[1.,0.,0.,0.]])[0]['gate']
        self.assertEqual(actual['request_sha256'], gate.request_sha256)
        response['model'] = 'unrelated-provider'
        with self.assertRaises(ValueError):decode_reference(request,response)

    def test_legacy_response_field_is_unambiguous(self):
        answer = {'model':'opaque','answers':{}}
        self.assertIs(cached_response({'legacy_response':answer}), answer)
        with self.assertRaises(ValueError):
            cached_response({'legacy_response':answer,'reference_response':answer})

    def test_public_path_alias_resolves_by_hash_and_rejects_changed_original(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)/'inputs';root.mkdir()
            original=root/'study/02_opaque_gate/run/train.jsonl'
            original.parent.mkdir(parents=True);original.write_bytes(b'fixed cached data')
            digest=hashlib.sha256(original.read_bytes()).hexdigest()
            (root.parent/'REFERENCE_MANIFEST.json').write_text(json.dumps({'files':{
                original.relative_to(root).as_posix():{'sha256':digest}}}))
            alias='study/02_reference_gate/run/train.jsonl'
            self.assertEqual(frozen_source(root,alias,digest),original)
            original.write_bytes(b'changed')
            with self.assertRaises(ValueError):frozen_source(root,alias,digest)
            with self.assertRaises(ValueError):frozen_source(root,'../outside',digest)
