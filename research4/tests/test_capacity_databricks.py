import json
from pathlib import Path
import tempfile
import unittest

import torch

from src.io_utils import dump,read,sha
from src.recognition_capacity_databricks import prepare
from src.recognition_capacity_worker import fingerprint


class CapacityDatabricksTests(unittest.TestCase):
    def test_amendment_preserves_original_and_prompt_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);original=root/'original';original.mkdir()
            dump(original/'CONFIG.json',{'precision':'bfloat16','quantization':None})
            (original/'NUMERIC_CASES.json').write_bytes(b'[{"fixture": true}]\n')
            dump(original/'MANIFEST.json',{p.name:sha(p) for p in original.iterdir() if p.is_file()})
            before=sha(original/'MANIFEST.json');prepare(original,root/'new')
            self.assertEqual(sha(original/'MANIFEST.json'),before)
            self.assertEqual((root/'new/NUMERIC_CASES.json').read_bytes(),(original/'NUMERIC_CASES.json').read_bytes())
            self.assertEqual(read(root/'new/CONFIG.json')['quantization']['bnb_4bit_quant_type'],'nf4')
            with self.assertRaises(FileExistsError):prepare(original,root/'new')

    def test_fingerprint_includes_buffers_and_quantization_metadata(self):
        class State:
            def __init__(self):self.stats=torch.tensor([1.,2.])
            def as_dict(self,packed):return {'stats':self.stats}
        model=torch.nn.Linear(2,2).requires_grad_(False)
        model.register_buffer('fixture',torch.tensor([0]))
        model.weight.quant_state=State()
        a=fingerprint(model,torch);self.assertEqual(a,fingerprint(model,torch))
        model.fixture[0]=1;b=fingerprint(model,torch);self.assertNotEqual(a,b)
        model.weight.quant_state.stats[0]=3
        self.assertNotEqual(b,fingerprint(model,torch))


if __name__=='__main__':unittest.main()
