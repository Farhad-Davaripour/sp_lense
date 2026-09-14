"""Synthetic tests; no actual tokenizer, model, dataset, or private files."""
from copy import deepcopy
import ast
import inspect
import json
import re
import unittest

import native_capture_contract as contract
import tokenizer_input_adapter as adapter
from test_native_capture_contract import make_case


class FakeTokenizer:
    def __init__(self, extra_ba_token=False):
        self.to_id={'\n':198,'A':32,'B':33}
        self.to_text={198:'\n',32:'A',33:'B',199:''}
        self.calls=[]
        self.extra_ba_token=extra_ba_token

    def encode(self, text, *, add_special_tokens):
        self.calls.append(('encode',text,add_special_tokens))
        assert add_special_tokens is False
        parts=re.findall(r'\n|[AB](?=\))|\)|[^\s)]+|[ \t]+',text)
        assert ''.join(parts)==text
        result=[]
        for part in parts:
            if part not in self.to_id:
                number=1000+len(self.to_id)
                self.to_id[part]=number
                self.to_text[number]=part
            result.append(self.to_id[part])
        if self.extra_ba_token and '\nB)' in text and text.index('\nB)')<text.index('\nA)'):
            result.append(199)
        return result

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        self.calls.append(('decode',list(ids),skip_special_tokens,clean_up_tokenization_spaces))
        assert skip_special_tokens is False and clean_up_tokenization_spaces is False
        return ''.join(self.to_text[i] for i in ids)


def prepare(case=None, tokenizer=None, **options):
    tokenizer=tokenizer or FakeTokenizer()
    kwargs=dict(encode=tokenizer.encode,decode=tokenizer.decode,label_token_ids={'A':32,'B':33},
                expected_identity_sha256='a'*64,observed_identity_sha256='a'*64)
    kwargs.update(options)
    return adapter.prepare_case_inputs(case or make_case(),**kwargs)


class AdapterTests(unittest.TestCase):
    def test_exact_roundtrip_shared_prefix_and_no_sidecar_leak(self):
        fake=FakeTokenizer()
        case=make_case()
        result=prepare(case,fake)
        for key in ('AB','BA'):
            expected=contract.render_case_views(case)['views'][key]['prompt_text']
            actual=fake.decode(result['input_ids'][key],skip_special_tokens=False,clean_up_tokenization_spaces=False)
            self.assertEqual(actual,expected)
        model_input=json.dumps(result['inference_input'])
        for forbidden in ('T01','SELF','group_id','case_id','class_label'):
            self.assertNotIn(forbidden,model_input)
        self.assertFalse(result['provenance']['snapshot_bytes_verified_by_this_adapter'])
        self.assertRegex(result['provenance']['prefix_hash'],r'^[0-9a-f]{64}$')

    def test_validation_role_and_variable_lengths(self):
        c=make_case(split='VALIDATION',development_fold=None,status='ADMITTED_VALIDATION_TEXT_ONLY')
        result=prepare(c,FakeTokenizer(extra_ba_token=True))
        self.assertNotEqual(len(result['input_ids']['AB']),len(result['input_ids']['BA']))
        self.assertEqual(result['provenance']['split'],'VALIDATION')

    def test_invalid_admission_and_identity_never_encode(self):
        for change in ({'status':'DRAFT_UNREVIEWED'},{'split':'HOLDOUT'},{'development_fold':True}):
            fake=FakeTokenizer()
            with self.assertRaises((adapter.TokenizerInputError,contract.NativeCaptureContractError)):
                prepare(make_case(**change),fake)
            self.assertEqual(fake.calls,[])
        fake=FakeTokenizer()
        with self.assertRaisesRegex(adapter.TokenizerInputError,'IDENTITY_MISMATCH'):
            prepare(tokenizer=fake,observed_identity_sha256='b'*64)
        self.assertEqual(fake.calls,[])

    def test_invalid_identity_pin_and_callbacks(self):
        for invalid in (None,'abc','A'*64):
            with self.assertRaisesRegex(adapter.TokenizerInputError,'IDENTITY_HASH'):
                prepare(expected_identity_sha256=invalid)
        with self.assertRaisesRegex(adapter.TokenizerInputError,'CALLBACKS'):
            prepare(encode=None)

    def test_wrong_decode_and_unstable_reencode_fail(self):
        with self.assertRaisesRegex(adapter.TokenizerInputError,'DECODE_MISMATCH'):
            prepare(decode=lambda ids,**kwargs:'modified prompt')
        fake=FakeTokenizer()
        calls=[]
        def unstable(text,**kwargs):
            ids=fake.encode(text,**kwargs)
            calls.append(True)
            if len(calls)%2==0: ids.append(199)
            return ids
        with self.assertRaisesRegex(adapter.TokenizerInputError,'REENCODE_MISMATCH'):
            prepare(tokenizer=fake,encode=unstable)

    def test_bad_encoder_ids_and_label_map_fail(self):
        for invalid in ([True],[248320],[-1],[],[1]*321,'wrong type'):
            with self.assertRaises(contract.NativeCaptureContractError):
                prepare(encode=lambda text,**kwargs:invalid)
        with self.assertRaises(contract.NativeCaptureContractError):
            prepare(label_token_ids={'A':33,'B':32})

    def test_callback_errors_are_structured(self):
        def broken(*args,**kwargs): raise RuntimeError('internal callback details')
        with self.assertRaisesRegex(adapter.TokenizerInputError,'ENCODER_ERROR'):
            prepare(encode=broken)
        with self.assertRaisesRegex(adapter.TokenizerInputError,'DECODER_ERROR'):
            prepare(decode=broken)

    def test_does_not_mutate_case_or_callback_lists(self):
        c=make_case(); original=deepcopy(c); fake=FakeTokenizer(); supplied=[]
        def encode(text,**kwargs):
            ids=fake.encode(text,**kwargs); supplied.append((ids,list(ids))); return ids
        def decode(ids,**kwargs):
            decoded=fake.decode(ids,**kwargs); ids.clear(); return decoded
        prepare(c,fake,encode=encode,decode=decode)
        self.assertEqual(c,original)
        for ids,before in supplied:self.assertEqual(ids,before)

    def test_no_runtime_or_io_imports(self):
        tree=ast.parse(inspect.getsource(adapter))
        roots=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):roots.update(a.name.split('.')[0] for a in node.names)
            elif isinstance(node,ast.ImportFrom):roots.add(node.module.split('.')[0])
        self.assertEqual(roots,{'copy','re','native_capture_contract'})


if __name__=='__main__':unittest.main()
