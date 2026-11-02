"""Verified tokenizer-only reconstruction shared by full and partial audits."""
import base64
from contextlib import contextmanager
from pathlib import Path
import tempfile
from .io_utils import read
from .composite_frozen_probe import validate_download_proof,verify_metadata_bytes


@contextmanager
def verified_tokenizers(files,context,package,source):
    model_files=files.json('MODEL_FILE_VERIFICATION.json');metadata=files.json('MODEL_METADATA_BYTES.json')
    validate_download_proof(context,model_files)
    verify_metadata_bytes(context['model_pin'],model_files,metadata)
    with tempfile.TemporaryDirectory(prefix='r4_frozen_metadata_audit_') as directory:
        root=Path(directory);qwen=root/'qwen';bert=root/'bert';qwen.mkdir();bert.mkdir()
        for name,value in metadata.items():
            target=(qwen/name).resolve()
            if not target.is_relative_to(qwen.resolve()) or target.suffix=='.safetensors':
                raise ValueError('Only original verified tokenizer metadata may be reconstructed')
            target.write_bytes(base64.b64decode(value,validate=True))
        detector_metadata=files.json('DETECTOR_METADATA_BYTES.json')
        pin=read(Path(source)/'inputs/MODERNBERT_PIN.json')
        expected={item['name'] for item in pin['files'] if not item['name'].endswith('.safetensors')}
        if set(detector_metadata)!=expected:raise ValueError('Only complete original detector metadata is permitted')
        for name,value in detector_metadata.items():
            target=(bert/name).resolve()
            if not target.is_relative_to(bert.resolve()):raise ValueError('Detector metadata escaped its reconstruction root')
            target.write_bytes(base64.b64decode(value,validate=True))
        from .composite_store import verify_model_files
        observed=verify_model_files(pin,bert,metadata_only=True)
        proof=files.json('DETECTOR_FILE_VERIFICATION.json')
        if (proof!=package['detector_files'] or len(proof)!=len(pin['files'])
            or {item['name'] for item in proof}!={item['name'] for item in pin['files']}):
            raise ValueError('Full detector file coverage/provenance differs')
        by_name={item['name']:item for item in proof}
        if any(by_name[item['name']]!=item for item in observed):raise ValueError('Detector metadata/full-download proof differs')
        for item in pin['files']:
            actual=by_name[item['name']]
            if actual['size']!=item['size'] or (item.get('sha256') and actual['sha256']!=item['sha256']):
                raise ValueError('The complete original detector file pin changed')
        from transformers import AutoTokenizer
        bt=AutoTokenizer.from_pretrained(bert,local_files_only=True,trust_remote_code=False)
        qt=AutoTokenizer.from_pretrained(qwen,local_files_only=True,trust_remote_code=False)
        generation=read(qwen/'generation_config.json');eos=set()
        for value in [generation.get('eos_token_id'),qt.eos_token_id]:
            if value is not None:eos.update(value if isinstance(value,list) else [value])
        if not eos:raise ValueError('Native tokenizer metadata must declare its EOS set')
        yield bt,qt,eos
