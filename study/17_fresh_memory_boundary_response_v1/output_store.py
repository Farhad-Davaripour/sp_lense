"""Read selected named full-vocabulary arrays without decoding unused states.

Every decoded array keeps its original dtype, shape and SHA256. There is no
token selection, projection, truncation or approximation in this I/O adapter.
"""
import hashlib,json
from pathlib import Path
import numpy as np
class OutputStore:
    def __init__(self,root):self.root=Path(root)
    def _read(self,name,wanted):
        path=self.root/name;assert path.resolve().is_relative_to(self.root.resolve())
        result={}
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z['__manifest__'].item()));assert meta['format']=='R4_LOSSLESS_ARRAYS_V1'
            for row in meta['arrays']:
                if row['name'] not in wanted:continue
                value=z[row['payload']];dtype=np.dtype(row['dtype']);shape=tuple(row['shape'])
                if row['encoding']=='xor':
                    reference=self._read(row['reference_id'],{row['name']})[row['name']]
                    assert reference.dtype==dtype and reference.shape==shape
                    assert hashlib.sha256(np.ascontiguousarray(reference).tobytes()).hexdigest()==row['reference_sha256']
                    words=np.dtype('<u4' if dtype.itemsize==4 else '<u8')
                    value=np.bitwise_xor(value,reference.view(words)).view(dtype).reshape(shape)
                else:assert row['encoding']=='raw'
                assert value.dtype==dtype and value.shape==shape
                assert hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()==row['sha256']
                result[row['name']]=value.copy()
        assert set(result)==set(wanted);return result
    def point(self,name):return {'logits':self._read(name,{'v:logits'})['v:logits']}
    def load_jet(self,name):
        a=self._read(name,{'q:logits','r:logits','t:logits'})
        return {k:{'logits':a[k+':logits']} for k in ['q','r','t']}
