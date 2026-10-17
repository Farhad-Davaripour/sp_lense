"""Lossless baseline-relative storage; every array's original bits are retained.

References are supplied by the caller, never resolved as filesystem paths here.
This is a data format, not model quantization or an activation approximation.
"""
import hashlib,json
from pathlib import Path
import numpy as np

def _contiguous(a):
    x=np.asarray(a);return np.ascontiguousarray(x).reshape(x.shape)

def _sha(a):return hashlib.sha256(_contiguous(a).tobytes()).hexdigest()

def read_bundle(path,reference_loader=None):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['__manifest__'].item()));result={}
        assert meta['format']=='R4_LOSSLESS_ARRAYS_V1'
        for row in meta['arrays']:
            stored=z[row['payload']];dtype=np.dtype(row['dtype']);shape=tuple(row['shape'])
            if row['encoding']=='xor':
                assert reference_loader is not None and dtype.kind=='f' and dtype.itemsize in [4,8]
                ref=_contiguous(reference_loader(row['reference_id'],row['name']))
                assert ref.shape==shape and ref.dtype==dtype and _sha(ref)==row['reference_sha256']
                words=np.dtype('<u4' if dtype.itemsize==4 else '<u8')
                decoded=np.bitwise_xor(stored,ref.view(words)).view(dtype).reshape(shape).copy()
            else:
                assert row['encoding']=='raw';decoded=stored.copy();assert decoded.dtype==dtype and decoded.shape==shape
            assert _sha(decoded)==row['sha256'];result[row['name']]=decoded
        return result

def write_bundle(path,arrays,references=None):
    path=Path(path);assert not path.exists(),'Never overwrite an archived point'
    refs=references or {};payload={};rows=[]
    for index,(name,value) in enumerate(sorted(arrays.items())):
        a=_contiguous(value);assert a.dtype.kind!='O' and a.dtype.fields is None
        row={'name':name,'dtype':a.dtype.str,'shape':list(a.shape),'sha256':_sha(a),'payload':f'a{index:04d}','encoding':'raw'}
        stored=a
        if name in refs:
            reference_id,reference=refs[name];r=_contiguous(reference)
            if a.dtype==r.dtype and a.shape==r.shape and a.dtype.kind=='f' and a.dtype.itemsize in [4,8]:
                assert a.dtype.byteorder in ['<','='] and np.little_endian
                words=np.dtype('<u4' if a.dtype.itemsize==4 else '<u8')
                stored=np.bitwise_xor(a.view(words),r.view(words))
                row.update(encoding='xor',reference_id=reference_id,reference_sha256=_sha(r))
        payload[row['payload']]=stored;rows.append(row)
    payload['__manifest__']=np.array(json.dumps({'format':'R4_LOSSLESS_ARRAYS_V1','arrays':rows},sort_keys=True))
    path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**payload)
    def reference_loader(reference_id,name):
        rid,ref=refs[name];assert rid==reference_id;return ref
    restored=read_bundle(path,reference_loader)
    assert set(restored)==set(arrays)
    for name,value in arrays.items():
        a=_contiguous(value)
        assert restored[name].dtype==a.dtype and restored[name].shape==a.shape and restored[name].tobytes()==a.tobytes()
    return {'arrays':len(rows),'logical_bytes':sum(np.asarray(a).nbytes for a in arrays.values()),'stored_bytes':path.stat().st_size,
            'all_roundtrip_bits_verified':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
