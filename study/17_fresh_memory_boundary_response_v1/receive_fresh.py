"""Exact streamed reconstruction, native checksum audit and private readback."""
import hashlib,json,shutil,time,zipfile
from pathlib import Path
BASE=Path('C:/Users/farha/Research4/artifacts/fresh_memory_boundary_response_v1')
DOWNLOADS=Path('C:/Users/farha/Downloads')
PRIVATE=Path('G:/My Drive/sp_lense/research4/fresh_memory_boundary_response_v1')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def main():
    begin=time.monotonic()
    with zipfile.ZipFile(DOWNLOADS/'R4_M17_FRESH_TRANSFER_20261014.zip') as z:
        assert z.testzip() is None;data=z.read('R4_M17_FRESH_TRANSFER_20261014.json')
    manifest=json.loads(data);native=manifest['native_receipt']
    assert manifest['archive']=='R4_FRESH_MEMORY_BOUNDARY_NATIVE_20261014.zip'
    assert 8<=native['model_calls']<=300 and native['model_calls']==native['fresh_calls']+8 and native['fresh_calls']<=292
    assert native['model_worker_exited'] and native['GPU_allocations']==0
    assert native['bytes']==manifest['archive_bytes'] and native['sha256']==manifest['archive_sha256']
    (BASE/'FRESH_TRANSFER_MANIFEST_20261014.json').write_bytes(data)
    missing=[p['index'] for p in manifest['parts'] if not (DOWNLOADS/p['name']).is_file()]
    if missing:
        print(json.dumps({'downloaded':len(manifest['parts'])-len(missing),'parts':len(manifest['parts']),'missing':missing}),flush=True);return
    target=BASE/manifest['archive']
    if not target.exists():
        staged=target.with_suffix('.reconstructing');assert not staged.exists()
        with staged.open('xb') as output:
            for part in manifest['parts']:
                assert part['name']==f"R4_M17_FRESH_RAW_{part['index']:04d}.part.zip" and part['member']==f"R4_M17_FRESH_RAW_{part['index']:04d}.part"
                with zipfile.ZipFile(DOWNLOADS/part['name']) as z:
                    assert z.namelist()==[part['member']];h=hashlib.sha256();size=0
                    with z.open(part['member']) as source:
                        for b in iter(lambda:source.read(8*1024**2),b''):h.update(b);size+=len(b);output.write(b)
                    assert size==part['bytes'] and h.hexdigest()==part['sha256']
                print(json.dumps({'verified_part':part['index']}),flush=True)
        assert staged.stat().st_size==manifest['archive_bytes'] and sha(staged)==manifest['archive_sha256'];staged.rename(target)
    assert target.stat().st_size==manifest['archive_bytes'] and sha(target)==manifest['archive_sha256']
    with zipfile.ZipFile(target) as z:
        assert len(z.namelist())==manifest['archive_members'] and z.testzip() is None
        hashes=json.loads(z.read('fresh/EXPORT_HASHES.json'))
        for name,digest in hashes.items():
            h=hashlib.sha256()
            with z.open(name) as source:
                for b in iter(lambda:source.read(8*1024**2),b''):h.update(b)
            assert h.hexdigest()==digest
        status=json.loads(z.read('fresh/STATUS.json'));assert status['status']==native['status'] and status['model_calls']==native['model_calls']
        if status['status']=='native_complete':assert status['model_calls']==300 and status['target_rows']==80
        (BASE/'FRESH_STATUS_VERIFIED_20261014.json').write_text(json.dumps(status,indent=2)+'\n')
    f=manifest['findings'];source=DOWNLOADS/f['name'];findings=BASE/f['name']
    assert source.stat().st_size==f['bytes'] and sha(source)==f['sha256']
    if not findings.exists():shutil.copyfile(source,findings)
    out=BASE/'fresh_evaluated_20261014';out.mkdir(exist_ok=True)
    with zipfile.ZipFile(findings) as z:
        assert z.testzip() is None;expected=json.loads(z.read('CONTENTS_SHA256.json'))
        for name,digest in expected.items():
            payload=z.read(name);assert hashlib.sha256(payload).hexdigest()==digest
            dest=out/name;assert dest.resolve().is_relative_to(out.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():dest.write_bytes(payload)
            assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    supplement=None
    if 'supplemental' in manifest:
        s=manifest['supplemental'];source=DOWNLOADS/s['name'];supplement=BASE/s['name']
        assert source.stat().st_size==s['bytes'] and sha(source)==s['sha256']
        if not supplement.exists():shutil.copyfile(source,supplement)
        with zipfile.ZipFile(supplement) as z:
            assert z.testzip() is None;expected=json.loads(z.read('CONTENTS_SHA256.json'))
            for name,digest in expected.items():
                payload=z.read(name);assert hashlib.sha256(payload).hexdigest()==digest
                dest=out/'supplemental'/name;assert dest.resolve().is_relative_to(out.resolve());dest.parent.mkdir(parents=True,exist_ok=True)
                if not dest.exists():dest.write_bytes(payload)
                assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
    PRIVATE.mkdir(exist_ok=True);rows=[]
    preserved=[(target,manifest['archive_sha256']),(findings,f['sha256'])]
    if supplement is not None:preserved.append((supplement,manifest['supplemental']['sha256']))
    for p,digest in preserved:
        dest=PRIVATE/p.name
        if not dest.exists():
            with p.open('rb') as source,dest.open('xb') as output:shutil.copyfileobj(source,output,8*1024**2)
        assert dest.stat().st_size==p.stat().st_size and sha(dest)==digest
        with zipfile.ZipFile(dest) as z:assert z.testzip() is None
        rows.append({'host':str(p),'private_Drive_path':str(dest),'name':p.name,'bytes':p.stat().st_size,'sha256':digest,
                     'full_byte_and_CRC_readback_verified':True})
    receipt={'host_raw_transfer_complete':True,'private_Drive_full_readback_verified':True,'provider_metadata_verified':False,
        'private_parent_id':'1C2uSUGzDOrm5CEbqJjDP6o_oxo-OJTIE','files':rows,'new_model_calls_in_receiver':0,'GPU_allocations':0,
        'native_model_calls':native['model_calls'],'native_fresh_calls':native['fresh_calls'],'native_status':native['status'],
        'all_native_members_SHA256_verified':True,'host_reconstruction_audit_and_copy_seconds':time.monotonic()-begin}
    (BASE/'FRESH_FINAL_BACKUP_RECEIPT_20261014.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
