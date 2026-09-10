"""Administrative CLI byte recorder; scientific containment remains the frozen owner."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
def main():
    cap=HERE/'capture';prep=HERE/'preparation';release=prep/'root_release/PREPARATION_RELEASE.json'
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert len(sys.argv)==3 and h(cap/'SOURCE_FREEZE.json')==sys.argv[1] and h(release)==sys.argv[2]
    assert not (cap/'ownership_attempt_001').exists() and not (prep/'preparation_attempt_001').exists()
    out=HERE/'actual_preparation_cli';out.mkdir(exist_ok=False)
    cmd=[sys.executable,'-E','-B',str(cap/'preparation_owner.py'),'--owner-source-sha256',sys.argv[1],'--approved-preparation-sha256',sys.argv[2]]
    started=time.time();result=subprocess.run(cmd,cwd=cap,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=360)
    payloads={'stdout.log':result.stdout,'stderr.log':result.stderr}
    receipt={'command':cmd,'exit_code':result.returncode,'started_unix':started,'finished_unix':time.time(),'source_sha256':sys.argv[1],'release_sha256':sys.argv[2]}
    payloads['CLI_RECEIPT.json']=(json.dumps(receipt,sort_keys=True)+'\n').encode()
    for name,raw in payloads.items():
        with (out/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    print(json.dumps(receipt,sort_keys=True));print(result.stdout.decode(errors='replace'));return result.returncode
if __name__=='__main__':raise SystemExit(main())
