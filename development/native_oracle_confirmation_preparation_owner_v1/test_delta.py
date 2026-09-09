"""Focused changed-namespace denial and one retained fake-child integration."""
import json,subprocess,sys,time
from owner import HERE,PREP,run,verify,sha,jb,require,sizes
def main():
    source=verify();config=json.loads((HERE/'OWNED_IDENTITY.json').read_bytes());root=HERE/'test_evidence'/str(time.time_ns());root.mkdir(parents=True)
    require(PREP.name=='native_oracle_confirmation_preparation_v1','NEW_PREP_PATH')
    p=subprocess.run([sys.executable,'-B',str(HERE/'owner.py'),'--owner-source-sha256',source,'--approved-preparation-sha256','0'*64],capture_output=True,text=True,timeout=10)
    require(p.returncode==2 and not (HERE/'ownership_attempt_001').exists(),'NO_REAL_RELEASE_DENIAL')
    out=root/'owner';prep=root/'fake_preparation'
    result=run([config['launch_image'],'-B',str(HERE/'fake_child.py'),'success',str(prep)],config,out,prep,{'owner_source_sha256':source,'text_lock_sha256':'0'*64,'synthetic_only':True},wait_seconds=5.,cleanup_seconds=2.)
    require(result['status']=='PASS' and result['quiescent'] and result['exit_code']==0,'RETAINED_FAKE_CLOSURE')
    require(sizes(out)<=32768 and sizes(out)+sizes(prep)<=16*1024**2,'DISJOINT_CAPS')
    receipt={'status':'PASS','groups':3,'source_sha256':source,'closure':result,'real_tokenizer_calls':0,'model_calls':0}
    (HERE/'TEST_RESULTS.json').write_bytes(jb(receipt));print(json.dumps({'status':'PASS','groups':3}))
if __name__=='__main__':main()
