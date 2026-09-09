"""Fixed local fake work only; no tokenizer/model imports, network or input reading."""
import json,sys,time
from pathlib import Path
def main():
    mode,out=sys.argv[1],Path(sys.argv[2]);time.sleep(.25)
    if mode=='timeout':time.sleep(30);return 0
    out.mkdir(exist_ok=False)
    (out/'RESULT.json').write_text(json.dumps({'status':'PASS' if mode!='nonzero' else 'FAIL','fake_only':True})+'\n',encoding='utf-8')
    if mode in ('stdout_overflow','stderr_overflow'):
        stream=sys.stdout if mode=='stdout_overflow' else sys.stderr
        stream.write('x'*20000);stream.flush();time.sleep(.25)
    else:print(json.dumps({'mode':mode,'fake_only':True}),flush=True)
    return 7 if mode=='nonzero' else 0
if __name__=='__main__':raise SystemExit(main())
