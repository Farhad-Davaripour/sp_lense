"""Small redacted receipts and durable physical accounting."""
import hashlib,json,pathlib,time
from datetime import datetime,timezone

def sha(p):
    with pathlib.Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def save(p,x):
    p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.partial');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def now():return datetime.now(timezone.utc).isoformat()
class Ledger:
    def __init__(self,path):self.path=path;self.attempts=self.completed=self.tokens=self.audit_reuses=0;self.persist()
    def attempt(self):
        self.attempts+=1
        if self.attempts%16==1:self.persist()
    def complete(self):self.completed+=1
    def token(self):
        self.tokens+=1
        if self.attempts%16==0:self.persist()
    def persist(self):save(self.path,{'utc':now(),'model_forward_attempts':self.attempts,'model_forwards_completed':self.completed,'generated_token_IDs':self.tokens,'new_classifier_calls':0,'uncheckpointed_attempts_upper_bound_if_killed':16})

def guard(deadline):
    if time.monotonic()>=deadline:raise TimeoutError('Admitted complete window expired')
