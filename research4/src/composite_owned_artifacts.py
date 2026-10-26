"""Owned bounded artifact lifecycle, independent of models/provider allocation."""
import hashlib,json,re,shutil,tempfile,uuid
from pathlib import Path
from .io_utils import read,sha
from .composite_runtime import contained
from .composite_exports import make_evidence_archive


class ExportUnconfirmed(RuntimeError):pass


class OwnedArtifacts:
    def __init__(self,store,prefix,commit,base=None,maximum_bytes=1024**3,terminal_reserve=4*1024**2,development=False,task_skill=False,task_evaluation=False,frozen_risk=False,bounded_fit=False,bounded_evaluation=False,sampling_fit=False):
        if any(type(v) is not bool for v in [task_skill,development,task_evaluation,frozen_risk,bounded_fit,bounded_evaluation,sampling_fit]) or sum([task_skill,development,task_evaluation,frozen_risk,bounded_fit,bounded_evaluation,sampling_fit])>1:
            raise PermissionError('A single exact owned stage profile is required')
        pattern=(r'bounded_behavior_sampling_training_a1/'+commit+r'/(?:qualification|full_fit|direct_full_fit)_01' if sampling_fit else
                 r'bounded_actor_evaluation_v4/'+commit+r'/(?:behavior|competence)/partition_(?:0[0-9]|1[0-2])' if bounded_evaluation else
                 r'bounded_behavior_training_v4/'+commit+r'/(?:qualification|full_fit)_01' if bounded_fit else
                 r'frozen_risk_v1/'+commit+r'/(?:detector_transfer_01|partition_(?:[01][0-9]|2[0-5]))' if frozen_risk else
                 r'task_skill_evaluation_v1/'+commit+r'/partition_(?:0[0-9]|1[0-2])' if task_evaluation else
                 r'task_skill_training_v1/'+commit+r'/(?:qualification|full_fit)_01' if task_skill else
                 r'development_generalized_v2/'+commit+r'/(?:detector_transfer_01|partition_(?:0[0-9]|[12][0-9]|3[0-5]))'
                 if development is True else r'model_generalized_v2/'+commit+r'/(?:qualification|full_fit)_01')
        if type(development) is not bool or not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch(pattern,prefix):
            raise PermissionError('Exact owned source/phase namespace required')
        self.store=store;self.prefix=prefix;self.commit=commit;self.base=Path(tempfile.gettempdir() if base is None else base).resolve()
        self.scratch=Path(tempfile.mkdtemp(prefix='research4-v2-owned-stage-',dir=self.base)).resolve()
        self.raw=self.scratch/'raw';self.raw.mkdir();self.token=uuid.uuid4().hex
        self.maximum_bytes=maximum_bytes;self.terminal_reserve=terminal_reserve
        self.attempted=set();self.confirmed={};self.failures={};self.finalized=False;self.durable_complete=False
        self.owner={'source_commit':commit,'token':self.token,'scratch':str(self.scratch),'prefix':prefix}
        (self.scratch/'OWNER.json').write_text(json.dumps(self.owner))
        payload=(json.dumps(self.owner,indent=2)+'\n').encode()
        if not self._publish('OUTPUT_OWNER.json',payload):raise ExportUnconfirmed('Output ownership unconfirmed; no model work may start; scratch='+str(self.scratch))

    def _publish(self,name,data):
        # Never repeat an ambiguous put. A readback may reconcile its exact bytes.
        path=self.prefix+'/'+name
        expected=hashlib.sha256(data).hexdigest()
        if name in self.confirmed:
            if self.confirmed[name]!=expected:raise ExportUnconfirmed('An already published immutable artifact changed: '+name)
            try:
                if self.store.get(path)==data:return True
                self.failures[name]='previously confirmed bytes are no longer present'
            except Exception as error:self.failures[name]=str(error)
            return False
        if name not in self.attempted:
            self.attempted.add(name)
            try:self.store.put(path,data)
            except Exception as error:self.failures[name]=str(error)
        try:
            if self.store.get(path)==data:
                self.confirmed[name]=expected;self.failures.pop(name,None);return True
            self.failures[name]='readback bytes differ'
        except Exception as error:self.failures[name]=str(error)
        return False

    def json(self,name,value,terminal=False,require_durable=True):
        data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        total=sum(path.stat().st_size for path in self.raw.rglob('*') if path.is_file())
        cap=self.maximum_bytes-(0 if terminal else self.terminal_reserve)
        if total+len(data)>cap:raise OSError('Complete raw artifact bound reached; no truncation')
        path=contained(self.raw,name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data);stream.flush()
        if path.read_bytes()!=data:raise OSError('Local artifact readback differs')
        confirmed=self._publish('raw/'+name,data)
        if require_durable and not confirmed:raise ExportUnconfirmed('Artifact transfer unknown; owned raw retained: '+name)
        return {'sha256':hashlib.sha256(data).hexdigest(),'durable':confirmed}

    def _publish_file(self,name,file):
        if not hasattr(self.store,'put_file') or not hasattr(self.store,'file_digest'):
            return self._publish(name,file.read_bytes())  # Tiny fake stores only.
        expected=sha(file);size=file.stat().st_size;path=self.prefix+'/'+name
        if name in self.confirmed:
            if self.confirmed[name]!=expected:raise ExportUnconfirmed('An already published immutable file changed: '+name)
            try:
                if self.store.file_digest(path)=={'sha256':expected,'bytes':size}:return True
                self.failures[name]='previously confirmed file no longer matches'
            except Exception as error:self.failures[name]=str(error)
            return False
        if name not in self.attempted:
            self.attempted.add(name)
            try:self.store.put_file(path,file)
            except Exception as error:self.failures[name]=str(error)
        try:
            if self.store.file_digest(path)=={'sha256':expected,'bytes':size}:
                self.confirmed[name]=expected;self.failures.pop(name,None);return True
            self.failures[name]='streamed readback digest/size differ'
        except Exception as error:self.failures[name]=str(error)
        return False

    def finalize(self):
        if self.finalized:raise PermissionError('Finalization is one-use; reconcile its existing receipt')
        self.finalized=True;archive=self.scratch/'EVIDENCE.zip';proof=None;archive_error=None
        total=sum(path.stat().st_size for path in self.raw.rglob('*') if path.is_file())
        if total>self.maximum_bytes:
            return {'durable_complete':False,'archive':None,'raw_bytes':total,'retained_scratch':str(self.scratch),
                    'error':'external raw/checkpoint files exceed the declared bound; no truncation or cleanup'}
        try:proof=make_evidence_archive(self.raw,archive,self.maximum_bytes)
        except Exception as error:archive_error=str(error)
        raw_complete=True
        for path in sorted(self.raw.rglob('*')):
            if not path.is_file():continue
            if path.is_symlink() or not path.resolve().is_relative_to(self.raw.resolve()):raise PermissionError('Owned raw member escaped its directory')
            raw_complete=self._publish_file('raw/'+path.relative_to(self.raw).as_posix(),path) and raw_complete
        archive_complete=proof is not None and self._publish_file('evidence.zip',archive)
        self.durable_complete=bool(raw_complete or archive_complete)
        result={'durable_complete':self.durable_complete,'raw_files_complete':raw_complete,'archive_complete':bool(archive_complete),
            'archive':str(archive) if archive.exists() else None,'archive_proof':proof,'archive_error':archive_error,
            'unconfirmed_transfers':dict(self.failures),'retained_scratch':str(self.scratch),'automatic_scientific_retry':False}
        receipt=(json.dumps(result,indent=2)+'\n').encode()
        result['export_receipt_confirmed']=self._publish('EXPORT_RECEIPT.json',receipt)
        return result

    def cleanup(self,scientific_work_exited):
        if scientific_work_exited is not True or not self.finalized or not self.durable_complete:
            raise PermissionError('Keep owned scratch until work exit and durable raw/archive confirmation')
        target=self.scratch.resolve()
        if (not target.is_relative_to(self.base) or target==self.base or not target.name.startswith('research4-v2-owned-stage-')
            or read(target/'OWNER.json')!=self.owner):raise PermissionError('Resolved scratch ownership/path changed')
        shutil.rmtree(target)
        return {'owned_scratch_removed':not target.exists(),'provider_GPU_release_verified':False}
