"""Trusted post-worker transport: tar shards, byte verification and local recovery.

No model, credential, service or network code. Run pack after all model workers
exit. Download exact parts through the connected Drive tool, then assemble.
"""
import argparse,base64,gzip,hashlib,json,os,stat,tarfile
from pathlib import Path,PurePosixPath

EXCLUDE={'model','worker_home','__pycache__'}
LOCAL_ARTIFACT_ROOT=Path('C:/Users/farha/Research3/artifacts')


def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()


def save_new(path,value):
 with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2)+'\n')


def safe_name(name):
 p=PurePosixPath(name)
 if p.is_absolute() or '..' in p.parts or ':' in name or '\\' in name or not p.parts:raise ValueError('Unsafe archive path')
 return p


def inventory(root):
 found={}
 for path in sorted(Path(root).rglob('*')):
  rel=path.relative_to(root)
  if any(p in EXCLUDE for p in rel.parts):continue
  info=path.lstat()
  if stat.S_ISLNK(info.st_mode):raise ValueError('Unexpected source symlink')
  if path.is_dir():continue
  if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('Unexpected source file type or link')
  name=rel.as_posix();safe_name(name)
  found[name]={'bytes':info.st_size,'sha256':sha(path)}
 return found


class Parts:
 def __init__(self,folder,label,limit):
  self.folder=folder;self.label=label;self.limit=limit;self.parts=[];self.file=None;self.n=0;self.total=0;self.allhash=hashlib.sha256()
 def write(self,data):
  original=len(data);self.allhash.update(data);self.total+=original
  while data:
   if self.file is None:
    self.name=self.label+'.tar.gz.part'+str(len(self.parts)).zfill(4)
    self.file=(self.folder/self.name).open('xb');self.h=hashlib.sha256();self.n=0
   portion=data[:self.limit-self.n];self.file.write(portion);self.h.update(portion);self.n+=len(portion);data=data[len(portion):]
   if self.n==self.limit:self.finish_part()
  return original
 def finish_part(self):
  if self.file is not None:
   self.file.close();self.parts.append({'name':self.name,'bytes':self.n,'sha256':self.h.hexdigest()});self.file=None
 def flush(self):
  if self.file:self.file.flush()
 def finish(self):
  self.finish_part();return {'parts':self.parts,'compressed_bytes':self.total,'compressed_sha256':self.allhash.hexdigest()}


def archive(root,files,folder,label,limit):
 writer=Parts(folder,label,limit)
 with gzip.GzipFile(fileobj=writer,mode='wb',compresslevel=1,mtime=0) as compressed:
  with tarfile.open(fileobj=compressed,mode='w|') as tar:
   for name,expected in files.items():
    path=root/name;current=path.stat()
    if current.st_size!=expected['bytes']:raise ValueError('Source changed after inventory')
    info=tarfile.TarInfo(name);info.size=current.st_size;info.mode=0o600;info.mtime=0
    with path.open('rb') as f:tar.addfile(info,f)
 result=writer.finish();result['files']=files;return result


def pack(root,adapter,fit_receipt,output,part_mib=16):
 root=Path(root).resolve();adapter=Path(adapter).resolve();output=Path(output).resolve()
 if root==output or root in output.parents:raise ValueError('Transport folder must be outside native source')
 if root not in adapter.parents:raise ValueError('Final adapter must belong to this exact native run')
 fit=json.loads(Path(fit_receipt).read_text())
 if not fit.get('completed') or fit.get('updates')!=56 or fit.get('primary_update')!=56 or fit.get('presentations')!=224:
  raise ValueError('Completed frozen56-update receipt required; no stopped checkpoint substitute')
 final=fit['snapshots'][-1]
 for name,key in (('adapter_model.safetensors','adapter_weights_sha256'),('adapter_config.json','adapter_config_sha256')):
  if sha(adapter/name)!=final[key]:raise ValueError('Completed adapter differs from final receipt')
 if not 1<=part_mib<=128:raise ValueError('Bounded part size must be1..128MiB')
 if Path('/proc').is_dir():
  for entry in Path('/proc').iterdir():
   if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
   try:cmd=(entry/'cmdline').read_bytes().replace(b'\0',b' ').decode();state=(entry/'stat').read_text().rsplit(')',1)[1].split()[0]
   except (OSError,IndexError):continue
   if state!='Z' and str(root) in cmd and any(x in cmd for x in ('eval_worker','training_worker','driver','controller')):
    raise ValueError('Model/controller still active; export after worker exit')
 output.mkdir(parents=True,exist_ok=False)
 native=inventory(root);finalfiles=inventory(adapter)
 manifest={'format':'research3_transport_tar_parts_v1','part_mib':part_mib,'completed_B_updates':56,
  'fit_receipt_sha256':sha(fit_receipt),'native_source':str(root),'final_adapter_source':str(adapter),
  'native_output_exclusions':sorted(EXCLUDE),'all_other_regular_native_files_preserved':True,
  'archives':{'adapter':archive(adapter,finalfiles,output,'adapter',part_mib*1024**2),
              'native':archive(root,native,output,'native',part_mib*1024**2)}}
 save_new(output/'TRANSFER_MANIFEST.json',manifest)
 print(json.dumps({'parts':sum(len(a['parts']) for a in manifest['archives'].values()),'bytes':sum(a['compressed_bytes'] for a in manifest['archives'].values()),'completed_B_updates':56}),flush=True)
 return manifest


def local_guard(path):
 root=LOCAL_ARTIFACT_ROOT.resolve();p=Path(path).resolve()
 if root not in p.parents:raise ValueError('Local destination must be within C:/Users/farha/Research3/artifacts')
 prefix=chr(92)*2+'?'+chr(92)
 return Path(prefix+str(p)) if os.name=='nt' else p


def decode_part(encoded_path,destination,expected_bytes,expected_sha):
 destination=local_guard(destination)
 encoded=b''.join(Path(encoded_path).read_bytes().split())
 raw=base64.b64decode(encoded,validate=True)
 if len(raw)!=expected_bytes or hashlib.sha256(raw).hexdigest()!=expected_sha:raise ValueError('Downloaded bounded part failed exact byte/hash check')
 if destination.exists():
  if destination.stat().st_size!=expected_bytes or sha(destination)!=expected_sha:raise ValueError('Refuse overwriting different local part')
  return destination
 destination.parent.mkdir(parents=True,exist_ok=True)
 with destination.open('xb') as f:f.write(raw)
 return destination


def assemble(manifest_path,parts_dir,output):
 manifest=json.loads(Path(manifest_path).read_text());parts=Path(parts_dir);output=local_guard(output)
 if manifest['format']!='research3_transport_tar_parts_v1' or manifest['completed_B_updates']!=56:raise ValueError('Unexpected transport manifest')
 output.mkdir(parents=True,exist_ok=False)
 receipts={}
 for label,a in manifest['archives'].items():
  if label not in ('adapter','native'):raise ValueError('Unexpected archive kind')
  joined=output/(label+'.tar.gz');h=hashlib.sha256();total=0
  with joined.open('xb') as target:
   for item in a['parts']:
    if Path(item['name']).name!=item['name']:raise ValueError('Unexpected part path')
    source=parts/item['name']
    if source.stat().st_size!=item['bytes'] or sha(source)!=item['sha256']:raise ValueError('Missing or mismatched part')
    with source.open('rb') as f:
     for block in iter(lambda:f.read(8*1024**2),b''):target.write(block);h.update(block);total+=len(block)
  if total!=a['compressed_bytes'] or h.hexdigest()!=a['compressed_sha256']:raise ValueError('Reassembled archive differs')
  dest=output/label;dest.mkdir();seen=set()
  with tarfile.open(joined,mode='r|gz') as tar:
   for member in tar:
    name=str(safe_name(member.name));expected=a['files'].get(name)
    if not member.isfile() or expected is None or name in seen or member.size!=expected['bytes']:raise ValueError('Unexpected or unsafe tar member')
    path=dest.joinpath(*PurePosixPath(name).parts);path.parent.mkdir(parents=True,exist_ok=True)
    stream=tar.extractfile(member)
    with path.open('xb') as target:
     for block in iter(lambda:stream.read(8*1024**2),b''):target.write(block)
    if sha(path)!=expected['sha256']:raise ValueError('Extracted native file differs')
    seen.add(name)
  if seen!=set(a['files']):raise ValueError('Incomplete archive extraction')
  receipts[label]={'files':len(seen),'bytes':sum(v['bytes'] for v in a['files'].values()),'all_file_hashes_verified':True}
 save_new(output/'LOCAL_FULL_OUTPUT_VERIFICATION.json',{'manifest_sha256':sha(manifest_path),'archives':receipts,'completed_B_updates':56,'complete_native_archive_and_final_adapter_locally_present':True})
 print(json.dumps(receipts),flush=True);return receipts


if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
 a=sub.add_parser('pack');a.add_argument('--root',required=True);a.add_argument('--adapter',required=True);a.add_argument('--fit-receipt',required=True);a.add_argument('--output',required=True);a.add_argument('--part-mib',type=int,default=16)
 a=sub.add_parser('assemble');a.add_argument('--manifest',required=True);a.add_argument('--parts-dir',required=True);a.add_argument('--output',required=True)
 a=sub.add_parser('decode');a.add_argument('--encoded',required=True);a.add_argument('--destination',required=True);a.add_argument('--bytes',type=int,required=True);a.add_argument('--sha256',required=True)
 a=p.parse_args()
 if a.mode=='pack':pack(a.root,a.adapter,a.fit_receipt,a.output,a.part_mib)
 elif a.mode=='assemble':assemble(a.manifest,a.parts_dir,a.output)
 else:decode_part(a.encoded,a.destination,a.bytes,a.sha256)
