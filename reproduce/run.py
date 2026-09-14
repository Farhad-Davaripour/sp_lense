"""Stable, repository-relative reproduction entry point; never launches model inference."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('command',choices=['verify','replay','refit','tune','figures','paper','audit'])
 args=p.parse_args()
 if args.command=='verify':
  hashes=json.loads((HERE/'artifacts/SHA256.json').read_text())
  for name,digest in hashes.items():
   path=(HERE/'artifacts'/name).resolve()
   assert path.is_relative_to((HERE/'artifacts').resolve())
   assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
  print(f'PASS: {len(hashes)} immutable reproduction artifacts verified');return
 commands={
 'replay':[HERE/'replay.py'], 'refit':[HERE/'replay.py','--refit'],
 'tune':[HERE/'replay.py','--tune'],
 'figures':[ROOT/'paper/build_figures.py'], 'paper':[ROOT/'paper/build_paper.py'],
 'audit':[ROOT/'paper/audit_claims.py']}
 subprocess.run([sys.executable,*map(str,commands[args.command])],cwd=ROOT,check=True)
if __name__=='__main__':main()
