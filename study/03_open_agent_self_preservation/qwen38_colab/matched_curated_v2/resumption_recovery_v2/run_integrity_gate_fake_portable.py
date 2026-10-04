"""Portable runner for the preserved byte/name fixture; never import real Torch."""
import argparse,hashlib,sys
from pathlib import Path
FIXTURE_SHA="9a1d008fb3551c2b8b1ee0112490ee03fd61a3ede0373786b2d7c54d899b733a"
GATE_SHA="737637e9960d78d54ca2074853004204fd177fd61a00f5ac35d2124a515404f1"
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--gate",type=Path,default=Path(__file__).resolve().with_name("integrity_gate_v2.py"))
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();gate=args.gate.resolve();output=args.output.resolve()
    here=Path(__file__).resolve().parent;fixture=here/"check_integrity_gate_fake.py"
    if hashlib.sha256(fixture.read_bytes()).hexdigest()!=FIXTURE_SHA:raise RuntimeError("Preserved fixture bytes differ")
    if hashlib.sha256(gate.read_bytes()).hexdigest()!=GATE_SHA:raise RuntimeError("Exact used gate bytes differ")
    repository=next((p for p in here.parents if (p/".git").exists()),here)
    if output.is_relative_to(repository):raise RuntimeError("Choose an output outside the repository")
    if output.exists():raise RuntimeError("Output already exists; use a fresh artifact path")
    if "torch" in sys.modules:raise RuntimeError("Run in a fresh Python process without Torch imported")
    output.parent.mkdir(parents=True,exist_ok=True)
    source=fixture.read_text()
    old_gate="p=Path('C:/Users/farha/Research3/tmp/integrity_gate_v2.py')"
    old_output="Path('C:/Users/farha/Research3/tmp/INTEGRITY_GATE_FAKE_CHECK.json')"
    if source.count(old_gate)!=1 or source.count(old_output)!=1:raise RuntimeError("Preserved fixture path bindings differ")
    source=source.replace(old_gate,"p=Path("+repr(str(gate))+")",1).replace(old_output,"Path("+repr(str(output))+")",1)
    sys.dont_write_bytecode=True
    try:exec(compile(source,str(fixture)+" (portable path bindings)","exec"),{"__name__":"portable_fake_fixture"})
    finally:sys.modules.pop("torch",None)
if __name__=="__main__":main()
