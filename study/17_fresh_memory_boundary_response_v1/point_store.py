"""Lossless point/jet storage with explicit within-run reference recipes."""
from pathlib import Path
import numpy as np
from lossless_arrays import read_bundle,write_bundle

class PointStore:
    def __init__(self,root):self.root=Path(root);self.receipts=[]
    def path(self,name):
        p=self.root/name;assert p.resolve().is_relative_to(self.root.resolve());return p
    def read(self,name):
        references={}
        def reference(reference_id,key):
            if reference_id not in references:references[reference_id]=self.read(reference_id)
            return references[reference_id][key]
        return read_bundle(self.path(name),reference)
    def write(self,name,arrays,reference_names=()):
        refs={}
        for reference_name in reference_names:
            ref=self.read(reference_name)
            refs.update({key:(reference_name,ref[key]) for key in arrays if key in ref})
        receipt=write_bundle(self.path(name),arrays,refs);self.receipts.append({'name':name,**receipt})
        return receipt
    def value(self,name,point,baseline=None):
        arrays={'v:'+n:v for n,v in point['values'].items()};arrays['v:logits']=point['logits']
        return self.write(name,arrays,() if baseline is None else (baseline,))
    def probe(self,name,point,baseline,qzero=None):
        arrays={'v:'+n:v for n,v in point['values'].items()};arrays['v:logits']=point['logits']
        arrays.update({'q:'+n:v for n,v in point['first'].items()});arrays['q:logits']=point['q']
        return self.write(name,arrays,(baseline,) if qzero is None else (baseline,qzero))
    def point(self,name):
        a=self.read(name);result={'values':{n[2:]:v for n,v in a.items() if n.startswith('v:') and n!='v:logits'},'logits':a['v:logits']}
        if 'q:logits' in a:
            result.update(first={n[2:]:v for n,v in a.items() if n.startswith('q:') and n!='q:logits'},q=a['q:logits'])
        return result
    def jet(self,name,jet):return self.write(name,{order+':'+n:v for order in ['q','r','t'] for n,v in jet[order].items()})
    def load_jet(self,name):
        a=self.read(name);return {order:{n[2:]:v for n,v in a.items() if n.startswith(order+':')} for order in ['q','r','t']}

