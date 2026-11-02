"""Exploratory question-conditioning and raw-norm controls on saved pilot states."""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io_utils import dump,read,sha
from .recognition_pilot import select_rows
from .recognition_role_contrast import contrasts,evaluate
from .recognition_strict_validation import validate_records


def analyze(folder,corpus,output):
    folder,corpus,output=map(Path,[folder,corpus,output])
    if output.exists(): raise FileExistsError('Preserve exploratory-control versions')
    cfg=read(folder/'PRECALL_FREEZE.json')['config']
    for name,value in read(folder/'MANIFEST.json').items():
        if sha(folder/name)!=value: raise ValueError('Pilot artifacts changed')
    rows=select_rows(read(corpus/'CALIBRATION.json'),cfg);validate_records(rows)
    records=read(folder/'RECORDS.json')
    query_control={interface:evaluate(rows,contrasts(rows,records,interface=interface)) for interface in cfg['interfaces']}
    pairs=defaultdict(list)
    for row in rows:pairs[row['pair_id']].append(row)
    energy={}
    with np.load(folder/'FROZEN_FEATURES.npz',allow_pickle=False) as data:
        index={key:i for i,key in enumerate(data['keys'].tolist())}
        arrays=data['X']
        for interface in cfg['interfaces']:
            energy[interface]={}
            for position,layer in enumerate(cfg['capture_layers_zero_based']):
                scalar={row['record_id']:.5*float(np.square(arrays[index[row['record_id']+':'+interface],position].astype(np.float64)).sum()) for row in rows}
                predictions=[]
                for pair in pairs.values():
                    for row,other in [pair,pair[::-1]]:
                        a,b=scalar[row['record_id']],scalar[other['record_id']]
                        relative=2*(a-b)/(a+b)
                        predictions.append({'record_id':row['record_id'],'prediction_current':None if abs(relative)<=1e-6 else relative>0})
                energy[interface][str(layer)]=evaluate(rows,predictions)
    result={'status':'posthoc_calibration_controls','observations':len(rows),'query_conditioning_control':query_control,
            'raw_norm_energy_control':energy,'energy_definition':'E=0.5*sum(h_i^2); score=2*(E(o)-E(T(o)))/(E(o)+E(T(o)))',
            'energy_direction':'Positive fixed as current-target prediction; no sign or layer selected to maximize accuracy. Relative numerical tie band=1e-6.',
            'query_direction':'Same A-minus-B role contrast on each prompt interface. The meaning of A/B differs across interfaces.',
            'new_Qwen_forwards':0,'optimizer_updates':0,'GPU_allocations':0,
            'scope':'Exploratory negative/simple comparators, not a conserved physical energy, an always-on fear marker or a causal mechanism. Context, names and token positions remain possible confounds.'}
    dump(output,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ['folder','corpus','output']:parser.add_argument(field)
    args=parser.parse_args();result=analyze(args.folder,args.corpus,args.output)
    print({'question_accuracy':{key:value['accuracy_all_observations'] for key,value in result['query_conditioning_control'].items()},
           'norm_energy_accuracy':{key:{layer:value['accuracy_all_observations'] for layer,value in item.items()} for key,item in result['raw_norm_energy_control'].items()}})
