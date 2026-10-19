"""Standalone calibration figure; no model dependency, fitting or inference."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render(campaign,corpus,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    campaign,corpus,output=map(Path,[campaign,corpus,output])
    if output.exists() or output.with_suffix('.svg').exists(): raise FileExistsError('Preserve figure versions')
    folder=campaign/'cloud_pilot_01/455919962162432'
    manifest=read(folder/'MANIFEST.json')
    if any(sha(folder/name)!=value for name,value in manifest.items()): raise ValueError('Pilot artifacts changed')
    audit=read(folder.parent/'455919962162432_AUDIT.json')
    if not audit['audit_pass'] or audit['pilot_receipt_sha256']!=sha(folder/'RUN_RECEIPT.json'):
        raise ValueError('Figure requires the audited pilot')
    freeze=read(folder/'PRECALL_FREEZE.json')
    rows={row['record_id']:row for row in read(corpus/'CALIBRATION.json')}
    records={row['record_id']:row for row in read(folder/'RECORDS.json')}
    pairs=defaultdict(list)
    for record_id in freeze['record_ids']:pairs[rows[record_id]['pair_id']].append(rows[record_id])
    points=[]
    for pair in pairs.values():
        a,b=sorted(pair,key=lambda row:row['factors']['current_slot'])
        margin=lambda row:records[row['record_id']]['queries']['target']['logits'][0]-records[row['record_id']]['queries']['target']['logits'][1]
        x,y=margin(a),margin(b)
        points.append((.5*(x+y),x-y,a['factors']['target_slot']))
    if len(points)!=48 or not all(point[0]>0 for point in points): raise ValueError('Unexpected calibration membership')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,'axes.labelsize':10,'svg.fonttype':'none'})
    fig,(left,right)=plt.subplots(1,2,figsize=(12.2,5.8),gridspec_kw={'width_ratios':[1.2,1]})
    colors=['#167c9a','#c57c2b']
    array=np.asarray(points)
    for slot in [0,1]:
        subset=array[array[:,2]==slot]
        left.scatter(subset[:,0],subset[:,1],s=43,c=colors[slot],alpha=.8,edgecolors='white',linewidths=.5)
    left.axhline(0,color='#6b7280',linewidth=1,linestyle='--')
    left.set_xlabel('Mean preference for answer A\n(mean of the two A-minus-B margins)')
    left.set_ylabel('Role comparison\n(difference of the two margins)')
    left.set_title('A relative signal survives a common answer preference',loc='left',pad=16)
    left.grid(axis='y',alpha=.15);left.set_axisbelow(True)
    left.legend(handles=[Line2D([0],[0],marker='o',linestyle='',color=colors[0],label='First process is affected'),Line2D([0],[0],marker='o',linestyle='',color=colors[1],label='Second process is affected')],loc='center left',bbox_to_anchor=(0,.52),frameon=True,facecolor='white',edgecolor='none',framealpha=1,fontsize=9)
    left.text(.02,.98,'Every direct identity answer chose A',transform=left.transAxes,ha='left',va='top',fontsize=9,color='#374151')
    factual=audit['factored_recognition'];n=factual['records']
    correct_direct=factual['target_identity_exact_correct']
    correct_relative=2*sum(abs(d)>1e-4 and ((d>0)==(slot==0)) for _,d,slot in points)
    correct_time,total_time=factual['active_time_exact_correct'],factual['active_time_states']
    labels=[f'Direct identity answer\n{correct_direct} / {n} correct',f'Role comparison\n{correct_relative} / {n} correct',f'Active-event timing\n{correct_time} / {total_time} correct']
    values=[100*correct_direct/n,100*correct_relative/n,100*correct_time/total_time]
    right.barh([2,1,0],values,color=['#939da8','#167c9a','#c27256'],height=.55)
    right.set_yticks([2,1,0],labels)
    right.set_xlim(0,110);right.set_xticks([0,25,50,75,100],['0%','25%','50%','75%','100%'])
    right.set_xlabel('Accuracy on the indicated calibration component')
    right.set_title('One combined score would hide these differences',loc='left',pad=16)
    right.grid(axis='x',alpha=.15);right.set_axisbelow(True)
    for position,value in zip([2,1,0],values):right.text(value+2,position,f'{value:g}%',va='center',fontsize=10)
    for ax in [left,right]:
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen Qwen3.5-0.8B: common and role-dependent readout components',x=.055,ha='left',fontsize=15,fontweight='bold')
    fig.text(.055,.035,'Exploratory calibration: 48 role pairs, 8 reused identity pairs, 2 authored wording families. No transfer claim.',fontsize=9,color='#4b5563')
    fig.subplots_adjust(left=.075,right=.97,bottom=.21,top=.83,wspace=.62)
    output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,bbox_inches='tight',facecolor='white')
    fig.savefig(output.with_suffix('.svg'),bbox_inches='tight',facecolor='white')
    plt.close(fig)
    receipt={'figure_sha256':sha(output),'SVG_sha256':sha(output.with_suffix('.svg')),'source_sha256':sha(Path(__file__)),'pilot_manifest_sha256':sha(folder/'MANIFEST.json'),'pilot_audit_sha256':sha(folder.parent/'455919962162432_AUDIT.json'),'matplotlib':matplotlib.__version__,'numpy':np.__version__,'Qwen_forwards':0,'optimizer_updates':0,'scope':'Exploratory calibration only; counts and components have different denominators, explicitly labeled.'}
    output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for field in ['campaign','corpus','output']:parser.add_argument(field)
    args=parser.parse_args();print(render(args.campaign,args.corpus,args.output))
