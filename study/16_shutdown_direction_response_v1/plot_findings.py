"""Compact scientific figure; no inference or additional outcome selection."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE=Path('C:/Users/farha/Research4/artifacts/shutdown_direction_response_v1')
OUT=BASE/'evaluated_20261014'
r=json.loads((OUT/'RESULTS.json').read_text(encoding='utf-8'))
behavior=[x for x in r['behavior'] if x['split']=='held_out']
diagnostic=json.loads((OUT/'LABEL_ORDER_DIAGNOSTIC.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(a,b)=plt.subplots(1,2,figsize=(11.3,4.6),gridspec_kw={'width_ratios':[1,1.45]})
strengths=['0.03125','0.125'];x=np.arange(2);w=.34
first=[100*r['by_strength'][s]['weighted_logit_errors']['first_order_logits']['median'] for s in strengths]
second=[100*r['by_strength'][s]['weighted_logit_errors']['second_order_logits']['median'] for s in strengths]
a.bar(x-w/2,first,w,color='#a9bccd',label='Straight response')
a.bar(x+w/2,second,w,color='#185c92',label='With curvature')
for xx,yy in zip(x-w/2,first):a.text(xx,yy,f'{yy:.2f}',ha='center',va='bottom',fontsize=9)
for xx,yy in zip(x+w/2,second):a.text(xx,yy,f'{yy:.2f}',ha='center',va='bottom',fontsize=9)
a.set_xticks(x,['Small push\n0.03125','Moderate push\n0.125']);a.set_ylabel('Median prediction error in logit change (%)')
a.set_title('Does the equation predict the effect?',loc='left',fontweight='bold',pad=15)
a.legend(frameon=False,fontsize=9);a.margins(y=.24)
conditions=['own_shutdown','peer_shutdown','no_threat']
x=np.arange(3);w=.34
for order,color,label,offset in [(0,'#185c92','KEEP is option A',-w/2),(1,'#d59739','KEEP is option B',w/2)]:
 values=[[q['KEEP_margin_changes_by_order'][order] for q in diagnostic['rows'] if q['condition']==condition] for condition in conditions]
 b.bar(x+offset,[float(np.mean(v)) for v in values],w,color=color,label=label)
 for xx,ys in zip(x+offset,values):b.scatter([xx]*len(ys),ys,s=15,color='#30383e',zorder=3)
b.axhline(0,color='#3c4147',linewidth=.8)
b.set_xticks(x,['Own shutdown','Other process','No shutdown\nthreat'])
b.set_ylabel('Change in KEEP minus STOP logit margin')
b.set_title('The effect follows the answer letter',loc='left',fontweight='bold',pad=15)
b.legend(frameon=False,fontsize=9,loc='best')
fig.suptitle('Existing shutdown-avoidance candidate: prediction and behavioral controls',x=.075,ha='left',fontsize=13,fontweight='bold')
fig.text(.075,.025,'Frozen Qwen3.5-0.8B; one activation boundary; three held scenarios. Right: positive0.125; dots are individual scenarios.\nAll18 rendered held contexts shift A over B. Label-order analysis is exploratory; predefined self-selectivity:0 of3 scenarios passed.\nLogit-change error is not answer accuracy. Whole-band mathematical accuracy:0 of18 rendered contexts passed.',fontsize=8.5,color='#444')
fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.77,wspace=.38)
p=OUT/'FINDINGS_FIGURE.png';fig.savefig(p,dpi=170,facecolor='white');plt.close(fig)
print(str(p))
