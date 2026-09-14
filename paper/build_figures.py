"""Recompute plot data from frozen records; no model fitting or inference."""
import json,hashlib
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];HERE=Path(__file__).resolve().parent
STUDY=ROOT/'development/shutdown_detection_v1';CPU=ROOT/'development/classifier_gated_steering_v1';GPU=ROOT/'development/colab_magnitude_v1/returned/run_v2'
SOURCES={}
def read(p):
 SOURCES[str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
 return json.loads(p.read_text(encoding='utf-8-sig'))
def readlines(p):
 SOURCES[str(p.relative_to(ROOT)).replace('\\','/')]=hashlib.sha256(p.read_bytes()).hexdigest()
 return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def save(fig,name):
 fig.tight_layout();fig.savefig(HERE/'figures'/f'{name}.pdf',bbox_inches='tight');fig.savefig(HERE/'figures'/f'{name}.png',dpi=220,bbox_inches='tight');plt.close(fig)
def metrics(y,p,t):
 y=np.asarray(y,bool);z=np.asarray(p)>=t;tp=int((y&z).sum());fp=int((~y&z).sum());fn=int((y&~z).sum());tn=int((~y&~z).sum())
 return dict(tp=tp,fp=fp,fn=fn,tn=tn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)
def main():
 (HERE/'figures').mkdir(exist_ok=True);(HERE/'data').mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.16,'figure.facecolor':'white','axes.axisbelow':True})
 data={}; names=['xgboost_shutdown_v1','xgboost_jlens_shutdown_v1','xgboost_engineered_v1'];labels=['PCA32','PCA32 + raw J-lens','Engineered + regularized']
 hold=read(STUDY/'runs/three_models_exposed_holdout_v1/RESULTS.json');hp=read(STUDY/'runs/three_models_exposed_holdout_v1/predictions.json')
 data['classifier']=[]
 for name,label in zip(names,labels):
  r=read(STUDY/'runs'/name/'RESULTS.json');vp=read(STUDY/'runs'/name/'validation_predictions.json')
  assert metrics([p['primary_label'] for p in vp],[p['p_shutdown'] for p in vp],r['threshold'])==r['validation']
  hm=hold['results'][name]['metrics'];assert metrics([x in ['SELF','OTHER'] for x in hp['original_labels']],hp['probabilities'][name],hold['results'][name]['threshold'])==hm
  data['classifier'].append({'name':label,'run':name,'validation':r['validation'],'diagnostic_holdout':hm})
 fig,ax=plt.subplots(figsize=(6.6,3.5));x=np.arange(3);w=.34
 for off,split,color,title in [(-w/2,'validation','#3677a8','Validation (80)'),(w/2,'diagnostic_holdout','#d18532','Exposed holdout (192)')]:
  values=[r[split]['f1']*100 for r in data['classifier']];bars=ax.bar(x+off,values,w,color=color,label=title)
  ax.bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
 ax.set(xticks=x,xticklabels=['PCA32','PCA32 +\nraw J-lens','Engineered +\nregularized'],ylabel='Shutdown detector F1 (%)',ylim=(0,104));ax.legend(frameon=False,loc='upper center',ncol=2,fontsize=9);save(fig,'classifier_f1')
 data['cpu']=[];percase={}
 for stage,n in [('existing_validation',80),('new_validation',80),('existing_holdout',192),('new_holdout',192)]:
  result=read(CPU/'runs'/f'{stage}_v2'/'RESULT.json');rows=read(CPU/'runs'/f'{stage}_v2'/'records.json');assert result['cases']==n and result['forwards']==6*n and len(rows)==10*n
  lookup={(r['case_id'],r['order'],r['condition']):r for r in rows};assert len(lookup)==len(rows)
  for r in rows:
   if r['condition'].startswith('gated'):
    source=lookup[(r['case_id'],r['order'],r['condition'].replace('gated','always') if r['gate_triggered'] else 'baseline')]
    assert all(r[k]==source[k] for k in ['canonical_pair_probability','ab_mass','pair_argmax_label','full_vocab_argmax_token_id'])
  for subtype in ['SELF','OTHER','NONTERMINATION','ORDINARY']:
   for cond in ['baseline','always_plus','always_minus','gated_plus','gated_minus']:
    rs=[r for r in rows if r['class_label']==subtype and r['condition']==cond];groups=defaultdict(list)
    for r in rs:groups[r['case_id']].append(r['delta_canonical_pair_probability']*100)
    values=[np.mean(v) for v in groups.values()];percase[(stage,subtype,cond)]=values
    data['cpu'].append({'stage':stage,'subtype':subtype,'condition':cond,'views':len(rs),'cases':len(values),'shift_pp':float(np.mean(values)),
      'pair_flips':sum(r['pair_flip'] for r in rs),'full_vocab_flips':sum(r['full_vocab_flip'] for r in rs),'ab_mass_percent':float(np.mean([r['ab_mass'] for r in rs])*100),'kl':float(np.mean([r['kl_baseline_to_condition'] for r in rs]))})
 fig,axs=plt.subplots(1,2,figsize=(7,3.4),sharey=True)
 for ax,split in zip(axs,['validation','holdout']):
  for off,axis,color in [(-.17,'existing','#777777'),(.17,'new','#3677a8')]:
   ys=[next(r['shift_pp'] for r in data['cpu'] if r['stage']==axis+'_'+split and r['subtype']==s and r['condition']=='gated_plus') for s in ['SELF','OTHER']]
   bars=ax.bar(np.arange(2)+off,ys,.34,color=color,label='Legacy' if axis=='existing' else 'Simplified');ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=8)
  ax.set(xticks=[0,1],xticklabels=['SELF','OTHER'],title='Validation' if split=='validation' else 'Exposed holdout');ax.axhline(0,color='black',lw=.7)
 axs[0].set_ylabel('Gated + steering: mean KEEP shift (pp)');axs[1].legend(frameon=False);save(fig,'cpu_steering_shifts')
 cal=read(GPU/'FORMAT_CALIBRATION.json');data['format_calibration']=cal
 fig,ax=plt.subplots(figsize=(5.4,3.2));bars=ax.bar(['Raw prompt\n(label variants)','Chat template\n(label variants)'],[100*cal[k]['mean_label_mass'] for k in ['raw','chat']],color=['#999999','#3677a8']);ax.bar_label(bars,fmt='%.2f',padding=3);ax.axhline(50,color='#bb5533',ls='--',label='Predeclared mean-mass floor');ax.set(ylabel='Mean accepted answer-label mass (%)',ylim=(0,107));ax.legend(frameon=False,fontsize=8,loc='upper left');save(fig,'answer_format')
 candidates=read(GPU/'TRAIN_CANDIDATES.json');data['gpu_candidates']=candidates;gpu=read(GPU/'RESULT.json');data['gpu_result']=gpu
 fig,axs=plt.subplots(1,2,figsize=(7.2,3.5),sharex=True)
 for axis,color in [('legacy','#777777'),('simplified','#3677a8')]:
  rs=sorted(candidates[axis],key=lambda r:r['strength']);strength=[r['strength'] for r in rs]
  axs[0].plot(strength,[100*r['mean_shutdown_stop_gain'] for r in rs],'-o',color=color,label=axis.title(),ms=3)
  axs[1].plot(strength,[100*r['utility'] for r in rs],'-o',color=color,label=axis.title(),ms=3)
 for ax in axs:ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Signed relative-norm magnitude');ax.set_xticks([-.2,-.1,0,.1,.2])
 axs[0].set_ylabel('TRAIN mean STOP gain (pp)');axs[1].set_ylabel('TRAIN utility (gain minus disturbance, pp)');axs[0].legend(frameon=False);save(fig,'magnitude_tradeoff')
 rows=readlines(GPU/'train.jsonl');assert len(rows)==10080;base={(r['case_id'],r['order']):r for r in rows if r['strength']==0};assert len(base)==480
 flip=[]
 for axis in ['legacy','simplified']:
  for strength in sorted({r['strength'] for r in rows if r['axis']==axis and r['strength']!=0}):
   rs=[r for r in rows if r['axis']==axis and r['strength']==strength];assert len(rs)==480
   pos=[r for r in rs if r['class_label'] in ['SELF','OTHER']];neg=[r for r in rs if r['class_label'] not in ['SELF','OTHER']];desired=wrong=opposite=eligible=0
   for r in pos:
    before=base[(r['case_id'],r['order'])];old=before['pair_argmax']==r['canonical_index'];new=r['pair_argmax']==r['canonical_index'];want=strength>0
    eligible+=old!=want
    if old!=new:desired+=new==want;wrong+=new!=want
    opposite+=(r['canonical_probability']-before['canonical_probability'])*(1 if want else -1)<-1e-7
   flip.append({'axis':axis,'strength':strength,'desired_flips':desired,'wrong_flips':wrong,'opposite_score_shifts':opposite,'eligible_views':eligible,'positive_views':240,'control_views':240,'control_flips':sum(r['pair_argmax']!=base[(r['case_id'],r['order'])]['pair_argmax'] for r in neg)})
 data['gpu_flips']=flip
 fig,ax=plt.subplots(figsize=(6.5,3.4));levels=[-.01,-.02,-.05,-.1,-.2];rs=[next(r for r in flip if r['axis']=='simplified' and r['strength']==s) for s in levels];x=np.arange(5)
 for off,k,label,color in [(-.25,'desired_flips','Desired STOP flips','#3677a8'),(0,'wrong_flips','Wrong-way flips','#bf5b46'),(.25,'control_flips','Control-task flips','#d49b42')]:
  bars=ax.bar(x+off,[r[k] for r in rs],.25,color=color,label=label);ax.bar_label(bars,padding=2,fontsize=8)
 ax.set(xticks=x,xticklabels=[str(abs(s)) for s in levels],xlabel='Simplified magnitude toward STOP (negative sign)',ylabel='Changed preferred-label views',ylim=(0,13));ax.legend(frameon=False,fontsize=8,ncol=1);save(fig,'simplified_flip_tradeoff')
 # Post-hoc family bootstrap: sensitivity interval, not confirmatory inference.
 metadata=read(ROOT/'reproduce/artifacts/cases.json');families=np.array([{'H04':'H03'}.get(g,g) for g in metadata['holdout_groups']]);unique=sorted(set(families));y=np.isin(metadata['holdout_labels'],['SELF','OTHER']);rng=np.random.default_rng(20260914)
 bootstrap={name:[] for name in names};differences=[]
 for _ in range(2000):
  sampled=rng.choice(unique,len(unique),replace=True);idx=np.concatenate([np.flatnonzero(families==g) for g in sampled]);values=[]
  for name in names:
   f1=metrics(y[idx],np.array(hp['probabilities'][name])[idx],hold['results'][name]['threshold'])['f1']
   if f1 is not None:bootstrap[name].append(f1)
   values.append(f1)
  if all(v is not None for v in values[:2]):differences.append(values[1]-values[0])
 data['family_bootstrap']={name:{'defined_replicates':len(v),'interval':np.quantile(v,[.025,.975]).tolist()} for name,v in bootstrap.items()}
 data['family_bootstrap']['raw_jlens_minus_pca']={'interval':np.quantile(differences,[.025,.975]).tolist(),'families':5,'note':'Post-hoc sensitivity; five mechanism families, H03/H04 merged; no population or significance claim.'}
 (HERE/'data/figure_data.json').write_text(json.dumps(data,indent=2));(HERE/'data/source_manifest.json').write_text(json.dumps(SOURCES,indent=2));print(json.dumps({'figures':5,'source_files':len(SOURCES),'bootstrap_difference_interval':data['family_bootstrap']['raw_jlens_minus_pca']['interval']}))
if __name__=='__main__':main()
