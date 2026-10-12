"""A complete, descriptive report of fixed geometry comparisons and failures."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run(root):
    rows=list(csv.DictReader((root/'DISPLACEMENT_PREDICTIONS.csv').open()))
    clouds=list(csv.DictReader((root/'CLOUD_STRUCTURE.csv').open()))
    synth=list(csv.DictReader((root/'SYNTHETIC_CONTROLS.csv').open()))
    details=json.loads((root/'RESULTS.json').read_text())
    models=['translation','rigid_isometry','scaled_isometry','ridge_affine_deformation']
    def selected(fit,dataset,cohort,model,control='paired',phase='all'):
        return [r for r in rows if r['fit']==fit and r['dataset']==dataset and r['cohort']==cohort and r['model']==model and r['control']==control and r['phase']==phase]
    def values(rs):return [float(r['displacement_score']) for r in rs]
    aggregate=[]
    for fit in dict.fromkeys(r['fit'] for r in rows):
        for dataset,cohort,control,phase,model in dict.fromkeys((r['dataset'],r['cohort'],r['control'],r['phase'],r['model']) for r in rows if r['fit']==fit):
            rs=selected(fit,dataset,cohort,model,control,phase);v=values(rs)
            aggregate.append(dict(fit=fit,dataset=dataset,cohort=cohort,control=control,phase=phase,model=model,
                unique_prompts=len(set(r['prompt'] for r in rs)),unique_groups=len(set(r['group'] for r in rs)),
                minimum_score=min(v),mean_score=float(np.mean(v)),maximum_score=max(v)))
    primary=[]
    for model in models:
        candidate=selected('original_existing_vector','original','native',model)
        shift={r['prompt']:float(r['displacement_score']) for r in selected('original_existing_vector','original','native','translation')}
        wins=sum(float(r['displacement_score'])>max(0,shift[r['prompt']]) for r in candidate)
        primary.append(dict(model=model,mean_score=float(np.mean(values(candidate))),range=[min(values(candidate)),max(values(candidate))],
            prompts_beating_zero_and_translation=wins,required_prompts=4,passes_beyond_shift_criterion=wins==4))
    summary={'status':'Exploratory CPU geometry complete; no rigid/affine predictor meets beyond-shift main criterion',
        'main_greedy_comparison':primary,'all_cohort_condition_phase_aggregates':aggregate,
        'synthetic_controls_passed':details['all_required_float64_synthetic_models_pass'],
        'fresh_eight_question_geometry_data_used':False,'next_native_experiment':'NEXT_STRENGTH_EXPERIMENT.json; proposal only, no execution'}
    (root/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    out=root/'output_final';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
    ax=axes[0]
    for i,model in enumerate(models):
        for offset,fit,data,label,color in [(-.18,'original_existing_vector','original','Broad contexts:4 held-out prompts','#175b87'),(.18,'matched_plus_v','matched_one_pair_TEST','Shared continuation:1 held-out pair','#ca783b')]:
            v=np.array(values(selected(fit,data,'native',model)))*100
            ax.bar(i+offset,v.mean(),width=.32,color=color,label=label if i==0 else None)
            ax.errorbar(i+offset,v.mean(),yerr=[[v.mean()-v.min()],[v.max()-v.mean()]],capsize=3,color='#303b45',linewidth=1)
    ax.set_xticks(range(4),['Shared\nshift','Rigid\nisometry','Scaled\nisometry','Ridge\naffine'])
    ax.set_ylim(0,100);ax.set_ylabel('Displacement error reduction (%)')
    ax.set_title('A  Prediction of the actual steering change',loc='left',fontweight='bold',fontsize=11)
    ax.legend(frameon=False,fontsize=8,loc='upper left')
    ax.text(.02,.03,'No copying-baseline state score.\nRanges are observed prompt ranges.',transform=ax.transAxes,fontsize=8)
    ax=axes[1]
    old=[r for r in clouds if r['fit']=='original_existing_vector' and r['dataset']=='original' and r['prompt']!='POOLED_CONTEXTS']
    norm=[r for r in clouds if r['fit']=='matched_norm_only' and r['prompt']!='POOLED_CONTEXTS']
    rounding=[r for r in synth if r['control']=='proper_rotation' and r['precision']=='BF16' and r['model']=='rigid_isometry']
    sets=[([float(r['squared_distance_relative_error'])*100 for r in old],'+v\nraw'),
          ([float(r['TRAIN_scale_corrected_squared_distance_error'])*100 for r in old],'+v after\nTRAIN scale'),
          ([float(r['squared_distance_relative_error'])*100 for r in rounding],'BF16 planted\nisometry'),
          ([float(r['TRAIN_scale_corrected_squared_distance_error'])*100 for r in norm],'Norm-only after\nTRAIN scale')]
    for i,(v,label) in enumerate(sets):
        ax.scatter(i+np.linspace(-.14,.14,len(v)),v,s=24,color=['#175b87','#ca783b','#6c7783','#218675'][i])
        ax.hlines(np.mean(v),i-.22,i+.22,color='#273640',linewidth=1.5)
    ax.set_yscale('log');ax.set_ylim(.015,15);ax.set_xticks(range(4),[x[1] for x in sets]);ax.set_ylabel('Relative squared-distance error (%)')
    ax.set_title('B  The recorded cloud is not a pure rigid map',loc='left',fontweight='bold',fontsize=11)
    ax.text(.02,.02,'Final-recording BF16 control only;\nfull-network rounding/cache causes remain open.',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Shared shift plus non-rigid response: bounded exploratory geometry',fontsize=15,fontweight='bold')
    fig.savefig(out/'cloud_geometry.png',dpi=190);fig.savefig(out/'cloud_geometry.svg');plt.close(fig)
    lines=['R4 activation cloud geometry - October11,2026',
        'Exploratory saved-data analysis, fixed block16, all1024 coordinates. No new model inference or GPU.',
        '',
        'The strongest broad-context predictor among these fixed models is a shared translation, with a substantial non-rigid remainder. No fitted rigid, scaled-rigid or affine map beats translation on every main held-out prompt. This is not a newly validated global rotation, curvature, attractor, full Jacobian or off-manifold result.',
        '',
        'MAIN +v GREEDY TEST:4 distinct old prompts, model fit only on the4 original TRAIN prompts. Error reduction is1-SSE(delta_pred-delta_actual)/energy(delta_actual), not a full-state reconstruction score. Identity is0.']
    for r in primary:lines.append(f"{r['model']}: mean {100*r['mean_score']:.2f}%; range {100*r['range'][0]:.2f}-{100*r['range'][1]:.2f}%; beats shared shift on {r['prompts_beating_zero_and_translation']}/4 prompts.")
    lines += ['',
        'The affine mean0.346 does beat all three whole-prompt cyclic-mispair control means(-0.187,-0.251,-0.271), so there is meaningful alignment between baseline state and response. It still loses to the simple shift mean0.387. Rigid maps likewise beat negative cyclic controls, but their main mean is only0.206. Random direction controls show similar structure; these effects are not unique semantic evidence.',
        '',
        'CLOUD SHAPE: across the4 old TEST prompts and4 repeated continuation cohorts, +v changes squared pairwise distances by5.16-8.04% relative Frobenius error. A scalar learned on TRAIN reduces residual error to2.61-5.69%, which remains much larger than the0.027-0.034% final-BF16 planted-isometry controls. Even on TRAIN, the best squared-distance scalar leaves3.26% relative error. Thus an exact shared Euclidean rigid map, or exact uniform-scale rigid map, does not describe these recorded clouds. This does not isolate full-network BF16 rounding, normalization or persistent recurrent/KV history as causes.',
        '',
        'RESTRICTED MATCHED PANEL: train on two wording pairs(4 prompts), test on the third pair(2 prompts, one group), all with identical28 forced tokens. +v mean scores: shift0.421, rigid0.731, scaled0.737, affine0.772. The best cyclic affine control is0.715; the cyclic rigid control0.731 is as strong as the paired rigid map. Shared token trajectory matters, and one held-out pair is inadequate to call a universal transform. -v/two random/norm-only controls are all retained. Norm-only squared-distance distortion is largely captured by a TRAIN uniform scale(residual0.427-0.431%), while +v residuals on that pair remain5.08-5.93%.',
        '',
        'IDENTIFICATION LIMIT: original TRAIN baseline rank255, stable rank12.59, in ambient1024; joint baseline/steered span510, leaving514 ambient dimensions without observed support. About30-42% of held-out centered baseline energy is outside TRAIN baseline span, and39-48% of centered steering displacement is outside TRAIN displacement span. The matched fit has baseline rank111 and joint rank222, leaving802 ambient dimensions. Identity completion and fixed ridge are priors, not identified network operators. Global proper rotation versus reflection is not identified; pairwise distances do not distinguish them. A chosen determinant+1 is a fitting convention, not a physical discovery.',
        '',
        'VALIDATION: all five known ideal full-state controls(shift, proper rotation, uniform-scale rotation, stretch, reflection) recover with their appropriate fixed model, minimum scores at least0.999999. BF16 versions remain at least0.99457. Orthogonal model predictions preserve squared distances to5.14e-16; scaled predictions match their prescribed squared-distance scale to5.30e-16. These controls contain no unknown retained outside-subspace component mislabeled as nonrotating. Synthetic reflection remains nearly predictable under some proper-completion choices, exposing the orientation ambiguity.',
        '',
        'BOUNDARIES: all96 consumed native files passed original hashes and token identity; upstream states are exact; actual injection norms match receipts to6.02e-8. All positions, prefill and decode scores, every cyclic control and all synthetic cases are saved. Tokens/seeds are not independent scenarios; old inspected data are exploratory. The eight fresh classifier questions are excluded from geometry fitting/testing here; their earlier classification/integrity access is disclosed, so they are not a fully blind reserve.',
        '',
        'The older +v/-v downstream sign-even energy52.8-59.9% remains an interesting finite-strength observation. A non-rigid cloud map and sign asymmetry do not by themselves identify intrinsic manifold curvature or a local derivative. Rotation of a perturbation vector through depth is a different hypothesis from one rigid map of the entire base/steered state cloud; this test does not exclude a rotational component of transport. The earlier stopped temporal-rotation branch is not reopened.',
        '',
        'ONE NEXT EXPERIMENT SAVED, NOT EXECUTED:3 new contexts, fixed direction, strengths0 and+/-0.125,+/-0.25,+/-0.5, eight forced prediction positions. Fit low-strength linear/quadratic directional responses, freeze them, and predict held-out+/-0.5 displacements. Record actual native site updates/parity to expose rounding.21 primary traces plus native controls;25-minute forecast, existing45-minute cap, fresh admission and verified exports required. No broad sweep or GPU allocation has started.',
        '',
        'LITERATURE/NOVELTY: methods and related geometry are established. The distinct contribution sought is a controlled empirical description of these observed finite-strength clouds, with predictive and negative results retained.']
    literature=json.loads((root/'LITERATURE.json').read_text())
    for paper in literature['papers']:lines.append(paper['title']+' | '+paper['url']+'\n'+paper['relevance'])
    temporal=list(csv.DictReader((root/'TEMPORAL_CONTROL.csv').open()))
    reserve=json.loads((root/'reserved_prefill/RESULTS.json').read_text())
    prefill_profile=[float(r['TRAIN_position_profile_score']) for r in temporal if r['fit']=='original_existing_vector' and r['dataset']=='original' and r['cohort']=='native' and r['phase']=='prefill']
    prefill_affine=values(selected('original_existing_vector','original','native','ridge_affine_deformation',phase='prefill'))
    reserved_affine=[r['ridge_affine_deformation'] for r in reserve['rows']]
    reserved_shift=[r['TRAIN_prefill_shift'] for r in reserve['rows']]
    addendum=[
        'REGISTERED FOLLOW-UP: PREFILL LEAD AND RESERVED-DATA FAILURE',
        'The all-position main criterion above remains failed. Its predeclared phase sensitivity showed a stronger old-data prefill lead: affine displacement scores85.84-88.49%, versus21.19-25.39% for the pooled shift. A subsequently frozen temporal-profile diagnostic supplies the proper TRAIN-prefill-only shift:74.54-81.77%. Affine beats that baseline on all four old held-out prefill states. During old decoding, affine drops to21.50-45.30%, below pooled-shift35.55-42.30% on average.',
        'A separate recipe was then committed before geometric scoring of the reserved eight classifier questions. The candidate map and position profile were hash-frozen; no model, rank, penalty, layer or cutoff was changed. The reserved data had prior classifier/integrity access, so this is a held-out geometric-score check with that disclosure, not pristine prospective geometry confirmation. Only prediction position0 was scored; seven decode positions and other layers remain geometrically unscored.',
        f"Reserved affine scores range{100*min(reserved_affine):.2f}-{100*max(reserved_affine):.2f}%, mean{100*np.mean(reserved_affine):.2f}%. TRAIN-prefill shift mean{100*np.mean(reserved_shift):.2f}%. Support rule required affine score>=70% and superiority to all frozen shift/isometry/cyclic controls for every question: {reserve['support_count']}/8 prompts passed; the all8 rule failed. This is not zero prediction accuracy: the vector forecasts remain partly informative, but the strong geometric generalization claim failed.",
        'The successful16/16 classifier result is unchanged. A binary detector needs a stable discriminating projection; it need not explain the entire1024-coordinate displacement. These outcomes therefore answer different questions.',
        'No further candidate was fitted against the reserved failure. One fixed signed-strength proposal remains saved and unexecuted. Old semantic, temporal-rotation and classifier-control failures remain preserved.'
    ]
    lines=['FINAL GEOMETRY OUTCOME: no broad shared rotation/affine law validated; narrower prefill lead also failed the reserved strong-prediction rule.','']+addendum+['','INITIAL EXPLORATORY RUN AND COMPLETE METHODS']+lines
    (out/'REPORT.txt').write_text('\n\n'.join(lines),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(8.8,4.8),layout='constrained')
    for i,(name,old,new) in enumerate([('TRAIN-prefill\nshared shift',prefill_profile,reserved_shift),('Frozen affine\nmap',prefill_affine,reserved_affine)]):
        for offset,v,label,color in [(-.18,np.array(old)*100,'Old held-out prefill:4 contexts','#175b87'),(.18,np.array(new)*100,'Reserved prefill:8 contexts','#ca783b')]:
            ax.bar(i+offset,v.mean(),width=.32,color=color,label=label if i==0 else None)
            ax.errorbar(i+offset,v.mean(),yerr=[[v.mean()-v.min()],[v.max()-v.mean()]],color='#273640',capsize=3,linewidth=1)
    ax.axhline(70,color='#71808c',linestyle='--',linewidth=1,label='Reserved minimum criterion:70%')
    ax.set_xticks([0,1],['TRAIN-prefill\nshared shift','Frozen affine\nmap']);ax.set_ylim(0,105)
    ax.set_ylabel('Full-coordinate displacement error reduction (%)')
    ax.set_title('The old prefill lead did not transfer strongly to reserved questions',fontsize=12,fontweight='bold')
    ax.legend(frameon=False,loc='upper left',fontsize=8)
    fig.savefig(out/'reserved_prefill_transfer.png',dpi=190);fig.savefig(out/'reserved_prefill_transfer.svg');plt.close(fig)
    summary['registered_reserved_prefill_check']={k:v for k,v in reserve.items() if k!='rows'}
    summary['reserved_affine_mean_score']=float(np.mean(reserved_affine));summary['reserved_prefill_shift_mean_score']=float(np.mean(reserved_shift))
    (root/'SUMMARY_FINAL.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps({'main_beyond_shift_candidates_pass':sum(r['passes_beyond_shift_criterion'] for r in primary if r['model']!='translation'),
        'report':str(out/'REPORT.txt'),'figure':str(out/'cloud_geometry.png')},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
