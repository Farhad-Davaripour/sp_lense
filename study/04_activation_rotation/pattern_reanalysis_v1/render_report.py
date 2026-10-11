"""Static scientific figures and a compact verified-data report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image

def run(root):
    r=json.loads((root/'RESULTS.json').read_text())
    robust=json.loads((root/'robustness/ROBUSTNESS.json').read_text())
    rows=[x for x in r['anchor_rows'] if x['phase']=='decode']
    out=root/'output';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white'})
    fig,axes=plt.subplots(2,2,figsize=(11.6,8.2),layout='constrained')
    navy='#175b87';orange='#d17032';green='#218675';gray='#77818c'
    # A: error reduction using an old, frozen fingerprint.
    groups=[('original','existing_vector','Old test\n+v'),('original','random_vector','Old test\nRandom 1'),
            ('matched','plus_v','Matched\n+v'),('matched','random1','Matched\nRandom 1')]
    ax=axes[0,0]
    for i,(dataset,arm,label) in enumerate(groups):
        sub=[x for x in rows if x['dataset']==dataset and x['condition']==arm]
        for offset,key,color in [(-.19,'template_score_vs_zero',navy),(.19,'unchanged_injection_score_vs_zero',gray)]:
            v=np.array([x[key] for x in sub])*100
            ax.bar(i+offset,v.mean(),width=.34,color=color,label=('TRAIN fingerprint' if offset<0 else 'Unchanged injection') if i==0 else None)
            ax.errorbar(i+offset,v.mean(),yerr=[[v.mean()-v.min()],[v.max()-v.mean()]],color='#303b45',capsize=3,linewidth=1)
    ax.set_xticks(range(4),[g[2] for g in groups]);ax.set_ylabel('Displacement error reduction (%)')
    ax.set_title('A  Fingerprints transfer across contexts',loc='left',fontsize=11);ax.set_ylim(0,56);ax.legend(fontsize=8,frameon=False,loc='upper right')
    # B: delivered-displacement-normalized response.
    ax=axes[0,1];arms=['plus_v','minus_v','random1','random2'];labels=['+v','-v','Random 1','Random 2']
    for i,arm in enumerate(arms):
        v=np.array([x['mean_gain_over_actual_injection'] for x in rows if x['dataset']=='matched' and x['condition']==arm])
        ax.bar(i,v.mean(),color=[navy,orange,gray,green][i],alpha=.85,width=.62)
        ax.scatter(i+np.linspace(-.14,.14,len(v)),v,color='#182833',s=16,zorder=4)
    ax.set_xticks(range(4),labels);ax.set_ylabel('Downstream change / actual injection')
    ax.set_title('B  Learned direction has greater gain',loc='left',fontsize=11);ax.set_ylim(0,3.5)
    # C: mirrored-sign geometry at every downstream block.
    ax=axes[1,0];layers=np.arange(10,24)
    for pid in dict.fromkeys(x['prompt'] for x in r['sign_symmetry']):
        values=[next(x['plus_minus_mirror_cosine'] for x in r['sign_symmetry'] if x['prompt']==pid and x['phase']=='decode' and x['block_zero_based']==l) for l in layers]
        ax.plot(layers+1,values,color=orange,alpha=.25,linewidth=1)
    values=[np.mean([x['plus_minus_mirror_cosine'] for x in r['sign_symmetry'] if x['phase']=='decode' and x['block_zero_based']==l]) for l in layers]
    ax.plot(layers+1,values,color=orange,marker='o',markersize=3,linewidth=2,label='Six-prompt mean')
    ax.axhline(0,color=gray,linewidth=.8,linestyle='--');ax.axvline(16,color=navy,linewidth=.8,linestyle=':')
    ax.set_ylim(-.25,1.12);ax.set_xticks([11,12,16,20,24]);ax.set_xlabel('Residual block (one-based; injection at 11)')
    ax.set_ylabel('Cosine(delta +v, -delta -v)');ax.set_title('C  Reversing steering does not reverse the response',loc='left',fontsize=11)
    ax.text(11.25,.91,'Nearly perfect opposites\nat injection',fontsize=8);ax.text(16.25,.36,'Almost perpendicular\nat block 16',fontsize=8)
    # D: full-pair holdout.
    ax=axes[1,1]
    for i,arm in enumerate(arms):
        sub=[x for x in robust['matched_leave_pair_out'] if x['condition']==arm]
        for offset,key,color in [(-.19,'position_matched_same_condition_score',navy),(.19,'position_matched_other_conditions_mean_score',gray)]:
            v=np.array([x[key] for x in sub])*100
            ax.bar(i+offset,v.mean(),width=.34,color=color,label=('Same direction' if offset<0 else 'Other directions mean') if i==0 else None)
            ax.errorbar(i+offset,v.mean(),yerr=[[v.mean()-v.min()],[v.max()-v.mean()]],color='#303b45',capsize=3,linewidth=1)
    ax.axhline(0,color='#303b45',linewidth=.7);ax.set_xticks(range(4),labels);ax.set_ylim(-40,100)
    ax.set_ylabel('Displacement error reduction (%)');ax.set_title('D  Fingerprints survive leaving out a wording pair',loc='left',fontsize=11)
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    fig.suptitle('Base versus steered activation: a repeatable, nonlinear response',fontsize=16,fontweight='bold')
    fig.savefig(out/'activation_patterns.png',dpi=190);fig.savefig(out/'activation_patterns.svg');plt.close(fig)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleR4',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=colors.HexColor(navy),spaceAfter=14))
    styles.add(ParagraphStyle(name='BodyR4',fontName='Helvetica',fontSize=10.5,leading=15,spaceAfter=9))
    styles.add(ParagraphStyle(name='SmallR4',fontName='Helvetica',fontSize=8.5,leading=12,spaceAfter=7,textColor=colors.HexColor('#44515d')))
    styles.add(ParagraphStyle(name='HeadR4',fontName='Helvetica-Bold',fontSize=13,leading=17,spaceBefore=10,spaceAfter=8,textColor=colors.HexColor(navy)))
    story=[];plain=[]
    def para(text,style='BodyR4'):
        story.append(Paragraph(text,styles[style]));plain.append(text.replace('<b>','').replace('</b>','').replace('&gt;','>').replace('&lt;','<'))
    def table(data,widths):
        wrapped=[[Paragraph(str(x),styles['SmallR4']) for x in row] for row in data]
        t=Table(wrapped,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(navy)),('BOTTOMPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
            ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce2e7'))]))
        story.append(t);story.append(Spacer(1,9));plain.extend(' | '.join(str(x) for x in row) for row in data)
    para('A repeatable steering fingerprint','TitleR4')
    para('Research 4 | Saved-data exploratory reanalysis | October 11, 2026','SmallR4')
    para('<b>The robust pattern in these saved activations is a direction-specific, context-stable response with strong sign asymmetry.</b> The learned direction produces a larger downstream change than the available random directions. The earlier pending-work semantic prediction remains failed.')
    para('1. Part of the displacement transfers across contexts','HeadR4')
    para('A single mean response profile per direction and block, built only from four original greedy TRAIN prompts, reduces full-coordinate displacement error on every TEST prompt and every matched +v prompt. There is no target-prompt amplitude fitting. It beats zero displacement, unchanged injection, the wrong-direction template and the mean of both templates in all 22 +v evaluations (four old TEST prompts across four correlated continuations, plus six matched prompts).')
    table([['Evaluation at block 16, decode','+v fingerprint score','Unchanged injection','Extra propagation score'],
        ['Original four TEST prompts, four continuations','27.8-42.4%','8.5-16.7%','20.5-35.9%'],
        ['Six matched prompts','28.1-32.7%','6.0-9.3%','23.5-26.1%']], [198,100,100,110])
    para('Scores are 1 - prediction squared error / target displacement energy, relative to predicting no change. They are uncentered error reductions, not variance explained in a fitted regression. Ranges are prompt/cohort ranges, not confidence intervals. Extra propagation subtracts the actual injected displacement, so success is not just detecting the injected vector.','SmallR4')
    para('2. Positive and negative steering do not mirror downstream','HeadR4')
    para('The cosine between delta(+v) and -delta(-v) is 0.999983-0.999984 at injection. At block 16 it is -0.107 to -0.019: the two actual downstream changes are approximately perpendicular, rather than opposite. The sign-even component accounts for 52.8-59.9% of the combined +/- displacement energy. Negating the original +v template fails for all six -v prompts.')
    para('This is a finite-strength, persistent-intervention result. It does not identify a local Jacobian, a single-layer mechanism, an attractor, or a behavioral preference. Prefill, recurrent/KV state, depth transformations and native BF16 computation all participate in the measured response.','SmallR4')
    story.append(PageBreak())
    para('The geometry in four views','TitleR4')
    story.append(Image(str(out/'activation_patterns.png'),width=508,height=359))
    para('All panels retain the fixed primary residual block 16 except C, which displays every downstream block. Panel A uses old TRAIN templates only. Panel B shows one point per matched prompt. Panel C shows all six prompts and their mean. Panel D excludes both members of one wording pair and fits on the remaining four prompts at the same forced-token positions. Error bars are observed minima/maxima.','SmallR4')
    para('3. Direction identity matters more than the status word','HeadR4')
    para('Changing only completed to pending preserves the +v response direction closely: cosine 0.972-0.986 across the three wording pairs. Transferring the completed response directly to the pending prompt reduces displacement error by 93.0-96.1%. Random controls show similar status stability, so this supports a generic response-geometry observation rather than the previous semantic story.')
    para('Leaving out a complete wording pair, a same-direction fingerprint at matching token positions reduces +v displacement error by 64.8-75.1%; for -v it gives 62.2-70.8%. The mean of other-condition templates gives negative scores. With position information removed, +v still gives 39.1-44.1%. Both context stability and the fixed continuation contribute.')
    para('Random 1 and Random 2 also transfer, with matching-position scores of 70.5-80.3% and 63.8-76.6%. The shared norm-only response scores 54.0-60.4%. Repeatable fingerprints are therefore a property of this intervention/model setup, not unique proof of meaning in v.','SmallR4')
    story.append(PageBreak())
    para('Magnitude, controls and robustness','TitleR4')
    para('4. The learned direction has higher downstream gain','HeadR4')
    table([['Matched condition','Change / actual injection','Mean base-to-steered angle'],
        ['+v','2.572-2.970','0.322-0.375 rad'],['-v','2.330-2.538','0.284-0.310 rad'],
        ['Random 1','1.644-1.710','0.203-0.210 rad'],['Random 2','1.742-1.833','0.217-0.228 rad'],
        ['Norm-only','Not a size-matched additive control','0.0160-0.0169 rad']], [118,195,195])
    para('At equal native additive displacement, +v gain exceeds the original random control in every old TEST prompt/cohort (1.21-1.63x) and both random controls in every matched prompt (1.40-1.78x). This refers to the measured activation change, not the usefulness or semantic correctness of steering. Native additive size mismatch stayed below 0.225%. Norm-only matches the hypothetical +v output norm, not its displacement.')
    para('For +v, 97.9-99.1% of relative displacement energy in the old TEST traces and 98.6-99.5% in the matched traces is accounted for by directional rather than norm-ratio change. The response is mainly a change in orientation, not a simple expansion or shrinkage of the activation.')
    para('Checks that survived, and limits that did not','HeadR4')
    para('Leaving out any one original TRAIN prompt preserves positive +v template scores: 24.0-43.7% in the old TEST traces and 27.2-32.7% in the matched traces. Both early and late decode halves remain positive. The matched +v template is positive at every individual decode position; original +v has a small number of negative individual positions. Tokens are not treated as independent replications.')
    para('Prefill gives smaller +v template scores (19.4-23.7% old, 16.6-18.7% matched) but still beats the same controls. Shared old prefill is counted once per prompt conceptually, not as four independent samples. Sign-even energy remains large in prefill and both decode halves; the second half ranges 47.5-53.2%, so "more than half everywhere" would be false.')
    para('Only 1.3-2.4% of old +v displacement energy and 1.8-2.6% of matched +v displacement energy falls in a fixed four-PC baseline TRAIN basis; projection onto the baseline token-transition tangent is also small. However, these four PCs cover only 25.1% of TRAIN baseline variance and 11.9-18.2% of centered evaluation baseline energy. This cannot establish that steering goes off the full activation manifold.')
    para('The previously observed full-attention/linear-attention depth pattern repeats for learned and random directions. Full-attention targets have larger average gain increments. This is an architectural covariate shared by the controls, not a direction-specific semantic result. All downstream blocks are retained; no best layer or rank was selected.','SmallR4')
    story.append(PageBreak())
    para('Methods and provenance','TitleR4')
    para('Saved data and separation','HeadR4')
    para('Qwen/Qwen3.5-0.8B, frozen revision 2fc06364715b967f1860aea9cf38778875588b17, BF16, no adapter or quantization. Persistent injection is at zero-based block 10. This reanalysis uses zero-based block 15 as its fixed anchor. Original discovery training is restricted to greedy r4_01-r4_04. Evaluation uses original r4_05-r4_08 with greedy and seeds 42/43/44, plus the six matched prompts in three wording pairs. Old seeds repeat prompts and shared prefill; the matched panel shares a fixed 28-token continuation. They are not 22 independent semantic replications.')
    para('Let d(l,t)=h_steered(l,t)-h_base(l,t), and A(t)=||d(10,t)||. For each TRAIN prompt, average d(l,t)/A(t) over decode positions; average those four prompt profiles equally to obtain T(l). Evaluate the fixed prediction A(t)T(l) with no target amplitude fit. The extra-propagation check uses d(l,t)/A(t)-d(10,t)/A(t) and T(l)-T(10). Wrong-direction and common templates use identical training and scoring.')
    para('Sign decomposition: odd=(d_plus-d_minus)/2 and even=(d_plus+d_minus)/2. Their squared energies sum to half the sum of the two displacement energies. The even fraction is an exact paired-geometry diagnostic at this intervention strength. It does not assign a causal semantic feature to either component.')
    para('Integrity and numerical checks','HeadR4')
    para('All 96 consumed native archives were SHA256-verified against their original export manifests. Forced/replayed token IDs match within each base/intervention comparison; all values are finite; every upstream coordinate remains exactly unchanged. The matched native study had 36/36 traces, 6/6 mechanics checks and two exact null/untreated calls. An independent explicit coordinate-sum calculation reproduces the symmetry statistics to 5.56e-17 and one full-space template score exactly.')
    para('Exploratory status and stopping boundaries','HeadR4')
    para('Prior aggregate results in both cohorts were already known. ANALYSIS_PLAN.json fixed the first geometry checks before these new computations; ROBUSTNESS_PLAN.json explicitly records later checks added after first results. This is reusable-data exploratory evidence, not a prospectively held-out confirmation. All failures are retained: -v is not predicted by negating the +v template, generic wrong-direction templates fail, the original pending-work primary hypothesis failed 0/3, and the stopped rotation-validation branch remains stopped.')
    para('No new model inference, GPU allocation, training, JEV, monitor, runtime reset or OneDrive research write occurred. The user-mounted Colab CPU notebook was left intact. Derived outputs are private under C:/Users/farha/Research4/artifacts/pattern_reanalysis_v1/analysis_20261011, with code and plans in the Research 4 checkout. Native source data and freezes are unchanged.','SmallR4')
    para('A focused future test would vary intervention strength and unseen continuation tokens to distinguish local linear response from the measured finite-strength asymmetry. It would require a separately specified experiment; none was launched here.','SmallR4')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#dce2e7'));canvas.line(52,43,560,43)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#63717c'))
        canvas.drawString(52,29,'R4 | Activation pattern reanalysis | Exploratory saved-data evidence')
        canvas.drawRightString(560,29,str(doc.page))
    doc=SimpleDocTemplate(str(out/'activation_pattern_report.pdf'),pagesize=letter,rightMargin=52,leftMargin=52,topMargin=46,bottomMargin=55,
        title='Base versus steered activation: robust response patterns',author='Research 4')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'REPORT.txt').write_text('\n\n'.join(plain),encoding='utf-8')
    (root/'REPORT_BUILD.json').write_text(json.dumps(dict(report=str(out/'activation_pattern_report.pdf'),
        report_sha256=hashlib.sha256((out/'activation_pattern_report.pdf').read_bytes()).hexdigest(),
        figure=str(out/'activation_patterns.png')),indent=2))
    print(str(out/'activation_pattern_report.pdf'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
