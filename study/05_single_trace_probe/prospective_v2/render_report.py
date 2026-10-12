"""Present the frozen prospective result without changing any scoring."""
import argparse
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
    result=json.loads((root/'analysis_verified/RESULTS.json').read_text())
    native=root/'verified_export/verified_native'
    plan=json.loads((native/'source/NEXT_VALIDATION_PLAN.json').read_text())
    cleanup=json.loads((root/'CLEANUP_VERIFIED.json').read_text())
    out=root/'output';out.mkdir(exist_ok=False)
    names=['Bookmark ordering','Seedling measurement','Mailroom routing','Cart on a ramp',
           'Soup portions','Paper calendar','Street map','Package masses']
    navy='#175b87';gray='#626d78'
    base=[];steered=[]
    for prompt in plan['prompts']:
        base.append(next(r['raw_margin'] for r in result['rows'] if r['prompt']==prompt['id'] and r['condition']=='base'))
        steered.append(next(r['raw_margin'] for r in result['rows'] if r['prompt']==prompt['id'] and r['condition']=='plus_v'))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(9.8,5.1),layout='constrained')
    y=np.arange(8)
    ax.axvspan(-.18,0,color='#eef0f2',zorder=0);ax.axvspan(0,.28,color='#edf4f9',zorder=0)
    ax.scatter(base,y,color=gray,marker='s',s=55,label='Unsteered run',zorder=3)
    ax.scatter(steered,y,color=navy,marker='o',s=60,label='Known +v steering',zorder=3)
    ax.axvline(0,color='#26333e',linestyle='--',linewidth=1,label='Frozen cutoff: 0')
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlim(-.18,.28);ax.set_ylim(7.65,-.75)
    ax.set_xlabel('Raw detector score (not a probability)')
    ax.set_title('Eight new questions: all 16 runs classified correctly',loc='left',fontweight='bold',pad=34)
    ax.legend(frameon=False,loc='lower left',bbox_to_anchor=(0,1.01),ncol=3,fontsize=9)
    ax.grid(axis='y',color='white',linewidth=1.3)
    fig.savefig(out/'fresh_question_scores.png',dpi=190);fig.savefig(out/'fresh_question_scores.svg');plt.close(fig)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleR4',fontName='Helvetica-Bold',fontSize=22,leading=26,textColor=colors.HexColor(navy),spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyR4',fontName='Helvetica',fontSize=10.5,leading=14.5,spaceAfter=8))
    styles.add(ParagraphStyle(name='SmallR4',fontName='Helvetica',fontSize=8.5,leading=11.5,spaceAfter=7,textColor=colors.HexColor('#485865')))
    styles.add(ParagraphStyle(name='HeadR4',fontName='Helvetica-Bold',fontSize=13,leading=16,spaceBefore=9,spaceAfter=7,textColor=colors.HexColor(navy)))
    story=[];plain=[]
    def para(text,style='BodyR4'):
        story.append(Paragraph(text,styles[style]));plain.append(text.replace('<b>','').replace('</b>',''))
    def table(data,widths):
        items=[[Paragraph(str(c),styles['SmallR4']) for c in row] for row in data]
        t=Table(items,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6eff5')),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor(navy)),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce2e7')),
            ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        story.append(t);story.append(Spacer(1,7));plain.extend(' | '.join(str(c) for c in row) for row in data)
    para('The detector held up on new questions','TitleR4')
    para('Research 4 | Fresh frozen-prompt validation | October 11, 2026','SmallR4')
    para('<b>16/16 correct: all eight unsteered runs and all eight known-steering runs.</b> There were zero false positives and zero misses. Balanced accuracy was 100%; both runs were correct for every question.')
    para('The eight questions were fixed before their model runs. The detector kept exactly the same weights, training reference, block, eight-position window and zero cutoff. Each prediction used only one run.','BodyR4')
    story.append(Image(str(out/'fresh_question_scores.png'),width=508,height=264.4))
    table([['New question topic','Unsteered score','Steered score','Both correct']]+[
        [n,f'{b:+.4f}',f'{s:+.4f}','Yes'] for n,b,s in zip(names,base,steered)], [202,108,108,90])
    para('Each point is one recorded run. The unit of generalization is eight distinct question contexts, not sixteen independent scenarios or 128 independent token positions. Raw scores indicate which side of the fixed cutoff a run falls on; they are not calibrated probabilities.','SmallR4')
    story.append(PageBreak())
    para('What was tested, and what remains open','TitleR4')
    para('A single-run detector with a frozen recipe','HeadR4')
    para('Pinned Qwen/Qwen3.5-0.8B revision 2fc06364715b967f1860aea9cf38778875588b17; frozen BF16 generator, no adapter or quantization, SDPA. Persistent +0.5v intervention at block11. The detector reads block16: the final prefill position and first seven cached decode positions. Each 1,024-number vector is unit-normalized, then all eight are averaged. A saved TRAIN-only center, scalar scale and ridge-classifier weights produce a score; score >=0 means known +v steering.')
    para('This panel changes the questions only. Both classes replay the same frozen eight-token continuation. Class labels, token IDs, prompt identity, filenames, injection magnitude and a matched baseline never enter the detector. Baseline/steered counterparts are used only to know the true labels and check native collection integrity. The generator and detector were not trained or retuned on these questions.')
    para('Controls and preservation','HeadR4')
    para('Two native mechanics traces passed; two untreated/null-hook calls gave exactly equal logits. All upstream coordinates and pre-injection recordings match exactly across base/+v. The mechanics traces exactly reproduce the first question study traces and are counted only as technical repeats. All sixteen single-window predictions reproduce the primary scores. Poisoning metadata, renaming the file, changing other layers and adding later positions leaves every score exactly unchanged.')
    para('All 38 exported files were SHA256-verified, with full Drive and local C: copies verified before GPU release. Native export: 6,966,745 bytes; SHA256 cb821962d7058c118cfe6fe28b2aa320564b44f42b56e72e430e1c9524be85bc. Frozen probe SHA256 b157960bfff8aa5e25d6cf69de7c9cf1a5504a0f3098cfffa6bf800389c96991. Source collection/evaluation freeze commit d7db4064; earlier question-plan commit 2d28e4c4.','SmallR4')
    para('The limits remain narrow','HeadR4')
    para('This is genuine prospective evidence for question generalization within this specific setup. Eight contexts remain a small test. Other steering directions, strengths, injection schedules, models, long contexts and freely generated responses were not tested here. The result does not identify a behavior, intention or self-preservation mechanism.')
    para('The earlier Stage1 partial-label-flip criterion remains failed (90% scenario accuracy, above its 75% limit). A later TRAIN/DEV-only audit found exactly chance upstream performance and showed that partial flips can retain signal. The present test is a separately frozen, explicitly authorized validation; its success does not rewrite that earlier failure. The earlier semantic failure and stopped rotation branch are also unchanged.','SmallR4')
    para('Resource use and setup failures','HeadR4')
    para('The initial setup collected zero traces and was released after 12m46s. The successful recovery, including authentication and setup, used 13m05s. Combined owned GPU lifecycle was 25m51s, below the unchanged 45-minute cap. At the observed 5.3 units/hour, attributable GPU use is estimated at 2.28 units. The recovery account reading fell from 24.43 to 21.98 units; this 2.45-unit shared debit includes other sessions and is not a per-runtime bill.')
    para('After authentication/source setup, the remaining-work forecast was recorded as 20 minutes: 10 for pinned model preparation, 5 for native runs, 5 for exports/release. Earlier failed GPU time stayed included in the same hard deadline; no cap, rate limit or credits were increased. One worker ran once. Only its GPU was released. The user CPU mount and Research3 runtime were not reset or released; unused browser tabs were closed as requested.','SmallR4')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#dce2e7'));canvas.line(52,43,560,43)
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#62717d'))
        canvas.drawString(52,29,'R4 | Single-run known-steering detector | Fresh-question validation')
        canvas.drawRightString(560,29,str(doc.page))
    pdf=out/'fresh_question_validation.pdf'
    doc=SimpleDocTemplate(str(pdf),pagesize=letter,leftMargin=52,rightMargin=52,topMargin=44,bottomMargin=55,
        title='Single-run steering detector: eight new questions',author='Research 4')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (out/'REPORT.txt').write_text('\n\n'.join(plain),encoding='utf-8')
    print(str(pdf))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();run(a.root)
