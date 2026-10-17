"""Freeze words, questions and source before any new backbone call."""
import os,time
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
BEGIN=time.monotonic()
import json,shutil,hashlib,zipfile
from pathlib import Path
from transformers import AutoTokenizer
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'21_coupled_memory_blocks_v1'
A=Path('C:/Users/farha/Research4/artifacts/semantic_memory_transport_v1');REF=A/'reference_20261015'
eligible=json.loads((REF/'TOKENIZER_ONLY_ADMISSION.json').read_text())['eligible'];assert len(eligible)>=16
pairs=[]
for j,q in enumerate(eligible[:16]):pairs.append({**q,'pair_index':j,'ids':[x[0] for x in q['ids']],'split':'word_training' if j<12 else 'held'})
system='Return exactly the requested noun in lowercase, with no explanation, quotes or punctuation.'
tokenizer=AutoTokenizer.from_pretrained('C:/Users/farha/Research4/cache/qwen35_08b_pinned',local_files_only=True,trust_remote_code=False)
questions=[]
def add(pair,template,number,split):
    noun=pair['words'][0];quantity='one' if number==0 else 'several'
    if template=='fit':text=f"Complete the phrase '{quantity} ___' using the noun '{noun}' in the correct form. Return only the missing word in lowercase."
    elif template=='a':text=f"Write the correct lowercase form of '{noun}' for '{quantity} ___'. Reply with just that word."
    else:text=f"A label must describe {quantity} {'item' if number==0 else 'items'} of type '{noun}'. Give only the lowercase noun with the correct grammatical number."
    pid=f"{split}_{pair['pair_index']:02d}_{template}_{number}"
    ids=tokenizer.apply_chat_template([{'role':'system','content':system},{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False)
    if hasattr(ids,'keys'):ids=ids['input_ids']
    if hasattr(ids,'tolist'):ids=ids.tolist()
    if ids and isinstance(ids[0],list):ids=ids[0]
    assert len(ids)<=128 and len(ids)>2
    questions.append({'id':pid,'pair_index':pair['pair_index'],'number':number,'template':template,'split':split,'text':text,'prefix_ids':ids,'word_ids':pair['ids'],'forms':pair['forms']})
for pair in pairs[:6]:
    for number in [0,1]:add(pair,'fit',number,'calibration' if pair['pair_index']==0 else 'fit')
for pair in pairs[12:16]:
    for template in ['a','b']:
        for number in [0,1]:add(pair,template,number,'held')
assert len(questions)==28 and sum(q['split']=='held' for q in questions)==16
assert len({q['prefix_ids'][-1] for q in questions})==1
plan={'objective':'Replicate a small output-space singular/plural concept-geometry test, then prospectively validate a contrastive block12 direction on new nouns/templates and predict its propagation through blocks12/13 and one following controlled token.','paper':'https://arxiv.org/abs/2311.03658','official_code_repository':'KihoPark/linear_rep_geometry','official_code_revision':json.loads((REF/'REPOSITORY_PIN.json').read_text())['revision'],'model_repository':'Qwen/Qwen3.5-0.8B','model_revision':'2fc06364715b967f1860aea9cf38778875588b17','model_loader':'Original BF16 load then FP32 promotion; frozen original matrices and nonlinearities; CPU MATH/reference kernels.','software':{'torch':'2.13.0+cpu','transformers':'5.19.0','numpy':'2.5.4'},'word_pairs':pairs,'word_selection':'First16 eligible pairs in official noun/plural file order, requiring distinct single-token lower/initial-capital forms. First12 fit output directions; last4 held. First6 fit internal contrastive direction. Filtering uses tokenizer only, before any forward.','system':system,'questions':questions,'replication_part':'Population covariance C of all original vocabulary output rows; M=C^-1, no ridge unless protocol fails admission. Mean singular/plural differences define output gamma; lambda=M*gamma normalized for comparisons. Compare Euclidean estimator, held pair projections and capitalization interference. This is an adaptation to Qwen with strict held pairs, not a reproduction of all original LLaMA experiments.','internal_direction':'Mean block12 pre-norm residual difference (plural minus singular) from six paired normal training contexts. Normalize Euclidean norm. A deterministic Gaussian direction(seed2026101522) orthogonal to it is the same-norm random control. No output-space metric is assumed valid inside the layer.','magnitude_rule':'alpha=clip(0.25*median(norm(plural-minus-singular block12 training differences)),0.125,2.0), fixed using training normal traces only; both signs used, no held tuning.','positions':'Token1 is the last assistant-prefix token, with the preceding prompt cached. This predicts the first answer word. Token2 consumes the original normal most-likely answer token. Reject EOS for that controlled continuation; do not substitute. Intervention is only before block12 at token1.','forecast_contract':'Separate NumPy-only process receives public pair parameters, frozen directions/magnitude, initial normal pair memories and two normal upstream block12 boundary inputs. It calculates both full blocks and carries its own memories. Seal all forecasts before any native steered target.','numerical_gates':{'all_complete_values':'<=1e-5*max(1,native norm)','all_complete_finite_changes':'<=max(0.01*actual change norm,20*float32 epsilon*max(1,normal norm,target norm)); norms and bound multiplication in FP64','resolved_key_targets':'Every held semantic-direction block13 complete recurrent state and residual output at both signs/positions must exceed5 numerical floors','calibration_stop':'All original numerical, strict-null, upstream and energy gates on both calibration contexts before held steered targets; no held-driven changes.'},'semantic_gates':{'natural_held_contexts':'At least13/16 native normal count choices correct conditional on the four words; median four-word mass>=0.05; all8 natural paired plurality log-odds differences positive. Otherwise retain failed semantic admission without claiming a meaningful direction.','internal_causal_effect':'At least13/16 held contexts have plural_logodds(+alpha)>normal>plural_logodds(-alpha); median signed half-swing>=0.01nats; semantic median swing>=2*median absolute random-control half-swing. Applies only at first-answer prediction.','internal_selectivity':'Median(max(abs(capitalization half-swing within singular),abs(capitalization half-swing within plural)))<=0.5*median signed plurality half-swing. Failure means no selective concept-direction claim.','output_subspace':'All four held noun/plural mean-direction projections under M positive; median percentile among1024 fixed random vocabulary-pair projections>=0.95. Report Euclidean comparison, do not tune metric.','output_intervention_selectivity':'For the normalized M-mapped output direction, held plural-minus-singular slopes positive for both capitalizations; median absolute capitalization slope<=0.5*median number slope. Also report Euclidean direction.'},'energy':'Same conditional memory force/work identity as study21, independently checked with dense K for every evaluation normal and steered head. Drivers change with token/context; no global conserved physical energy claim.','secondary_geometry':'Project predicted/native pair outputs and full memory changes onto fixed unit block13 directions learned only from six natural training pairs. Report size/angle and finite decomposition; these projections alone do not establish semantic meaning.','budget':{'CPU_only':True,'planned_backbone_calls':222,'breakdown':{'training_prefix_and_token1':24,'calibration_normal_token2':2,'held_normal_prefix_and_two_steps':48,'strict_calibration_null':4,'calibration_targets':16,'held_targets':128},'forward_ceiling':300,'charged_execution_seconds_ceiling':2700,'worker_stop_with_prior':2300,'setup_stop_seconds':1200,'minimum_available_RAM_before_load':8589934592,'Torch_threads':4,'BLAS_threads':2,'GPU_allocations':0,'conservative_prior_setup_seconds':60},'interpretation':'No semantic claim unless held causality and selectivity gates pass. Local pair prediction retains all original matrices. Native final-word effects are measured; whole-backbone final logits are not forecast by the two-block equations. Token2 is controlled, not autonomous generation.','preservation':'Keep original failures, full normal/steered pair fields, full logits/final embeddings, sealed forecasts, all complete finite-term hashes and regeneration inputs. Verify host/Drive CRC and SHA, private provider size/parent/shared=false; release owned workers. No Research3 edits, no monitor, new task or automation.'}
(HERE/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
shutil.copyfile(OLD/'equations.py',HERE/'equations.py');shutil.copyfile(OLD/'preflight.py',HERE/'preflight.py')
native=(OLD/'native_teacher.py').read_text()
start=native.index('    def prefix(self,ids):');end=native.index('    def step(',start)
native=native[:start]+'''    def prefix(self,ids):
        self.count()
        with torch.no_grad(),sdpa_kernel(SDPBackend.MATH):
            answer=self.text(input_ids=torch.tensor([ids]),attention_mask=torch.ones((1,len(ids)),dtype=torch.long),use_cache=True,return_dict=True)
        return answer.past_key_values,None
'''+native[end:]
native=native.replace("candidate=int(self.model.lm_head(answer.last_hidden_state[0,-1]).argmax()) if choose_next else None","logits=self.model.lm_head(answer.last_hidden_state[0,-1])\n                result['final_embedding']=array(answer.last_hidden_state[0,-1]);result['logits']=array(logits)\n                candidate=int(logits.argmax()) if choose_next else None")
native=native.replace('assert len(result)==14','assert len(result)==16')
(HERE/'native_teacher.py').write_text(native)
predict=(OLD/'predict.py').read_text();predict=predict.replace("from equations import trajectory,finite_terms,potential","from equations import trajectory,finite_terms,potential")
predict=predict.replace("        controls={k:trajectory(boundaries,initial,parameters,direction,alpha,k,normal) for k in ['frozenK','frozenKF','reset']}\n",'')
predict=predict.replace("            for mode,output in controls.items():\n                for name in ['b13_S','b13_output']:a[mode+':'+name]=output[ti]['fields'][name]\n",'')
predict=predict.replace('assert len(metadata)==96','assert len(metadata)==len(cases)*4').replace("'normal_question_packets':6,'trajectories':24","'normal_question_packets':len(inputs['questions']),'trajectories':len(cases)")
(HERE/'predict.py').write_text(predict)
prepared=A/'prepared_20261015';prepared.mkdir(exist_ok=False);source=prepared/'source';source.mkdir()
for p in HERE.iterdir():
    if p.is_file():
        if p.suffix=='.py':compile(p.read_text(),str(p),'exec')
        shutil.copyfile(p,source/p.name)
shutil.copyfile(HERE.parent/'04_activation_rotation/model_pin.json',source/'model_pin.json')
for n in ['REPOSITORY_PIN.json','TOKENIZER_ONLY_ADMISSION.json','[noun - plural].txt','[lower - upper].txt']:shutil.copyfile(REF/n,source/n)
receipt={'no_model_forwards':True,'questions':28,'held_contexts':16,'planned_backbone_calls':222,'preparation_seconds':time.monotonic()-BEGIN,'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()}}
(prepared/'PREPARATION_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
zpath=A/'R4_SEMANTIC_TRANSPORT_SOURCE_20261015.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
    for p in source.iterdir():z.write(p,'source/'+p.name)
    z.write(prepared/'PREPARATION_RECEIPT.json','PREPARATION_RECEIPT.json')
print('PREPARED',zpath,zpath.stat().st_size,hashlib.sha256(zpath.read_bytes()).hexdigest())
