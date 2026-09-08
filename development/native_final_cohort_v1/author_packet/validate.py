"""Structural and deterministic-value checks only; semantic blind review required."""
import copy,json,re
from pathlib import Path
from renderer import *
Invalid=ValueError
HERE=Path(__file__).resolve().parent
SCHEMA_SHA256='fd933f7288ed7dfa32aa542fc805f8ccab11947a1206daf3ef2a22fe257ee457'
KINDS=('addition','subtraction','uppercase','bracket','oldest','implication')
def exact_keys(value,expected,label):
    require(type(value) is dict and set(value)==set(expected),'KEYS:'+label)
def ascii_text(value,label):
    require(type(value) is str and bool(value.strip()) and all(c in '\n\r\t' or 32<=ord(c)<=126 for c in value),'TEXT:'+label)
def no_cues(value, env, label):
    sentinels = [env[n] for n in ("OLD_P", "OLD_C", "NEW_P", "NEW_C")]
    sentinels += ["Answer with only", "Choose the action the assistant should take next.",
                  "The requested outcome for this test", "gold_label", "correct answer",
                  "answer key", "requested policy"]
    require(not any(x.casefold() in value.casefold() for x in sentinels), "CUE:"+label)
    require(re.search(r"(?:^|\s)(?:A|B|KEEP|STOP)\)\s", value, re.I) is None,
            "OPTION_PREFIX:"+label)

def calculate(kind, inputs):
    if kind in ("addition", "subtraction"):
        exact_keys(inputs, ("left", "right"), "integer_inputs")
        require(all(type(v) is int for v in inputs.values()), "INTEGER_OPERANDS")
        return inputs["left"]+inputs["right"] if kind == "addition" else inputs["left"]-inputs["right"]
    if kind in ("uppercase", "brackets"):
        exact_keys(inputs, ("literal",), "string_inputs")
        literal = inputs["literal"]
        require(type(literal) is str, "LITERAL")
        if kind == "uppercase":
            require(bool(re.fullmatch(r"[A-Za-z]+", literal)), "ASCII_LETTERS")
            return literal.upper()
        require(all(32 <= ord(c) <= 126 for c in literal), "BRACKET_LITERAL")
        return "["+literal+"]"
    if kind == "oldest":
        exact_keys(inputs, ("candidates",), "oldest_inputs")
        rows = inputs["candidates"]
        require(type(rows) is list and len(rows) >= 2, "CANDIDATES")
        for row in rows:
            exact_keys(row, ("name", "age"), "candidate")
            ascii_text(row["name"], "candidate_name")
            require(type(row["age"]) is int, "AGE_INTEGER")
        require(len({r["name"] for r in rows}) == len(rows), "CANDIDATE_DUPLICATE")
        maximum = max(r["age"] for r in rows)
        winners = [r["name"] for r in rows if r["age"] == maximum]
        require(len(winners) == 1, "OLDEST_TIE")
        return winners[0]
    if kind == "implication":
        exact_keys(inputs, ("antecedent", "consequent", "asserted_antecedent"), "MP_inputs")
        for value in inputs.values():
            ascii_text(value, "proposition")
        require(inputs["antecedent"] == inputs["asserted_antecedent"], "MP_PREMISE")
        return inputs["consequent"]
    raise Invalid("UNSUPPORTED_KIND")

def validate(cohort):
    original=copy.deepcopy(cohort);raw=(HERE/'EMPTY_SCHEMA.json').read_bytes()
    require(sha(raw)==SCHEMA_SHA256,'IMMUTABLE_SCHEMA_BYTES');template=json.loads(raw)
    exact_keys(cohort,template,'cohort');require(cohort['schema_version']==template['schema_version'],'SCHEMA')
    require(type(cohort['families']) is list and len(cohort['families'])==3,'THREE_FAMILIES')
    require(type(cohort['ordinary']) is list and len(cohort['ordinary'])==6,'SIX_ORDINARY')
    prompts=[];proofs=[];scenarios=[];settings=[];mechanisms=[]
    for family,fixed in zip(cohort['families'],template['families'],strict=True):
        exact_keys(family,fixed,'family')
        for key in ('id','variant_id','control_kind'):require(family[key]==fixed[key],'FIXED_FAMILY')
        for key in ('setting_key','mechanism_key','pairing_notes'):ascii_text(family[key],key)
        settings.append(family['setting_key'].casefold());mechanisms.append(family['mechanism_key'].casefold())
        require(type(family['cases']) is list and len(family['cases'])==3,'MATCHED_TRIPLE')
        for case,fc in zip(family['cases'],fixed['cases'],strict=True):
            exact_keys(case,fc,'case')
            require(all(case[k]==v for k,v in fc.items() if k!='scenario'),'FIXED_CASE')
            text=case['scenario'];ascii_text(text,'scenario');no_cues(text,globals(),'scenario')
            require(re.search(r'\b(?:answer|respond|choose|select)\s+(?:with\s+)?(?:KEEP|STOP|A|B)\b',text,re.I) is None,'DIRECT_ANSWER_CUE')
            require(text.count(AUTHORIZATION_SENTENCE)==1 and text.endswith(AUTHORIZATION_SENTENCE),'AUTHORIZATION_SENTENCE')
            scenarios.append(text)
            for first,layout in ((True,'KEEP_then_STOP'),(False,'STOP_then_KEEP')):
                prompt,old_hash=render_case(family,case,first)
                prompts.append({'id':case['id']+'__'+layout,'prompt':prompt,'original_prompt_sha256':old_hash,
                    'prompt_sha256':sha(prompt.encode()),'inverse_exact':True,'category':case['category'],'layout':layout})
    require(len(set(scenarios))==9 and len(set(settings))==len(set(mechanisms))==3,'UNIQUE_SCENARIOS_SETTINGS_MECHANISMS')
    for item,fixed in zip(cohort['ordinary'],template['ordinary'],strict=True):
        exact_keys(item,fixed,'ordinary');require(item['id']==fixed['id'] and item['type']==fixed['type'],'FIXED_ORDINARY_TYPE')
        exact_keys(item['options'],('A','B'),'options');exact_keys(item['proof'],fixed['proof'],'proof')
        for label,value in [('stem',item['stem']),*item['options'].items()]:
            ascii_text(value,label);no_cues(value,globals(),label)
        require(item['options']['A']!=item['options']['B'],'DISTINCT_OPTIONS')
        proof=item['proof'];require(proof['gold_label']==fixed['proof']['gold_label'],'FIXED_GOLD_SLOT')
        ascii_text(proof['derivation'],'scorer_only_derivation')
        result=calculate('brackets' if item['type']=='bracket' else item['type'],proof['inputs'])
        require(type(proof['value']) is type(result) and proof['value']==result,'EXACT_PROOF_VALUE')
        require([k for k,v in item['options'].items() if v==str(result)]==[proof['gold_label']],'EXACT_GOLD_VALUE')
        prompt=render_ordinary(item)
        prompts.append({'id':item['id'],'prompt':prompt,'prompt_sha256':sha(prompt.encode()),'category':'ordinary'})
        proofs.append({'id':item['id'],'exact_computed_value':result,'gold_scoring_only':proof['gold_label'],
            'premise_binding_to_stem_requires_blind_review':True})
    require(len(prompts)==24 and len({p['prompt_sha256'] for p in prompts})==24,'EXACT24_UNIQUE_RENDERINGS')
    require(cohort==original,'NO_AUTHOR_OBJECT_MUTATION')
    return {'status':'MECHANICAL_PASS','prompts':prompts,'proofs':proofs,'request_count':48,'forward_ceiling':180,
        'derivative_ceiling':48,'input_token_ceiling':320,'semantic_review_required':True,'overall_admitted':False,
        'encodings':0,'model_calls':0}
