"""Fixed12 scenario identity/status mapping, not a new authoring policy."""
from plan import slots
from renderer import need,sha,render_case
def validate(cohort):
    need(type(cohort) is dict and set(cohort)=={'cases'},'EXACT_SCENARIO_CONTAINER')
    cases=cohort['cases'];planned=slots();need(type(cases) is list and len(cases)==12,'EXACT12_SCENARIOS')
    prompts=[]
    for case,slot in zip(cases,planned,strict=True):
        need(set(case)=={'id','family_id','category','current_assistant_status','scenario'},'EXACT_SCENARIO_FIELDS')
        need(case['id']==slot['id'] and case['family_id']==slot['family_id'] and case['category']==slot['category'],'EXACT12_ORDER')
        need(case['current_assistant_status']==('ends' if slot['gold_label']=='A' else 'continues'),'GOLD_STATUS_CATEGORY_AGREE')
        need(type(case['scenario']) is str and 0<len(case['scenario'])<=10000,'SCENARIO_STRING_BOUND')
        prompt=render_case(case)
        prompts.append({'id':case['id'],'prompt':prompt,'prompt_sha256':sha(prompt.encode())})
    need(len({p['prompt_sha256'] for p in prompts})==12,'DISTINCT12_PROMPTS')
    return {'status':'MECHANICAL_PASS','prompts':prompts}
