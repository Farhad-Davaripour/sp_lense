"""Read-only broader prior-input collision audit; no model/service dependencies."""
import json
import re
from pathlib import Path
from prepare import ARTIFACT, HERE, SOURCE, sha, write_new


def main():
    corpus = [SOURCE/'training_data/plain_train.json', SOURCE/'EVAL_FREEZE.json']
    for path in HERE.parent.rglob('*.json'):
        if 'same_family_confirmation_v1' in path.parts: continue
        if any(x in path.parts for x in ('results','completed_20261004','recovery','unqueried_candidate_archive')): continue
        if any(x in path.parts for x in ('data','data_frozen','input_frozen')):
            corpus.append(path)
    corpus=sorted(set(corpus)); pairs=set(); triples=set(); tasks=set(); ids=set()
    def walk(x):
        if isinstance(x,dict):
            numeric=[v for v in x.values() if isinstance(v,int) and not isinstance(v,bool)]
            if len(x)==2 and len(numeric)==2:pairs.add(tuple(sorted(numeric)))
            for k in ('id','self_id','key','reserve','activate'):
                if isinstance(x.get(k),str):ids.add(x[k])
            if isinstance(x.get('parts'),list) and len(x['parts'])==3:triples.add(tuple(sorted(x['parts'])))
            for v in x.values():walk(v)
        elif isinstance(x,list):
            for v in x:walk(v)
        elif isinstance(x,str):
            tasks.add(re.sub(r'\s+',' ',x).strip().casefold())
            try:walk(json.loads(x))
            except (ValueError,TypeError):pass
    for p in corpus:walk(json.loads(p.read_bytes()))
    cases=json.loads((ARTIFACT/'cases10.json').read_bytes()); rows=[]
    for c in cases:
        row={'id':c['id'],'case_id_collision':c['id'] in ids,'visible_self_id_collision':c.get('self_id') in ids,
             'initial_numeric_pair_overlap':tuple(sorted(c['records'].values())) in pairs,
             'pending_numeric_multiset_overlap':bool(c.get('parts')) and tuple(sorted(c['parts'])) in triples,
             'exact_task_wording_collision':re.sub(r'\s+',' ',c['task']).strip().casefold() in tasks}
        rows.append(row)
    receipt={'model_free':True,'prior_input_files_checked':len(corpus),
             'input_files':[{'path':str(p),'sha256':sha(p)} for p in corpus],
             'case_fixture_sha256':sha(ARTIFACT/'cases10.json'),'cases':rows,
             'no_neural_or_API_calls':True}
    receipt['all_no_collisions']=not any(v for row in rows for k,v in row.items() if k!='id')
    write_new(HERE/'BROAD_COLLISION_AUDIT.json',receipt)
    print(json.dumps({'files_checked':len(corpus),'cases':rows,'all_no_collisions':receipt['all_no_collisions']},separators=(',',':')))
    assert receipt['all_no_collisions'], 'Revise before querying; preserve the candidate freeze'


if __name__=='__main__':main()
