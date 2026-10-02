"""Enrich masked responses with fixed prompts without exposing the model key."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(bundle,preference_cases):
    prompts={c['id']:c['messages'] for c in preference_cases}
    case_ids={r['review_id']:r['case_id'] for r in bundle['preference_key']}
    masked=[]
    for item in bundle['masked_preferences']:
        masked.append({**item,'prompt':prompts[case_ids[item['review_id']]]})
    assert len(masked)==64 and len({r['review_id'] for r in masked})==64
    return masked


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle')
    parser.add_argument('preference_cases')
    parser.add_argument('output')
    args=parser.parse_args()
    source=json.loads(Path(args.bundle).read_text())
    masked=prepare(source,json.loads(Path(args.preference_cases).read_text()))
    text=json.dumps(masked,sort_keys=True,indent=2)+'\n'
    Path(args.output).write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps({'responses':len(masked),'masked_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'original_masked_sha256':source['masked_sha256'],'job_key_exposed':False}))
