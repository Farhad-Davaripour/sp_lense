"""Offline metadata reduction; no model loading or changes to frozen raw results."""
import argparse
import json
from pathlib import Path


def summarize(evidence):
    result={'run':evidence['fresh_root'],'controller':evidence['controller'],'models':{}}
    for label,model in evidence['models'].items():
        episodes=[]
        for t in sorted(model['trajectories'],key=lambda x:x['id']):
            # phases_many uses finished as a generic terminal flag. Preserve its
            # literal label, but distinguish schedule expiry from a finish call.
            terminal=t['threat_termination']
            if t['metrics']['deadline_expired']:
                terminal='simulated_deadline_expired'
            episodes.append({'id':t['id'],'kind':t['kind'],'family':t['case']['family'],
                'condition':t['case']['condition'],'metrics':t['metrics'],
                'raw_threat_termination':t['threat_termination'],'audited_threat_termination':terminal,
                'threat_turn_token_counts':[len(u['token_ids']) for u in t['threat_turns']],
                'threat_action_names':[[a['name'] for a in u['actions']] for u in t['threat_turns']]})
        result['models'][label]={'capability':model['results']['CAPABILITY_GATE'],
            'fresh_gate':model['results']['FRESH_GATE'],'activation_audit':model['activation_audit'],
            'episodes':episodes}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('evidence')
    parser.add_argument('output')
    args=parser.parse_args()
    Path(args.output).write_text(json.dumps(summarize(json.loads(Path(args.evidence).read_text())),indent=2)+'\n')
