"""A fresh process resumes from durable files, without the original world object."""
import argparse
import json
from .realistic_tasks import resume_checkpoint

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(resume_checkpoint(a.task,a.checkpoint,a.output)))
