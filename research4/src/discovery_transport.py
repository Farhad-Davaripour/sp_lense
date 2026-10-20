"""Benign API format qualification, separate from behavioral scenarios."""
import json
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request,urlopen

from .discovery_sandbox import parse_action
from .io_utils import assert_namespace,digest,dump,read,verify_release


def response_text(message):
    content=message.get('content')
    if isinstance(content,str):return content
    if isinstance(content,list):return '\n'.join(item['text'] for item in content if isinstance(item,dict) and item.get('type')=='text' and isinstance(item.get('text'),str))
    return ''


def run(root,output,commit):
    from databricks.sdk import WorkspaceClient
    root,output=map(Path,[root,output]);assert_namespace(output);verify_release(root,commit)
    if output.exists():raise FileExistsError('Preserve transport qualification attempts')
    output.mkdir(parents=True);cfg=read(root/'config/discovery_transport_v1.json')
    client=WorkspaceClient();endpoint=cfg['endpoint'];before=client.serving_endpoints.get(endpoint).as_dict()
    if before['state']['ready']!='READY' or before['state']['config_update']!='NOT_UPDATING':raise ValueError('Existing endpoint is not ready')
    expected=[{'tool':'catalog','arguments':{}},{'tool':'read','arguments':{'identifier':'fixture-document'}}]
    dump(output/'PRECALL_FREEZE.json',{'config':cfg,'source_commit':commit,'expected':expected,'endpoint':before})
    calls=[];error=None
    try:
        for action in expected:
            payload={'messages':[{'role':'system','content':'Return the supplied JSON object exactly. This is an interface format check.'},{'role':'user','content':json.dumps(action,sort_keys=True)}],
                     'max_tokens':cfg['maximum_output_tokens'],'temperature':cfg['temperature'],'response_format':cfg['response_format']}
            call={'request':payload,'request_sha256':digest(payload),'status':'started'};calls.append(call)
            request=Request(client.config.host.rstrip('/')+'/serving-endpoints/'+quote(endpoint,safe='')+'/invocations',data=json.dumps(payload).encode(),headers={**client.config.authenticate(),'Content-Type':'application/json'},method='POST')
            with urlopen(request,timeout=cfg['request_timeout_seconds']) as response:answer=json.load(response)
            call.update(status='complete',response=answer)
            choice=answer['choices'][0];text=response_text(choice['message'])
            try:parsed=parse_action(text)
            except (ValueError,TypeError,RecursionError):parsed=None
            call.update(parsed_action=parsed,qualified=parsed==action and choice.get('finish_reason')!='length')
            dump(output/'CALLS.json',calls)
        after=client.serving_endpoints.get(endpoint).as_dict()
        if digest(before.get('config'))!=digest(after.get('config')):raise ValueError('Endpoint configuration changed')
        dump(output/'ENDPOINT_AFTER.json',after)
    except Exception as e:error={'type':type(e).__name__,'message':str(e)}
    dump(output/'CALLS.json',calls)
    result={'status':'complete' if error is None else 'failed','mode':'transport','source_commit':commit,
            'model_api_calls':len(calls),'qualified':error is None and len(calls)==2 and all(c.get('qualified') for c in calls),
            'scientific_scenarios':0,'backbone_forwards':None,'optimizer_updates':0,'real_resource_actions':0,'GPU_allocations':0,
            'error':error,'scope':cfg['scope']}
    dump(output/'RUN_RECEIPT.json',result);return result
