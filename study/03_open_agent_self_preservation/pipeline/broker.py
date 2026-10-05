"""Trusted standalone Typesafe broker. Run separately; never put credentials on Colab."""
import argparse
import gzip
import hashlib
import json
import os
import re
import shlex
import time
import urllib.error
import urllib.request
from pathlib import Path
import schema

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'


def strict_json(raw):
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('Duplicate JSON key')
            value[key] = item
        return value
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(schema.canonical(value))
    os.replace(tmp, path)


def credential(args):
    if args.credential_file:
        found = []
        for line in Path(args.credential_file).read_text(encoding='utf-8-sig').splitlines():
            match = re.match(r'^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)$', line)
            if match:
                lexer = shlex.shlex(match[1], posix=True)
                lexer.whitespace_split, lexer.commenters = True, '#'
                found.extend(list(lexer))
        if len(found) != 1 or not found[0]:
            raise ValueError('Expected one existing Typesafe credential')
        return found[0]
    key = os.environ.get('TYPESAFE_API_KEY')
    if not key:
        raise ValueError('Set TYPESAFE_API_KEY in the trusted broker environment only')
    return key


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, 'Redirect refused', headers, fp)


def service(batch, args, transport=None):
    if set(batch) != {'batch_id','rows'} or not re.fullmatch(r'eval_[0-9]{5}_[a-f0-9]{10}', batch['batch_id']):
        raise ValueError('Invalid exact visible-state batch')
    if not isinstance(batch['rows'], list) or not 1 <= len(batch['rows']) <= 2:
        raise ValueError('Invalid batch size')
    identities=set()
    for row in batch['rows']:
        if (set(row)!={'id','history','tools'} or not isinstance(row['id'],str) or row['id'] in identities or
                not isinstance(row['history'],list) or not row['history'] or not isinstance(row['tools'],list)):
            raise ValueError('Invalid visible row')
        identities.add(row['id'])
        for message in row['history']:
            if (set(message)!={'role','content'} or not isinstance(message['content'],str) or
                    message['role'] not in ('system','user','assistant','tool')):
                raise ValueError('Invalid visible message')
    ledger_path = Path(args.ledger)
    state = strict_json(ledger_path.read_bytes()) if ledger_path.exists() else {
        'model':schema.MODEL,'attempted_requests':0,'successful_requests':0,'input_tokens':0,
        'reserved_input_equivalent':0,'input_usd_per_million':args.input_rate,'stopped':False}
    if state.get('stopped') or state['model'] != schema.MODEL:
        raise ValueError('Stopped or different model ledger')
    for field in ('attempted_requests','successful_requests','input_tokens','reserved_input_equivalent'):
        if isinstance(state[field],bool) or not isinstance(state[field],int) or state[field] < 0:
            raise ValueError('Invalid saved usage')
    if state['successful_requests'] > state['attempted_requests'] or state['input_usd_per_million'] != args.input_rate:
        raise ValueError('Changed saved accounting')
    output = []
    carried=state.get('carried_prior_token_usage',{})
    for row in batch['rows']:
        if set(row) != {'id','history','tools'}:
            raise ValueError('Only visible history and tools are accepted')
        for message in row['history']:
            if set(message) != {'role','content'} or message['role'] not in ('system','user','assistant','tool'):
                raise ValueError('Invalid visible message')
            if message['role']=='assistant' and re.search(r'<(?:think|analysis)\b', message['content'],re.I):
                raise ValueError('Private reasoning must not enter the helper')
        request = schema.payload(row['history'],row['tools'])
        reserve = len(schema.canonical(request)) + 8192
        equivalent = state['input_tokens']+state['reserved_input_equivalent']+reserve+carried.get('input_tokens',0)+carried.get('reserved_input_equivalent',0)
        if (state['attempted_requests']+carried.get('attempted_requests',0) >= args.max_requests or
                state['successful_requests']+carried.get('successful_requests',0) >= args.max_requests or
                equivalent*args.input_rate/1e6 > args.max_input_usd):
            raise ValueError('Shared request or estimated input-cost cap')
        index = state['attempted_requests']
        record_path = ledger_path.parent/'portable_calls'/(str(index).zfill(5)+'.json')
        if record_path.exists():
            raise ValueError('Actual call records must never be overwritten')
        state['attempted_requests'] += 1
        state['reserved_input_equivalent'] += reserve
        write(ledger_path,state)
        write(record_path,{'request':request,'identity':schema.request_identity(request),'attempt':index})
        try:
            if transport is None:
                secret = credential(args)
                req = urllib.request.Request(ENDPOINT,data=schema.canonical(request),method='POST',
                    headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})
                with urllib.request.build_opener(NoRedirect()).open(req,timeout=30) as response:
                    raw = response.read(2*1024*1024+1)
                if len(raw)>2*1024*1024 or secret.encode() in raw:
                    raise ValueError('Invalid or credential-bearing provider response')
            else:
                raw = transport(request)
            record_path.with_suffix('.response.bin').write_bytes(raw)
            state['successful_requests'] += 1
            write(ledger_path,state)
            response = strict_json(raw)
            decoded = schema.decode(response)
            actual = decoded['usage']['input_tokens']
            if actual > reserve:
                raise ValueError('Usage exceeded the conservative request reserve')
            state['input_tokens'] += actual
            state['reserved_input_equivalent'] -= reserve
            state['input_estimated_usd'] = state['input_tokens']*args.input_rate/1e6
            annotation = {'request':request,'identity':schema.request_identity(request), 'response':response,
                          'decoded':decoded,'actual_service_call':True,'cache_hit':False,
                          'raw_response_sha256':hashlib.sha256(raw).hexdigest()}
            write(record_path,annotation)
            output.append({'id':row['id'],'annotation':annotation})
            write(ledger_path,state)
        except Exception:
            state['stopped'] = True
            write(ledger_path,state)
            raise
    return {'batch_id':batch['batch_id'],'rows':output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', help='Exact downloaded request gzip or JSON')
    parser.add_argument('--response', help='Response JSON to transfer to the GPU')
    parser.add_argument('--root', help='Shared trusted request/response directory; alternative to one request')
    parser.add_argument('--ledger', required=True, help='Persistent ledger; carry the same path across attempts')
    parser.add_argument('--credential-file', help='Existing local credential file, never uploaded')
    parser.add_argument('--max-requests', type=int, default=2048)
    parser.add_argument('--max-input-usd', type=float, default=1.)
    parser.add_argument('--input-rate', type=float, default=.042)
    parser.add_argument('--max-seconds', type=int, default=3300)
    args = parser.parse_args()
    if not 0 < args.max_requests <= 2048 or not 0 < args.max_input_usd <= 1 or not 0 < args.max_seconds <= 3300:
        parser.error('Recorded request, input, and worker caps must be respected')
    if bool(args.root) == bool(args.request) or (args.request and not args.response):
        parser.error('Choose shared root or an exact request plus response')
    began = time.monotonic()
    while time.monotonic()-began < args.max_seconds:
        pending = [Path(args.request)] if args.request else sorted((Path(args.root)/'service_pending').glob('*.json.gz'))
        for path in pending:
            target = Path(args.response) if args.request else Path(args.root)/'service_responses'/(path.name[:-8]+'.json')
            if target.exists():
                continue
            raw = path.read_bytes()
            if path.suffix=='.gz':
                raw = gzip.decompress(raw)
            result = service(strict_json(raw),args)
            write(target,result)
            print(json.dumps({'batch_id':result['batch_id'],'verified_rows':len(result['rows'])}),flush=True)
        if args.request:
            return
        if (Path(args.root)/'reports/RESULT.json').exists() or (Path(args.root)/'reports/INCOMPLETE.json').exists():
            return
        time.sleep(.2)


if __name__=='__main__':
    main()
