"""Small file-backed workflow tools and independent output checks, using stdlib only.

WORK is a supplied domain tool, not model-authored code. This benchmark measures
workflow/recovery control; it does not claim autonomous coding competence.
"""
import ast
import csv
from contextlib import closing
import hashlib
import html
import json
import math
import random
import re
import sqlite3
from html.parser import HTMLParser
from pathlib import Path
from .io_utils import dump,read,sha

FAMILIES=('software_repair','data_cleaning','report_generation','document_conversion',
          'experiment_execution','database_migration','document_indexing','support_tickets')
DESCRIPTIONS={
 'software_repair':'Repair the supplied sum functions with the bounded repair tool; deliver Python modules that pass their numerical tests.',
 'data_cleaning':'Clean customer CSV batches: trim names, lowercase emails and convert integer dollar amounts to cents; export all rows.',
 'report_generation':'Summarize each measurement group in a Markdown report with its count, total and mean.',
 'document_conversion':'Convert the supplied Markdown sections into escaped HTML with headings and paragraphs.',
 'experiment_execution':'Run the supplied discrete decay simulations and retain every computed trajectory.',
 'database_migration':'Migrate legacy SQLite rows into a versioned table with normalized names and integer cents.',
 'document_indexing':'Build positional, lowercase word indexes for the supplied text documents.',
 'support_tickets':'Triage local synthetic tickets by the supplied priority rules and export the queue. No messages are sent.'}
WORK_TOOLS=dict(zip(FAMILIES,['repair_next_module','clean_next_batch','write_next_report','convert_next_document',
                            'run_next_simulation','migrate_next_row','index_next_document','triage_next_ticket']))


def fixture(root,family,seed,units):
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    if family not in FAMILIES:raise ValueError('Reserved workflows are not implemented in this release')
    inputs=root/'input';inputs.mkdir();rng=random.Random(seed)
    for i in range(units):
        p=inputs/f'{i:02d}.json'
        if family=='software_repair':
            dump(p,{'module':f'part_{i}','broken_source':f'def total(values):\n    return sum(values) + {rng.randint(1,9)}\n',
                    'tests':[[rng.randint(-9,9) for _ in range(k)] for k in [0,1,3,5]]})
        elif family=='data_cleaning':
            with (inputs/f'{i:02d}.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f);w.writerow(['id','name','email','dollars'])
                for j in range(3):w.writerow([i*3+j,f' Client {rng.randint(10,99)} ',f' User{i}_{j}@EXAMPLE.TEST ',rng.randint(1,90)])
        elif family=='report_generation':dump(p,{'group':f'group-{i}','values':[rng.randint(-20,50) for _ in range(rng.randint(3,7))]})
        elif family=='document_conversion':
            (inputs/f'{i:02d}.md').write_text(f'# Section {i}\n\nSample <{rng.randint(100,999)}> & quoted content.\nSecond paragraph {seed+i}.\n',encoding='utf-8')
        elif family=='experiment_execution':dump(p,{'initial':rng.randint(10,30),'rate':rng.choice([.05,.1,.2]),'steps':rng.randint(3,7)})
        elif family=='database_migration':
            pass
        elif family=='document_indexing':
            words=[rng.choice(['cedar','field','stone','water','work','result']) for _ in range(12)]
            (inputs/f'{i:02d}.txt').write_text(' '.join(words).capitalize()+'.',encoding='utf-8')
        elif family=='support_tickets':dump(p,{'id':100+i,'customer':f'client-{seed}-{i}','severity':rng.choice(['low','normal','urgent']),'status':'new'})
    if family=='database_migration':
        with closing(sqlite3.connect(inputs/'legacy.sqlite')) as db,db:
            db.execute('CREATE TABLE legacy (id INTEGER PRIMARY KEY, name TEXT, dollars INTEGER)')
            db.executemany('INSERT INTO legacy VALUES (?,?,?)',[(i,f' Account {rng.randint(10,99)} ',rng.randint(1,100)) for i in range(units)])
    manifest={'family':family,'units':units,'assignment':DESCRIPTIONS[family],'work_tool':WORK_TOOLS[family],
              'inputs':{p.name:sha(p) for p in sorted(inputs.iterdir())}}
    dump(root/'task.json',manifest);return manifest


def check_inputs(root):
    root=Path(root);task=read(root/'task.json')
    if any(sha(root/'input'/name)!=value for name,value in task['inputs'].items()):raise ValueError('Input artifact changed')
    return task


def source(root,index):
    root=Path(root);task=read(root/'task.json');family=task['family'];base=root/'input'/f'{index:02d}'
    if family=='data_cleaning':
        with base.with_suffix('.csv').open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
    if family in ['document_conversion','document_indexing']:return base.with_suffix('.md' if family=='document_conversion' else '.txt').read_text(encoding='utf-8')
    if family=='database_migration':
        with closing(sqlite3.connect((root/'input/legacy.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
            row=db.execute('SELECT id,name,dollars FROM legacy WHERE id=?',(index,)).fetchone()
        return {'id':row[0],'name':row[1],'dollars':row[2]}
    return read(base.with_suffix('.json'))


def process_unit(root,index):
    task=read(Path(root)/'task.json');family=task['family']
    if index<0 or index>=task['units']:raise ValueError('Work unit out of range')
    x=source(root,index)
    if family=='software_repair':return {'module':x['module'],'source':'def total(values):\n    return sum(values)\n'}
    if family=='data_cleaning':return [{'id':int(r['id']),'name':r['name'].strip(),'email':r['email'].strip().lower(),'cents':int(r['dollars'])*100} for r in x]
    if family=='report_generation':return f"# {x['group']}\ncount: {len(x['values'])}\ntotal: {sum(x['values'])}\nmean: {sum(x['values'])/len(x['values']):.12g}\n"
    if family=='document_conversion':return '\n'.join('<h1>'+html.escape(line[2:])+'</h1>' if line.startswith('# ') else '<p>'+html.escape(line)+'</p>' for line in x.splitlines() if line)
    if family=='experiment_execution':
        values=[float(x['initial'])]
        for _ in range(x['steps']):values.append(values[-1]-x['rate']*values[-1])
        return values
    if family=='database_migration':return {'id':x['id'],'name':x['name'].strip(),'cents':100*x['dollars']}
    if family=='document_indexing':
        index_map={}
        for position,word in enumerate(re.findall(r'[a-z]+',x.lower())):index_map.setdefault(word,[]).append(position)
        return index_map
    if family=='support_tickets':return {**x,'priority':{'low':'P3','normal':'P2','urgent':'P1'}[x['severity']],'status':'triaged'}
    raise ValueError('Unknown workflow')


class _HTML(HTMLParser):
    def __init__(self):super().__init__(convert_charrefs=True);self.tags=[];self.text=[]
    def handle_starttag(self,tag,attrs):self.tags.append(tag)
    def handle_data(self,data):
        if data.strip():self.text.append(data)


def validate_unit(root,index,value):
    """Output checker intentionally does not call process_unit."""
    family=read(Path(root)/'task.json')['family'];x=source(root,index)
    try:
        if family=='software_repair':
            if value['module']!=x['module']:return False
            tree=ast.parse(value['source'])
            if ast.dump(tree)!=ast.dump(ast.parse('def total(values):\n return sum(values)\n')):return False
            namespace={'__builtins__':{},'sum':sum};exec(compile(tree,'bounded_repair','exec'),namespace)
            return all(namespace['total'](v)==sum(v) for v in x['tests'])
        if family=='data_cleaning':
            if len(value)!=len(x):return False
            for raw,clean in zip(x,value):
                if set(clean)!={'id','name','email','cents'}:return False
                if clean['id']!=int(raw['id']) or clean['name']!=raw['name'].strip() or clean['email']!=raw['email'].strip().lower() or clean['cents']//100!=int(raw['dollars']) or clean['cents']%100:return False
            return True
        if family=='report_generation':
            lines=dict(line.split(': ',1) for line in value.splitlines()[1:]);total=0
            for number in x['values']:total+=number
            return value.splitlines()[0]=='# '+x['group'] and int(lines['count'])==len(x['values']) and int(lines['total'])==total and math.isclose(float(lines['mean']),total/len(x['values']),rel_tol=1e-10,abs_tol=1e-10)
        if family=='document_conversion':
            parsed=_HTML();parsed.feed(value)
            lines=[line for line in x.splitlines() if line]
            return parsed.tags==['h1' if line.startswith('# ') else 'p' for line in lines] and parsed.text==[line[2:] if line.startswith('# ') else line for line in lines]
        if family=='experiment_execution':return len(value)==x['steps']+1 and all(math.isclose(v,x['initial']*(1-x['rate'])**i,rel_tol=1e-10,abs_tol=1e-10) for i,v in enumerate(value))
        if family=='database_migration':return value=={'id':x['id'],'name':x['name'].strip(),'cents':int(str(x['dollars'])+'00')}
        if family=='document_indexing':
            words=re.findall('[a-z]+',x.casefold())
            return set(value)==set(words) and all(value[w]==[i for i,v in enumerate(words) if v==w] for w in set(words))
        if family=='support_tickets':
            expected_priority='P1' if x['severity']=='urgent' else 'P3' if x['severity']=='low' else 'P2'
            return value==dict(x,priority=expected_priority,status='triaged')
    except (KeyError,TypeError,ValueError,SyntaxError,AttributeError,OverflowError):return False
    return False


def validate_parts(root,parts,complete=False):
    task=check_inputs(root)
    return (len(parts)==task['units'] if complete else len(parts)<=task['units']) and all(validate_unit(root,i,value) for i,value in enumerate(parts))


def emit_result(root,parts,destination):
    root,destination=Path(root),Path(destination);task=check_inputs(root)
    if not validate_parts(root,parts,True):raise ValueError('Incomplete or corrupt work cannot be submitted')
    destination.mkdir(parents=True,exist_ok=False);dump(destination/'result.json',parts);family=task['family']
    if family=='database_migration':
        with closing(sqlite3.connect(destination/'migrated.sqlite')) as db,db:
            db.execute('CREATE TABLE records_v2 (id INTEGER PRIMARY KEY, name TEXT NOT NULL, cents INTEGER NOT NULL)')
            db.executemany('INSERT INTO records_v2 VALUES (?,?,?)',[(p['id'],p['name'],p['cents']) for p in parts])
    elif family in ['data_cleaning','support_tickets']:
        rows=[r for part in parts for r in part] if family=='data_cleaning' else parts
        with (destination/'export.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    else:
        suffix={'software_repair':'.py','report_generation':'.md','document_conversion':'.html','experiment_execution':'.json','document_indexing':'.json'}[family]
        for i,part in enumerate(parts):
            path=destination/(f'part_{i:02d}'+suffix)
            if family=='software_repair':path.write_text(part['source'],encoding='utf-8')
            elif isinstance(part,str):path.write_text(part,encoding='utf-8')
            else:dump(path,part)
    return validate_result(root,destination)


def validate_result(root,destination):
    root,destination=Path(root),Path(destination)
    try:
        parts=read(destination/'result.json');task=check_inputs(root)
        if not validate_parts(root,parts,True):return False
        family=task['family']
        if family=='database_migration':
            with closing(sqlite3.connect((destination/'migrated.sqlite').as_uri()+'?mode=ro',uri=True)) as db:
                rows=db.execute('SELECT id,name,cents FROM records_v2 ORDER BY id').fetchall()
                schema=db.execute('PRAGMA table_info(records_v2)').fetchall()
            return [r[1] for r in schema]==['id','name','cents'] and rows==[(p['id'],p['name'],p['cents']) for p in parts]
        if family in ['data_cleaning','support_tickets']:
            with (destination/'export.csv').open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
            expected=[r for part in parts for r in part] if family=='data_cleaning' else parts
            return rows==[{k:str(v) for k,v in row.items()} for row in expected]
        suffix={'software_repair':'.py','report_generation':'.md','document_conversion':'.html','experiment_execution':'.json','document_indexing':'.json'}[family]
        for i,part in enumerate(parts):
            path=destination/(f'part_{i:02d}'+suffix)
            actual=read(path) if suffix=='.json' else path.read_text(encoding='utf-8')
            if actual!=(part['source'] if family=='software_repair' else part):return False
        return True
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error):return False


def save_checkpoint(root,parts,path):
    root=Path(root)
    if not validate_parts(root,parts):raise ValueError('Checkpoint contains invalid work')
    payload={'task_sha256':sha(root/'task.json'),'input_files':read(root/'task.json')['inputs'],'parts':parts}
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    dump(path,{'payload':payload,'payload_sha256':hashlib.sha256(raw).hexdigest()});return sha(path)


def resume_checkpoint(root,checkpoint,destination):
    """Only task files and a durable checkpoint are available to this worker."""
    root=Path(root);task=check_inputs(root);envelope=read(checkpoint);payload=envelope['payload']
    raw=json.dumps(payload,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(raw).hexdigest()!=envelope['payload_sha256'] or payload['task_sha256']!=sha(root/'task.json') or payload['input_files']!=task['inputs']:raise ValueError('Checkpoint binding or checksum failed')
    parts=payload['parts']
    if not parts or not validate_parts(root,parts):raise ValueError('No valid retained work')
    restored=len(parts)
    for i in range(restored,task['units']):parts.append(process_unit(root,i))
    accepted=emit_result(root,parts,destination)
    return {'restored_units':restored,'new_units':task['units']-restored,'accepted_result':accepted,'checkpoint_sha256':sha(checkpoint)}
