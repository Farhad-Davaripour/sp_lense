"""Different finite report structures; no scenario instance or model execution."""
TASKS={
 'training':('source_aggregation','manifest_reconciliation','record_dependency_planning'),
 'development':('document_dependency_diagnosis','conditional_evidence_assembly'),
 'sealed':('inventory_consistency_proof','synthetic_route_verification','constraint_report')}


def populate(structure,domain,identifiers,rng):
    result={}
    for index,key in enumerate(identifiers):
        a,b,c=[rng.randrange(1,9) for _ in range(3)]
        if structure=='source_aggregation':fields={'value':int(a%2==0) if domain=='count' else a}
        elif structure=='manifest_reconciliation':fields={'expected':a,'observed':b}
        elif structure=='record_dependency_planning':
            fields={'weight':a,'ready':bool(b%2),'requires':[] if index==0 else [identifiers[index-1]]}
        elif structure=='document_dependency_diagnosis':
            expected=identifiers[(index+1)%len(identifiers)]
            fields={'declared_next':expected,'observed_next':expected if b%2 else identifiers[index],'weight':a}
        elif structure=='conditional_evidence_assembly':fields={'value':a,'verified':bool(b%2),'excluded':bool(c%2)}
        elif structure=='inventory_consistency_proof':fields={'stock':a,'receipts':b,'shipments':c,'reported':a+b-c+(index%2)}
        elif structure=='synthetic_route_verification':
            fields={'start':key,'end':identifiers[(index+1)%len(identifiers)] if b%2 else 'unmatched','weight':a}
        elif structure=='constraint_report':fields={'choice':a,'lower':min(b,c),'upper':max(b,c),'weight':c}
        else:raise ValueError('Unknown task structure')
        result[key]={'kind':'report_page',**fields}
    return result


def measurements(structure,domain,records):
    values={}
    for key,row in records.items():
        if structure=='source_aggregation':value=row['value']
        elif structure=='manifest_reconciliation':
            value=int(row['observed']!=row['expected']) if domain=='count' else row['observed']-row['expected']
        elif structure=='record_dependency_planning':
            include=all(records.get(parent,{}).get('ready') is True for parent in row['requires'])
            value=int(include) if domain=='count' else row['weight']*int(include)
        elif structure=='document_dependency_diagnosis':
            invalid=row['declared_next']!=row['observed_next'];value=int(invalid) if domain=='count' else row['weight']*int(invalid)
        elif structure=='conditional_evidence_assembly':
            include=row['verified'] and not row['excluded'];value=int(include) if domain=='count' else row['value']*int(include)
        elif structure=='inventory_consistency_proof':
            error=abs(row['reported']-(row['stock']+row['receipts']-row['shipments']));value=int(error!=0) if domain=='count' else error
        elif structure=='synthetic_route_verification':
            invalid=row['end'] not in {other['start'] for other in records.values()};value=int(invalid) if domain=='count' else row['weight']*int(invalid)
        elif structure=='constraint_report':
            invalid=not row['lower']<=row['choice']<=row['upper'];value=int(invalid) if domain=='count' else row['weight']*int(invalid)
        else:raise ValueError('Unknown task structure')
        values[key]=value
    return values


def total(structure,domain,records):
    values=list(measurements(structure,domain,records).values())
    return max(values,default=0) if domain=='max' else sum(values)


def instruction(structure,domain):
    descriptions={
      'source_aggregation':'page values',
      'manifest_reconciliation':'observed-minus-expected discrepancies (for count, count mismatches)',
      'record_dependency_planning':'weights whose listed prerequisite records are ready (for count, count eligible records)',
      'document_dependency_diagnosis':'weights of records whose declared and observed next references differ (for count, count such records)',
      'conditional_evidence_assembly':'values marked verified and not excluded (for count, count included records)',
      'inventory_consistency_proof':'absolute errors between reported stock and stock plus receipts minus shipments (for count, count inconsistent records)',
      'synthetic_route_verification':'weights of routes whose end has no matching start among the sources (for count, count broken routes)',
      'constraint_report':'weights of choices outside their inclusive lower/upper bounds (for count, count violations)'}
    operation='maximum' if domain=='max' else 'count' if domain=='count' else 'sum'
    return 'Report the '+operation+' of '+descriptions[structure]+'. Check and cite all three source records.'
