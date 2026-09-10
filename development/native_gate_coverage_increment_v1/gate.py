"""Fixed binary64 minimum-norm affine hard-margin dual active-set solver."""
from dataclasses import dataclass
from pathlib import Path
import hashlib,json,math,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
CONTRACT={'checkpoint':'Qwen/Qwen3.5-0.8B@2fc06364715b967f1860aea9cf38778875588b17',
    'native_target':'model.language_model.layers.10','position':'final_input','residual_dtype':'float32','width':1024}
METHOD={'name':'minimum_norm_hard_margin_affine_v1','objective':'0.5*||w||^2',
    'constraints':'y_i*(w.dot(x_i)+b)>=1','intercept_penalty':0,'class_weights':False,'slack':False,
    'centering':'unweighted_training_rows_only','normalization':'row_l2','on_rule':'score>0',
    'threshold':0.,'arithmetic':'binary64','solver':'dual_feasible_active_set_v1',
    'numpy_version':'2.5.2','iteration_limit':10000,'linear_tolerance':1e-10,
    'active_tolerance':1e-12,'certificate_tolerance':1e-8,
    'tie_rule':'most_negative_reduced_gradient_then_smallest_index;first_ratio_then_smallest_index',
    'intercept_rule':'midpoint_of_training_feasible_bounds;gap_tolerance1e-8',
    'singular_policy':'UNCERTIFIED_OPTIMIZATION_NO_FALLBACK'}
def require(ok,code):
    if not ok:raise ValueError(code)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False,ensure_ascii=True).encode('ascii')
def decode(raw):
    def unique(pairs):
        out={}
        for k,v in pairs:
            require(k not in out,'DUPLICATE_JSON_KEY');out[k]=v
        return out
    def bad(v):raise ValueError('JSON_NONFINITE')
    return json.loads(raw,object_pairs_hook=unique,parse_constant=bad)
def dot(a,b):return math.fsum(u*v for u,v in zip(a,b,strict=True))
def finite_rows(rows):
    require(type(rows) in (list,tuple) and len(rows)>=2,'ROWS')
    d=len(rows[0]);require(1<=d<=1024,'WIDTH')
    require(all(len(r)==d and all(type(v) in (int,float) and math.isfinite(v) for v in r) for r in rows),'FINITE_ROWS')
    return tuple(tuple(float(v) for v in r) for r in rows)
def transform(row,mu):
    require(len(row)==len(mu) and all(type(v) in (int,float) and math.isfinite(v) for v in row),'TRANSFORM_ROW')
    delta=tuple(float(v)-m for v,m in zip(row,mu));norm=math.sqrt(dot(delta,delta))
    require(norm>0 and math.isfinite(norm),'ZERO_OR_NONFINITE_NORM')
    return tuple(v/norm for v in delta)
def preprocess(rows):
    rows=finite_rows(rows);mu=tuple(math.fsum(r[j] for r in rows)/len(rows) for j in range(len(rows[0])))
    return mu,tuple(transform(r,mu) for r in rows)
def numpy_runtime():
    path=ROOT/'.venv/Lib/site-packages'
    if 'numpy' not in sys.modules:
        sys.path.insert(0,str(path))
        try:import numpy
        finally:sys.path.remove(str(path))
    np=sys.modules['numpy']
    require(np.__version__=='2.5.2' and Path(np.__file__).resolve()==(path/'numpy/__init__.py').resolve(),'PINNED_NUMPY_RUNTIME')
    return np
class OptimizationFailure(ValueError):
    def __init__(self,code,classification='UNCERTIFIED_OPTIMIZATION',iterations=0):
        super().__init__(code);self.code=code;self.classification=classification;self.iterations=iterations
def solve(rows,labels,*,deadline,max_iterations=10000):
    x=finite_rows(rows);n=len(x)
    require(len(labels)==n and all(type(v) is int and v in (-1,1) for v in labels)
        and set(labels)=={-1,1},'BOTH_BINARY_CLASSES')
    require(type(max_iterations) is int and 0<=max_iterations<=10000,'ITERATION_LIMIT_BOUND')
    for i in range(n):
        for j in range(i):
            if labels[i]!=labels[j] and x[i]==x[j]:
                raise OptimizationFailure('EXACT_CONTRADICTORY_DUPLICATE','CERTIFIED_INFEASIBLE',0)
    np=numpy_runtime();y=np.asarray(labels,dtype=np.float64)
    Q=np.asarray([[labels[i]*labels[j]*dot(x[i],x[j]) for j in range(n)] for i in range(n)],dtype=np.float64)
    require(np.isfinite(Q).all() and np.max(np.abs(Q-Q.T))<=1e-10,'FINITE_SYMMETRIC_Q')
    alpha=np.zeros(n,dtype=np.float64)
    free=sorted([labels.index(1),labels.index(-1)])
    iterations=0
    while True:
        if time.monotonic()>=deadline:raise OptimizationFailure('SOLVER_DEADLINE',iterations=iterations)
        if iterations>=max_iterations:raise OptimizationFailure('SOLVER_ITERATION_LIMIT',iterations=iterations)
        iterations+=1
        if min(alpha)<-1e-12 or abs(float(y@alpha))>1e-10:
            raise OptimizationFailure('DUAL_FEASIBLE_ITERATE_LOST',iterations=iterations)
        k=len(free);K=np.zeros((k+1,k+1),dtype=np.float64)
        K[:k,:k]=Q[np.ix_(free,free)];K[:k,k]=y[free];K[k,:k]=y[free]
        rhs=np.ones(k+1,dtype=np.float64);rhs[k]=0.
        try:answer=np.linalg.solve(K,rhs)
        except np.linalg.LinAlgError:
            raise OptimizationFailure('SINGULAR_RESTRICTED_KKT',iterations=iterations) from None
        if not np.isfinite(answer).all() or np.max(np.abs(K@answer-rhs))>1e-10:
            raise OptimizationFailure('RESTRICTED_KKT_UNCERTIFIED',iterations=iterations)
        target=np.zeros(n,dtype=np.float64);target[free]=answer[:k]
        bad=[i for i in free if target[i]<-1e-12]
        if bad:
            direction=target-alpha
            ratios=[(float(alpha[i]/(-direction[i])),i) for i in free if direction[i]<-1e-12]
            if not ratios:raise OptimizationFailure('NO_FEASIBLE_BOUND_STEP',iterations=iterations)
            step,blocker=min(ratios);step=max(0.,min(1.,step))
            alpha=alpha+step*direction
            if np.min(alpha)<-1e-10:raise OptimizationFailure('BOUND_STEP_UNCERTIFIED',iterations=iterations)
            alpha[np.abs(alpha)<1e-12]=0.;alpha[blocker]=0.;free.remove(blocker)
            if not free:raise OptimizationFailure('EMPTY_ACTIVE_SET',iterations=iterations)
            continue
        alpha=target;alpha[np.abs(alpha)<1e-12]=0.
        reduced=Q@alpha-1.+answer[k]*y
        excluded=[i for i in range(n) if i not in free]
        entering=min(excluded,key=lambda i:(float(reduced[i]),i)) if excluded else None
        if entering is not None and reduced[entering]<-1e-12:
            free=sorted([*free,entering]);continue
        w=tuple(math.fsum(float(alpha[i])*labels[i]*x[i][j] for i in range(n)) for j in range(len(x[0])))
        lower=max(1.-dot(w,x[i]) for i in range(n) if labels[i]==1)
        upper=min(-1.-dot(w,x[i]) for i in range(n) if labels[i]==-1)
        if lower>upper+1e-8:raise OptimizationFailure('NO_CERTIFIED_INTERCEPT_INTERVAL',iterations=iterations)
        b=(lower+upper)/2.
        from checker import verify
        try:certificate=verify(x,tuple(labels),w,b,tuple(float(v) for v in alpha))
        except ValueError:raise OptimizationFailure('FINAL_KKT_UNCERTIFIED',iterations=iterations) from None
        if time.monotonic()>=deadline:raise OptimizationFailure('SOLVER_DEADLINE',iterations=iterations)
        return {'status':'CERTIFIED','w':list(w),'b':b,'alpha':[float(v) for v in alpha],
            'iterations':iterations,'intercept_lower':lower,'intercept_upper':upper,'certificate':certificate}
@dataclass(frozen=True)
class Gate:
    mu:tuple
    w:tuple
    b:float
    def __post_init__(self):
        require(type(self.mu) is tuple and type(self.w) is tuple and 1<=len(self.mu)==len(self.w)<=1024,'IMMUTABLE_GATE_SHAPE')
        require(all(type(v) in (int,float) and math.isfinite(v) for v in (*self.mu,*self.w,self.b)),'IMMUTABLE_GATE_FINITE')
    def score(self,row):return dot(self.w,transform(row,self.mu))+self.b
    def route(self,row):return 'ON' if self.score(row)>0 else 'OFF'
def artifact(model,bindings):
    require(type(model.mu) is tuple and type(model.w) is tuple and len(model.mu)==len(model.w),'ARTIFACT_SHAPE')
    require(all(math.isfinite(v) for v in (*model.mu,*model.w,model.b)),'ARTIFACT_FINITE')
    return canonical({'schema':'hardmargin_familydev_gate.v1','method':METHOD,'feature_contract':CONTRACT,
        'bindings':bindings,'parameters':{'mu':model.mu,'w':model.w,'b':model.b}})
def load_artifact(raw,*,expected_sha256,expected_bindings):
    require(digest(raw)==expected_sha256,'ARTIFACT_HASH');v=decode(raw)
    require(canonical(v)==raw and set(v)=={'schema','method','feature_contract','bindings','parameters'},'ARTIFACT_CANONICAL_SCHEMA')
    require(v['schema']=='hardmargin_familydev_gate.v1' and v['method']==METHOD and v['feature_contract']==CONTRACT
        and v['bindings']==expected_bindings,'ARTIFACT_BINDINGS')
    p=v['parameters'];require(set(p)=={'mu','w','b'},'PARAMETERS')
    require(type(p['mu']) is list and type(p['w']) is list and 1<=len(p['mu'])==len(p['w'])<=1024
        and all(type(z) in (int,float) and math.isfinite(z) for z in (*p['mu'],*p['w'],p['b'])),'PARAMETER_FINITE_SHAPE')
    return Gate(tuple(p['mu']),tuple(p['w']),float(p['b']))
