"""Small CPU checks for weighted solves, multi-layer patches and causal cache replay."""
import copy,json
import numpy as np
import torch
from dense_math import fit,replay_fit,predict_numpy,Predictor,FIT_RANK,INFERENCE_RANK
from dense_hook import DenseHook

torch.set_num_threads(2);torch.manual_seed(23)
rng=np.random.default_rng(23);x=rng.normal(size=(40,12)).astype(np.float32)
active=np.arange(40)%2==0;y=np.zeros_like(x);y[active,:8]=x[active,:8]*.3+.2
w=np.concatenate([np.full(n,40/(3*n)) for n in (9,13,18)]).astype(np.float32)
a=fit(x,y,active,w,device='cpu',input_rank=4,output_rank=FIT_RANK,ridge=1.)
assert FIT_RANK==8 and INFERENCE_RANK==4
assert a['basis'].shape==(12,8) and a['weights'].shape==(9,8)
assert np.linalg.norm(a['weights'][:,4:])>1e-6
error=np.linalg.norm(replay_fit(a)-a['weights'])/np.linalg.norm(a['weights']);assert error<1e-5
batch=predict_numpy(x,a);single=np.concatenate([predict_numpy(row[None],a) for row in x]);assert np.allclose(batch,single,atol=1e-6)

# Give every direction a distinct coefficient so dropping direction 3 or 4 fails.
probe={'x_mean':np.zeros(12,dtype=np.float32),'input_basis':np.eye(12,4,dtype=np.float32),
       'scale':np.ones(4,dtype=np.float32),'basis':np.eye(12,8,dtype=np.float32),
       'weights':np.zeros((9,8),dtype=np.float32)}
probe['weights'][-1]=np.arange(1,9,dtype=np.float32)
expected=np.zeros((1,12),dtype=np.float32);expected[0,:4]=[1,2,3,4]
assert np.array_equal(predict_numpy(np.zeros((1,12),dtype=np.float32),probe),expected)
deployed=Predictor(probe,'cpu');assert deployed.rank==4
assert np.array_equal(deployed(torch.zeros((1,12))).numpy(),expected)
upper=copy.deepcopy(probe);upper['weights'][:,4:]+=100
assert np.array_equal(Predictor(upper,'cpu')(torch.zeros((1,12))).numpy(),expected)

class Layer(torch.nn.Module):
    def forward(self,x,previous):
        total=x.cumsum(1)+previous[:,None,:]
        return x+.05*total,total[:,-1].clone()
class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.embedding=torch.nn.Embedding(17,4);self.layers=torch.nn.ModuleList([Layer() for _ in range(3)])
    def forward(self,ids,past=None):
        h=self.embedding(ids);states=[]
        for i,layer in enumerate(self.layers):
            previous=torch.zeros((1,4)) if past is None else past[i]
            h,state=layer(h,previous);states.append(state)
        return h,states
class P:
    width=4
    def __call__(self,h):return h.float()*.02+.1
model=Toy().eval().requires_grad_(False);before={k:v.clone() for k,v in model.state_dict().items()}
tokens=[1,2,3,4,5,6,7];prefix=3
def sequence(sites,cache,zero=False):
    hook=DenseHook(model.layers,{i:P() for i in range(3)},sites,zero=zero);hook.begin_response(prefix,True)
    past=None;values=[]
    try:
        for count in range(prefix,len(tokens)+1):
            cached=cache and past is not None;hook.prepare(count,cached)
            ids=torch.tensor([tokens[count-1:count] if cached else tokens[:count]])
            with torch.inference_mode():h,next_past=model(ids,past if cached else None)
            hook.after_forward();past=next_past if cache else None;values.append(h[0,-1].numpy().copy())
        records=copy.deepcopy(hook.deliveries)
    finally:hook.close()
    return np.asarray(values),records
for sites in ([0],[1],[2],[0,1,2]):
    cached,records=sequence(sites,True);full,_=sequence(sites,False)
    assert np.allclose(cached,full,atol=2e-6),sites
    assert all(r['nonzero_layer_patches']==len(sites) for r in records)
zero,_=sequence([0,1,2],True,True)
plain=[];past=None
for count in range(prefix,len(tokens)+1):
    ids=torch.tensor([tokens[count-1:count] if past is not None else tokens[:count]])
    with torch.inference_mode():h,past=model(ids,past)
    plain.append(h[0,-1].numpy().copy())
assert np.array_equal(zero,np.asarray(plain))
assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
print(json.dumps({'CPU_only':True,'full_Qwen_loaded':False,'weighted_solver_replay_relative_L2':float(error),
    'fit_directions':FIT_RANK,'deployed_directions':INFERENCE_RANK,'all_four_deployed_directions_verified':True,
    'all_eight_fit_columns_present':True,
    'batch_singleton_predictions_match':True,'all_single_and_joint_cache_replay_match':True,'zero_hook_exact':True,
    'toy_weights_unchanged':True,'new_JEV_calls':0,'paid_experiments':0}))
