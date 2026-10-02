"""Read-only selected-layer activations aligned to emitted-token predictions."""
import re
from pathlib import Path


class Capture:
    def __init__(self, model, root):
        self.root = Path(root)
        layers = [(name,module) for name,module in model.named_modules()
                  if re.search(r'\.language_model\.layers\.\d+$',name)]
        if not layers:
            raise RuntimeError('No language decoder layers found for activation capture')
        indices = sorted(set((0,len(layers)//3,2*len(layers)//3,len(layers)-1)))
        self.names = [layers[i][0] for i in indices]
        self.buffers = {}
        self.handles = []
        for i in indices:
            name,module = layers[i]
            def hook(_module,_args,output,name=name):
                value = output[0] if isinstance(output,tuple) else output
                if value.ndim!=3:
                    raise RuntimeError('Unexpected activation shape')
                self.buffers[name].append(value[:,-1,:].detach().clone())
            self.handles.append(module.register_forward_hook(hook))

    def generate(self, original, model, tokenizer, conversations, cap, tools=None):
        import torch
        from safetensors.torch import save_file
        self.buffers = {name:[] for name in self.names}
        turns = original(model,tokenizer,conversations,cap,tools)
        maximum = max(len(turn['token_ids']) for turn in turns)
        if any(len(values)!=maximum for values in self.buffers.values()):
            raise RuntimeError('Activation/generation-step alignment mismatch')
        stacked = {name:torch.stack(values) for name,values in self.buffers.items()}
        for i,turn in enumerate(turns):
            count = len(turn['token_ids'])
            path = self.root/'activations'/(turn['batch_id']+'_row'+str(i)+'.safetensors')
            path.parent.mkdir(parents=True,exist_ok=True)
            tensors = {'layer_'+str(j):values[:count,i,:].cpu().contiguous() for j,values in enumerate(stacked.values())}
            save_file(tensors,str(path),metadata={'alignment':'row t predicts generated token t; t=0 uses final prompt position'})
            turn['activation_record'] = {'path':str(path.relative_to(self.root)),'layers':self.names,
                'rows':count,'dtype':str(next(iter(tensors.values())).dtype),
                'alignment':'predictor state for each emitted token, including EOS; prefill stores its final position only',
                'observational_only':True}
        self.buffers = {}
        return turns
