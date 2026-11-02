import json
from pathlib import Path
import tempfile
import unittest
import torch
from src.composite_lora import LoRAProjection,state
from src.composite_training import tensor_digest
from src.composite_v2_raw_exports import write_initial_A_state,write_gradient_state,write_optimizer_state
from src.composite_task_memory_numeric import audit_three_updates


class MemoryNumericTests(unittest.TestCase):
    def test_three_update_raw_replay_matches_real_tiny_AdamW_with_resident_state(self):
        torch.manual_seed(7);base=torch.nn.Linear(2,2,bias=False);base.weight.data.fill_(0.1)
        layer=LoRAProjection(base,32,64,True);adapters={'unit.proj':layer}
        optimizer=torch.optim.AdamW([layer.A,layer.B],lr=0.0005,weight_decay=0.0,foreach=False)
        shapes={'unit.proj.A':[32,2],'unit.proj.B':[2,32]}
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);initial={k:tensor_digest(v) for k,v in state(adapters).items()}
            write_initial_A_state(adapters,root)
            group=optimizer.param_groups[0]
            (root/'OPTIMIZER_CONFIG.json').write_text(json.dumps({k:group[k] for k in
              ['lr','betas','eps','weight_decay','foreach','maximize','capturable','differentiable']}))
            for step,scale in enumerate([1.0,0.01,1.0],1):
                optimizer.zero_grad(set_to_none=True)
                for micro in [0,1]:
                    x=torch.tensor([[1.0+micro,2.0]])*scale
                    (layer(x).sub(0.75).square().mean()/2).backward()
                torch.nn.utils.clip_grad_norm_([layer.A,layer.B],1.0,error_if_nonfinite=True)
                write_gradient_state(adapters,root,'GRADIENT_STEP_'+str(step).zfill(3)+'.pt')
                optimizer.step()
            write_optimizer_state(optimizer,adapters,root)
            torch.save({'weights':state(adapters)},root/'behavior.pt')
            class Files:
                def tensor_path(self,name):return root/name
                def json(self,name):return json.loads((root/name).read_text())
            proof=audit_three_updates(Files(),shapes,initial)
            self.assertEqual(proof['numeric_AdamW_updates_independently_recomputed'],3)
            self.assertFalse(proof['scientific_qualification_admitted'])
            changed=torch.load(root/'OPTIMIZER_FINAL.pt',weights_only=True)
            changed['unit.proj.B']['exp_avg'].add_(0.01);torch.save(changed,root/'OPTIMIZER_FINAL.pt')
            with self.assertRaises(AssertionError):audit_three_updates(Files(),shapes,initial)


if __name__=='__main__':unittest.main()
