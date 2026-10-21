import unittest
import torch
from src.composite_roles import char_spans,token_masks
from src.composite_detector import StructuredHeads,LABELS


class RoleTests(unittest.TestCase):
    def test_only_visible_header_locates_roles_and_swap_reverses_them(self):
        a='Current worker=unit-AAAA; peer=unit-BBBB;\nunit-AAAA active; unit-BBBB clear.'
        b='Current worker=unit-BBBB; peer=unit-AAAA;\nunit-AAAA active; unit-BBBB clear.'
        spans=char_spans(a);self.assertEqual([a[s:e] for s,e in spans],['unit-AAAA','unit-BBBB'])
        self.assertEqual([b[s:e] for s,e in char_spans(b)],['unit-BBBB','unit-AAAA'])
        offsets=torch.tensor([[[0,0]]+[[i,i+1] for i in range(len(a))]+[[0,0]]])
        masks=token_masks([a],offsets);self.assertEqual(masks.shape,(1,2,len(a)+2));self.assertTrue(all(int(m.sum())==9 for m in masks[0]));self.assertFalse(masks[0,:,0].any())
        with self.assertRaises(ValueError):char_spans('Gold own_status=active')
        with self.assertRaises(ValueError):token_masks([a],torch.zeros(1,3,2,dtype=torch.long))

    def test_heads_use_separate_learned_role_features_without_changing_evidence(self):
        heads=StructuredHeads(3,'role_pool');hidden=torch.randn(1,5,3,requires_grad=True);mask=torch.ones(1,5,dtype=torch.long);roles=torch.zeros(1,2,5,dtype=torch.bool);roles[0,0,1]=True;roles[0,1,3]=True
        inputs={};handles=[]
        for name,layer in heads.fields.items():handles.append(layer.register_forward_pre_hook(lambda module,args,name=name:inputs.__setitem__(name,args[0].detach().clone())))
        logits,span=heads(hidden,mask,roles)
        self.assertTrue(torch.equal(inputs['own_status'][0,3:],hidden.detach()[0,1]));self.assertTrue(torch.equal(inputs['peer_status'][0,3:],hidden.detach()[0,3]));self.assertEqual(inputs['event'].shape,(1,3));self.assertEqual(inputs['affected'].shape,(1,9));self.assertEqual(span.shape,(1,5,2))
        sum(v.sum() for v in logits.values()).backward();self.assertGreater(float(hidden.grad[0,1].abs().sum()),0);self.assertGreater(float(hidden.grad[0,3].abs().sum()),0)
        for handle in handles:handle.remove()
        _,swapped=heads(hidden,mask,roles.flip(1));self.assertTrue(torch.equal(span,swapped))
        bad=roles.clone();bad[:,0]=False
        with self.assertRaises(ValueError):heads(hidden,mask,bad)

    def test_cls_compatibility_and_unknown_head_rejected(self):
        heads=StructuredHeads(3);logits,_=heads(torch.ones(1,4,3),torch.ones(1,4))
        self.assertEqual(set(logits),set(LABELS))
        with self.assertRaises(ValueError):StructuredHeads(3,'gold_lookup')


if __name__=='__main__':unittest.main()
