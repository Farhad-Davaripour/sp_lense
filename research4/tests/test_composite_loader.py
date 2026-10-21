"""Pinned loader contract with synthetic modules, never pretrained weights."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from src.composite_detector import Detector


class LoaderTests(unittest.TestCase):
    def test_modernbert_no_removed_constructor_arguments(self):
        encoder=nn.Module()
        encoder.config=SimpleNamespace(hidden_size=4,num_hidden_layers=2)
        encoder.layers=nn.ModuleList([nn.ModuleDict({'Wqkv':nn.Linear(4,12),'Wo':nn.Linear(4,4)}) for _ in range(2)])
        cfg={'hidden_size':4,'layers':2,'lora_last_layers':1,'lora_modules':['Wqkv','Wo'],'rank':2,'alpha':4,'maximum_tokens':32}
        # The installed v5 constructor accepts config only. This stub models
        # its from_pretrained consumed kwargs and rejects leaked model kwargs.
        def load(path,*,local_files_only,trust_remote_code,use_safetensors,dtype,attn_implementation):
            self.assertEqual(path,'synthetic-no-files')
            self.assertTrue(local_files_only and use_safetensors)
            self.assertFalse(trust_remote_code)
            self.assertEqual(dtype,torch.float32)
            self.assertEqual(attn_implementation,'sdpa')
            return encoder
        with patch('src.composite_data.require_databricks'),patch('transformers.AutoTokenizer.from_pretrained',return_value=object()),patch('transformers.AutoModel.from_pretrained',side_effect=load):
            detector=Detector('synthetic-no-files',cfg,device='cpu')
        self.assertEqual(set(detector.adapters),{'layers.1.Wqkv','layers.1.Wo'})
        self.assertTrue(all(not p.requires_grad for _,p in detector.base_references))


if __name__=='__main__':unittest.main()
