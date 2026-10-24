"""Tiny meta architecture and fake parameter records; no pretrained weights."""
import base64,hashlib,io,json,unittest
from copy import deepcopy
import torch
from transformers import AutoModel,ModernBertConfig
from src.composite_capacity import detector_targets
from src.composite_detector import StructuredHeads
from src.composite_training import tensor_digest
from src.composite_v2_model_audit import detector_integrity,detector_parameter_roster


class DetectorIntegrityTests(unittest.TestCase):
    def fixture(self):
        config=ModernBertConfig(vocab_size=32,hidden_size=8,num_hidden_layers=2,num_attention_heads=2,
            intermediate_size=16,pad_token_id=0,bos_token_id=1,eos_token_id=2,cls_token_id=1,sep_token_id=2,max_position_embeddings=32)
        with torch.device('meta'):
            model=AutoModel.from_config(config,trust_remote_code=False,attn_implementation='sdpa')
            heads=StructuredHeads(8,'role_pool')
        cfg={'hidden_size':8,'layers':2,'lora_last_layers':2,'rank':2,'alpha':4,'scope':'all_blocks',
             'lora_modules':['Wqkv','Wo','Wi']}
        targets=detector_targets(model,cfg);lora={}
        for name in targets:
            module=model.get_submodule(name)
            lora[name+'.A']=torch.ones(2,module.in_features);lora[name+'.B']=torch.ones(module.out_features,2)
        saved={'lora':lora,'heads':{n:torch.ones(v.shape) for n,v in heads.state_dict().items()},
               'config':cfg,'placements':targets,'head_mode':'role_pool'}
        checkpoint=io.BytesIO();torch.save(saved,checkpoint);raw=checkpoint.getvalue()
        metadata=json.dumps(config.to_dict()).encode();metadata_sha=hashlib.sha256(metadata).hexdigest()
        pin={'files':[{'name':'config.json','size':len(metadata),'sha256':metadata_sha}]}
        class Files:
            def __init__(self):self.records={
                'DETECTOR_FILE_VERIFICATION.json':[{'name':'config.json','size':len(metadata),'sha256':metadata_sha}],
                'DETECTOR_METADATA_BYTES.json':{'config.json':base64.b64encode(metadata).decode()},
                'DETECTOR_RUNTIME.json':{'config':cfg,'head_mode':'role_pool','projection_names':targets,
                    'projection_scales':{name:2.0 for name in targets},'all_modules_in_eval_mode':True,'full_attention':'sdpa'}}
            def json(self,name):return deepcopy(self.records[name])
        files=Files();roster=detector_parameter_roster(files,pin,saved)
        trained={**{'encoder.'+n:v for n,v in saved['lora'].items()},**{'heads.'+n:v for n,v in saved['heads'].items()}}
        records=[{'name':n,'shape':shape,'dtype':'torch.float32','device':'cuda:0','requires_grad':False,'gradient_is_none':True,
                  'sha256':tensor_digest(trained[n]) if n in trained else 'a'*64} for n,shape in roster.items()]
        files.records['DETECTOR_PARAMETERS_BEFORE.json']=records
        files.records['DETECTOR_PARAMETERS_AFTER.json']=deepcopy(records)
        return files,pin,raw,hashlib.sha256(raw).hexdigest(),set(trained)

    def test_complete_tiny_record_consistency_is_not_a_scientific_permit(self):
        files,pin,raw,sha,_=self.fixture()
        result=detector_integrity(files,pin,raw,sha)
        self.assertGreater(result['frozen_parameters'],result['selected_checkpoint_tensors'])
        self.assertNotIn('scientific_qualification_admitted',result)

    def test_matching_before_after_without_base_or_with_wrong_scale_is_rejected(self):
        files,pin,raw,sha,trained=self.fixture()
        name=next(v['name'] for v in files.records['DETECTOR_PARAMETERS_BEFORE.json'] if v['name'] not in trained)
        for key in ['DETECTOR_PARAMETERS_BEFORE.json','DETECTOR_PARAMETERS_AFTER.json']:
            files.records[key]=[v for v in files.records[key] if v['name']!=name]
        with self.assertRaisesRegex(ValueError,'Complete'):detector_integrity(files,pin,raw,sha)
        files,pin,raw,sha,_=self.fixture()
        first=next(iter(files.records['DETECTOR_RUNTIME.json']['projection_scales']))
        files.records['DETECTOR_RUNTIME.json']['projection_scales'][first]=0.0
        with self.assertRaisesRegex(ValueError,'runtime'):detector_integrity(files,pin,raw,sha)


if __name__=='__main__':unittest.main()
