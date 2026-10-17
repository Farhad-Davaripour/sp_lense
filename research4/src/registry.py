"""MLflow/UC gate package; the same trainables file also holds LoRA weights."""
from pathlib import Path
def register(checkpoint,root,experiment,metadata,example_feature):
    import mlflow,mlflow.pyfunc,torch,pandas as pd
    from mlflow.models.signature import ModelSignature
    from mlflow.types.schema import Schema,ColSpec,Array,DataType
    from .learners import load_gate,GateState
    from .io_utils import load_torch
    class GateArtifact(mlflow.pyfunc.PythonModel):
        def load_context(self,context):
            import torch
            from src.learners import load_gate
            self.c=torch.load(context.artifacts['trainables'],map_location='cpu',weights_only=True);self.h,self.r=load_gate(self.c)
        def predict(self,context,model_input,params=None):
            import torch,pandas as pd
            result=[]
            for _,row in model_input.iterrows():
                with torch.no_grad():
                    x=(torch.tensor(row['features'],dtype=torch.float32)-self.c['mean'])/self.c['scale'];p=self.h(x).sigmoid();prev=row.get('previous_hidden',None)
                    hidden=torch.tensor(prev,dtype=torch.float32).reshape(1,1,-1) if prev is not None and len(prev) else None
                    y,next_h=self.r(p[None,None],hidden);prob=float(y.sigmoid()[0,0])
                result.append({'threat_probability':float(p[0]),'unfinished_probability':float(p[1]),'useful_probability':float(p[2]),'gate_probability':prob,'gate_on':prob>=self.c['threshold'],'next_hidden':next_h[0,0].tolist()})
            return pd.DataFrame(result)
    mlflow.set_tracking_uri('databricks');mlflow.set_registry_uri('databricks-uc');mlflow.set_experiment(experiment)
    signature=ModelSignature(inputs=Schema([ColSpec(Array(DataType.double),'features'),ColSpec(Array(DataType.double),'previous_hidden',required=False)]),outputs=Schema([ColSpec(DataType.double,n) for n in ['threat_probability','unfinished_probability','useful_probability','gate_probability']]+[ColSpec(DataType.boolean,'gate_on'),ColSpec(Array(DataType.double),'next_hidden')]))
    with mlflow.start_run(run_name='research4-semantic-gate-pilot') as run:
        mlflow.log_params({k:str(v) for k,v in metadata.items()});mlflow.log_artifact(str(checkpoint),'checkpoints')
        info=mlflow.pyfunc.log_model(name='gate_bundle',python_model=GateArtifact(),artifacts={'trainables':str(checkpoint)},code_paths=[str(Path(root)/'src')],signature=signature,pip_requirements=['torch==2.11.0+cpu','mlflow==3.1.0','numpy','pandas'])
        version=mlflow.register_model(info.model_uri,'workspace.research4.semantic_gate_pilot')
        loaded=mlflow.pyfunc.load_model(info.model_uri);prediction=loaded.predict(pd.DataFrame({'features':[list(map(float,example_feature))],'previous_hidden':[[]]}))
        saved=load_torch(checkpoint);expected_p,expected_probability,expected_on=GateState(saved).update(example_feature)
        actual=prediction.iloc[0];observed=[actual[n] for n in ['threat_probability','unfinished_probability','useful_probability']]
        if max(abs(float(a)-float(b)) for a,b in zip(observed,expected_p))>1e-6 or abs(float(actual['gate_probability'])-expected_probability)>1e-6 or bool(actual['gate_on'])!=expected_on:raise ValueError('Registered artifact differs from saved checkpoint')
        return {'mlflow_run_id':run.info.run_id,'model_uri':info.model_uri,'registered_model':'workspace.research4.semantic_gate_pilot','registered_version':str(version.version),'registered_consumer_matches_saved_checkpoint':True,'artifact_consumer_check':prediction.to_dict(orient='records')}
