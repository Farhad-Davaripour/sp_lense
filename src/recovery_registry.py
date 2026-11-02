"""Register the small adapter artifact; its consumer applies projection deltas."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import mlflow
from mlflow.models import infer_signature
from .io_utils import load_torch


class AdapterDelta(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.weights = load_torch(context.artifacts['adapter'])['lora']
        checkpoint = load_torch(context.artifacts['adapter'])
        model = checkpoint['model_config']
        self.scale = model['lora_alpha'] / model['lora_rank']

    def predict(self, context, model_input, params=None):
        rows = []
        for _, row in model_input.iterrows():
            index = int(row['projection'])
            a, b = self.weights[str(index) + '.A'], self.weights[str(index) + '.B']
            x = torch.as_tensor(np.asarray(row['features']), dtype=torch.float32)
            if x.ndim != 1 or x.shape[0] != a.shape[1]:
                raise ValueError('Wrong projection input dimension')
            with torch.no_grad():
                delta = (b @ (a @ x)) * self.scale if bool(row['enabled']) else torch.zeros(b.shape[0])
            rows.append({'delta': delta.numpy().tolist()})
        return pd.DataFrame(rows)


def register(checkpoint, root, source_commit, dataset_version):
    saved = load_torch(checkpoint)
    generator = torch.Generator().manual_seed(17)
    examples = []
    expected = []
    scale = saved['model_config']['lora_alpha'] / saved['model_config']['lora_rank']
    for index, enabled in [(0, True), (2, True), (2, False)]:
        a, b = saved['lora'][str(index) + '.A'], saved['lora'][str(index) + '.B']
        x = torch.randn(a.shape[1], generator=generator)
        examples.append({'projection': index, 'features': x.numpy().tolist(), 'enabled': enabled})
        expected.append(((b @ (a @ x)) * scale if enabled else torch.zeros(b.shape[0])).numpy())
    frame = pd.DataFrame(examples)
    local = AdapterDelta()
    local.weights = saved['lora']
    local.scale = scale
    signature = infer_signature(frame, local.predict(None, frame))
    mlflow.set_registry_uri('databricks-uc')
    mlflow.set_experiment('/Shared/research4-semantic-gate/experiments/recovery_development')
    with mlflow.start_run(run_name='oracle-recovery-adapter-development') as active:
        mlflow.log_params({'source_commit': source_commit, 'dataset_version': dataset_version,
                           'protocol': saved['protocol'], 'base_frozen': True, 'development_only': True})
        info = mlflow.pyfunc.log_model(
            name='adapter_projection_delta', python_model=AdapterDelta(),
            artifacts={'adapter': str(checkpoint)}, signature=signature,
            code_paths=[str(Path(root) / 'src')],
            pip_requirements=['torch==2.11.0+cpu', '--extra-index-url https://download.pytorch.org/whl/cpu',
                              'numpy==2.2.2', 'pandas==2.2.3', 'mlflow==3.1.0'],
            registered_model_name='workspace.research4.recovery_adapter')
        consumed = mlflow.pyfunc.load_model(info.model_uri).predict(frame)
        differences = [float(np.max(np.abs(np.asarray(actual) - target)))
                       for actual, target in zip(consumed['delta'], expected)]
        if max(differences) > 1e-6:
            raise ValueError('Registered adapter consumer differs from the saved artifact')
        return {'mlflow_run_id': active.info.run_id, 'model_uri': info.model_uri,
                'registered_model': 'workspace.research4.recovery_adapter',
                'registered_version': str(info.registered_model_version),
                'consumer': 'Projection deltas only; the frozen Qwen base and episode controller are separate.',
                'consumer_matches_saved_adapter': True, 'maximum_difference': max(differences),
                'serving_endpoint_created': False}
