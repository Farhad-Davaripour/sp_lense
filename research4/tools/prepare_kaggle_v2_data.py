"""Exact source CPU-data bundle for the human-selected Kaggle continuation."""
import argparse,json,shutil,sys,zipfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.io_utils import read,dump,sha
from tools.prepare_v2_data import prepare as prepare_data


def prepare(repo,commit,model_inputs,human_scope,kaggle_scope,output,notebook):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    result=prepare_data(repo,commit,model_inputs,human_scope,output/'prepared','kaggle',notebook=notebook,kaggle_scope=kaggle_scope)
    bundle=output/'bundle';bundle.mkdir();shutil.copytree(output/'prepared/source',bundle/'source')
    manifest={'version':'kaggle-generalized-data-bundle-v1','source_commit':commit,'notebook':notebook,
        'files':{p.relative_to(bundle).as_posix():sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}}
    dump(bundle/'KAGGLE_V2_PACKAGE_FILES.json',manifest)
    archive=output/'KAGGLE_V2_DATA_PACKAGE.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as stream:
        for path in bundle.rglob('*'):
            if path.is_file():stream.write(path,path.relative_to(bundle).as_posix())
    receipt={**result,'platform':'kaggle','account':'farhad89','notebook':notebook,
        'status':'prepared_Kaggle_CPU_data_bundle_not_submitted','bundle_sha256':sha(archive),
        'bundle_manifest_sha256':sha(bundle/'KAGGLE_V2_PACKAGE_FILES.json'),
        'profile_sha256':sha(bundle/'source/inputs/KAGGLE_REMAINDER_PROFILE.json'),
        'original_reference_core_commit':'e49a75d95739ad0ad106f4e737bc6a4b35f6b216',
        'scientific_GPU_count':0,'model_weights_downloaded':False,'model_forwards':0,
        'actual_scientific_cohorts_generated':False,'compute_submissions':0,'automatic_scientific_retries':0}
    cell=(bundle/'source/notebooks/composite_kaggle_v2_data.py').read_text()
    cell=cell.replace('SOURCE_COMMIT_PLACEHOLDER',commit).replace('PACKAGE_MANIFEST_SHA256_PLACEHOLDER',receipt['bundle_manifest_sha256'])
    notebook_json={'cells':[{'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],
        'source':cell.splitlines(keepends=True)}],'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},
        'nbformat':4,'nbformat_minor':5}
    dump(output/'research4_generalized_data.ipynb',notebook_json)
    receipt['notebook_file_sha256']=sha(output/'research4_generalized_data.ipynb')
    dump(output/'KAGGLE_PREPARATION.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ['repo','commit','model-inputs','human-scope','kaggle-scope','output','notebook']:parser.add_argument('--'+key,required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))
