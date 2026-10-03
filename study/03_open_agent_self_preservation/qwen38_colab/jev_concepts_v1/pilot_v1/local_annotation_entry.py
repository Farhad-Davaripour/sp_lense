"""Local trusted dotenv invocation; never transfers or prints the credential."""
import argparse,json,re,shlex
from pathlib import Path
from pilot_fixtures import load_preparation
from annotate_boundaries import run


def credential_provider(dotenv_path):
    # Read only the requested assignment. No interpolation, eval or environment export.
    secret=None
    with Path(dotenv_path).open(encoding="utf-8-sig") as source:
        for line in source:
            match=re.match(r"^\s*(?:export\s+)?TYPESAFE_API_KEY\s*=\s*(.*)$",line)
            if not match:continue
            lexer=shlex.shlex(match.group(1),posix=True)
            lexer.whitespace_split=True;lexer.commenters="#"
            tokens=list(lexer)
            if len(tokens)!=1:raise RuntimeError("Invalid requested credential assignment")
            secret=tokens[0]
    if not secret:raise RuntimeError("Requested existing credential is absent")
    return secret


def main(dotenv,fixtures,readiness_path,output):
    _,client=load_preparation()
    ready=json.loads(Path(readiness_path).read_text(encoding="utf-8"))
    client.validate_readiness(ready)
    # Private closure only; no notebook, cache, output, model env or other key reads.
    provider=lambda:credential_provider(dotenv)
    if not provider():raise RuntimeError("Existing credential unavailable")
    print(json.dumps({"existing_key_found":True,"key_output":False,"key_exported_to_environment":False}),flush=True)
    result=run(fixtures,output,ready,client.make_official_transport(provider))
    safe={"completed_boundary_states":len(result["states"]),"authenticated_pinned_model_access_verified":result["completed"],
          "api_requests":result["api"]["attempted_deduplicated_requests"],"input_tokens":result["api"]["input_tokens"],
          "estimated_input_usd":result["api"]["input_estimated_usd"],"qwen_model_queries":0,"credential_transferred_to_colab":False}
    print(json.dumps(safe),flush=True)
    return safe


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dotenv",required=True);parser.add_argument("--fixtures",required=True)
    parser.add_argument("--readiness",required=True);parser.add_argument("--output",required=True)
    args=parser.parse_args()
    try:main(args.dotenv,args.fixtures,args.readiness,args.output)
    except Exception as error:
        print(json.dumps({"stage":"local_annotation_stopped","error_type":type(error).__name__,"secret_value_output":False}),flush=True)
        raise SystemExit(1) from None
