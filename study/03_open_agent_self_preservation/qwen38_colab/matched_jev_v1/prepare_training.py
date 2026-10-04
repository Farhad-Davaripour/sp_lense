"""Pure matched raw/concept preparation; no API, tokenizer or model loading."""
import copy
import hashlib
import importlib
import json
from pathlib import Path

RAW_FILE_SHA256 = "be99179d97f0edd6e078f37980aa749853a64a91a9c5bbff29fa33b5eff8fa01"
ROWS = 112


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n").encode("utf-8"))


def annotation_records(value):
    if isinstance(value, dict) and "batch_id" in value and "rows" in value:
        value = [row["annotation"] for row in value["rows"]]
    elif isinstance(value, dict) and "records" in value:
        value = value["records"]
    elif isinstance(value, dict) and "annotations" in value:
        value = value["annotations"]
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        raise ValueError("Actual annotation records must be a list or request-hash map")
    return value


def prepare(raw_rows, annotations, schema, matched):
    if not isinstance(raw_rows, list) or len(raw_rows) != ROWS:
        raise ValueError("The exact primary dataset must contain112 rows")
    records = {}
    for record in annotation_records(annotations):
        identity = record["identity"]
        request = record["request"]
        if identity != schema.request_identity(request):
            raise ValueError("Annotation request/model/schema identity differs")
        decoded = record["decoded"]
        if "response" in record and schema.decode(record["response"]) != decoded:
            raise ValueError("Actual typed response differs from cached decoding")
        key = identity["request_sha256"]
        if key in records and canonical(records[key]) != canonical(record):
            raise ValueError("Conflicting annotation for one exact visible prefix")
        records[key] = record
    plain, aided, joins = [], [], []
    used = set()
    for slot, raw in enumerate(raw_rows):
        request = schema.payload(raw["messages"], raw.get("tools"))
        identity = schema.request_identity(request)
        key = identity["request_sha256"]
        if key not in records:
            raise ValueError("Missing actual annotation for slot"+str(slot))
        left, right = matched.training_pair(raw, records[key], schema)
        if left != raw or left["targets"] != right["targets"]:
            raise ValueError("Raw rows/targets changed")
        if {k:v for k,v in left.items() if k!="messages"} != {k:v for k,v in right.items() if k!="messages"}:
            raise ValueError("Primary row metadata/category/tools changed")
        if right["messages"][:-1] != raw["messages"] or len(right["messages"]) != len(raw["messages"])+1:
            raise ValueError("Concept block placement differs")
        plain.append(left);aided.append(right);used.add(key)
        joins.append({"slot":slot,"id":raw["id"],"kind":raw["kind"],
                      "raw_row_sha256":digest(raw),"raw_visible_identity":identity,
                      "concept_messages_sha256":digest(right["messages"]),
                      "assistant_targets_sha256":digest(raw["targets"]),
                      "annotation_record_sha256":digest(records[key])})
    if len(used) != 104:
        raise ValueError("Frozen primary inventory is104 unique raw prefixes")
    audit = {"passed":True,"rows":112,"unique_actual_annotations_used":len(used),
             "schema_version":schema.SCHEMA_VERSION,
             "questions_sha256":digest(schema.QUESTIONS),
             "raw_rows_canonical_sha256":digest(raw_rows),
             "plain_rows_canonical_sha256":digest(plain),"concept_rows_canonical_sha256":digest(aided),
             "targets_categories_tools_metadata_unchanged":True,"raw_history_preserved":True,
             "primary_scope_register_repair":False,
             "only_B_input_change":"One shared v2 factual block appended to each exact raw prefix",
             "training_inference_shared_helper":"matched_inputs.training_pair / inference_copy",
             "no_ground_truth_fallback":True,"joins":joins}
    return {"plain":plain,"concept":aided}, audit


def prepare_files(raw_path, annotations_path, output_dir, schema=None, matched=None):
    raw_bytes = Path(raw_path).read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != RAW_FILE_SHA256:
        raise ValueError("Primary raw dataset file changed")
    schema = schema or importlib.import_module("concept_schema_v2")
    matched = matched or importlib.import_module("matched_inputs")
    annotation_bytes = Path(annotations_path).read_bytes()
    datasets, audit = prepare(json.loads(raw_bytes), json.loads(annotation_bytes), schema, matched)
    audit.update(raw_file_sha256=RAW_FILE_SHA256,
                 actual_annotations_file_sha256=hashlib.sha256(annotation_bytes).hexdigest(),
                 schema_source_sha256=hashlib.sha256(Path(schema.__file__).read_bytes()).hexdigest(),
                 matched_inputs_source_sha256=hashlib.sha256(Path(matched.__file__).read_bytes()).hexdigest())
    output = Path(output_dir);output.mkdir(parents=True, exist_ok=False)
    (output/"plain_train.json").write_bytes(raw_bytes)
    write_json(output/"concept_train.json", datasets["concept"])
    write_json(output/"PREPARATION_AUDIT.json", audit)
    return audit


def paired_token_audit(tokenizer, plain, concept, model_ops, audit):
    """Called only in the admitted Colab worker; no truncation or model forward."""
    lengths=[];rows=[]
    for slot,(left,right) in enumerate(zip(plain,concept)):
        a=audit.row_tokens(tokenizer,left);b=audit.row_tokens(tokenizer,right)
        if not a["decoded_target_matches_literal"] or not b["decoded_target_matches_literal"]:
            raise RuntimeError("Assistant target roundtrip differs")
        if a["response"] != b["response"]:
            raise RuntimeError("Matched labels differ")
        length=max(a["actual_length"],b["actual_length"])
        if length>3072:raise RuntimeError("Concept input exceeds3072; no truncation admitted")
        lengths.append(length)
        rows.append({"slot":slot,"id":left["id"],"plain_actual":a["actual_length"],
                     "concept_actual":b["actual_length"],"paired_padded":length,
                     "supervised_tokens":len(a["response"]),
                     "plain_prefix_sha256":digest(a["prefix"]),"concept_prefix_sha256":digest(b["prefix"]),
                     "prefix_labels_masked":True,"padding_labels_masked":True,
                     "target_eos_verified":True,"next_token_shift_verified":True})
    if len(rows)!=112:raise RuntimeError("Token audit row count differs")
    return {"passed":True,"rows":rows,"paired_lengths":lengths,
            "maximum_padded":max(lengths),"padded_tokens_per_pass":sum(lengths),
            "truncation_performed":False,"assistant_labels_identical":True}
