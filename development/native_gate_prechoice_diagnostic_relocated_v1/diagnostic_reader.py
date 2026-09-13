"""Relocated reader/selector adapter for the fixed sixteen PRECHOICE diagnostic views.

Model-free authentication boundary only. It reuses the frozen selector contract
(``diagnostic_capture/index_contract.py``) and the frozen fit verification
(``diagnostic_gate.py``); it adds no new mathematics and reads no activation rows.

The adapter answers one question before any float row is accepted:

* every one of the sixteen fixed views carries the certified pre-option selector
  metadata bound to the frozen boundary certificate, and
* its declared symbols are bound to the certified option token ids, and
* it is not a legacy ``final_input`` readout masquerading as a pre-option readout.
"""
import hashlib, json,sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _repository_root():
    """Locate the relocated repository root from this file's own path, cwd-independent."""
    for candidate in HERE.parents:
        if (candidate / ".git").exists() and (candidate / "development/native_gate_prechoice_readout_v1").is_dir():
            return candidate
    raise RuntimeError("RELOCATED_REPOSITORY_ROOT_NOT_FOUND")


ROOT = _repository_root()
READOUT = ROOT / "development/native_gate_prechoice_readout_v1"
DIAGNOSTIC_CAPTURE = READOUT / "diagnostic_capture"
FIT = READOUT / "fit"
CERTIFICATE = READOUT / "BOUNDARY_CERTIFICATE.json"
CERTIFICATE_SHA256 = "57e700c03c46243b2cee9f1cd9dd6dd4fd0a556fec472a325af5e390a848571f"
DIAGNOSTIC_GATE_SHA256 = "93b51449aae1215173bc7d46510e20e29b9b478393ff282b52591ead01398f5d"
FREEZE_SHA256 = "433f7c1aea7b4016bf2352c894d3061f6ba4e13ea6bdf48a691c1a41dd477a7f"
# Fixed historical records from the already independently accepted checkpoint. They are
# asserted against the freeze-pinned document; no pin is ever regenerated from current bytes.
ACCEPTED_CHECKPOINT_SHA256 = "57726ab7c5564ac690f4c77c4e73f650672a925569a5e98baaae534ed3e5838c"
INDEPENDENT_REVIEW_SHA256 = "fbc2c71411e02a9b6ef30d8dae9a3651dc261b7ea404460c09941495db17e973"
CHECKPOINT_REUSE_SCHEMA = "prechoice_relocated_checkpoint_reuse.v1"
# Historical source-lock pin, resolved from diagnostic_capture/SOURCE_FREEZE.json
# (source_sha256["index_contract.py"]); the frozen lock is the authority, never a
# recomputed current hash. It is asserted against the on-disk bytes before import.
INDEX_CONTRACT_SHA256 = "61bb517b142d72a313cb3efae29bbb00d0490aa1d9f4e19bed84c95efc287db9"
POSITION = "last_shared_preoption_input"
LEGACY_POSITION = "final_input"
LABEL_ID_SYMBOLS = {50057: "KEEP", 48964: "STOP", 32: "A", 33: "B"}
SCHEMA_ROWS = "prechoice_diagnostic_scoring_rows.v1"
SCHEMA_BUNDLE = "prechoice_relocated_diagnostic_bundle.v1"
CASES = 8
VIEWS = 16
SYMBOL_PAIRS = {
    "KEEP_then_STOP": ("KEEP", "STOP"),
    "STOP_then_KEEP": ("STOP", "KEEP"),
    "A_then_B": ("A", "B"),
    "B_then_A": ("B", "A"),
}
# Accepted prepared-input reuse boundary. The old approved preparation release and its
# nineteen blobs are already frozen and independently accepted; this reader re-verifies
# the exact pinned bytes and re-observes the canonical pre-observation digest instead of
# re-executing the preparation/train authorizers. The data-lock, release and certificate
# pins are fixed historical records, never a caller-supplied value or a re-pin from
# current bytes.
DATA_LOCK = DIAGNOSTIC_CAPTURE / "DATA_LOCK.json"
DATA_LOCK_SHA256 = "47d47c09df7fa127b7588b02614a977b4a70bb795a6cc21e7c6a2e6c6372c7cc"
ACCEPTED_PREPARATION_RELEASE_SHA256 = "c707f335c4b1025e00896c3569c59cbad42cfe38dc0c2db2131ebaa749d72c19"
PREPARED_INPUTS_SCHEMA = "prechoice_diagnostic_inputs.v1"
PREPARED_INPUTS_ROLE = "DIAGNOSTIC_PRECHOICE"
PREPARED_INPUTS_SHA256 = "8a899b396d504cc0ccbac2e373cc0e933329803478f6320fc97cbc305c0a7b0f"
PREPARED_REUSE_SCHEMA = "prechoice_prepared_input_reuse.v1"
PREPARATION_VIEWS = 16
PREPARATION_FILES = 19
COMMON_PREPARATION_FILES = ("inputs.json", "RESULT.json", "operations.jsonl")
PREPARED_VIEW_KEYS = tuple(
    f"{family}_{category}__{order}"
    for family in ("G10", "G11")
    for category in ("self_shutdown", "other_shutdown", "non_termination_control")
    for order in ("KEEP_then_STOP", "STOP_then_KEEP")
) + tuple(f"{key}__{order}" for key in ("O09", "O10") for order in ("A_then_B", "B_then_A"))


def require(ok, code):
    if not ok:
        raise ValueError(code)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def read_bytes(path, code, cap=8 * 1024 ** 2):
    path = Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size <= cap, code)
    return path.read_bytes()


def _contract_module():
    """Load the frozen selector contract additively, restoring sys.modules and sys.path."""
    import importlib
    names = ("index_contract",)
    saved = {name: sys.modules.get(name) for name in names}
    path = list(sys.path)
    try:
        for name in names:
            sys.modules.pop(name, None)
        raw = read_bytes(DIAGNOSTIC_CAPTURE / "index_contract.py", "INDEX_CONTRACT_FILE")
        require(sha(raw) == INDEX_CONTRACT_SHA256, "INDEX_CONTRACT_HASH")
        sys.path.insert(0, str(DIAGNOSTIC_CAPTURE))
        return importlib.import_module("index_contract")
    finally:
        for name in names:
            sys.modules.pop(name, None)
            if saved[name] is not None:
                sys.modules[name] = saved[name]
        sys.path[:] = path


def case_keys():
    """The eight fixed case keys, read from the certified diagnostic pairs."""
    pairs = certificate_pairs()
    keys = [pair["view_keys"][0].split("__", 1)[0] for pair in pairs]
    require(len(keys) == CASES and len(set(keys)) == CASES, "EXACT_EIGHT_CERTIFIED_CASES")
    return tuple(keys)


def certificate_pairs():
    """Certified metadata-only view of the eight exposed diagnostic pairs (no rows)."""
    raw = read_bytes(CERTIFICATE, "BOUNDARY_CERTIFICATE")
    require(sha(raw) == CERTIFICATE_SHA256, "BOUNDARY_CERTIFICATE_HASH")
    document = json.loads(raw)
    return [pair for pair in document["pairs"] if pair["role"] == "EXPOSED_DIAGNOSTIC"]


def verified_frozen():
    """Reuse the frozen fit verification; the artifact itself is never re-read here."""
    import importlib.util
    raw = read_bytes(READOUT / "diagnostic_gate.py", "DIAGNOSTIC_GATE_FILE")
    require(sha(raw) == DIAGNOSTIC_GATE_SHA256, "DIAGNOSTIC_GATE_HASH")
    require(FIT.is_dir(), "FROZEN_FIT_DIRECTORY")
    spec = importlib.util.spec_from_file_location("relocated_diagnostic_gate", READOUT / "diagnostic_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_frozen(FREEZE_SHA256)


def _frozen_bytes(path, expected, code="FROZEN_FILE_HASH"):
    """Read one frozen file and bind it to its historical pin before any use."""
    raw = read_bytes(path, "FROZEN_FILE_BOUND", cap=5 * 1024 ** 2)
    require(sha(raw) == expected, code)
    return raw


def _frozen_document(path, expected):
    return json.loads(_frozen_bytes(path, expected))


def verify_accepted_checkpoint(fit=None):
    """Reuse the already independently accepted PRECHOICE29 checkpoint, model-free.

    This is a reuse-only gate, not a new reproduction. ``verified_frozen`` above is the
    historical full-history verifier and is preserved unchanged; candidate future
    diagnostic admission still needs separate prospective review and a new lock. The
    local checkpoint checks of the original frozen ``diagnostic_gate`` are re-expressed
    here so they can run without importing fit code or reopening training inputs: the
    pinned freeze is hash-checked *before* it is parsed, the exact four fit source hashes
    are checked before any fit import (none occurs), and release/manifest/lock bytes,
    the 3-head/29-training RESULT, retained owner terminal and finalization, the
    independent review hash plus its PASS and artifact/result/source/release joins and
    all three certificate/exact-reload flags, and the artifact bytes/result join with
    the release-derived artifact bindings are all re-checked. The retained input data
    lock is compared to the on-disk ``train_capture/DATA_LOCK.json`` hash instead of
    replaying history. Every pin is the already fixed historical record; nothing is
    re-pinned from current bytes. This does not claim equivalence to re-running every
    historical guard and grants no model, tokenizer, evaluation or diagnostic permission.
    """
    fit = FIT if fit is None else Path(fit)
    expected_freeze_sha256 = FREEZE_SHA256
    freeze_raw = read_bytes(fit / "FROZEN_PRECHOICE_ARTIFACT.json", "FROZEN_FILE_BOUND", cap=5 * 1024 ** 2)
    require(sha(freeze_raw) == expected_freeze_sha256, "FROZEN_FILE_HASH")
    freeze = json.loads(freeze_raw)
    require(freeze.get("schema") == "prechoice29_accepted_artifact.v1" and freeze.get("approved") is True
            and freeze.get("arm") == "PRECHOICE29" and freeze.get("diagnostic_inputs_used") is False
            and freeze.get("diagnostic_execution_permission") is False, "NEW_TRAINING_ARTIFACT_ONLY")
    require(freeze.get("artifact_sha256") == ACCEPTED_CHECKPOINT_SHA256, "ACCEPTED_CHECKPOINT_PIN")
    require(freeze.get("independent_review_sha256") == INDEPENDENT_REVIEW_SHA256, "INDEPENDENT_REVIEW_PIN")
    sources = _frozen_document(fit / "CORE_SOURCE_LOCK.json", freeze["source_sha256"])
    require(set(sources) == {"gate.py", "checker.py", "source_auth.py", "construction.py"}, "EXACT_FIT_SOURCE_SET")
    for name, expected in sources.items():
        _frozen_bytes(fit / name, expected, "EXACT_FIT_SOURCE_HASH")
    release_raw = _frozen_bytes(fit / "RELEASE.json", freeze["release_sha256"])
    release = json.loads(release_raw)
    require(release.get("approved") is True and release.get("operation") == "one_construction_fit"
            and release.get("source_sha256") == freeze["source_sha256"]
            and all(release.get(k) is False for k in ("model_permission", "tokenizer_permission", "evaluation_permission")),
            "EXACT_TRAINING_ONLY_RELEASE")
    _frozen_bytes(fit / "TRAINING_MANIFEST.json", release["training_manifest_sha256"])
    _frozen_bytes(fit / "CONSTRUCTION_LOCK_DRAFT.json", release["construction_lock_sha256"])
    result = _frozen_document(fit / "construction_attempt_001/RESULT.json", freeze["result_sha256"])
    require(result.get("scientific_pass") is True and result.get("status") == "PRECHOICE29_TRAINING_PASS"
            and result.get("fits_attempted") == 3, "CERTIFIED_THREE_HEAD_TRAINING")
    stage = result.get("stages")
    require(type(stage) is list and len(stage) == 1 and type(stage[0]) is dict and stage[0].get("id") == "PRECHOICE29"
            and stage[0]["training_count"] == stage[0]["training_correct"] == 29
            and len(stage[0]["heads"]) == 3 and all(h["status"] == "PASS" for h in stage[0]["heads"]),
            "ALL29_TRAINING_CORRECT")
    terminal = _frozen_document(fit / "fit_owner_attempt_001/TERMINAL.json", freeze["owner_terminal_sha256"])
    final = _frozen_document(fit / "fit_owner_attempt_001/FINALIZATION.json", freeze["owner_finalization_sha256"])
    require(terminal.get("process_technical_complete") is True and terminal.get("exit_code") == 0
            and final.get("terminal_sha256") == freeze["owner_terminal_sha256"]
            and not final["deadline_fault"] and not final["storage_fault"], "CLOSED_RETAINED_FIT_OWNER")
    review = _frozen_document(fit / "INDEPENDENT_ACTUAL_FIT_REVIEW.json", freeze["independent_review_sha256"])
    require(review.get("schema") == "prechoice_fit_independent_review.v1" and review.get("status") == "PASS"
            and all(review.get(k) == freeze.get(k)
                    for k in ("artifact_sha256", "result_sha256", "source_sha256", "release_sha256"))
            and review.get("all_three_certificates_verified") is True and review.get("all29_training_correct") is True
            and review.get("exact_artifact_reload") is True, "INDEPENDENT_CERTIFICATION_REQUIRED")
    _frozen_bytes(fit / "construction_attempt_001/PRECHOICE29_GATE.json", freeze["artifact_sha256"], "FROZEN_ARTIFACT_HASH")
    require(result.get("artifacts") == {"PRECHOICE29": freeze["artifact_sha256"]}, "RESULT_ARTIFACT_JOIN")
    retained = sha(read_bytes(fit.parent / "train_capture/DATA_LOCK.json", "RETAINED_INPUT_DATA_LOCK_FILE"))
    frozen_bindings = freeze.get("artifact_bindings")
    require(type(frozen_bindings) is dict and retained == frozen_bindings.get("retained_input_data_lock_sha256"),
            "RETAINED_INPUT_DATA_LOCK")
    bindings = {
        "input_contract_sha256": release["training_manifest_sha256"],
        "construction_lock_sha256": release["construction_lock_sha256"],
        "source_sha256": release["source_sha256"],
        "retained_input_data_lock_sha256": retained,
        "development_only": True,
        "arm": "PRECHOICE29",
    }
    require(freeze.get("artifact_bindings") == bindings, "EXACT_RELEASE_DERIVED_ARTIFACT_BINDINGS")
    return {
        "schema": CHECKPOINT_REUSE_SCHEMA,
        "status": "REUSED_ALREADY_ACCEPTED_CHECKPOINT",
        "accepted_checkpoint_sha256": freeze["artifact_sha256"],
        "artifact_bindings": bindings,
        "freeze_sha256": expected_freeze_sha256,
        "independent_review_sha256": freeze["independent_review_sha256"],
        "result_sha256": freeze["result_sha256"],
        "release_sha256": freeze["release_sha256"],
        "retained_input_data_lock_sha256": retained,
        "historical_training_replayed": False,
        "independent_reproduction": False,
        "reuse_of_accepted_checkpoint": True,
        "model_permission": False,
        "tokenizer_permission": False,
        "evaluation_permission": False,
        "diagnostic_execution_permission": False,
        "scope": ("Reuse of the already independently accepted and pinned checkpoint only; not a re-run of the "
                  "historical training guards, not a new independent reproduction, and not diagnostic or model-run permission."),
    }


def _witness(case_key):
    for pair in certificate_pairs():
        if pair["view_keys"][0].split("__", 1)[0] == case_key:
            return pair
    raise ValueError("UNCERTIFIED_CASE_KEY")


def _symbol_names(case_key, selector):
    """Bind the declared symbols to this exact view's certified option token ids.

    The certified pair carries two views whose divergent option labels are the two
    certified label ids in canonical pair order. Each selector must bind to its own
    view slot: ``first_option_label_id`` must equal the witness's divergent label id
    for that view, whose mapped symbol is ``declared[0]``, and the paired view's
    divergent label id must map to ``declared[1]``. This rejects opposite-view label
    substitution rather than accepting loose membership in the divergent pair.
    """
    witness = _witness(case_key)
    view_keys = witness["view_keys"]
    require(type(view_keys) is list and len(view_keys) == 2, "EXACT_TWO_CERTIFIED_VIEWS")
    ids = witness["first_divergent_label_ids"]
    require(type(ids) is list and len(ids) == 2, "EXACT_TWO_DIVERGENT_LABELS")
    declared = _declared_symbols(selector["case_key"])
    view = view_keys.index(selector["case_key"])
    require(selector["first_option_label_id"] == ids[view], "EXACT_CERTIFIED_SYMBOL_BINDING")
    require(LABEL_ID_SYMBOLS.get(ids[view]) == declared[0], "EXACT_CERTIFIED_SYMBOL_BINDING")
    require(LABEL_ID_SYMBOLS.get(ids[1 - view]) == declared[1], "EXACT_CERTIFIED_SYMBOL_BINDING")
    return declared


def bind_selector(case_key, selector):
    """Accept one certified pre-option selector document for one fixed case key."""
    anchor = _contract_module()
    require(tuple(selector["feature_contract"].items()) == tuple(anchor.CONTRACT.items()), "EXACT_PRECHOICE_CONTRACT")
    require(selector["feature_contract"]["position"] == POSITION, "NOT_PRECHOICE_POSITION")
    require(selector["feature_contract"]["position"] != LEGACY_POSITION, "LEGACY_FINAL_INPUT_READOUT")
    require(selector["feature_contract"]["final_logit_position"] == LEGACY_POSITION, "EXACT_FINAL_LOGIT_POSITION")
    require(selector["certificate_sha256"] == CERTIFICATE_SHA256, "CERTIFICATE_HASH")
    require(selector["case_key"].split("__", 1)[0] == case_key, "SELECTOR_CASE_KEY")
    witness = _witness(case_key)
    require(selector["case_key"] in witness["view_keys"], "SELECTOR_VIEW_KEY_CERTIFIED")
    require(selector["witness_sha256"] == sha(canonical(witness)), "WITNESS_HASH")
    require(len(selector["input_ids_sha256"]) == 64 and len(selector["shared_prefix_ids_sha256"]) == 64,
            "EXACT_SELECTOR_INPUT_JOIN")
    require(selector["readout_index"] == witness["readout_index"], "CERTIFIED_READOUT_INDEX")
    require(selector["readout_index"] < selector["final_input_index"], "PREOPTION_INDEX_BOUND")
    require(selector["final_input_index"] == witness["final_input_indices"][witness["view_keys"].index(selector["case_key"])],
            "CERTIFIED_FINAL_INPUT_INDEX")
    require(selector["shared_prefix_ids_sha256"] == witness["shared_prefix_ids_sha256"], "EXACT_SHARED_PREFIX_IDS")
    declared = _symbol_names(case_key, selector)
    return {"schema": "prechoice_relocated_selector.v1", "case_key": selector["case_key"],
            "symbols": list(declared), "first_option_label_id": selector["first_option_label_id"],
            "readout_index": selector["readout_index"], "final_input_index": selector["final_input_index"],
            "certificate_sha256": CERTIFICATE_SHA256, "witness_sha256": selector["witness_sha256"]}


def validate_bundle(bundle):
    """Accept the fixed sixteen-view record set only when every view carries a certified selector."""
    document = bundle if isinstance(bundle, dict) else json.loads(bundle)
    require(document.get("schema") == SCHEMA_BUNDLE, "BUNDLE_SCHEMA")
    require(document.get("role") == "EXPOSED_DIAGNOSTIC", "BUNDLE_ROLE")
    require(document.get("fresh_confirmation") is False, "BUNDLE_FRESH_CONFIRMATION")
    require(document.get("certificate_sha256") == CERTIFICATE_SHA256, "BOUNDARY_CERTIFICATE_HASH")
    views = document.get("views")
    require(type(views) is list and len(views) == VIEWS, "EXACT_SIXTEEN_VIEWS")
    accepted = []
    for index, view in enumerate(views):
        require(type(view) is dict and set(view) == {"case_key", "view_key", "symbols", "selector"}, "VIEW_FIELDS")
        require(view["view_key"].split("__", 1)[0] == view["case_key"], "VIEW_CASE_KEY")
        bound = bind_selector(view["case_key"], view["selector"])
        require(bound["case_key"] == view["view_key"], "VIEW_KEY_CERTIFIED")
        require(list(view["symbols"]) == bound["symbols"], "VIEW_SYMBOLS_CERTIFIED")
        accepted.append(bound)
    expected = [pair["view_keys"][slot] for pair in certificate_pairs()
                for slot in range(len(pair["view_keys"]))]
    require([view["view_key"] for view in views] == expected, "EXACT_SIXTEEN_VIEW_ORDER")
    return accepted

def _row_slot_expectation(case_key, slot):
    """Certified view key plus the row-convention and bundle-convention symbols for a slot.

    The certified witness fixes the view key for each slot. ``diagnostic_scoring``'s row
    schema declares ``symbols`` as the ordered pair of view-suffix tokens (slot 0 canonical,
    slot 1 reversed), while the bundle schema carries the split symbol names bound to the
    certified label ids. A row whose declared key or suffix-token symbols do not match its
    certified slot is rejected rather than silently relabelled.
    """
    view_keys = _witness(case_key)["view_keys"]
    require(type(view_keys) is list and len(view_keys) == 2, "EXACT_TWO_CERTIFIED_VIEWS")
    suffixes = tuple(view_key.split("__", 1)[1] for view_key in view_keys)
    view_key = view_keys[slot]
    row_symbols = suffixes if slot == 0 else tuple(reversed(suffixes))
    return view_key, row_symbols, tuple(_declared_symbols(view_key))


def bundle_from_rows(rows_document, selectors):
    """Additive bridge from the already validated scoring rows to the sixteen-view bundle.

    Row values are carried through untouched; only the case/view/symbol/selector metadata
    is authenticated here. No activation or outcome value is read. The rows must already
    be the fixed eight cases/two views in certified order with role ``EXPOSED_DIAGNOSTIC``
    and ``fresh_confirmation`` false; a row whose case/view/symbol is outside its certified
    slot is rejected rather than silently relabelled (the row ``symbols`` field keeps the
    scoring schema's ordered view-suffix-token convention). Each selector must be the exact
    certified selector for its own row view (``bind_selector`` runs before the view is
    emitted), and the assembled bundle is validated before it is returned.
    """
    document = rows_document if isinstance(rows_document, dict) else json.loads(rows_document)
    require(document.get("schema") == SCHEMA_ROWS, "ROWS_SCHEMA")
    require(set(document) == {"schema", "role", "fresh_confirmation", "cases"}, "ROWS_EXACT_FIELDS")
    require(document.get("role") == "EXPOSED_DIAGNOSTIC", "ROWS_ROLE")
    require(document.get("fresh_confirmation") is False, "ROWS_FRESH_CONFIRMATION")
    cases = document.get("cases")
    require(type(cases) is list and len(cases) == CASES, "EXACT_EIGHT_CASES")
    keys = case_keys()
    require(type(selectors) is dict and set(selectors) == set(keys), "EXACT_SELECTOR_CASE_SET")
    views = []
    for index, case in enumerate(cases):
        require(type(case) is dict and set(case) == {"case_key", "views"}, "ROWS_CASE_FIELDS")
        key = case["case_key"]
        require(key == keys[index], "FIXED_CASE_ORDER")
        case_views = case["views"]
        require(type(case_views) is list and len(case_views) == 2, "EXACT_TWO_VIEWS")
        slot_selectors = selectors[key]
        require(type(slot_selectors) in (list, tuple) and len(slot_selectors) == 2, "EXACT_TWO_SELECTORS")
        for slot, view in enumerate(case_views):
            expected_key, row_symbols, bundle_symbols = _row_slot_expectation(key, slot)
            require(type(view) is dict and set(view) == {"view_key", "symbols", "row"}, "ROWS_VIEW_FIELDS")
            require(view["view_key"] == expected_key, "ROW_VIEW_KEY_POSITION")
            require(tuple(view["symbols"]) == row_symbols, "ROW_VIEW_SYMBOLS_POSITION")
            bound = bind_selector(key, slot_selectors[slot])
            require(bound["case_key"] == expected_key, "SELECTOR_VIEW_KEY_MATCH")
            views.append({"case_key": key, "view_key": expected_key,
                          "symbols": list(bundle_symbols), "selector": slot_selectors[slot]})
    bundle = {"schema": SCHEMA_BUNDLE, "role": "EXPOSED_DIAGNOSTIC", "fresh_confirmation": False,
              "certificate_sha256": CERTIFICATE_SHA256, "views": views}
    validate_bundle(bundle)
    return bundle


def _declared_symbols(view_key):
    """The canonical declared symbol pair for one certified view key suffix."""
    suffix = view_key.split("__", 1)[1]
    require(suffix in SYMBOL_PAIRS, "EXACT_SYMBOL_PAIR")
    return SYMBOL_PAIRS[suffix]


def reuse_accepted_prepared_inputs():
    """Reuse the already accepted prepared diagnostic inputs, model-free and history-free.

    This is a reuse-only gate, not a new production or provenance reproduction. The fixed
    sixteen pre-option views were already prepared and independently accepted under the
    frozen transfer release; the underlying accepted raw bytes stay pinned and no
    preparation or training authorizer is re-executed. The old approved release is
    hash-checked *before* it is parsed, its exact nineteen preparation blobs (three common
    plus the sixteen expected views) are bounded in-root and hash-checked before any JSON
    parse, the text lock and preparation closure are joined by hash plus PASS flags, and the
    inputs/result joins are re-checked. Only then are the accepted inputs parsed, each
    existing case is bound to the certified pre-option selector through the unchanged pinned
    ``index_contract.bind``, and the fixed sixteen-key order is re-observed. A digest
    mismatch is fatal: no pin is regenerated and no label is forced to PASS. The returned
    record carries every execution permission false and grants no model, tokenizer,
    provider, activation, outcome, fit or scoring authority. This function takes no
    caller-supplied pin, so no arbitrary caller value can stand in for prior acceptance.
    """
    lock_raw = read_bytes(DATA_LOCK, "DATA_LOCK_FILE", cap=5 * 1024 ** 2)
    require(sha(lock_raw) == DATA_LOCK_SHA256, "ACCEPTED_DATA_LOCK_HASH")
    lock = json.loads(lock_raw)
    require(lock.get("schema") == "prechoice_retained_diagnostic_inputs.v1"
            and lock.get("role") == PREPARED_INPUTS_ROLE
            and lock.get("training_chain_indexes") == [0, 1, 2, 3]
            and lock.get("diagnostic_chain_indexes") == [4], "EXACT_DATA_ROLES")
    require(lock.get("certificate_sha256") == CERTIFICATE_SHA256, "ACCEPTED_CERTIFICATE_PIN")
    chains = lock.get("chains")
    require(type(chains) is list, "PREPARATION_CHAIN_LIST")
    diagnostic_pins = [chains[index] for index in lock["diagnostic_chain_indexes"]]
    require(len(diagnostic_pins) == 1, "EXACT_ONE_PREPARATION_CHAIN")
    pin = diagnostic_pins[0]
    require(pin.get("release_sha256") == ACCEPTED_PREPARATION_RELEASE_SHA256, "ACCEPTED_PREPARATION_RELEASE_PIN")

    base = ROOT / pin["namespace"] / "root_release"
    require(base.resolve().is_relative_to(ROOT.resolve()), "PREPARED_INPUT_ROOT")
    release_raw = read_bytes(base / "RELEASE.json", "PREPARATION_RELEASE_FILE", cap=5 * 1024 ** 2)
    require(sha(release_raw) == pin["release_sha256"], "ACCEPTED_PREPARATION_RELEASE_HASH")
    release = json.loads(release_raw)
    require(release.get("schema") == "frozen_transfer_capture_release.v1" and release.get("approved") is True
            and release.get("source_freeze_sha256") == pin.get("source_freeze_sha256"), "OLD_PREPARATION_ACCEPTED")

    preparation = release.get("preparation_files")
    require(type(preparation) is dict
            and set(preparation) == {key + ".json" for key in PREPARED_VIEW_KEYS} | set(COMMON_PREPARATION_FILES)
            and len(preparation) == PREPARATION_FILES, "EXACT_PREPARATION_FILE_SET")
    prep_root = (base / "preparation").resolve()
    require(prep_root.is_relative_to(base.resolve()), "PREPARED_INPUT_ROOT")
    blobs = {}
    for name in sorted(preparation):
        require(type(name) is str and name not in ("", ".", "..") and "/" not in name and "\\" not in name
                and not Path(name).is_absolute(), "PREPARED_INPUT_NAME")
        path = (prep_root / name).resolve()
        require(path.is_relative_to(prep_root) and path != prep_root, "PREPARED_INPUT_PATH")
        blobs[name] = read_bytes(path, "PREPARATION_FILE_HASH", cap=5 * 1024 ** 2)
        require(sha(blobs[name]) == preparation[name], "PREPARATION_FILE_HASH")

    text_raw = read_bytes(base / "TEXT_LOCK.json", "PREPARATION_TEXT_LOCK_FILE", cap=5 * 1024 ** 2)
    require(sha(text_raw) == release.get("text_lock_sha256"), "PREPARATION_TEXT_LOCK_HASH")
    text_lock = json.loads(text_raw)
    require(text_lock.get("schema") == "native_final_text_lock.v1"
            and text_lock.get("final_text_locked") is True
            and text_lock.get("blind_semantic_review_approved") is True, "FROZEN_TEXT_LOCK_INTERFACE")
    require(release.get("admitted_submission_sha256") == text_lock.get("admitted_submission_sha256"),
            "RELEASE_ADMITTED_SUBMISSION_JOIN")

    closure_raw = read_bytes(base / "PREPARATION_CLOSURE.json", "PREPARATION_CLOSURE_FILE", cap=5 * 1024 ** 2)
    require(sha(closure_raw) == release.get("preparation_closure_sha256"), "PREPARATION_CLOSURE_HASH")
    closure = json.loads(closure_raw)
    require(closure.get("schema") == "root_preparation_closure.v1" and closure.get("status") == "PASS"
            and closure.get("preparation_status") == "PASS" and closure.get("exit_code") == 0
            and closure.get("timed_out") is False and closure.get("one_shot") is True
            and closure.get("preparation_result_sha256") == preparation["RESULT.json"]
            and closure.get("text_lock_sha256") == release.get("text_lock_sha256"), "ROOT_PREPARATION_CLOSED")

    require(preparation["inputs.json"] == release.get("inputs_sha256"), "RELEASE_INPUT_HASH_JOIN")
    result_doc = json.loads(blobs["RESULT.json"])
    require(result_doc.get("status") == "PASS" and result_doc.get("complete_input_publication") is True
            and result_doc.get("completed_cases") == PREPARATION_VIEWS
            and result_doc.get("inputs_sha256") == preparation["inputs.json"]
            and result_doc.get("failed_operations") == 0 and result_doc.get("unrun_operations") == 0
            and all(result_doc.get(key) == 0 for key in
                    ("real_model_loads", "real_model_forwards", "real_model_derivatives")),
            "COMPLETE_BOUNDED_PREPARATION")
    accepted = json.loads(blobs["inputs.json"])
    require(accepted.get("schema_version") == "native_final_prepared_inputs_v1"
            and accepted.get("real_model_authorized") is False
            and accepted.get("scope") == "OFFLINE_FINAL_PREPARATION"
            and accepted.get("input_token_ceiling") == 320, "PREPARED_INPUT_INTERFACE")

    certified_pairs = certificate_pairs()
    certified = tuple(view_key for pair in certified_pairs for view_key in pair["view_keys"])
    require(certified == PREPARED_VIEW_KEYS, "EXACT_SIXTEEN_CERTIFIED_VIEW_ORDER")
    witnesses = {view_key: pair for pair in certified_pairs for view_key in pair["view_keys"]}
    require(len(witnesses) == PREPARATION_VIEWS, "EXACT_SIXTEEN_CERTIFIED_VIEWS")

    anchor = _contract_module()
    cases = {}
    for case in accepted.get("cases", []):
        key = case.get("case_key")
        require(key in witnesses and key not in cases, "UNIQUE_PREPARED_CASE")
        selector = anchor.bind(case["input"]["input_ids"], witnesses[key], key, CERTIFICATE_SHA256,
                               expected_input_ids_sha256=case["input_binding"]["derived_input_int64_le_sha256"])
        cases[key] = {**case, "readout_selector": selector}
    require(set(cases) == set(PREPARED_VIEW_KEYS), "COMPLETE_SIXTEEN_PREPARED_INPUTS")

    result = {"schema": PREPARED_INPUTS_SCHEMA, "role": PREPARED_INPUTS_ROLE,
              "cases": [cases[key] for key in PREPARED_VIEW_KEYS],
              "certificate_sha256": CERTIFICATE_SHA256, "data_lock_sha256": DATA_LOCK_SHA256}
    require(sha(canonical(result) + b"\n") == PREPARED_INPUTS_SHA256, "PREPARED_INPUTS_REOBSERVED")
    return {
        "schema": PREPARED_REUSE_SCHEMA,
        "status": "REUSED_ALREADY_ACCEPTED_PREPARED_INPUTS",
        "prepared_inputs": result,
        "prepared_inputs_sha256": PREPARED_INPUTS_SHA256,
        "accepted_preparation_release_sha256": pin["release_sha256"],
        "preparation_files_verified": PREPARATION_FILES,
        "certificate_sha256": CERTIFICATE_SHA256,
        "data_lock_sha256": DATA_LOCK_SHA256,
        "historical_preparation_replayed": False,
        "independent_reproduction": False,
        "reuse_of_accepted_prepared_inputs": True,
        "model_permission": False,
        "tokenizer_permission": False,
        "provider_permission": False,
        "activation_permission": False,
        "logit_permission": False,
        "outcome_permission": False,
        "fit_permission": False,
        "scoring_permission": False,
        "diagnostic_execution_permission": False,
        "scope": ("Reuse of the already accepted and pinned prepared diagnostic inputs only; not a re-run of the "
                  "preparation or training authorizers, not a new independent provenance reproduction, and not "
                  "model, tokenizer, provider, activation, logit, outcome, fit or scoring permission."),
    }
