"""Exact CPU token counting for Root-normalized text chat; never loads a model."""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import sys

from .research_paths import fine_tune_root

REFERENCE_ROOT = fine_tune_root() / ".local/reference-tokenizer"
DEPENDENCY_SITE = fine_tune_root() / ".local/reference-tokenizer-venv/Lib/site-packages"
PINNED_FILES = {
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "chat_template.jinja": "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
    "tokenizer_config.json": "792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b",
}
CONTEXT_TOTAL_TOKENS = 131072
OUTPUT_RESERVATION_TOKENS = 8192
MAX_PROMPT_TOKENS = CONTEXT_TOTAL_TOKENS - OUTPUT_RESERVATION_TOKENS


class ContextBudgetExceeded(ValueError):
    """The full formatted history does not fit; no truncation or retry is allowed."""


def _sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_messages(messages):
    if type(messages) is not list or not messages:
        raise ValueError("A nonempty Root-normalized text-message list is required")
    clean = []
    for index, message in enumerate(messages):
        if (type(message) is not dict or set(message) != {"role", "content"}
                or message["role"] not in {"system", "user", "assistant"}
                or type(message["content"]) is not str
                or message["role"] == "system" and index != 0):
            raise ValueError("Count only Root-normalized text messages; no hidden channels")
        clean.append(dict(message))
    if clean[-1]["role"] != "user":
        raise ValueError("The final Root-normalized message must be user")
    return clean


@lru_cache(maxsize=1)
def _resources():
    for name, expected in PINNED_FILES.items():
        if _sha(REFERENCE_ROOT / name) != expected:
            raise ValueError("Pinned CPU tokenizer/template metadata changed")
    # Use the already-installed CPU reference dependencies, without installing or
    # changing the interpreter's persistent search order.
    sys.path.insert(0, str(DEPENDENCY_SITE))
    try:
        tokenizers = importlib.import_module("tokenizers")
        jinja2 = importlib.import_module("jinja2")
        sandbox = importlib.import_module("jinja2.sandbox")
        markupsafe = importlib.import_module("markupsafe")
    finally:
        sys.path.remove(str(DEPENDENCY_SITE))
    for module, version in ((tokenizers, "0.23.3"), (jinja2, "3.1.6")):
        if (module.__version__ != version
                or not Path(module.__file__).resolve().is_relative_to(DEPENDENCY_SITE.resolve())):
            raise ValueError("Exact existing CPU reference dependency required")
    if not Path(markupsafe.__file__).resolve().is_relative_to(DEPENDENCY_SITE.resolve()):
        raise ValueError("Reference MarkupSafe dependency required")
    tokenizer = tokenizers.Tokenizer.from_file(str(REFERENCE_ROOT / "tokenizer.json"))
    if tokenizer.truncation is not None or tokenizer.padding is not None:
        raise ValueError("CPU reference tokenizer must not truncate or pad")
    environment = sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
        extensions=["jinja2.ext.loopcontrols"])
    def raise_exception(message):
        raise ValueError(str(message))
    environment.globals["raise_exception"] = raise_exception
    environment.filters["tojson"] = lambda value, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(
        value, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
    template = environment.from_string((REFERENCE_ROOT / "chat_template.jinja").read_text(encoding="utf-8"))
    provenance = {
        "reference_root": REFERENCE_ROOT.as_posix(), "files_sha256": dict(PINNED_FILES),
        "dependency_site": DEPENDENCY_SITE.as_posix(),
        "tokenizers_version": tokenizers.__version__, "jinja2_version": jinja2.__version__,
        "tokenizers_module": Path(tokenizers.__file__).as_posix(),
        "jinja2_module": Path(jinja2.__file__).as_posix(),
        "add_special_tokens": False, "add_generation_prompt": True,
        "enable_thinking": True, "reasoning_effort": "low",
        "scope": "CPU reference only; no model load, network call or live native dependency claim",
    }
    return tokenizer, template, provenance


def render_formatted_prompt(messages):
    """Render the pinned template exactly; input must already be Root-normalized."""
    clean = _text_messages(messages)
    _, template, _ = _resources()
    return template.render(messages=clean, tools=None, add_generation_prompt=True,
                           enable_thinking=True, reasoning_effort="low")


def count_formatted_prompt_tokens(messages) -> int:
    tokenizer, _, _ = _resources()
    formatted = render_formatted_prompt(messages)
    return len(tokenizer.encode(formatted, add_special_tokens=False).ids)


def validate_context_budget(messages, token_counter=None) -> dict:
    """Require the complete formatted prompt plus8192 output inside131072."""
    clean = _text_messages(messages)
    count = (count_formatted_prompt_tokens if token_counter is None else token_counter)(clean)
    if type(count) is not int or count < 0:
        raise ValueError("An exact nonnegative integer token count is required")
    if count > MAX_PROMPT_TOKENS:
        raise ContextBudgetExceeded("Full formatted prompt exceeds122880 tokens; no history truncation")
    return {"prompt_tokens": count, "output_reservation_tokens": OUTPUT_RESERVATION_TOKENS,
            "context_total_tokens": CONTEXT_TOTAL_TOKENS}


def tokenizer_provenance() -> dict:
    """Return immutable source/dependency identities used by the cached CPU counter."""
    return json.loads(json.dumps(_resources()[2]))
