"""Exact checked diagnostic component, including its own frozen source binding."""
from support import SOURCES
COMMIT="84bfd749876c44dc9bb1bc1e75db89f3e491c159"
PREFIX="diagnostics/fresh_confirmation_loader_diagnostics_v1/"
SOURCES.read(COMMIT,PREFIX+"SOURCE_FREEZE.json")
_component=SOURCES.load("setup_checked_loader_diagnostics",COMMIT,PREFIX+"loader_diagnostics.py")
Context=_component.Context
new_context=_component.new_context
encoded=_component.encoded
sha=_component.sha
DiagnosticStopped=_component.DiagnosticStopped
PREDICATES=_component.PREDICATES
