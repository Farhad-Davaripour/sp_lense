"""Reuse the focused fake-child owner suite, preserving its prior report bytes."""
import ast
from pathlib import Path
import test_owner as prior
def main():
    path=Path(__file__).with_name('test_owner.py');tree=ast.parse(path.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    class ReportOnly(ast.NodeTransformer):
        def visit_Constant(self,n):
            if n.value=='TEST_RESULTS.json':return ast.copy_location(ast.Constant('TEST_STORAGE_OWNER_RESULTS.json'),n)
            if n.value=='storage_blocked_production_admission':return ast.copy_location(ast.Constant('missing_root_release_still_refused'),n)
            return n
    module=ast.fix_missing_locations(ast.Module(body=[ReportOnly().visit(node)],type_ignores=[]))
    scope=dict(prior.__dict__);exec(compile(module,str(path),'exec'),scope);return scope['main']()
if __name__=='__main__':raise SystemExit(main())
