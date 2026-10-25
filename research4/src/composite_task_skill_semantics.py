"""Keep fitted native prompts, decoder and inert task semantics unchanged."""
import ast
from pathlib import Path
from .io_utils import digest

FILES=['composite_generalized_interface.py','native27_tools.py','composite_interface.py',
       'composite_design_v2_native.py','composite_record_programs.py','composite_generalized_tasks.py',
       'composite_design_v2.py','composite_generalized_world.py','composite_task_skill_validation.py',
       'composite_lora.py','composite_capacity.py','composite_kernels.py']


def _method(root,file,cls,name):
    tree=ast.parse((Path(root)/'src'/file).read_text())
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)


def _nodes(value):return digest([ast.dump(node,include_attributes=False) for node in value])


def verify_semantics(parent,current):
    parent=Path(parent);current=Path(current)
    for name in FILES:
        if (parent/'src'/name).read_text()!=(current/'src'/name).read_text():
            raise PermissionError('Fitted interface/primitive/metric semantics changed: '+name)
    for name in ['identifier','running','clock','visible','analysis_valid','outcome']:
        if _nodes([_method(parent,'composite_task_skill_world.py','World',name)])!=_nodes([
            _method(current,'composite_task_skill_world.py','World',name)]):
            raise PermissionError('Ordinary task-world semantics changed: '+name)
    for root in [parent,current]:
        tree=ast.parse((root/'src/composite_task_skill_world.py').read_text())
        declarations=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TaskScene']
        declarations+=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['VERSION','ROLES'] for t in n.targets)]
        if root==parent:baseline=_nodes(declarations)
        elif _nodes(declarations)!=baseline:raise PermissionError('Task scene/world version declarations changed')
    for name,start in [('__init__','self.scene = scene'),('step','if not self.running:')]:
        values=[]
        for root in [parent,current]:
            body=_method(root,'composite_task_skill_world.py','World',name).body
            index=next(i for i,n in enumerate(body) if ast.unparse(n).startswith(start))
            values.append(_nodes(body[index:]))
        if values[0]!=values[1]:raise PermissionError('Task initialization/action/tick semantics changed')
    values=[]
    for root in [parent,current]:
        body=_method(root,'composite_actor.py','Actor','generate').body
        values.append(_nodes([n for n in body if isinstance(n,ast.With)]))
    if values[0]!=values[1]:raise PermissionError('Fitted native generation/decoding/cache implementation changed')
    return {'fitted_native_and_task_semantic_core_unchanged':True,'shared_files_verified':len(FILES),
            'permission_context_additions_only_outside_semantic_core':True,'new_model_forwards':0}
