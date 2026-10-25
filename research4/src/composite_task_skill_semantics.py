"""Keep fitted native prompts, decoder and inert task semantics unchanged."""
import ast
import hashlib
from pathlib import Path
from .io_utils import digest

FILES=['composite_generalized_interface.py','native27_tools.py','composite_interface.py',
       'composite_design_v2_native.py','composite_record_programs.py','composite_generalized_tasks.py',
       'composite_design_v2.py','composite_task_skill_validation.py',
       'composite_lora.py','composite_capacity.py','composite_kernels.py',
       'composite_world.py','composite_controller.py','composite_memory.py',
       'composite_detector.py','composite_roles.py','composite_generalized_teacher.py',
       'composite_generalized_data.py','composite_v2_controller.py']


def _method(root,file,cls,name):
    tree=ast.parse((Path(root)/'src'/file).read_text(encoding='utf-8-sig'))
    node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    return next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name==name)


def _stable_ast(value):
    # Python3.12 adds empty type_params fields to ordinary functions/classes.
    # Empty optional typing syntax must not change a scientific runtime freeze.
    if isinstance(value,ast.AST):
        return {'node':type(value).__name__,'fields':{k:_stable_ast(v) for k,v in ast.iter_fields(value)
                                                   if not (k=='type_params' and not v)}}
    if isinstance(value,list):return [_stable_ast(v) for v in value]
    return value


def _nodes(value):return digest([_stable_ast(node) for node in value])


def _risk_world(root):
    tree=ast.parse((Path(root)/'src/composite_generalized_world.py').read_text(encoding='utf-8-sig'))
    class RuntimeArgument(ast.NodeTransformer):
        def visit_FunctionDef(self,node):
            if node.name=='step':
                guard=ast.parse('from .composite_frozen_risk_runtime import recorded_action_guard\nrecorded_action_guard(self.scene,action)\n').body
                if len(node.body)>=2 and _nodes(node.body[:2])==_nodes(guard):node.body=node.body[2:]
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Name) and node.func.id=='require_v2_runtime':
                if node.args:raise PermissionError('Unregistered world runtime arguments')
                if node.keywords:
                    if (len(node.keywords)!=1 or node.keywords[0].arg!='scene'
                            or not isinstance(node.keywords[0].value,ast.Name) or node.keywords[0].value.id!='scene'):
                        raise PermissionError('Only the exact scene may be added to the runtime guard')
                    node.keywords=[]
            return self.generic_visit(node)
    return digest(_stable_ast(RuntimeArgument().visit(tree)))


def semantic_fingerprint(root):
    root=Path(root);value={'shared_files':{n:hashlib.sha256((root/'src'/n).read_text(encoding='utf-8-sig').encode()).hexdigest() for n in FILES},
                         'risk_world_normalized_AST':_risk_world(root)}
    value['task_methods']={n:_nodes([_method(root,'composite_task_skill_world.py','World',n)])
      for n in ['identifier','running','clock','visible','analysis_valid','outcome']}
    tree=ast.parse((root/'src/composite_task_skill_world.py').read_text(encoding='utf-8-sig'))
    declarations=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TaskScene']
    declarations+=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['VERSION','ROLES'] for t in n.targets)]
    value['task_declarations']=_nodes(declarations)
    for name,start in [('__init__','self.scene = scene'),('step','if not self.running:')]:
        body=_method(root,'composite_task_skill_world.py','World',name).body
        index=next(i for i,n in enumerate(body) if ast.unparse(n).startswith(start))
        value['task_'+name]=_nodes(body[index:])
    body=_method(root,'composite_actor.py','Actor','generate').body
    value['native_generation']=_nodes([n for n in body if isinstance(n,ast.With)])
    return digest(value)


def verify_semantics(parent,current):
    parent=Path(parent);current=Path(current)
    for name in FILES:
        if (parent/'src'/name).read_text(encoding='utf-8-sig')!=(current/'src'/name).read_text(encoding='utf-8-sig'):
            raise PermissionError('Fitted interface/primitive/metric semantics changed: '+name)
    for name in ['identifier','running','clock','visible','analysis_valid','outcome']:
        if _nodes([_method(parent,'composite_task_skill_world.py','World',name)])!=_nodes([
            _method(current,'composite_task_skill_world.py','World',name)]):
            raise PermissionError('Ordinary task-world semantics changed: '+name)
    for root in [parent,current]:
        tree=ast.parse((root/'src/composite_task_skill_world.py').read_text(encoding='utf-8-sig'))
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
    if _risk_world(parent)!=_risk_world(current):raise PermissionError('Original nine-condition risk-world semantics changed')
    return {'fitted_native_and_task_semantic_core_unchanged':True,'shared_files_verified':len(FILES),
            'original_nine_condition_risk_world_core_unchanged':True,'semantic_core_sha256':semantic_fingerprint(current),
            'permission_context_additions_only_outside_semantic_core':True,'new_model_forwards':0}
