"""Import Step22B unchanged; adapt only T3 lr/delta membership in memory."""
import os
import sys
sys.dont_write_bytecode = True
for _key in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ.pop(_key, None)
os.environ.update(OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE')
import io_b2 as io
sys.path.insert(0, str(io.OLD))
import worker22b_v3 as old
import ast
import copy

# Imported helper functions retain their original code; only their namespace bindings move.
old.__dict__.update(io.exports())

def bind_t3(function_name):
    tree = ast.parse((io.OLD / 'worker22b_v3.py').read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name))
    class SettingBinding(ast.NodeTransformer):
        def __init__(self):
            self.changed = []
        def visit_Compare(self, n):
            self.generic_visit(n)
            if (len(n.ops) == 1 and isinstance(n.ops[0], ast.Eq) and len(n.comparators) == 1
                and isinstance(n.comparators[0], ast.Constant) and n.comparators[0].value in ['T1', 'T2']
                and ast.unparse(n.left) in ["job['setting']", 'setting']):
                original = n.comparators[0].value
                self.changed.append(dict(line=n.lineno, original=original, expanded=[original, 'T3']))
                n.ops = [ast.In()]
                n.comparators = [ast.List(elts=[ast.Constant(original), ast.Constant('T3')], ctx=ast.Load())]
            return n
    patch = SettingBinding()
    node = patch.visit(node)
    assert len(patch.changed) == (2 if function_name == 'train' else 1), patch.changed
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(io.OLD / 'worker22b_v3.py'), 'exec'), old.__dict__)
    return dict(function=function_name, source=io.rel(io.OLD / 'worker22b_v3.py'),
                original_source_sha256=io.sha(io.OLD / 'worker22b_v3.py'), changes=patch.changed,
                bound_ast_sha256=io.hashlib.sha256(ast.dump(module, include_attributes=False).encode()).hexdigest())

BINDINGS = [bind_t3('train'), bind_t3('evaluate')]

if __name__ == '__main__':
    action, encoded, destination = sys.argv[1:]
    job = io.json.loads(encoded)
    assert job['setting'] == 'T3' and not job['gate'] and job['model'] in ['static', 'adaptive', 'lstm']
    folder = io.target(io.Path(destination))
    folder.mkdir(parents=True, exist_ok=True)
    io.js(folder / 'T3_BINDINGS.json', dict(bindings=BINDINGS, protocol_sha256=io.check_protocol(), imported_original=True))
    io.js(folder / 'POWER_START.json', dict(utc=io.utc(), snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
    try:
        if action == 'train':
            old.train(job, folder)
        elif action == 'evaluate':
            old.evaluate(job, folder)
        else:
            raise ValueError(action)
        io.js(folder / 'POWER_END.json', dict(utc=io.utc(), snapshot=old.subprocess.run(['pmset','-g','batt'],capture_output=True,text=True).stdout))
        io.js(folder / 'ADAPTER_COMPLETE.json', dict(protocol_sha256=io.check_protocol(), utc=io.utc(), job=job,
            artifacts=[io.artifact(folder/n) for n in ['COMPLETE.json','T3_BINDINGS.json','POWER_START.json','POWER_END.json']]))
    except Exception:
        io.text(folder / ('UNHANDLED_' + io.stamp() + '.txt'), old.traceback.format_exc())
        raise
