"""Import the Step22D data gate, substituting its stored sigma instead of estimating SD."""
import sys,ast,copy
sys.dont_write_bytecode=True
import io_e as io
sys.path.insert(0,str(io.D));sys.path.insert(0,str(io.ROOT))
d=io.module('_22e_prepare_d',io.D/'prepare_d.py')
from reuse_d import window_loader

def main():
    pd=d.pd
    source=io.read(io.D/'SIGMAS.json')
    assert source['ddof']==1 and source['arithmetic']=='float64 from frozen float32 TRAIN labels'
    assert source['values']==io.read(io.D.parent/'SIGMAS.json')['values']
    rows=pd.read_csv(io.D/'TRAIN_TARGET_SD.csv',float_precision='round_trip')
    lookup={(r['window'],int(r['h'])):r for r in rows.to_dict('records')}
    for w in ['W1','W2']:
        for h in io.HS:
            assert lookup[w,h]['sd_ddof1']==lookup[w,h]['selected_delta']==source['values'][w][str(h)]
    tree=ast.parse((io.D/'prepare_d.py').read_text())
    fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main'))
    changes=[]
    class Binding(ast.NodeTransformer):
        def visit_Assign(self,n):
            self.generic_visit(n)
            if len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='sd':
                assert ast.unparse(n.value)=='float(y.std(ddof=1))'
                n.value=ast.parse("SOURCE_SIGMAS[w][str(h)]",mode='eval').body;changes.append('stored_sigma')
            return n
        def visit_Call(self,n):
            self.generic_visit(n)
            if isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='stats' and n.func.attr=='append':
                n.args=[ast.parse('SOURCE_SD_ROWS[w,h].copy()',mode='eval').body];changes.append('stored_sigma_table')
            return n
        def visit_Constant(self,n):
            if n.value=='PROTOCOL_STEP22D.md':return ast.copy_location(ast.Constant('PROTOCOL_STEP22E.md'),n)
            return n
    fn=Binding().visit(fn);assert sorted(changes)==['stored_sigma','stored_sigma_table']
    env=dict(vars(d));env.update(io.exports());env.update(load=window_loader(),SOURCE_SIGMAS=source['values'],SOURCE_SD_ROWS=lookup)
    node=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]))
    exec(compile(node,str(io.D/'prepare_d.py'),'exec'),env)
    env['main']()
    mine=pd.read_csv(io.OUT/'data_hash_checks.csv').sort_values('path').reset_index(drop=True)
    old=pd.read_csv(io.D/'data_hash_checks.csv').sort_values('path').reset_index(drop=True)
    assert mine[['path','expected_sha256','actual_sha256']].equals(old[['path','expected_sha256','actual_sha256']])
    assert io.read(io.OUT/'SIGMAS.json')['values']==source['values']
    if not (io.OUT/'SIGMA_IDENTITY_GATE.json').exists():
        io.js(io.OUT/'SIGMA_IDENTITY_GATE.json',dict(utc=io.utc(),passed=True,sigmas_recomputed=False,values=source['values'],
          source_json=io.artifact(io.D/'SIGMAS.json'),source_table=io.artifact(io.D/'TRAIN_TARGET_SD.csv'),
          data_gate_module=io.artifact(io.D/'prepare_d.py'),gate_bindings=changes,window_tensor_hashes_equal_to_step22d=True,
          no_test_deserialization=True,bound_ast_sha256=io.hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()))
    print('Step22E data/sigma gate PASS; TEST not deserialized; sigma copied exactly, not re-estimated.')

if __name__=='__main__':main()
