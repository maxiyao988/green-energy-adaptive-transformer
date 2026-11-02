"""Import unchanged Step22D nested statistics definitions and data binding."""
import sys
sys.dont_write_bytecode=True
import io_e as io
import ast

def window_loader():
    m=io.module('_22e_window_data_d',io.D/'window_data.py')
    m.__dict__.update(io.exports())
    return m.load

def statistics(dest):
    sys.path.insert(0,str(io.D));sys.path.insert(0,str(io.ROOT/'revision_work/scripts/analysis16'))
    sys.path.insert(0,str(io.OLD));sys.path.append(str(io.ROOT/'revision_work/outputs/_deps_pyarrow15'))
    d=io.module('_22e_statistics_d',io.D/'analyze_d_v3.py')
    env=dict(vars(d));env.update(io.exports())
    env.update(dest=dest,outputs=[],protected={},all_labels={},B=20000,BASE=20261022,SEEDS=io.SEEDS)
    tree=ast.parse((io.D/'analyze_d_v3.py').read_text())
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    names=['protect','save','binary','replicates','readcsv','strict_mean','summary','draws','date_boot','joint_boot','series_result','per_seed']
    nodes=[n for n in main.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(io.D/'analyze_d_v3.py'),'exec'),env)
    return env
