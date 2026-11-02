"""Read-only source cache and exact CSV locators; no estimation routines imported."""
import sys,json,hashlib,csv
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step23b';R=ROOT/'revision_work/outputs';SOURCES={};CACHE={};DERIVED=[]
sys.path.append(str(R/'_deps_pyarrow15'))
import pandas as pd,numpy as np
def path(p):return p if isinstance(p,Path) and p.is_absolute() else ROOT/Path(p)
def read(p):
 p=path(p);b=p.read_bytes();key=str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p);SOURCES[key]=dict(path=key,bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return b
def js(p):return json.loads(read(p))
P=js(R/'step23/payload23_v2.json');ALIASES={b['key']:b['source'] for b in P['blocks']}
ALIASES.update({'da_table4':str(R/'step21/table4_da_summary.csv'),'da_references':str(R/'step21/references.csv'),'da_pair':str(R/'step21/paired_da_tests.csv'),'E':str(R/'analysis16/E_paired_tests.csv'),'A':str(R/'analysis16/A_paired_tests.csv'),'D':str(R/'analysis16/D_paired_tests.csv'),'B':str(R/'analysis16/clarified/B_forecast_skill_final.csv'),'bsum':str(R/'analysis16/B_PRIMARY_seed_summary.csv'),'G':str(R/'analysis16/G_PRIMARY_with_existing_O.csv'),'GRS':str(R/'analysis16/G_GRS.csv'),'OGRS':str(R/'analysis16/G_existing_O_GRS_reference.csv'),'H':str(R/'analysis16/H_bootstrap_tests_v2.csv'),'Hrouter':str(R/'analysis16/H_router_diagnostics.csv'),'subgroup':str(R/'analysis16/clarified/C_subgroups_summary.csv'),'learn':str(R/'manuscript19/learning_summary.csv'),'labelscale':str(ROOT/'revision_audit/training_label_scale.csv'),'timing':str(R/'data_audit/D_timing_summary.csv'),'precision':str(R/'causal_features/precision_distributions.csv'),'labelimpact':str(R/'causal_features/label_impact_summary.csv'),'labelmse':str(R/'causal_features/saved_predictions_new_label_mse.csv'),'ratios':str(R/'causal_features/ratio_step_context.csv'),'universe':str(R/'universe/survivorship_summary.csv'),'profile':str(R/'analysis16/I_4_4_C_numerical_profile.csv'),'a3joint':str(R/'skill_analysis/a3_original/a3_joint_seed_date_bootstrap.csv'),'a3seed':str(R/'skill_analysis/a3_original/a3_seed_level_tests.csv'),'a1':str(R/'skill_analysis/a1/a1_benchmark_mse.csv'),'ensemble':str(R/'skill_analysis/ensemble/ensemble_metrics.csv'),'power':str(R/'skill_analysis/power/power_summary.csv')})
def table(k):
 p=ALIASES.get(k,k);p=path(p)
 if str(p) not in CACHE:read(p);CACHE[str(p)]=pd.read_csv(p,float_precision='round_trip')
 return CACHE[str(p)]
def select(k,**kwargs):
 f=table(k)
 for c,v in kwargs.items():f=f[f[c].isin(v) if isinstance(v,(list,tuple,set)) else f[c].eq(v)]
 assert len(f)>0,(k,kwargs)
 return f
def vals(k,c,scale=1,**kwargs):
 f=select(k,**kwargs);a=f[c].to_numpy(float)*scale;p=path(ALIASES.get(k,k));rel=str(p.relative_to(ROOT));refs=[rel+':'+str(i+2)+' ['+c+']' for i in f.index];return a,refs
def csvwrite(name,f):
 with (OUT/name).open('x',newline='') as h:f.to_csv(h,index=False)
def dump(name,v):
 with (OUT/name).open('x') as h:json.dump(v,h,indent=2,ensure_ascii=False,allow_nan=False)
def checkpoint_sources(name):csvwrite(name,pd.DataFrame(SOURCES.values()))
