"""Read-only AST inventory and recorded provenance, no experiment execution."""
import sys,ast,hashlib,json,csv,datetime
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step24';R=ROOT/'revision_work/outputs';P=ROOT/'scripts/python';W=ROOT/'revision_work/scripts';FULL=R/'fullrun/sessions'
G={k:[] for k in ['E'+str(i) for i in range(1,13)]}
def add(e,*paths):
 for x in paths:
  p=Path(x);p=p if p.is_absolute() else ROOT/p
  if p.is_dir():G[e]+=list(p.glob('*.py'))
  else:G[e].append(p)
for e in ['E1','E3','E9']:add(e,P/'panel_common.py',P/'panel_build_multiscale_dataset.py',W/'causal/core.py',W/'causal/build_and_audit.py',W/'causal/build_stage3c_asof.py')
add('E1',P/'panel_download_daily_weekly.py',ROOT/'revision_audit/download_revision_data.py')
for e in ['E2','E3']:add(e,*[P/f for f in ['panel_pathformer_main_v2.py','panel_pathformer_experiment3.py','panel_pathformer_experiment4_multiseed.py','panel_baseline_lstm.py','panel_baseline_vanilla_transformer.py','panel_baseline_swim_cpu.py','panel_baseline_ridge.py']])
add('E2',W/'skill/a1_benchmarks.py',W/'skill/skill_common.py')
add('E3',W/'causal/fullrun14a_worker_v2.py',R/'step22b/worker22b_v3.py',R/'step22b2/worker_b2_v2.py',R/'step22f/worker_f.py')
for e in ['E4','E8']:add(e,W/'skill/skill_common.py',W/'skill/a2_skill.py',W/'skill/a3_hierarchical.py',W/'analysis16/statistics16.py',W/'analysis16/statistics16_v2.py',W/'analysis16/statistics16_v3.py',W/'analysis16/common16.py',W/'analysis16/registry16.py',W/'analysis16/reconcile16_v2.py',W/'analysis16/clarified16.py',W/'analysis16/ensembles16.py')
add('E4',W/'analysis17',R/'step21')
add('E8',W/'ensemble/ensemble_analysis.py',W/'ensemble/ensemble_robust.py',P/'panel_pathformer_experiment4_inference.py',R/'frozen/scripts/frozen_report.py')
for session,es in [('14c_day_20261017T050034Z_12db13f7',['E2','E3']),('14c_B2_20261017T101438Z_1a8d5041',['E1','E9']),('14c_B2_supplement_20261017T162556Z_bfb4b1be',['E9']),('14c_B3_20261017T201559Z_f47c5c3c',['E5']),('14c_B5_20261018T142713Z_6908ba80',['E2','E3']),('14c_B5fix_20261019T055730Z_6da4918c',['E2','E3'])]:
 for e in es:add(e,FULL/session/'scripts')
add('E5',W/'analysis16/statistics16_v3.py')
add('E6',R/'frozen/scripts',R/'step22a/worker22.py',R/'step22a/analyze22.py',R/'step22a/io22.py')
add('E7',W/'interventions/router_interventions.py',W/'analysis16/interventions16.py',W/'analysis16/intervention_stats16_v2.py')
add('E9',W/'analysis16/clarified16.py',W/'analysis16/statistics16.py',R/'step22d/session_v3/window_data.py',R/'step22d/session_v3/prepare_d.py')
add('E10',W/'portfolio/engine.py',W/'analysis16/portfolio16_v2.py',R/'step22g/analyze_g_v3.py')
add('E11',P/'panel_pathformer_experiment5_router_diagnostics.py',W/'analysis16/intervention_stats16_v2.py',R/'step22a/analyze22.py',R/'step22f/analyze_f.py',R/'step22g/analyze_g_v3.py')
for st in ['step22a','step22b','step22b2','step22c','step22d/session_v3','step22e','step22f','step22g']:add('E12',R/st)
# Register exact source hashes recorded by the producing protocols/run manifests when present.
refs={};manifests=[]
for st in ['step22a','step22b','step22b2','step22d/session_v3','step22e','step22f','step22g','analysis16']:
 q=R/st/'RNG_MANIFEST.json'
 if not q.exists():continue
 j=json.loads(q.read_text());manifests.append(dict(path=str(q.relative_to(ROOT)),sha256=hashlib.sha256(q.read_bytes()).hexdigest()))
 def walk(v):
  if isinstance(v,dict):
   if 'path' in v and 'sha256' in v:refs.setdefault(str(v['path']),[]).append(dict(manifest=str(q.relative_to(ROOT)),sha256=v['sha256']))
   for a in v.values():walk(a)
  elif isinstance(v,list):
   for a in v:walk(a)
 walk(j)
reports={
'E1':['revision_audit/AUDIT_REPORT.md','revision_work/outputs/causal_features/REPORT_STEP4_FINAL.md','revision_work/outputs/data_audit/REPORT_STEP13.md'],
'E2':['revision_work/outputs/fullrun/REPORT_STEP14A.md','revision_work/outputs/fullrun/REPORT_STEP14C_DAY.md','revision_work/outputs/fullrun/REPORT_STEP14C_B5FIX.md'],
'E3':['revision_work/outputs/repro_threads/REPORT_STEP3B.md','revision_work/outputs/fullrun/REPORT_STEP14A.md','revision_work/outputs/step22b/REPORT_STEP22B.md'],
'E4':['revision_work/outputs/analysis16/REPORT_STEP16_FINAL.md','revision_work/outputs/analysis17/REPORT_STEP17.md','revision_work/outputs/step21/REPORT_STEP21.md'],
'E5':['revision_work/outputs/fullrun/REPORT_STEP14C_B3.md','revision_work/outputs/analysis16/REPORT_STEP16.md'],
'E6':['revision_work/outputs/frozen/REPORT_STEP15.md','revision_work/outputs/step22a/REPORT_STEP22A_FINAL.md'],
'E7':['revision_work/outputs/interventions/original/REPORT_STEP9.md','revision_work/outputs/analysis16/REPORT_STEP16.md'],
'E8':['revision_work/outputs/skill_analysis/REPORT_STEP5.md','revision_work/outputs/skill_analysis/ensemble/REPORT_STEP7.md','revision_work/outputs/skill_analysis/ensemble_robust/REPORT_STEP7B.md'],
'E9':['revision_work/outputs/fullrun/REPORT_STEP14C_B2_5SEED.md','revision_work/outputs/analysis16/REPORT_STEP16.md'],
'E10':['revision_work/outputs/portfolio/original_rp/REPORT_STEP8.md','revision_work/outputs/analysis16/REPORT_STEP16.md','revision_work/outputs/step22g/REPORT_STEP22G_FINAL.md'],
'E11':['revision_work/outputs/analysis16/REPORT_STEP16.md','revision_work/outputs/step22a/REPORT_STEP22A_FINAL.md'],
'E12':[]}
for st in ['step22a','step22b','step22b2','step22c','step22d','step22e','step22f','step22g']:reports['E12'] +=[str(q.relative_to(ROOT)) for q in (R/st).glob('REPORT*.md')]
outputs={
'E1':['dataset/multiscale_dataset/panel/*/{X_daily.npy,X_weekly.npy,meta.csv,y_*d.npy}','revision_work/outputs/causal_features/tensors/*','revision_work/outputs/fullrun/sessions/14c_B2_20261017T101438Z_1a8d5041/{design.json,tensors/**}'],
'E2':['revision_audit/parameter_counts.csv','dataset/audit/main_v2/**','revision_work/outputs/fullrun/**/trainer_summary.csv'],
'E3':['revision_work/outputs/fullrun/**/best_state.pt','revision_work/outputs/step22{b,b2,e,f,g}/fits/**/COMPLETE.json'],
'E4':['revision_work/outputs/analysis16/clarified/B_forecast_skill_final.csv','revision_work/outputs/analysis17/test_results.csv','revision_work/outputs/step21/*.csv'],
'E5':['revision_work/outputs/analysis16/D_paired_tests.csv','revision_work/outputs/fullrun/sessions/14c_B3_20261017T201559Z_f47c5c3c/design*'],
'E6':['revision_work/outputs/frozen/backbones/**','revision_work/outputs/step22a/{BINDINGS.json,cache_replay_verification.csv,selected_learning_rates.csv,joint_bootstrap_tests.csv}'],
'E7':['revision_work/outputs/analysis16/H_bootstrap_tests_v2.csv','revision_work/outputs/analysis16/H/**'],
'E8':['revision_work/outputs/skill_analysis/a3_original/*','revision_work/outputs/analysis16/{A,C,D,E}_paired_tests.csv','revision_work/outputs/skill_analysis/ensemble_robust/*'],
'E9':['revision_work/outputs/analysis16/labels/*.parquet','revision_work/outputs/analysis16/clarified/C_subgroups_summary.csv'],
'E10':['revision_work/outputs/analysis16/G_PRIMARY_with_existing_O.csv','revision_work/outputs/portfolio/original_rp/{weights.csv,rebalances.csv,metrics_all.csv,grs_tests.csv}'],
'E11':['dataset/audit/experiment5_router_diagnostics/*.csv','revision_work/outputs/analysis16/H_router_diagnostics.csv'],
'E12':['revision_work/outputs/step22*/{RNG_MANIFEST.json,PROTOCOL*.md,ANALYSIS_COMPLETE.json}','revision_work/outputs/step22*/analysis/**']}
rows=[]
for e,paths in G.items():
 for p in sorted(set(paths)):
  rel=str(p.relative_to(ROOT))
  if not p.exists():rows.append(dict(experiment=e,path=rel,status='UNLOCATED',functions='',sha256='',reports='; '.join(reports[e]),stored_outputs='; '.join(outputs[e]),provenance='Named candidate not found; not assumed used'));continue
  data=p.read_bytes()
  try:tree=ast.parse(data);syntax=''
  except SyntaxError as er:tree=ast.parse('');syntax=f'{er.msg} at line {er.lineno}'
  functions=['.'.join([n.name])+':'+str(n.lineno)+'-'+str(n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))];h=hashlib.sha256(data).hexdigest();reg=refs.get(rel,[]);same=[r for r in reg if r['sha256']==h]
  rows.append(dict(experiment=e,path=rel,status='LOCATED',syntax_error=syntax,functions='; '.join(functions),sha256=h,reports='; '.join(x for x in reports[e] if (ROOT/x).exists()),stored_outputs='; '.join(outputs[e]),provenance=json.dumps(reg),registered_current_hash_matches=bool(same) if reg else 'NOT RECORDED',role='registered_source' if same else 'shared implementation or version candidate; runtime/source linkage checked in priority review'))
with (OUT/'CODE_MAP.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=sorted(set().union(*(r.keys() for r in rows))));w.writeheader();w.writerows(rows)
with (OUT/'INVENTORY_MANIFEST.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(rows),experiments=sorted(G),source_manifest_hashes=manifests,note='AST inventory only; no experiment executed. Located legacy versions are candidates unless production provenance is recorded; none is silently assumed executed.'),f,indent=2)
print('CODE_MAP',len(rows),'rows;',len(set(r['path'] for r in rows)),'paths; unlocated',[(r['experiment'],r['path']) for r in rows if r['status']=='UNLOCATED'])
