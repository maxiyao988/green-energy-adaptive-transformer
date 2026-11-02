"""Final artifact/source validation and exclusive immutable manifest."""
import sys,os,json,platform,importlib.metadata,re
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
sys.path.append(str(ROOT/'revision_work/outputs/_deps_pyarrow15'))
import pandas as pd,numpy as np
# This supplement corrects an illustrative command only, preserving the write-once report.
text('REPORT_STEP13_ERRATA.md',
 '# Step13 command erratum\n\nThe illustrative shell block in REPORT_STEP13.md contains literal backslash-n text. Use this equivalent single-line command, with a fresh output label if rerunning. The numerical results and tables are unchanged; no additional intervention was run for this erratum.\n\n```sh\nOMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 10 .venv/bin/python -B revision_work/outputs/data_audit/scripts/router_interventions_with_inputset.py --input-set causal --smoke-only --run-label NEW_EXCLUSIVE_LABEL\n```\n')
A=pd.read_csv(OUT/'A_event_verification.csv');cash=pd.read_csv(OUT/'A_cash_dividend_spotchecks.csv');C=pd.read_csv(OUT/'C_event_window_features_annotated.csv');D=pd.read_csv(OUT/'D_decision_dates.csv');E=json.loads((OUT/'causal_i0_smoke_strict_profile/verification.json').read_text());F=json.loads((OUT/'F_seed_provenance_conclusion.json').read_text());source=pd.read_csv(OUT/'source_catalog.csv');http=json.loads((OUT/'http_compliance.json').read_text());cons=json.loads((OUT/'frozen_consistency.json').read_text())
checks={
 '8_vendor_events':len(A)==8,
 '7_match_1_mismatch':A.status.value_counts().to_dict()=={'MATCH':7,'MISMATCH':1},
 '17_cash_tickers_266_events':len(cash)==17 and cash['count'].sum()==266,
 '7_issuer_cash_spotchecks':cash.source_id.notna().sum()==7,
 '3_exact_cash_amount_mismatches':(cash.exact_amount_status=='MISMATCH').sum()==3,
 '352_event_channel_rows':len(C)==352,
 '8_events_11_anchors':bool((C.groupby(['ticker','event_date']).anchor_date.nunique()==11).all()),
 '269_step8_decision_cells':len(D)==269,
 '13719_labels_exact_frozen':cons['D_label_values']==13719 and cons['D_original_exact_frozen'] and cons['D_asof_exact_frozen'],
 'I0_EXACT_strict_light_profile':E['classification']=='EXACT' and E['max_abs_delta_prediction']==0 and E['max_abs_delta_router_weights']==0 and E['torch_num_threads']<=2 and E['OMP_NUM_THREADS']=='1' and E['nice']==10,
 'seed_numeric_rationale_NOT_FOUND':F['rationale_for_particular_values']=='NOT FOUND',
 'SEC_UA_and_rate':http['all_user_agents_match'] and http['peak_requests_started_in_any_rolling_second']<=5,
 'all_final_source_http_200':bool((source.status==200).all()),
 'all_evidence_needles_found':bool((pd.read_csv(OUT/'source_evidence_locations.csv').status=='FOUND').all()),
 'adapter_help_cli_checked':(OUT/'router_adapter_help.log').read_text().startswith('usage:'),
 'report_starts_BLOCKING':(OUT/'REPORT_STEP13.md').read_text().startswith('# BLOCKING'),
}
protections=[]
for r in pd.read_csv(OUT/'protected_inputs_CD.csv').to_dict('records'):
 p=ROOT/r['path'];protections.append(dict(path=r['path'],sha256_before=r['sha256_before'],sha256_now=sha(p),unchanged=r['sha256_before']==sha(p)))
for r in json.loads((OUT/'causal_i0_smoke_strict_profile/protected_inputs.json').read_text()):
 p=Path(r['path']);protections.append(dict(path=str(p.relative_to(ROOT)),sha256_before=r['sha256_before'],sha256_now=sha(p),unchanged=r['sha256_before']==sha(p)))
pr=rows('protected_inputs_final.csv',pd.DataFrame(protections).drop_duplicates('path'));checks['all_accessed_protected_inputs_unchanged']=bool(pr.unchanged.all())
for r in source.to_dict('records'):
 p=ROOT/r['cached_path'];assert sha(p)==r['sha256'],r['source_id']
checks['all_source_body_hashes_valid']=True
report=(OUT/'REPORT_STEP13.md').read_text();localrefs=re.findall(r'`((?:A_|B_|C_|D_|E_|F_)[^`]+\.(?:csv|json|txt))`',report)
checks['report_referenced_tables_exist']=all((OUT/v).is_file() for v in localrefs)
js('FINAL_VALIDATION.json',dict(utc=utc(),all_required_checks_pass=all(checks.values()),checks=checks,n_protected_inputs=len(pr),no_training=True,smoke_checkpoint_cells=1,non_I0_interventions=0,output_namespace=str(OUT.relative_to(ROOT)),fullrun_paths_opened_by_audit=0,report_command_erratum='REPORT_STEP13_ERRATA.md',initial_attempt_logs_preserved=True))
assert all(checks.values()),checks
versions={k:importlib.metadata.version(k) for k in ['torch','numpy','pandas','scipy','scikit-learn','certifi']}
files=[]
for p in sorted(OUT.rglob('*')):
 if p.is_file():files.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
# Exclude the finalizer's currently open shell log: it is finalized when this process exits.
files=[r for r in files if r['path']!='finalize_audit.log']
js('MANIFEST.json',dict(created_utc=utc(),python=platform.python_version(),platform=platform.platform(),package_versions=versions,SEC_User_Agent=UA,sec_max_requests_per_second=5,actual_peak_requests_per_second=http['peak_requests_started_in_any_rolling_second'],OMP_NUM_THREADS=os.environ['OMP_NUM_THREADS'],nice=os.getpriority(os.PRIO_PROCESS,0),task='PROMPT13 no training',n_files=len(files),files=files,manifest_self_hash_excluded=True,active_finalizer_log_hash_excluded=True,protected_input_table='protected_inputs_final.csv',validation='FINAL_VALIDATION.json',command_erratum='REPORT_STEP13_ERRATA.md'))
print(json.dumps(dict(all_required_checks_pass=True,n_protected_inputs=len(pr),n_manifest_files=len(files),event_matches=7,event_mismatches=1,I0='EXACT',max_pred_delta=E['max_abs_delta_prediction'],report=str(OUT/'REPORT_STEP13.md'))),flush=True)
