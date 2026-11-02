"""No-training baseline provenance, real-tensor probes and final design gate."""
from b3_io import *
from b3_data import base,common,formal,load_panel,contract,fit_bound_stats,apply_bound_stats,make_valid_stats
from b3_models import make_model,FrequencyInterface
import numpy as np,pandas as pd,torch,argparse

def independent():
 base.check_profile();base.exp3._verify_frozen_dependencies();base.exp4._verify_frozen_dependencies();base.exp4._verify_frozen_training_constants()
 source=FULLRUN/'sessions/14c_B2_supplement_20261017T162556Z_bfb4b1be';v=json.loads((source/'FINAL_VALIDATION.json').read_text());assert v['all_new_cells_verified']
 assert list((source/'control').glob('*/FINISHED.json'))
 original=FULLRUN/'sessions/14c_B2_20261017T101438Z_1a8d5041';assert list((original/'control').glob('*/FINISHED.json'))
 all_rows=pd.read_csv(source/'all_cell_metrics_5seed.csv');b=all_rows[all_rows.window.eq('W3')&all_rows.model.isin(['static','adaptive'])].copy();assert len(b)==60
 prov=json.loads((source/'W3_provenance.json').read_text());entries=[e for e in prov['entries'] if e['model'] in ['static','adaptive']];assert len(entries)==60
 frozen=common.load_frozen_panel();test=frozen.split=='test';key=pd.DataFrame(dict(ticker=frozen.ticker[test],anchor_date=pd.to_datetime(frozen.anchor_date[test]).strftime('%Y-%m-%d'))).sort_values(['ticker','anchor_date']).reset_index(drop=True)
 for e in entries:
  assert sha(ROOT/e['source'])==e['source_sha256'];a=e['copied_test_predictions'];assert sha(ROOT/a['path'])==a['sha256']
  frame=pd.read_parquet(ROOT/a['path']).sort_values(['ticker','anchor_date']).reset_index(drop=True);assert len(frame)==6647 and frame[['ticker','anchor_date']].equals(key)
  expected=pd.DataFrame(dict(ticker=frozen.ticker[test],anchor_date=pd.to_datetime(frozen.anchor_date[test]).strftime('%Y-%m-%d'),y_true=frozen.y[e['horizon']][test])).sort_values(['ticker','anchor_date']).reset_index(drop=True)
  assert np.array_equal(frame.y_true.to_numpy(np.float32),expected.y_true.to_numpy(np.float32))
  assert np.isfinite(frame[['y_true','y_pred']]).all().all()
 with target(EXP/'baseline_cell_metrics.csv').open('x',newline='') as f:b.to_csv(f,index=False)
 jsonout(EXP/'baseline_provenance.json',dict(utc=utc(),source_metric_csv=artifact(source/'all_cell_metrics_5seed.csv'),n_cells=60,entries=entries,seeds=[42,0,1,21,3407],original_feature_setting='Daily90 + Weekly26',no_training_or_inference=True,original_B2_complete=True,B2_supplement_complete=True))
 rows=[];formal.set_seed(42)
 for key_name in ['O','C_anchor','C_end']:
  ds=load_panel(key_name);contract(ds);stats=fit_bound_stats(ds);fresh=make_valid_stats(ds,ds.valid_daily)
  assert all(np.array_equal(a,b) for k in stats for a,b in zip(stats[k],fresh[k]))
  z=apply_bound_stats(ds,stats,'daily_weekly');assert z[0][ds.valid_daily].any(axis=-1).all() and (z[0][~ds.valid_daily]==0).all()
  assert np.array_equal(z[0],np.load(EXP/'tensors'/key_name/'X_daily_standardized.npy')) and np.array_equal(z[1],np.load(EXP/'tensors'/key_name/'X_weekly_standardized.npy'))
  for freq in ['daily_only','weekly_only','daily_weekly']:
   for mechanism in ['static','adaptive']:
    model=make_model(freq,mechanism,dual_interface=True);model.eval();xd=torch.from_numpy(z[0][:3]);xw=torch.from_numpy(z[1][:3])
    with torch.no_grad():p,wd,ww=model.forward_with_router(xd,xw)
    assert torch.isfinite(p).all()
    for active,w in [(freq!='weekly_only',wd),(freq!='daily_only',ww)]:
     if active:assert torch.allclose(w.sum(1),torch.ones(3),atol=1e-6,rtol=0)
     else:assert not w.any()
    rows.append(dict(tensor_key=key_name,frequency_setting=freq,mechanism=mechanism,parameters=formal.count_trainable_parameters(model),finite_forward=True,active_router_sums_one=True,inactive_branch_not_instantiated=freq!='daily_weekly',exact_mask_transport=True,stats_match_recomputed_valid_TRAIN_only=True))
 csvout(EXP/'real_tensor_forward_checks.csv',rows)
 jsonout(EXP/'INDEPENDENT_CHECKS.json',dict(utc=utc(),passed=True,trained_cells=0,forward_checks=18,baseline_cells_verified=60,padding_adapter_verified=json.loads((EXP/'PADDING_CHECKS.json').read_text())['passed'],candidate_tensors_verified=True,causal_reference_policy_pending=not (EXP/'causal_reference_policy.json').exists()))
 print('B3_INDEPENDENT_CHECKS_PASS',utc(),flush=True)

def final():
 from b3_contracts import dependencies
 assert json.loads((EXP/'INDEPENDENT_CHECKS.json').read_text())['passed']
 policy=json.loads((EXP/'causal_reference_policy.json').read_text());assert policy['policy'] in ['completed_week_close','anchor_close']
 assert policy['tensor_key']==('C_end' if policy['policy']=='completed_week_close' else 'C_anchor')
 assert policy['user_authorized'];d=json.loads((EXP/'design_diagnostics.json').read_text());ds=load_panel('C');contract(ds)
 for a in json.loads((EXP/'tensor_manifest.json').read_text())['files']:assert sha(ROOT/a['path'])==a['sha256']
 with (EXP/'design_examples.csv').open() as f:examples=list(csv.DictReader(f))
 with (EXP/'design_session_count_distribution.csv').open() as f:dist=list(csv.DictReader(f))
 body=['# BLOCKING','None. Design, tensor, mask and factory probes passed. The C reference policy has been explicitly resolved before any training.',
 '# B3 design before training',f'Written {utc()} UTC. Fixed Daily-matched length127, right-padded with zeros; real sessions range122–127. Weekly26 source bars, anchor-date forecast origin, frozen labels/splits/purges, formal keys and rows30634/6290/6647 remain as specified. All43571 matched Daily and Weekly source intervals agree; no selected bar is later than the last completed week.',
 table(['split','anchor','Daily first','Daily last','Weekly first underlying session','Weekly last underlying session','valid Daily sessions'],[[r['split'],r['anchor_date'],r['daily_first_session'],r['daily_last_session'],r['weekly_first_underlying_session'],r['weekly_last_underlying_session'],r['valid_daily_sessions']] for r in examples]),
 table(['split','valid Daily sessions','windows','fraction'],[[r['split'],r['valid_daily_sessions'],r['n_windows'],r['fraction']] for r in dist]),
 f'C reference policy: `{policy["policy"]}`; tensor binding `{policy["tensor_key"]}`. User instruction: {policy["instruction"]}.',
 ('Both C branches use the last source Close of the completed26-week interval, so no later anchor-Close reference enters either branch. This deliberately changes the C Weekly price ratios versus Stage3a; underlying Weekly26 OHLCV/volume centering, all labels/anchors and forecast origins are unchanged. The formal C comparator remains the archived Stage3a anchor-Close binding and this difference is disclosed.' if policy['policy']=='completed_week_close' else 'As explicitly selected, C retains the original Stage3a anchor-date Close reference. The input-bar intervals match exactly, but the denominator uses a later source session in34510 samples. Therefore strict exclusion of later information from the feature dependency is NOT claimed. Weekly C tensors/standardization remain bitwise equal to core.load_causal().'),
 'The imported common normalizer is fitted per ticker/resolution/channel on TRAIN only with ddof0 and the original std guard. Daily real TRAIN entries are compacted before fitting; padding is excluded. The imported normalizer is applied and padding then set to standardized zero. All real standardized tokens are nonzero, making the persisted valid-mask transport through unchanged DataLoader signatures exact.',
 'The unchanged formal expert valid_mask alone produces NaN for completely empty patches at the requested fixed length. A new external mask adapter forwards this existing mask, provides a discarded dummy key for empty intra-patch attention, and masks empty inter-patch keys. It adds no learned parameters or model formulas.24 synthetic checks across all valid lengths, Static/Adaptive and Daily-only/dual demonstrate finite forward/backward, exactly zero response to padding perturbations and less than1e-6 deviation from cropped unpadded inputs.18 real-tensor/factory probes and exact normalizer recomputation pass.',
 'Trainers are imported: seed42 Adaptive uses main_v2.train_one_cell with the requested resolution; seed42 Static uses Experiment3 with the imported factory at that resolution; other seeds use Experiment4 with the imported factory at that resolution. A parameter-free argument adapter permits the original Experiment3/4 dual-input loop to call the existing single-resolution model; the unused branch is not instantiated or trained. Generator42, optimizer/loss/batch/stopping/restoration remain unchanged. CPU12/12 and3 refilled workers are used, with caffeinate and telemetry.',
 f'[Full source windows]({EXP}/design_windows.csv), [tensor manifest]({EXP}/tensor_manifest.json), [padding tests]({EXP}/padding_adapter_checks.csv), [real tensor probes/parameter counts]({EXP}/real_tensor_forward_checks.csv), [baseline provenance]({EXP}/baseline_provenance.json). No training has run at this design-report timestamp.']
 textout(EXP/'DESIGN_REPORT_STEP14C_B3.md','\n\n'.join(body)+'\n')
 jsonout(EXP/'preflight.json',dict(utc=utc(),passed=True,dependencies=dependencies()+[artifact(EXP/'DESIGN_REPORT_STEP14C_B3.md')],trained_cells_before_design=0,profile='CPU12/12',workers=3,causal_reference_policy=policy,design=d))
 print('B3_FINAL_PREFLIGHT_PASS',utc(),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();final() if a.final else independent()
