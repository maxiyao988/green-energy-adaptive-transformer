"""Strict descriptive audit. No prediction evaluation, resampling or statistical testing."""
from sources23b import *
import re,math
EX=js(OUT/'EXTRACTED_TEXT.json');AUD=[];COVER=[]
SA=['static','adaptive'];AGG=['single','fixed','same_patch_fixed','static','adaptive'];POOL='W1+W2+W3'
def evidence(k,c,scale=1,**filters):
 a,r=vals(k,c,scale,**filters);return a,r
V=evidence
def join(*es):return np.concatenate([e[0] for e in es]),sum([e[1] for e in es],[])
def arith(k,c1,c2,formula,**filters):
 a,r=V(k,c1,**filters);b,s=V(k,c2,**filters);assert len(a)==len(b);return formula(a,b),r+s

def ratio_models(k,col,left,right,**filters):
 a,r=V(k,col,model=left,**filters);b,s=V(k,col,model=right,**filters);return 100*(a/b-1),r+s

def add(x,claim,e,kind,expected,digits=None,fix='',scope=''):
 a,refs=e;a=np.asarray(a,float).reshape(-1);a=a[np.isfinite(a)];assert len(a),(claim,refs)
 if kind=='range':
  dg=[digits,digits] if isinstance(digits,int) else digits;actual=[round(float(a.min()),dg[0]),round(float(a.max()),dg[1])];ok=actual==list(expected);cl='RANGE_OK' if ok else 'RANGE_MISMATCH';suggest=f'Use the rounded range {actual[0]}–{actual[1]} (ascending order).'
 elif kind=='eq':
  actual=np.round(a,digits) if digits is not None else a;ok=bool(np.all(actual==expected));cl='STRICT_OK' if ok else 'WRONG';suggest='Use stored values at the stated precision; see source_values.'
 else:
  ops={'ge':np.greater_equal,'gt':np.greater,'le':np.less_equal,'lt':np.less};ok=bool(ops[kind](a,expected).all());rounded=bool(ops[kind](np.round(a,digits),expected).all()) if digits is not None else False
  cl='STRICT_OK' if ok else 'ROUNDING_ONLY' if rounded else 'WRONG';actual=[float(a.min()),float(a.max())]
  if kind in ['ge','gt']:
   n=digits if digits is not None else 6;v=math.floor(a.min()*10**n)/10**n;suggest=f'Replace with ≥ {v:g}; minimum stored value = {a.min():.17g}.'
  else:
   n=digits if digits is not None else 6;v=math.ceil(a.max()*10**n)/10**n;suggest=f'Replace with ≤ {v:g}; maximum stored value = {a.max():.17g}.'
 AUD.append(dict(candidate_id=x['candidate_id'],document=x['document'],location=x['location'],paragraph=x['paragraph'],sentence=x['sentence'],claim=claim,source_values=json.dumps(a.tolist()),source_min=float(a.min()),source_max=float(a.max()),source_file_cell='; '.join(refs),comparison=kind,claimed_value=json.dumps(expected),display_digits=str(digits),classification=cl,proposed_fix='' if cl in ['STRICT_OK','RANGE_OK'] else fix or suggest,scope=scope))

def bound(x,claim,e,kind,n,d=2,fix='',scope=''):add(x,claim,e,kind,n,d,fix,scope)
def ran(x,claim,e,lo,hi,d):add(x,claim,e,'range',[lo,hi],d)
def unsupported(x,claim,reason,refs='',fix=''):
 AUD.append(dict(candidate_id=x['candidate_id'],document=x['document'],location=x['location'],paragraph=x['paragraph'],sentence=x['sentence'],claim=claim,source_values='NOT STORED / insufficient scope',source_min=None,source_max=None,source_file_cell=refs,comparison='source_scope',claimed_value='',display_digits='',classification='NOT_VERIFIABLE',proposed_fix=fix or 'Remove or qualify this claim until the named unrounded source is supplied.',scope=reason))

def explicit(x,claim,values,refs,cl='STRICT_OK',fix='',scope=''):
 AUD.append(dict(candidate_id=x['candidate_id'],document=x['document'],location=x['location'],paragraph=x['paragraph'],sentence=x['sentence'],claim=claim,source_values=json.dumps(values,ensure_ascii=False),source_min=None,source_max=None,source_file_cell=refs,comparison='recorded_definition_or_scope',claimed_value='',display_digits='',classification=cl,proposed_fix=fix,scope=scope))

# Additional immutable raw sources.
ALIASES.update(asof=str(R/'causal_features/stage3bc_queue/asof_contrasts.csv'),factors=str(R/'causal_features/adjustment_factor_split_summary.csv'),routerO=str(ROOT/'dataset/audit/experiment5_router_diagnostics/experiment5_router_diagnostics_summary.txt'),T7=str(R/'manuscript18/T7_router_diagnostics.csv'))
router_text=read(ALIASES['routerO']).decode();router_lines=router_text.splitlines()
def ro(field):
 a=[];refs=[]
 for i,line in enumerate(router_lines):
  m=re.match(r'\s*'+re.escape(field)+r':\s*([-+0-9.eE]+)\s*$',line)
  if m:a.append(float(m[1]));refs.append(str(Path(ALIASES['routerO']).relative_to(ROOT))+':'+str(i+1))
 assert a,field
 return np.array(a),refs

def docs_evidence(filename,pattern):
 t=read(ROOT/filename).decode(errors='replace');out=[]
 for i,line in enumerate(t.splitlines()):
  if re.search(pattern,line,re.I):out.append((i+1,line))
 return out

# Precise source selectors.
def pc(k='d_contrast',col='p_mbb_holm',**f):return V(k,col,window=POOL,input_set='C',**f)
def di(col,**f):return V('d_ic',col,window=POOL,**f)
def sg(col,**f):return V('subgroup',col,**f)

# Structural/reference data, extracted from stored reports only.
TECH=[
 ('122–127','revision_work/outputs/fullrun/REPORT_STEP14C_B3.md',r'real sessions range122|Fixed Daily-matched length127'),
 ('26 completed weeks','revision_work/outputs/fullrun/REPORT_STEP14C_B3.md',r'Weekly26|last completed week'),
 ('90 trading-day','revision_audit/AUDIT_REPORT.md',r'90|26'),
 ('at most 100 epochs','revision_audit/AUDIT_REPORT.md',r'100|patience'),
 ('at most 50 epochs','revision_work/outputs/step22a/PROTOCOL_STEP22A.md',r'50|patience'),
 ('up to 50 epochs','revision_work/outputs/step22a/PROTOCOL_STEP22A.md',r'50|patience'),
 ('2015 and 2018','revision_audit/AUDIT_REPORT.md',r'BLNK|zero.volume'),
 ('2,745','revision_audit/AUDIT_REPORT.md',r'2745|2,745'),
 ('2,603','revision_audit/AUDIT_REPORT.md',r'2603|2,603'),
]
# Resolve response comment context from paragraph positions (headings only).
response_context={};ctx='Response letter'
for p in EX['paragraphs']:
 if p['document']!='response':continue
 t=p['text']
 if re.match(r'^(Reviewer\s*#?\d+|Comment\s*#?\d+)',t,re.I):ctx=t[:160]
 response_context[p['paragraph']]=ctx

for x0 in EX['candidates']:
 x=dict(x0);s=x['sentence'];start=len(AUD);cid=x['candidate_id'];pnum=int(re.search(r'\d+',cid)[0]);exclusion=''
 if x['document']=='response':x['location']=response_context.get(x['paragraph'],'Response letter')+'; '+x['location']
 if x['document']=='manuscript' and pnum>=1484:exclusion='Bibliographic page/date range; not an empirical result or statistical bound; retained in coverage inventory.'
 # Scientific exponents, model names and grammatical matches are not numerical bounds.
 if not exclusion and (s in ['As Fixed, plus router 128–32–4','ΔMSE (×10−4)','Test MSE (×10−2)','Mean difference (×10−4)'] or ('Learning rate {' in s) or ('6–1 momentum' in s and not any(w in s for w in ['R²','negative','below']))):exclusion='Architecture dimensions, scientific notation or a named signal; not a numeric bound/range assertion.'
 if not exclusion and not re.search(r'(?:\d\s*[–−-]\s*\d|[<>≤≥≈]|at most|at least|up to|below|above|less than|more than|ranges? from|ranging|approximately|about \d|within.{0,15}\d|between.{0,20}\d)',s,re.I):exclusion='Broad keyword hit without a numeric bound/range (e.g. within-model or within-resolution).'
 if exclusion:COVER.append(dict(candidate_id=cid,location=x['location'],sentence=s,disposition='EXCLUDED',reason=exclusion));continue
 # Table/statistical inequalities.
 if 'p ≥ 0.067' in s:bound(x,'C h5 neural−Static/Adaptive MSE joint Holm p ≥ .067',V('E','p_holm_3h',input_set='C',h=5,metric='squared_loss',method='joint_seed_date'),'ge',.067,3)
 if ('p ≥ 0.16' in s):bound(x,'Frozen Adaptive−constant gate, all metrics/input/backbones',V('a_tests','p_holm_3h',contrast='adaptive-minus-constant_gate'),'ge',.16,2)
 if ('at least 0.12' in s or 'p ≥ 0.12' in s):bound(x,'Core mechanism joint Holm p (MSE and IC)',V('A','p_holm_3h',method='joint_seed_date'),'ge',.12,2)
 if 'p ≥ 0.26' in s:bound(x,'I1–I4 joint MSE/IC Holm p',V('H','p_holm_3h',method='joint_seed_date',contrast=['I1_equal−I0','I2_train_mean−I0','I3_ticker_train_mean−I0','I4_date_cross_mean−I0']),'ge',.26,2)
 if 'p ≥ 0.81' in s:
  bound(x,'T1/T2/T3 Static−Adaptive MSE/IC Holm p',join(V('b_tests','p_holm_3h',contrast='static-minus-adaptive',metric=['mse','rank_ic']),V('b2_tests','p_holm_3h',contrast='static-minus-adaptive',metric=['mse','rank_ic'])),'ge',.81,2)
 if 'p ≥ 0.89' in s:bound(x,'Pooled C Static−Adaptive MSE/IC Holm p',pc(),'ge',.89,2)
 if 'p ≥ 0.70' in s:bound(x,'Held-out C Static−Adaptive MSE/IC Holm p',V('e_route','p_mbb_holm'),'ge',.70,2)
 if 'p ≥ 0.34' in s:bound(x,'Static-R100−Fixed MSE Holm p',V('f_tests','p_holm_3h',contrast='Static-R100-minus-Fixed',metric='mse'),'ge',.34,2)
 if 'p ≥ 0.78' in s:bound(x,'Static-R100−Adaptive-R100 MSE/IC Holm p',V('f_tests','p_holm_3h',contrast='Static-R100-minus-Adaptive-R100'),'ge',.78,2)
 if 'p ≥ 0.21' in s:bound(x,'CS LSTM primary NW/MBB and LSTM superiority joint tests',join(V('g_ic','p_nw_holm',family='BENCHMARK'),V('g_ic','p_mbb_holm',family='BENCHMARK'),V('g_tests','p_holm_3h',family='SECONDARY-C')),'ge',.21,2)
 if 'p ≥ 0.52' in s or s.strip()=='≥ 0.52':bound(x,'O test GRS p (both stored all-seed/ensemble groups)',V('OGRS','p_value',split='test'),'ge',.52,2)
 if s.strip()=='≥ 0.90':bound(x,'O h10 test GRS p',V('OGRS','p_value',split='test',h=10),'ge',.90,2)
 if 'p ≥ 0.57' in s:bound(x,'C ensemble GRS p',V('GRS','p_value',input_set='C',group='ensembles'),'ge',.57,2)
 if 'p between 0.52 and 0.99' in s:
  bound(x,'O GRS lower bound .52',V('OGRS','p_value',split='test'),'ge',.52,2);bound(x,'O GRS upper bound .99',V('OGRS','p_value',split='test'),'le',.99,2)
 if 'p ≤ 0.007' in s:
  bound(x,'SA held-out CW Holm p, both NW and MBB',join(V('e_cw','p_nw_holm'),V('e_cw','p_mbb_holm')),'le',.007,3,scope='No method named in bound; includes NW and MBB.')
 if 'p = 0.015, 0.015 and < 0.001' in s:
  for h in [5,10]:
   e=join(V('d_confirm','p_nw_holm',family='CONFIRMATORY_CW',h=h),V('d_confirm','p_mbb_holm',family='CONFIRMATORY_CW',h=h))
   explicit(x,f'LSTM CW h={h}: displayed p=.015 for unnamed method',e[0].tolist(),'; '.join(e[1]),'WRONG' if np.any(np.round(e[0],3)!=.015) else 'STRICT_OK',fix='Name NW explicitly: h5/h10 NW Holm p=.015; MBB Holm p=.016. Both methods reject at .05. Exact raw values are in the cited files.')
  bound(x,'LSTM CW h20 p<.001, NW and MBB',join(V('d_confirm','p_nw_holm',family='CONFIRMATORY_CW',h=20),V('d_confirm','p_mbb_holm',family='CONFIRMATORY_CW',h=20)),'lt',.001,3)
 if 'p ≤ 0.041' in s:
  bound(x,'SA held-out Rank IC Holm p, NW and MBB',join(V('e_ic','p_nw_holm',setting='T3'),V('e_ic','p_mbb_holm',setting='T3')),'le',.041,3)
 if '≤ 0.006' in s:
  bound(x,'LSTM held-out Rank IC Holm p, NW and MBB',join(V('d_confirm','p_nw_holm',family='CONFIRMATORY_RANK_IC'),V('d_confirm','p_mbb_holm',family='CONFIRMATORY_RANK_IC')),'le',.006,3)
 if s.strip()=='<0.001':
  # Table11 cells P1131 /1153 correspond to h20 CW; both models' methods meet strict bound.
  bound(x,'Table11 h20 CW p<.001',join(V('e_cw','p_nw_holm',h=20),V('d_confirm','p_nw_holm',family='CONFIRMATORY_CW',h=20)),'lt',.001,3)
 if 'all 738' in s or 'all 738' in s.lower() or 'Over all 738' in s or 'Across all 738' in s:
  f=table('B');explicit(x,'738 cells; positive R2; nominal IC; adjusted CW/IC',dict(n=len(f),positive_R2=int((f.r2os_vs_global_train_mean>0).sum()),nominal_IC_MBB_p_lt_05=int((f.rank_ic_p_mbb<.05).sum()),min_CW_NW_Holm=float(f.cw_p_nw_one_sided_holm_all_cells.min()),min_CW_MBB_Holm=float(f.cw_p_mbb_one_sided_holm_all_cells.min()),min_IC_MBB_Holm=float(f.rank_ic_p_mbb_holm_all_cells.min())),str(Path(ALIASES['B']).relative_to(ROOT))+':2–739 [specified columns]',scope='Nominal IC count refers to the stored MBB IC p; no new p-values calculated.')
 if 'Two of the 72' in s:
  e=V('a_tests','p_holm_3h');explicit(x,'2 of72 Holm p<.05',dict(n=len(e[0]),n_below_05=int((e[0]<.05).sum()),significant_values=e[0][e[0]<.05].tolist()),'; '.join(e[1]))
 # Precision / data-audit bounds.
 if ('within 3×10' in s):bound(x,'Timing-check max absolute difference ≤3e−16',V('timing','max_abs_difference'),'le',3e-16,17)
 if '4.2×10' in s:
  bound(x,'Covariance max absolute difference ≤4.2e−17',V('timing','max_abs_difference',quantity='covariance'),'le',4.2e-17,18)
  f=table('timing');ix=f[f.quantity!='covariance'].index;bound(x,'Signals/labels ≤2.8e−16',(f.loc[ix,'max_abs_difference'].to_numpy(float),[ALIASES['timing']+':'+str(i+2) for i in ix]),'le',2.8e-16,17)
 if '2.0×10' in s:
  f=table('precision');printunit=False
  # Source precision diagnostics explicitly exclude step-crossing windows.
  e=V('precision','max',scale='standardized') if False else None
  ff=f[f.scale.eq('standardized')];bound(x,'Non-step standardized feature max |delta| ≤2.0e−5',(ff['max'].to_numpy(),[ALIASES['precision']+':'+str(i+2) for i in ff.index]),'le',2.0e-5,6,fix='Across the snapshots, the maximum standardized difference is below 2.1×10−5 outside the two step windows (documented numerical tolerance).')
 if 'at most 1.9%' in s:bound(x,'As-of-level relative MSE change ≤1.9%',V('asof','asof_relative_mse_change',100),'le',1.9,1,scope='Stored descriptive diagnostic compares archived and 3/1-thread as-of runs; numerical profile is not held fixed.')
 if 'up to 2.5%' in s:
  e=arith('profile','delta_mse_12_minus_3','mse_3',lambda a,b:100*np.abs(a/b));bound(x,'Thread-profile absolute relative MSE change ≤2.5%',e,'le',2.5,1)
 if 'within 3×10⁻⁸' in s or 'within 3×10−8' in s:
  f=table('ratios');c='step_minus_dividend_prediction';bound(x,'Ratio step minus dividend prediction absolute ≤3e−8',(np.abs(f[c].to_numpy()),[ALIASES['ratios']+':'+str(i+2)+' ['+c+']' for i in f.index]),'le',3e-8,9)
 if 'at most 0.012%' in s:bound(x,'New-label absolute MSE change ≤.012%',(np.abs(V('labelmse','relative_mse_change',100)[0]),V('labelmse','relative_mse_change')[1]),'le',.012,3)
 if '1.3×10' in s:
  # all-anchor summary and stored explicit affected list only; use stored delta column without new labels.
  k=str(R/'causal_features/label_comparison_all_anchors.csv');f=table(k);dc=next(c for c in f.columns if c in ['delta','delta_label','label_delta']);printunit=False
  # Affected anchors are explicitly listed by the existing builder.
  af=table(str(R/'causal_features/label_affected_anchors.csv'));keys=[c for c in ['ticker','anchor_date','horizon'] if c in af and c in f];ii=pd.MultiIndex.from_frame(f[keys]);ai=pd.MultiIndex.from_frame(af[keys]);outside=~ii.isin(ai)
  bound(x,'Unaffected label |delta| <1.3e−6',(np.abs(f.loc[outside,dc].to_numpy(float)),[k+' ['+dc+'; all rows excluding stored affected-anchor keys]']),'lt',1.3e-6,7)
  bound(x,'All label |delta| ≤.011',V('labelimpact','max_abs_delta'),'le',.011,3)
 if '99.6–99.9%' in s:
  e=V('labelscale','fraction_abs_label_minus_train_mean_gt_1');ran(x,'Quadratic-region fraction from stored TRAIN-mean residual proxy',(100*(1-e[0]),e[1]),99.6,99.9,1)
  unsupported(x,'Scope: all model training residuals vs stored TRAIN-mean residual proxy','Stored audit is |label−training mean|, not every model/epoch residual.',ALIASES['labelscale'],'With δ=1, 99.6–99.9% of TRAIN labels lie within one unit of their horizon-specific training mean; this is a target-scale diagnostic, not a fitted-residual audit.')
 # Forecast/DA/ranges.
 if 'squared bias' in s and '0.02%' in s:
  # Stored full-precision Step16 reference decomposition.
  f=table('B');z=f[(f.model=='global_train_mean')]
  if len(z):bound(x,'Train-Mean squared-bias share <.02%',(100*z.bias2.to_numpy()/z.mse.to_numpy(),[ALIASES['B']+':'+str(i+2)+' [bias2/mse]' for i in z.index]),'lt',.02,2)
  else:unsupported(x,'Train-Mean squared bias / TEST MSE','No reference rows in Step16 B; locate the stored mean-shift decomposition.','revision_work/outputs/manuscript18/T1_mean_shift.csv')
 if '0.01006–0.01009' in s:
  ran(x,'C h5 neural mean MSE',V('bsum','mse_mean',group='neural',input_set='C',h=5,config='daily_weekly'),.01006,.01009,5)
  ran(x,'C h5 Static/Adaptive mean MSE',V('bsum','mse_mean',group='core',input_set='C',h=5,model=SA),.01027,.01031,5)
  ran(x,'C h5 neural paired seed wins',V('E','seed_wins',input_set='C',h=5,method='joint_seed_date',metric='squared_loss'),4,5,0)
 if '52.5–53.1%' in s:ran(x,'Always-up / positive realized return share',V('da_references','positive_return_share',100),52.5,53.1,1)
 if '45.6%' in s and '49.5%' in s:
  ran(x,'O aggregation DA percent',V('da_table4','da_mean',100,input_set='O',model=AGG),45.6,49.5,1);ran(x,'C aggregation DA percent',V('da_table4','da_mean',100,input_set='C',model=AGG),48.2,50.9,1)
 if '5–22%' in s:
  ran(x,'Train loss % fall',(-V('learn','mean_train_huber_pct_change',mechanism=SA)[0],V('learn','mean_train_huber_pct_change',mechanism=SA)[1]),5,22,0)
  ran(x,'Validation loss % increase',V('learn','mean_val_huber_pct_change',mechanism=SA),1,98,0)
 if 'at most +33%' in s:bound(x,'C validation Huber increase ≤33%',V('b2_lossmean','mean_val_pct_change',setting='T0',model=SA),'le',33,0)
 if 'cross-sectional R² at most 0.14%' in s:
  f=table('B');f=f[f.group.isin(['core','neural']) & f.config.eq('daily_weekly')];bound(x,'Learned per-seed cross-sectional R2 ≤.14%',(100*f.xs_r2.to_numpy(),[ALIASES['B']+':'+str(i+2)+' [xs_r2]' for i in f.index]),'le',.14,2)
 if ('factor of about 5 to 23' in s or 'factor of about 5 to 20' in s):
  models=SA if 'both mechanisms' in s else AGG;f=table('bsum');f=f[f.model.isin(models)&f.group.eq('core')&f.config.eq('daily_weekly')];a=f[f.input_set.eq('O')].sort_values(['model','h']);b=f[f.input_set.eq('C')].sort_values(['model','h']);ran(x,'O/C seed-MSE SD ratio',(a.mse_sd.to_numpy()/b.mse_sd.to_numpy(),[ALIASES['bsum']+':'+str(i+2)+' [mse_sd]' for i in list(a.index)+list(b.index)]),5,20 if '5 to 20' in s else 23,0)
 if 'within 0.6%' in s:
  a=table('A');a=a[a.input_set.eq('C')&a.metric.eq('squared_loss')&a.method.eq('joint_seed_date')&a.contrast.isin(['single−fixed','single−same_patch_fixed','same_patch_fixed−fixed'])]
  bound(x,'C Single/same-patch/Fixed pairwise relative MSE ≤.6%',(100*np.abs(a.theta_hat.to_numpy()/a.second_mean.to_numpy()),[ALIASES['A']+':'+str(i+2)+' [theta_hat/second_mean]' for i in a.index]),'le',.6,1)
 if 'at most 0.5%' in s and 'Static and Fixed' in s:
  a=select('A',contrast='fixed−static',metric='squared_loss',method='joint_seed_date');bound(x,'Static vs Fixed mean MSE percent ≤.5%',(100*np.abs(a.theta_hat.to_numpy()/a.first_mean.to_numpy()),[ALIASES['A']+':'+str(i+2)+' [theta_hat/first_mean]' for i in a.index]),'le',.5,1)
 if '2.5–5.1%' in s:
  a=select('A',contrast='static−adaptive',metric='squared_loss',method='joint_seed_date',input_set='O');ran(x,'O Adaptive MSE excess vs Static percent',(100*(-a.theta_hat.to_numpy())/a.first_mean.to_numpy(),[ALIASES['A']+':'+str(i+2) for i in a.index]),2.5,5.1,1)
  a=select('A',contrast='static−adaptive',metric='squared_loss',method='joint_seed_date',input_set='C');bound(x,'C Static/Adaptive relative MSE difference ≤1.2%',(100*np.abs(a.theta_hat.to_numpy()/a.second_mean.to_numpy()),[ALIASES['A']+':'+str(i+2) for i in a.index]),'le',1.2,1)
 if '0.027–0.049' in s:
  ran(x,'Static-R100 weight MAD',V('f_move','weight_mad_mean',arm='Static-R100'),.027,.049,3);ran(x,'T0 Static weight MAD',V('f_move','weight_mad_mean',arm='Static'),.0003,.0007,4)
 if 'bounds limit' in s and 'at most 0.5%' in s:bound(x,'Static-R100 vs Fixed MSE improvement upper bound ≤.5%',V('f_tests','mse_improvement_upper95_pct',contrast='Static-R100-minus-Fixed',metric='mse'),'le',.5,1)
 if 'within 2.1%' in s:
  f=select('bsum',group='B3',input_set='C');a=[]
  for _,q in f.groupby(['model','h']):a.append(100*(q.mse_mean.max()/q.mse_mean.min()-1))
  bound(x,'C matched three configurations max/min MSE spread ≤2.1%',(a,[ALIASES['bsum']+' [B3 C; group by model,h; max/min]']),'le',2.1,1)
 if '0.005–0.020' in s:ran(x,'Tuned frozen Static weight MAD',V('a_router','mean_abs_deviation_mean',variant='static',regime='tuned'),.005,.020,3);ran(x,'Tuned constant-gate weight MAD',V('a_router','mean_abs_deviation_mean',variant='constant_gate',regime='tuned'),.026,.094,3)
 if 'up to 0.21' in s:
  bound(x,'Tuned Adaptive weight MAD ≤.21',V('a_router','mean_abs_deviation_mean',variant='adaptive',regime='tuned'),'le',.21,2);bound(x,'Base Adaptive MAD ≤.10',V('a_router','mean_abs_deviation_mean',variant='adaptive',regime='Step16_lr1e-4'),'le',.10,2)
  e=arith('a_tuned_base','mse_mean_tuned','mse_mean_Step16',lambda a,b:100*(a/b-1),variant='adaptive');ran(x,'Tuned/base Adaptive MSE % change',e,-1.6,3.5,1)
 if 'at most 0.33%' in s:
  f=select('a_summary',input_set='O',backbone='S1',variant=['static','fixed']);a=f[f.variant.eq('static')].sort_values('h');b=f[f.variant.eq('fixed')].sort_values('h');bound(x,'OS1 Static−Fixed absolute MSE% difference',(100*np.abs(a.mse_mean.to_numpy()/b.mse_mean.to_numpy()-1),[ALIASES['a_summary']+':'+str(i+2) for i in f.index]),'le',.33,2)
  if 'in this setting' in s:explicit(x,'Scope of .33% limit',[],ALIASES['a_summary'],'WRONG',fix='With price-level inputs and Fixed-trained representations, Static and Fixed differ by at most 0.33%; this bound is not across all input/backbone cells.',scope='Unqualified frozen-primary claim includes C and S2; .33% is OS1-only.')
 if '6–15-fold' in s:ran(x,'Loss interval joint/date width ratio',V('a3joint','ci_width_ratio_joint_over_date_only',metric='loss'),6,15,0)
 if '1.7–2.1' in s:
  a=select('bsum',input_set='O',group='core',model='adaptive').sort_values('h');b=select('bsum',input_set='O',group='core',model='static').sort_values('h');ran(x,'O Adaptive/Static seed MSE SD ratio',(a.mse_sd.to_numpy()/b.mse_sd.to_numpy(),[ALIASES['bsum']+':'+str(i+2) for i in list(a.index)+list(b.index)]),1.7,2.1,1)
 if '60–190' in s:unsupported(x,'Approximate paired-run power requirement 60–190','No stored unrounded power calculation located; no new power analysis permitted.','revision_work/outputs/skill_analysis/a3_original/a3_seed_level_tests.csv','Remove the numerical 60–190 claim or supply the saved power calculation, its target power and assumptions.')
 if '3.2–5.2%' in s:
  ran(x,'O Adaptive ensemble MSE reduction %',V('ensemble','ensemble_mse_gain_pct',100,version='a_all_seeds',mechanism='adaptive'),3.2,5.2,1);ran(x,'O Static ensemble MSE reduction %',V('ensemble','ensemble_mse_gain_pct',100,version='a_all_seeds',mechanism='static'),1.8,2.7,1)
 if 'at most 1.0%' in s:
  bound(x,'T1/T2/T3 Adaptive MSE improvement bound ≤1%',join(V('c_settings','adaptive_mse_improvement_upper95_pct',contrast='static-minus-adaptive'),V('b2_bounds','adaptive_mse_improvement_upper95_pct_second_mse',contrast='static-minus-adaptive',metric='mse')),'le',1.,1)
 if '1.5–1.9%' in s:
  for model,lo,hi in [('static',1.5,1.9),('adaptive',.8,2.1),('lstm',.4,.9)]:
   a,r=V('b2_summary','mse_mean',setting='T0',model=model);b,t=V('b2_summary','mse_mean',setting='T3',model=model);ran(x,'T3 MSE reduction % '+model,(100*(1-b/a),r+t),lo,hi,1)
  bound(x,'T3 Static/Adaptive MSE difference %',ratio_models('b2_summary','mse_mean','static','adaptive',setting='T3'),'le',.3,1)
  for setting,lo,hi in [('T0',3,33),('T3',2,5)]:ran(x,'Validation loss rise % '+setting,V('b2_lossmean','mean_val_pct_change',setting=setting,model=SA),lo,hi,0)
 if 'by 0.4–2.1%' in s:
  a,r=V('b2_summary','mse_mean',setting='T0');b,t=V('b2_summary','mse_mean',setting='T3');ran(x,'All three models T3 MSE reduction %',(100*(1-b/a),r+t),.4,2.1,1)
 if '−1.2% to −0.6%' in s:ran(x,'T3 Static/Adaptive R2 percent',V('b2_summary','r2os_mean',100,setting='T3',model=SA),-1.2,-.6,1)
 if '0.010–0.014' in s:
  ran(x,'T0 h5 mean IC',V('b2_summary','rank_ic_mean',setting='T0',h=5),.010,.014,3);ran(x,'T2 h5 mean IC',V('b2_summary','rank_ic_mean',setting='T2',h=5),.027,.032,3);ran(x,'T3 Static/Adaptive IC',V('b2_summary','rank_ic_mean',setting='T3',model=SA),.010,.026,3)
 if '0.975–0.997' in s:ran(x,'C W2 Static/Adaptive relative MSE h5/20',(1-V('d_summary','r2os_train_mean_mean',window='W2',input_set='C',model=SA,h=[5,20])[0],V('d_summary','r2os_train_mean_mean',window='W2',input_set='C',model=SA,h=[5,20])[1]),.975,.997,3)
 if '1.41–1.49' in s:
  for model,lo,hi,d in [('adaptive',1.41,1.49,2),('ridge',4.7,10.3,1)]:
   a,r=V('d_summary','r2os_train_mean_mean',window='W1',input_set='O',model=model);ran(x,'O W1 '+model+' relative MSE',(1-a,r),lo,hi,d)
 if '0.98 to 1.05' in s:
  a,r=V('d_summary','r2os_train_mean_mean',input_set='C',model=SA,window=['W1','W2','W3']);ran(x,'C rolling S/A relative MSE',(1-a,r),.98,1.05,2)
  a,r=V('d_summary','r2os_train_mean_mean',input_set='O',model='static',window=['W1','W2','W3']);ran(x,'O rolling Static relative MSE',(1-a,r),1.06,1.17,2)
 if '0.011–0.078' in s:ran(x,'C windowwise S/A IC',V('d_ic','mean',window=['W1','W2','W3'],input_set='C',model=SA),.011,.078,3)
 if '0.94–0.96' in s:
  # subgroup name is stored verbatim.
  low=next(z for z in table('subgroup').subgroup.unique() if 'low' in str(z).lower());ran(x,'C W2 low-vol S/A relative MSE',sg('relative_mse_mean',window='W2',input_set='C',model=SA,subgroup=low),.94,.96,2);ran(x,'C W2 low-vol S/A Rank IC',sg('rank_ic_mean',window='W2',input_set='C',model=SA,subgroup=low),.09,.19,2);ran(x,'C W2 low-vol Zero relative MSE',sg('relative_mse_mean',window='W2',input_set='C',model='zero',subgroup=low),.92,.98,2)
 if '2.4–4.0' in s:
  for c,lo,hi,d in [('relative_mse_mean',2.4,4.,1),('mean_forecast_bias_pooled_seeds',.04,.11,2)]:ran(x,'O W1 Adaptive BEP/CWEN/NEE '+c,sg(c,window='W1',input_set='O',model='adaptive',subgroup=['BEP','CWEN','NEE']),lo,hi,d)
 if '26–57' in s:
  ran(x,'O W1 Ridge PLUG relative MSE',sg('relative_mse_mean',window='W1',input_set='O',model='ridge',subgroup='PLUG'),26,57,0);ran(x,'O W1 Ridge PLUG forecast bias, printed range descending',sg('mean_forecast_bias_pooled_seeds',window='W1',input_set='O',model='ridge',subgroup='PLUG'),-.42,-1.57,2)
 if '0.013–0.018' in s:ran(x,'Pooled C Adaptive IC improvement upper95',pc(col='adaptive_improvement_upper95',metric='rank_ic'),.013,.018,3)
 if '0.026–0.051' in s:
  ran(x,'Pooled C S/A IC',di('mean',input_set='C',model=SA),.026,.051,3)
  if 'Newey–West' in s:ran(x,'Pooled C raw NW p',di('p_nw',input_set='C',model=SA),.010,.048,3)
  else:
   ran(x,'Pooled C unqualified raw p includes NW/MBB',join(di('p_nw',input_set='C',model=SA),di('p_mbb',input_set='C',model=SA)),.010,.048,3)
   explicit(x,'No adjusted significance, unnamed method',dict(NW_min=float(di('p_nw_holm',input_set='C',model=SA)[0].min()),MBB_min=float(di('p_mbb_holm',input_set='C',model=SA)[0].min())),str(Path(ALIASES['d_ic']).relative_to(ROOT)),'WRONG',fix='No pooled C Rank IC is significant by the primary NW Holm criterion (minimum p=0.059); MBB is a robustness check and its minimum Holm p=0.043. State the disagreement explicitly.')
 if '0.042–0.087' in s:
  ran(x,'Pooled C Ridge IC',di('mean',input_set='C',model='ridge'),.042,.087,3);a,r=di('mean',input_set='O',model=SA);bound(x,'Pooled O |IC|<.012',(np.abs(a),r),'lt',.012,3)
 if '0.061–0.064' in s:ran(x,'Pooled O S/A MSE Holm p',V('d_contrast','p_mbb_holm',window=POOL,input_set='O',metric='mse'),.061,.064,3)
 if '0.03–2.0%' in s:
  ran(x,'Held-out S/A pooled ensemble R2 percent',V('e_ensemble','r2os_train_mean',100,setting='T3',window='W1+W2',model=SA),.03,2.0,[2,1])
 if '1.8–4.7%' in s:ran(x,'Held-out S/A W2 ensemble R2 percent',V('e_ensemble','r2os_train_mean',100,setting='T3',window='W2',model=SA),1.8,4.7,1)
 if '−0.4% to −2.4%' in s:ran(x,'Held-out S/A pooled zero-relative R2, printed descending',V('e_ensemble','r2os_zero',100,setting='T3',window='W1+W2',model=SA),-.4,-2.4,1)
 if '0.031–0.075' in s:
  ran(x,'Held-out T3 S/A IC',V('e_ic','mean',setting='T3'),.031,.075,3);ran(x,'Held-out T0 S/A IC',V('e_ic','mean',setting='T0'),.031,.065,3)
 if 'at most 2.7%' in s:
  e=join(V('c_end','adaptive_mse_improvement_upper95_pct',input_set='C'),V('c_settings','adaptive_mse_improvement_upper95_pct',contrast='static-minus-adaptive'),V('b2_bounds','adaptive_mse_improvement_upper95_pct_second_mse',contrast='static-minus-adaptive',metric='mse'),pc(col='adaptive_mse_improvement_upper95_pct',metric='mse'),V('e_route','adaptive_mse_improvement_upper95_pct',metric='mse'));bound(x,'C MSE improvement upper95 bound across stated scopes ≤2.7%',e,'le',2.7,1);bound(x,'C pooled bound ≤.6%',pc(col='adaptive_mse_improvement_upper95_pct',metric='mse'),'le',.6,1)
 if '0.013–0.040' in s:ran(x,'C Adaptive IC improvement bounds W3/T1/T2/T3/pooled',join(V('c_end','adaptive_rank_ic_improvement_upper95',input_set='C'),V('c_settings','adaptive_rank_ic_improvement_upper95',contrast='static-minus-adaptive'),V('b2_bounds','adaptive_improvement_upper95_raw',contrast='static-minus-adaptive',metric='rank_ic'),pc(col='adaptive_improvement_upper95',metric='rank_ic')),.013,.040,3)
 # Portfolio stored ranges (C gamma5 ensembles, all seven models).
 def gp(c,h):return V('G',c,input_set='C',split='test',cost_bp=10,layer='L2_g5',strategy=[m+'_ens' for m in ['single','fixed','static','adaptive','lstm','vanilla','swim']],h=h)
 if '1.09–1.46' in s:ran(x,'C h10 gamma5 ensemble Sharpe',gp('sharpe',10),1.09,1.46,2);ran(x,'C h10 maximum drawdown, printed descending',gp('max_drawdown',10),-.16,-.24,2)
 if '0.91–1.28' in s:
  ran(x,'C h5 gamma5 ensemble Sharpe',gp('sharpe',5),.91,1.28,2);ran(x,'C h20 gamma5 ensemble Sharpe',gp('sharpe',20),.67,1.13,2)
 if '0.59–0.70' in s:ran(x,'C h10 gamma5 mean turnover',gp('avg_turnover_ex_initial',10),.59,.70,2)
 if 'within sampling error' in s:unsupported(x,'Sharpe differences lie within sampling error','No saved sampling distribution, interval or test of the stated Sharpe differences was located; descriptive results do not establish this.',ALIASES['G'],'With about 19 months of TEST data, the Sharpe comparisons are descriptive; no inference about their differences is reported.')
 # Exact O router diagnostics, unrounded original text.
 for phrase,field,lo,hi,d in [('0.0434 to 0.0583','sample_dependence_index',.0434,.0583,4),('0.0108 to 0.0151','within_ticker_temporal_tv',.0108,.0151,4),('0.0431 to 0.0578','within_date_crosssectional_tv',.0431,.0578,4),('0.9619 to 0.9767','normalized_entropy_mean',.9619,.9767,4),('3.80 to 3.87','effective_scales_mean',3.80,3.87,2),('0.3275 to 0.3598','max_weight_mean',.3275,.3598,4),('0.1510 to 0.1760','cross_seed_mean_pairwise_seed_tv',.1510,.1760,4),('0.4592 to 0.5611','cross_seed_dominant_scale_agreement',.4592,.5611,4),('−0.1016 to 0.0612','cross_seed_sample_spearman',-.1016,.0612,4),('−0.1220 to 0.0512','cross_seed_date_spearman',-.1220,.0512,4)]:
  if phrase in s:ran(x,'O router '+field,ro(field),lo,hi,d)
 # Structural records with literal source evidence.
 if len(AUD)==start:
  for phrase,file,pattern in TECH:
   if phrase in s and (ROOT/file).exists():
    a=docs_evidence(file,pattern)
    if a:explicit(x,'Recorded input/training design',a,'; '.join(file+':'+str(i) for i,t in a),scope='Recorded design, not a newly evaluated model statistic.');break
 if len(AUD)==start:
  unsupported(x,'Unresolved numeric range/bound','Candidate retained explicitly; no strict pass inferred from rounded manuscript values.')
 COVER.append(dict(candidate_id=cid,location=x['location'],sentence=s,disposition='AUDITED',reason=str(len(AUD)-start)+' atomic claim(s)'))

csvwrite('AUDIT_BOUNDS_STEP23B.csv',pd.DataFrame(AUD));csvwrite('BOUND_COVERAGE_STEP23B.csv',pd.DataFrame(COVER));checkpoint_sources('SOURCES_BOUNDS.csv')
print('Audit rows',len(AUD),pd.Series([r['classification'] for r in AUD]).value_counts().to_dict())
for r in AUD:
 if r['classification']=='NOT_VERIFIABLE':print('UNRESOLVED',r['candidate_id'],r['sentence'][:180])
