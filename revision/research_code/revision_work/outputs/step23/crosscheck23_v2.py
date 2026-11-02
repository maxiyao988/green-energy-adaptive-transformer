"""Manuscript rounding/lookup checks only; no new tests or bootstrap."""
import sys,json,math
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.append(str(Path.cwd()/'revision_work/outputs/_deps_pyarrow15'))
import pandas as pd,numpy as np
D=Path('revision_work/outputs/step23');raw=json.loads((D/'catalog23.json').read_text());loc=json.loads((D/'locations23_v2.json').read_text())
T={k:pd.DataFrame(v['rows'],columns=v['columns']) for k,v in raw.items()};checks=[];computed=[]
def get(k,c,**f):
 x=T[k]
 for n,v in f.items():x=x[x[n].isin(v) if isinstance(v,list) else x[n].eq(v)]
 assert len(x)>0,(k,c,f)
 return x[c],'; '.join(loc.get(f'{k}|{i}|{c}','NOT LOCATED') for i in x.index)
def log(id,claim,vals,refs,condition,rounded=None,operation='lookup and manuscript rounding'):
 a=np.asarray(vals).reshape(-1).tolist();a=[v.item() if hasattr(v,'item') else v for v in a]
 checks.append(dict(item_id=id,manuscript_value=claim,workbook_value=json.dumps(a,ensure_ascii=False,allow_nan=False),rounded_value=json.dumps(rounded,ensure_ascii=False) if rounded is not None else '',sheet_cells=refs,status='MATCH' if condition else 'MISMATCH',operation=operation))
def check(id,k,c,expected,op='eq',digits=3,scale=1,**f):
 a,refs=get(k,c,**f);v=a.to_numpy(float)*scale;w=np.round(v,digits) if digits is not None else v
 if op=='eq':ok=bool(np.all(w==expected));claim=str(expected)
 elif op=='lt':ok=bool(np.all(w<expected));claim='< '+str(expected)
 elif op=='le':ok=bool(np.all(w<=expected));claim='≤ '+str(expected)
 elif op=='ge':ok=bool(np.all(w>=expected));claim='≥ '+str(expected)
 elif op=='range':ok=bool(np.array_equal([w.min(),w.max()],expected));claim=str(expected[0])+' to '+str(expected[1])
 else:raise ValueError(op)
 log(id,claim,v,refs,ok,w.tolist(),f'lookup;×{scale};round({digits});{op}')
def calc(id,claim,args,fn,test,digits=None):
 series=[];refs=[]
 for k,c,f in args:
  a,r=get(k,c,**f);series.append(a.to_numpy());refs.append(r)
 values=np.asarray(fn(*series)).reshape(-1);w=np.round(values.astype(float),digits) if digits is not None else values
 ok=bool(test(w));operation='Descriptive cross-check arithmetic on stored summary values only;no evaluation/test'
 log(id,claim,values,'; '.join(refs),ok,w.tolist(),operation)
 computed.append(dict(item=id,operation=operation,formula_or_claim=claim,value=json.dumps(values.tolist()),source_cells='; '.join(refs)))
def arg(k,c,**f):return (k,c,f)
def one(k,c,**f):return get(k,c,**f)[0].iloc[0]
sa=['static','adaptive'];agg=['single','fixed','same_patch_fixed','static','adaptive']
# Table11, every listed numeric cell (W1/W2,W1+W2;T3).
numbers={'static':{5:[-.85,2.02,.61,.003,-.36,.031],10:[-2.50,3.67,.73,.006,-1.28,.072],20:[-.71,4.65,1.92,.003,-2.36,.075]},'adaptive':{5:[-1.84,1.83,.03,.003,-.95,.046],10:[-2.03,3.73,.98,.003,-1.02,.069],20:[.94,3.15,2.02,'<.001',-2.25,.057]},'lstm':{5:[-.99,1.73,.39,.015,-.58,.049],10:[-.97,1.79,.47,.015,-1.54,.077],20:[-.26,2.83,1.25,'<.001',-3.05,.078]}}
for model,rows in numbers.items():
 for h,values in rows.items():
  for i,window in enumerate(['W1','W2','W1+W2']):check(f'T11.{model}.{h}.R2.{window}','e_ensemble','r2os_train_mean',values[i],digits=2,scale=100,model=model,h=h,setting='T3',window=window)
  k='d_confirm' if model=='lstm' else 'e_cw';f=dict(h=h,model=model)
  if model=='lstm':f['family']='CONFIRMATORY_CW'
  check(f'T11.{model}.{h}.CW',k,'p_nw_holm',.001 if isinstance(values[3],str) else values[3],op='lt' if isinstance(values[3],str) else 'eq',digits=None if isinstance(values[3],str) else 3,**f)
  check(f'T11.{model}.{h}.R2zero','e_ensemble','r2os_zero',values[4],digits=2,scale=100,model=model,h=h,setting='T3',window='W1+W2')
  check(f'T11.{model}.{h}.RankIC','e_summary','mean_rank_ic_mean',values[5],digits=3,model=model,h=h,setting='T3',window='W1+W2')
check('T11.note.SA.IC.Holm','e_ic','p_nw_holm',.041,op='le',setting='T3')
check('T11.note.LSTM.IC.Holm','d_confirm','p_nw_holm',.006,op='le',family='CONFIRMATORY_RANK_IC')
for k,f in [('e_cw',{}),('d_confirm',dict(family='CONFIRMATORY_CW'))]:calc('T11.note.CW.agreement.'+k,'NW and MBB reject at .05 in every same cell',[arg(k,'p_nw_holm',**f),arg(k,'p_mbb_holm',**f)],lambda a,b:(a.astype(float)<.05)==(b.astype(float)<.05),lambda v:v.all())
# Alternative settings.
for k in ['b_tests','b2_tests']:check('4.4.alt.SA.Holm.'+k,k,'p_holm_3h',.81,op='ge',digits=2,contrast='static-minus-adaptive',metric=['mse','rank_ic'])
check('4.4.alt.bounds.T1T2','c_settings','adaptive_mse_improvement_upper95_pct',1.,op='le',digits=1,contrast='static-minus-adaptive')
check('4.4.alt.bounds.T3','b2_bounds','adaptive_mse_improvement_upper95_pct_second_mse',1.,op='le',digits=1,contrast='static-minus-adaptive',metric='mse')
for k,settings in [('b_tests',['T1']),('b2_tests',['T3'])]:check('4.4.alt.LSTMwins.'+k,k,'seed_losses',5,digits=0,setting=settings,contrast='adaptive-minus-lstm',metric='mse')
for model,limits in [('static',[1.5,1.9]),('adaptive',[.8,2.1]),('lstm',[.4,.9])]:
 calc('4.4.alt.MSEreduction.'+model,f'T3 vsT0 reduction range {limits} percent',[arg('b2_summary','mse_mean',setting='T0',model=model),arg('b2_summary','mse_mean',setting='T3',model=model)],lambda a,b:100*(1-b.astype(float)/a.astype(float)),lambda v:np.array_equal([v.min(),v.max()],limits),digits=1)
calc('4.4.alt.SAdiff.T3','abs(Static/Adaptive-1) ≤0.3%', [arg('b2_summary','mse_mean',setting='T3',model='static'),arg('b2_summary','mse_mean',setting='T3',model='adaptive')],lambda a,b:100*np.abs(a.astype(float)/b.astype(float)-1),lambda v:(v<=.3).all(),digits=1)
for setting,limits in [('T0',[3.,33.]),('T3',[2.,5.])]:check('4.4.alt.valrise.'+setting,'b2_lossmean','mean_val_pct_change',limits,op='range',digits=0,setting=setting,model=sa)
for h,m,e in [(5,0.,.07),(10,.03,.09),(20,.25,.32)]:
 check('4.4.alt.LSTM.R2.'+str(h),'b2_summary','r2os_mean',m,digits=2,scale=100,setting='T3',model='lstm',h=h)
 check('4.4.alt.LSTM.ensembleR2.'+str(h),'b2_ensemble','r2os',e,digits=2,scale=100,setting='T3',model='lstm',h=h)
check('4.4.alt.SA.R2','b2_summary','r2os_mean',[-1.2,-.6],op='range',digits=1,scale=100,setting='T3',model=sa)
for setting,limits in [('T0',[.010,.014]),('T2',[.027,.032])]:check('4.4.alt.h5.IC.'+setting,'b2_summary','rank_ic_mean',limits,op='range',digits=3,setting=setting,h=5)
check('4.4.alt.T3.SA.IC','b2_summary','rank_ic_mean',[.010,.026],op='range',digits=3,setting='T3',model=sa)
# Pooled rolling windows.
po=dict(window='W1+W2+W3');cp=dict(input_set='C',**po);op=dict(input_set='O',**po)
check('4.4.pool.C.SA.p','d_contrast','p_mbb_holm',.89,op='ge',digits=2,**cp)
for h,v in [(5,.56),(10,.48),(20,.37)]:check('4.4.pool.C.MSEbound.'+str(h),'d_contrast','adaptive_mse_improvement_upper95_pct',v,digits=2,h=h,metric='mse',**cp)
check('4.4.pool.C.ICbound','d_contrast','adaptive_improvement_upper95',.018,op='le',digits=3,metric='rank_ic',**cp)
check('4.4.pool.C.IC','d_ic','mean',[.026,.051],op='range',digits=3,model=sa,**cp)
check('4.4.pool.C.IC.rawp','d_ic','p_nw',[.010,.048],op='range',digits=3,model=sa,**cp)
for c,v in [('p_nw_holm',.059),('p_mbb_holm',.043)]:calc('4.4.pool.C.min.'+c,f'min={v}',[arg('d_ic',c,model=sa,**cp)],lambda a:[a.astype(float).min()],lambda a:a[0]==v,digits=3)
check('4.4.pool.C.RidgeIC','d_ic','mean',[.042,.087],op='range',digits=3,model='ridge',**cp)
calc('4.4.pool.O.absIC','pooled O absIC≤.011',[arg('d_ic','mean',model=sa,**op)],lambda a:np.abs(a.astype(float)),lambda a:(a<=.011).all(),digits=3)
check('4.4.pool.O.Holm','d_ic','p_nw_holm',1.,digits=2,model=sa,**op)
check('4.4.pool.O.MSEp','d_contrast','p_mbb_holm',[.061,.064],op='range',digits=3,metric='mse',**op)
calc('4.4.pool.C.W2concentration','each W2 IC exceeds its W1 and W3 counterpart',[arg('d_ic','mean',window=w,input_set='C',model=sa) for w in ['W1','W2','W3']],lambda a,b,c:(b.astype(float)>a.astype(float))&(b.astype(float)>c.astype(float)),lambda a:a.all())
# Held-out windows.
check('4.4.heldout.CW.SA','e_cw','p_nw_holm',.007,op='le',digits=3)
check('4.4.heldout.CW.LSTM','d_confirm','p_nw_holm',.015,op='le',digits=3,family='CONFIRMATORY_CW')
calc('4.4.heldout.R2pool','pooled ensemble R2 range .03% to2.0%,minimum2 decimals/maximum1 decimal',[arg('e_ensemble','r2os_train_mean',setting='T3',window='W1+W2')],lambda a:[100*a.astype(float).min(),100*a.astype(float).max()],lambda a:round(a[0],2)==.03 and round(a[1],1)==2.0)
check('4.4.heldout.W2.R2','e_ensemble','r2os_train_mean',[1.8,4.7],op='range',digits=1,scale=100,setting='T3',window='W2',model=sa)
calc('4.4.heldout.W1.negative','5 of6 Static/Adaptive W1 R2 negative',[arg('e_ensemble','r2os_train_mean',setting='T3',window='W1',model=sa)],lambda a:[(a.astype(float)<0).sum()],lambda a:a[0]==5)
check('4.4.heldout.SA.zeroR2','e_ensemble','r2os_zero',[-2.4,-.4],op='range',digits=1,scale=100,setting='T3',window='W1+W2',model=sa)
check('4.4.heldout.individual.h5','e_summary','r2os_train_mean_mean',0.,op='lt',digits=None,setting='T3',window='W1+W2',h=5,model=sa)
check('4.4.heldout.SA.Holm','e_route','p_mbb_holm',.71,op='ge',digits=2)
calc('4.4.heldout.T3T0negative','5 of6 T3-minus-T0 differences negative',[arg('e_setting','estimate')],lambda a:[(a.astype(float)<0).sum()],lambda a:a[0]==5)
check('4.4.heldout.Adaptive10.effect','e_setting','estimate_pct',-3.7,digits=1,model='adaptive',h=10)
check('4.4.heldout.Adaptive10.p','e_setting','p_mbb_holm',.008,digits=3,model='adaptive',h=10)
check('4.4.heldout.IC.T3','e_ic','mean',[.031,.075],op='range',digits=3,setting='T3')
check('4.4.heldout.IC.T3.p','e_ic','p_nw_holm',.041,op='le',digits=3,setting='T3')
check('4.4.heldout.IC.T0','e_ic','mean',[.031,.065],op='range',digits=3,setting='T0')
bargs=[arg('c_end','adaptive_mse_improvement_upper95_pct',input_set='C'),arg('c_settings','adaptive_mse_improvement_upper95_pct',contrast='static-minus-adaptive'),arg('b2_bounds','adaptive_mse_improvement_upper95_pct_second_mse',contrast='static-minus-adaptive',metric='mse'),arg('d_contrast','adaptive_mse_improvement_upper95_pct',metric='mse',**cp),arg('e_route','adaptive_mse_improvement_upper95_pct',metric='mse')]
calc('5.2.max.AdaptiveMSEbound','C W3T0/T1/T2/T3/pooled/heldout bound≤2.7%',bargs,lambda *a:np.concatenate([v.astype(float) for v in a]),lambda a:(a<=2.7).all(),digits=1)
check('5.2.pooled.bound','d_contrast','adaptive_mse_improvement_upper95_pct',.6,op='le',digits=1,metric='mse',**cp)
# Larger scale learning rates.
check('4.2.F.R100MAD','f_move','weight_mad_mean',[.027,.049],op='range',digits=3,arm='Static-R100')
check('4.2.F.T0MAD','f_move','weight_mad_mean',.0007,op='le',digits=4,arm='Static')
for h,v in [(5,-.12),(10,.30),(20,.11)]:check('4.2.F.R100Fixed.'+str(h),'f_tests','estimate_pct',v,digits=2,contrast='Static-R100-minus-Fixed',metric='mse',h=h)
check('4.2.F.R100Fixed.p','f_tests','p_holm_3h',.35,op='ge',digits=2,contrast='Static-R100-minus-Fixed',metric='mse')
check('4.2.F.R100Fixed.bound','f_tests','mse_improvement_upper95_pct',.45,op='le',digits=2,contrast='Static-R100-minus-Fixed',metric='mse')
a,refs=get('f_gate','status',arm='Static-R10',h=[5,10]);log('4.2.F.R10.manipulation','FAIL at5/10',a,refs,a.eq('FAIL').all())
check('4.2.F.R10.h20.p','f_tests','p_holm_3h',.93,digits=2,contrast='Static-R10-minus-Fixed',metric='mse',h=20)
check('4.2.F.AdaptiveSD','f_move','within_model_sd_mean',.022,digits=3,arm='Adaptive',h=5,branch='daily')
check('4.2.F.AdaptiveR100SD','f_move','within_model_sd_mean',.0002,digits=4,arm='Adaptive-R100',h=5,branch='daily')
check('4.2.F.SR100AR100p','f_tests','p_holm_3h',.79,op='ge',digits=2,contrast='Static-R100-minus-Adaptive-R100')
# Frozen representations.
check('4.3.AC.allp','a_tests','p_holm_3h',.16,op='ge',digits=2,contrast='adaptive-minus-constant_gate')
for h,v in [(5,.3),(10,.3),(20,4.9)]:calc('4.3.OS1.AminusS.'+str(h),f'Adaptive-minus-Static MSE {v}%', [arg('a_summary','mse_mean',input_set='O',backbone='S1',variant='adaptive',h=h),arg('a_summary','mse_mean',input_set='O',backbone='S1',variant='static',h=h)],lambda a,b:100*(a.astype(float)/b.astype(float)-1),lambda a:a[0]==v,digits=1)
check('4.3.OS1.h20.Awins','a_tests','seed_losses',0,digits=0,input_set='O',backbone='S1',h=20,contrast='static-minus-adaptive',metric='mse')
check('4.3.OS1.h20.p','a_tests','p_holm_3h',.28,digits=2,input_set='O',backbone='S1',h=20,contrast='static-minus-adaptive',metric='mse')
for model,v in [('fixed',.03738),('static',.03714),('adaptive',.03681),('constant_gate',.03664)]:check('4.3.CS1.h20.'+model,'a_summary','mse_mean',v,digits=5,input_set='C',backbone='S1',h=20,variant=model)
calc('4.3.twoof72','2 of72 Holm p<.05',[arg('a_tests','p_holm_3h')],lambda a:[len(a),(a.astype(float)<.05).sum()],lambda a:np.array_equal(a,[72,2]))
check('4.3.OS2.h5.MSEp','a_tests','p_holm_3h',.042,digits=3,input_set='O',backbone='S2',h=5,contrast='static-minus-adaptive',metric='mse')
for model,v in [('adaptive',.028),('static',-.002)]:check('4.3.CS2.h5.IC.'+model,'a_summary','rank_ic_mean',v,digits=3,input_set='C',backbone='S2',h=5,variant=model)
check('4.3.CS2.h5.ICp','a_tests','p_holm_3h',.041,digits=3,input_set='C',backbone='S2',h=5,contrast='static-minus-adaptive',metric='rank_ic')
for input_,metric,v in [('O','mse',.35),('C','rank_ic',.30)]:check('4.3.S2.ACp.'+input_,'a_tests','p_holm_3h',v,digits=2,input_set=input_,backbone='S2',h=5,contrast='adaptive-minus-constant_gate',metric=metric)
for variant,limits in [('static',[.005,.020]),('constant_gate',[.026,.094])]:check('4.3.MAD.'+variant,'a_router','mean_abs_deviation_mean',limits,op='range',digits=3,variant=variant,regime='tuned')
calc('4.3.AdaptiveMAD.max','tuned Adaptive max MAD=.21',[arg('a_router','mean_abs_deviation_mean',variant='adaptive',regime='tuned')],lambda a:[a.astype(float).max()],lambda a:a[0]==.21,digits=2)
check('4.3.AdaptiveMAD.base','a_router','mean_abs_deviation_mean',.10,op='le',digits=2,variant='adaptive',regime='Step16_lr1e-4')
calc('4.3.Ccollapse','6 of12 branch-cells collapsed,in2to5 seeds',[arg('a_router','n_constant_seeds',variant='adaptive',regime='tuned',input_set='C')],lambda a:[len(a),int((a.astype(float)>0).sum()),a[a.astype(float)>0].astype(float).min(),a.astype(float).max()],lambda a:np.array_equal(a,[12,6,2,5]))
calc('4.3.tuning.AdaptiveMSE','tuned/base change -1.6 to+3.5%', [arg('a_tuned_base','mse_mean_tuned',variant='adaptive'),arg('a_tuned_base','mse_mean_Step16',variant='adaptive')],lambda a,b:100*(a.astype(float)/b.astype(float)-1),lambda a:np.array_equal([a.min(),a.max()],[-1.6,3.5]),digits=1)
calc('4.3.OS1.StaticFixed','abs(Static/Fixed-1)≤.33%', [arg('a_summary','mse_mean',input_set='O',backbone='S1',variant='static'),arg('a_summary','mse_mean',input_set='O',backbone='S1',variant='fixed')],lambda a,b:100*np.abs(a.astype(float)/b.astype(float)-1),lambda a:(a<=.33).all(),digits=2)
checks.append(dict(item_id='4.3.manuscript.Table7.full',manuscript_value='Every manuscript Table7 cell',workbook_value='PENDING: manuscript Table7 not provided;user authorized deferral',rounded_value='',sheet_cells="'Table S6'!A1",status='PENDING',operation='No inferred manuscript values'))
# Direction accuracy.
check('4.1.positiveShare','da_references','positive_return_share',[52.5,53.1],op='range',digits=1,scale=100)
for input_,limits in [('O',[45.6,49.5]),('C',[48.2,50.9])]:check('4.1.aggregationDA.'+input_,'da_table4','da_mean',limits,op='range',digits=1,scale=100,input_set=input_,model=agg)
check('4.1.CLSTM20DA','da_table4','da_mean',52.9,digits=1,scale=100,input_set='C',model='lstm',h=20)
check('4.1.h20positiveShare','da_references','positive_return_share',52.7,digits=1,scale=100,h=20)
for model,v in [('adaptive',45.6),('static',48.4)]:check('4.1.Oh20DA.'+model,'da_table4','da_mean',v,digits=1,scale=100,input_set='O',h=20,model=model)
calc('4.1.largestSminusADA','largest Static-minus-Adaptive DA difference=2.8pp',[arg('da_pair','estimate_pp',scope='table4',contrast='static-minus-adaptive')],lambda a:[a.astype(float).max()],lambda a:a[0]==2.8,digits=1)
check('4.1.DA.Oh20p','da_pair','p_holm_3h',.49,digits=2,scope='table4',contrast='static-minus-adaptive',input_set='O',h=20)
# Response letter bounds.
for h,v in [(5,.48),(10,.71),(20,2.50)]:check('response.C.T0.MSEbound.'+str(h),'c_end','adaptive_mse_improvement_upper95_pct',v,digits=2,input_set='C',h=h)
calc('response.ICbounds','C Static-minus-Adaptive IC bounds across W3T0,T1/T2/T3,pooled .013 to .040', [arg('c_end','adaptive_rank_ic_improvement_upper95',input_set='C'),arg('c_settings','adaptive_rank_ic_improvement_upper95',contrast='static-minus-adaptive'),arg('b2_bounds','adaptive_improvement_upper95_raw',contrast='static-minus-adaptive',metric='rank_ic'),arg('d_contrast','adaptive_improvement_upper95',metric='rank_ic',**cp)],lambda *a:np.concatenate([v.astype(float) for v in a]),lambda a:np.array_equal([a.min(),a.max()],[.013,.040]),digits=3)
calc('response.T3.MSEreduction','all models T3 reductions .4to2.1%', [arg('b2_summary','mse_mean',setting='T0'),arg('b2_summary','mse_mean',setting='T3')],lambda a,b:100*(1-b.astype(float)/a.astype(float)),lambda a:np.array_equal([a.min(),a.max()],[.4,2.1]),digits=1)
# Step22G addendum. Every block remains EXPLORATORY.
for arm,values in [('Static-CS',[.017,-.002,.019]),('Adaptive-CS',[.008,.006,.012]),('LSTM-CS',[.044,.033,.069])]:
 for h,v in zip([5,10,20],values):check('4.4.G.IC.'+arm+'.'+str(h),'g_summary','mean_rank_ic_mean',v,digits=3,arm=arm,h=h)
for colname in ['p_nw_holm','p_mbb_holm']:check('4.4.G.PRIMARY.'+colname,'g_ic',colname,1.,digits=2,family='PRIMARY')
for fam in ['SECONDARY-A','SECONDARY-B']:check('4.4.G.'+fam+'.allHolm','g_tests','p_holm_3h',1.,digits=2,family=fam)
check('4.4.G.BENCHMARK.NW','g_ic','p_nw_holm',.24,op='ge',digits=2,family='BENCHMARK')
check('4.4.G.BENCHMARK.MBB','g_ic','p_mbb_holm',.24,op='ge',digits=2,family='BENCHMARK')
check('4.4.G.LSTMwins','g_tests','seed_losses',5,digits=0,family='SECONDARY-C')
check('4.4.G.SECONDARYC.Holm','g_tests','p_holm_3h',.22,op='ge',digits=2,family='SECONDARY-C')
check('4.4.G.longshortSharpe','g_portfolio','sharpe',0.,op='lt',digits=None,portfolio='top5_minus_bottom5')
with (D/'CROSSCHECK_STEP23_v2.csv').open('x',newline='') as f:pd.DataFrame(checks).to_csv(f,index=False)
with (D/'COMPUTED_CROSSCHECK_v2.csv').open('x',newline='') as f:pd.DataFrame(computed).to_csv(f,index=False)
text='# BLOCKING\n\nFull manuscript Table7 remains pending,user-approved. Every other listed claim has a lookup/rounding check below;MISMATCH entries require manuscript review. No manuscript was edited.\n\nManuscript precision is applied before comparison. Thus p=.217189 is .22 at2 decimals;an inequality written as p>=.22 here is checked at that displayed precision,not asserted for the unrounded p. Raw underlying values and exact workbook cells are preserved in the CSV.\n\n'
text+='Counts: '+str(pd.Series([r['status'] for r in checks]).value_counts().to_dict())+'\n\n'
text+='| Item | Manuscript | Workbook raw | Rounded | Status | Cells |\n|---|---|---|---|---|---|\n'
for r in checks:text+='| '+' | '.join(str(r[c]).replace('|','/').replace('\n',' ') for c in ['item_id','manuscript_value','workbook_value','rounded_value','status','sheet_cells'])+' |\n'
with (D/'CROSSCHECK_STEP23_v2.md').open('x') as f:f.write(text)
print('CROSSCHECK counts',pd.Series([r['status'] for r in checks]).value_counts().to_dict())
for r in checks:
 if r['status']=='MISMATCH':print(r['item_id'],r['manuscript_value'],r['rounded_value'] or r['workbook_value'])
