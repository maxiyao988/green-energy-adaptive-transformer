"""Compare every numeric manuscript cell in Tables 4,9,10,11 with independent replay."""
from review24_common import *
T={x['table']:x['rows'] for x in js('revision_work/outputs/step23b/MANUSCRIPT_TABLES.json')};rows=[]
def clean(s):return s.replace('−','-').replace('–','-').replace('+','').replace(' ','')
def ck(t,r,c,expected,source):
 got=T[t][r][c];rows.append(dict(table=t,row=r,column=c,printed=got,independent_expected=expected,passed=clean(got)==clean(expected),source=source))
S=df(OUT/'INDEPENDENT_TABLE4_SUMMARY.csv');S=S[S.group.isin(['core','ridge','neural','samepatch'])];M=df(OUT/'INDEPENDENT_STEP16_METRICS_738.csv')
names=['ridge','lstm','vanilla','swim','single','fixed','same_patch_fixed','static','adaptive']
for inp,start in [('O',6),('C',16)]:
 for i,model in enumerate(names):
  for j,h in enumerate([5,10,20]):
   q=S[(S.input_set==inp)&(S.model==model)&(S.config=='daily_weekly')&S.h.eq(h)];assert len(q)==1;z=q.iloc[0]
   a=f'{100*z.mse_mean:.4f}';b=f'{z.rank_ic_mean:.3f}'
   if z.n_seeds==5:a+=f' ({100*z.mse_sd:.3f})';b+=f' ({z.rank_ic_sd:.3f})'
   ck(4,start+i,j+1,a,'INDEPENDENT_TABLE4_SUMMARY.csv');ck(4,start+i,j+4,b,'INDEPENDENT_TABLE4_SUMMARY.csv')
for r,model in [(3,'zero'),(4,'global_train_mean')]:
 for j,h in enumerate([5,10,20]):
  q=M[(M.model==model)&M.h.eq(h)];assert q.mse.max()-q.mse.min()<1e-12
  ck(4,r,j+1,f'{100*q.mse.iloc[0]:.4f}','INDEPENDENT_STEP16_METRICS_738.csv')
N=df(OUT/'INDEPENDENT_TABLE9.csv')
for inp,start in [('O',2),('C',5)]:
 for j,h in enumerate([5,10,20]):
  a=N[(N.input_set==inp)&N.h.eq(h)&N.method.eq('date_only')].iloc[0];b=N[(N.input_set==inp)&N.h.eq(h)&N.method.eq('joint_seed_date')].iloc[0];r=start+j
  ck(9,r,1,str(h),'independent label');ck(9,r,2,f'{1e4*a.theta:.2f}','INDEPENDENT_TABLE9.csv')
  for c,z in [(3,a),(5,b)]:ck(9,r,c,f'[{1e4*z.lo:.2f}, {1e4*z.hi:.2f}]','INDEPENDENT_TABLE9.csv')
  ck(9,r,4,f'{a.p_holm_3h:.4f}' if inp=='O' else f'{a.p_holm_3h:.2f}','INDEPENDENT_TABLE9.csv')
  ck(9,r,6,f'{b.p_holm_3h:.2f}','INDEPENDENT_TABLE9.csv');ck(9,r,7,f'{a.seed_wins_static}/5','INDEPENDENT_TABLE9.csv')
Z=df(OUT/'INDEPENDENT_TABLE10_METRICS.csv');Q=df(OUT/'INDEPENDENT_TABLE10_TESTS.csv')
for w,start in [('W1',2),('W2',6),('W3',10)]:
 for inp,offset in [('O',2),('C',5)]:
  for j,h in enumerate([5,10,20]):
   for k,model in enumerate(['static','adaptive','zero']):
    z=Z[(Z.window==w)&Z.input_set.eq(inp)&Z.model.eq(model)&Z.h.eq(h)].iloc[0];ck(10,start+k,offset+j,f'{z.mean_relative_mse:.2f}','INDEPENDENT_TABLE10_METRICS.csv')
   z=Q[Q.window.eq(w)&Q.input_set.eq(inp)&Q.h.eq(h)].iloc[0];ck(10,start+3,offset+j,f'{z.p_holm_3h:.2f}','INDEPENDENT_TABLE10_TESTS.csv')
Z=df(OUT/'INDEPENDENT_TABLE11.csv');E=ROOT/js('revision_work/outputs/step22e/ANALYSIS_COMPLETE.json')['analysis_directory'];D=ROOT/js('revision_work/outputs/step22d/session_v3/ANALYSIS_COMPLETE.json')['analysis_directory'];ecw=df(E/'E2_CONFIRMATORY_CW.csv');eic=df(E/'E1_SECONDARY_rank_ic.csv');dc=df(D/'confirmatory_tests.csv');numeric=[]
for model,start in [('static',2),('adaptive',5),('lstm',8)]:
 for j,h in enumerate([5,10,20]):
  z=Z[Z.model.eq(model)&Z.h.eq(h)].iloc[0];r=start+j
  ck(11,r,1,str(h),'independent label')
  for c,k in [(2,'r2_W1'),(3,'r2_W2'),(4,'r2_W1_W2'),(6,'r2_zero_W1_W2')]:ck(11,r,c,f'{100*z[k]:+.2f}','INDEPENDENT_TABLE11.csv')
  ck(11,r,5,'<0.001' if z.cw_p_nw_holm<.001 else f'{z.cw_p_nw_holm:.3f}','INDEPENDENT_TABLE11.csv');ck(11,r,7,f'{z.mean_rank_ic:.3f}','INDEPENDENT_TABLE11.csv')
  cw=(ecw[ecw.model.eq(model)&ecw.h.eq(h)] if model!='lstm' else dc[dc.family.eq('CONFIRMATORY_CW')&dc.h.eq(h)]).iloc[0]
  ic=(eic[eic.model.eq(model)&eic.h.eq(h)&eic.setting.eq('T3')] if model!='lstm' else dc[dc.family.eq('CONFIRMATORY_RANK_IC')&dc.h.eq(h)]).iloc[0]
  for key,a,b in [('CW mean',z.cw_mean,cw['mean']),('CW SE',z.cw_se,cw.nw_se),('CW raw p',z.cw_p_nw,cw.p_nw),('CW Holm p',z.cw_p_nw_holm,cw.p_nw_holm),('IC seed-mean',z.mean_rank_ic,ic['mean'])]:numeric.append(dict(check=key,model=model,h=h,max_abs_difference=abs(a-b),passed=abs(a-b)<=1e-10))
write('LITERAL_TABLES_4_9_10_11.csv',pd.DataFrame(rows));write('TABLE11_ADDITIONAL_NUMERIC_CHECKS.csv',pd.DataFrame(numeric));source_manifest('SOURCES_LITERAL_TABLES.csv');print('LITERAL',len(rows),'failed',sum(not x['passed'] for x in rows));print([x for x in rows if not x['passed']]);print('Additional numeric failures',sum(not x['passed'] for x in numeric))
