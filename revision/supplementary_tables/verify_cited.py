"""Check every number the manuscript/letter attributes to Supplementary Tables S2-S18 against the CLEAN workbook."""
import sys, openpyxl, pandas as pd, numpy as np
wb = openpyxl.load_workbook(sys.argv[1])
D = '–'
def panels(sheet):
    rows = list(wb[sheet].iter_rows(values_only=True)); out = {}; i = 0
    while i < len(rows):
        r = rows[i]
        if r[0] and isinstance(r[0], str) and (r[0][:2] in ('A.','B.','C.','D.','E.','F.','G.','H.','I.') or r[0] in ('Rank IC contrasts','Subgroup results','Mean realized returns and the Global Train-Mean','Annualized performance by cost level and risk aversion')) and all(v is None for v in r[1:]):
            hdr = [h for h in rows[i+1] if h is not None]; j = i + 2; data = []
            while j < len(rows) and any(v is not None for v in rows[j]): data.append(rows[j][:len(hdr)]); j += 1
            df = pd.DataFrame(data, columns=hdr).replace(D, np.nan)
            out[r[0].split('.')[0] if r[0][1] == '.' else r[0]] = df; i = j
        else: i += 1
    return out
res = []
def chk(name, cond, detail=''):
    res.append((bool(cond), name, detail)); 
def rng(s): s = pd.to_numeric(s); return '%.4f..%.4f' % (s.min(), s.max())

P = panels('S2')['Mean realized returns and the Global Train-Mean']
chk('S2 5-day validation mean -0.91%', round(P.loc[P['Horizon (days)']==5,'Validation mean'].iloc[0]*100,2)==-0.91)
P = panels('S3'); A = P['A']
r = A[(A.Model.str.startswith('Momentum'))&(A.Model.str.contains('calibrated'))&(A['Horizon (days)']==10)].iloc[0]
chk('S3 momentum 10d Rank IC -0.082, t -2.40', round(r['Mean Rank IC'],3)==-0.082 and round(r['Rank IC t (NW)'],2)==-2.40)
x = A[A.Model=='Short-term reversal (calibrated)']['R² vs Global Train-Mean']*100
chk('S3 reversal R2 +0.09..+0.13%', round(x.min(),2)==0.09 and round(x.max(),2)==0.13, rng(x))
B = P['B']; x = B[(B['Input specification']=='Price-level')&(B.Configuration=='Daily+Weekly')]
chk('Ridge O R2 +0.02/+0.04/+0.10%', list((x['R² vs Global Train-Mean']*100).round(2))==[0.02,0.04,0.10])
chk('Ridge O Rank IC -0.014..0.011', round(x['Mean Rank IC'].min(),3)==-0.014 and round(x['Mean Rank IC'].max(),3)==0.011)
x = B[(B['Input specification']=='Window-relative')&(B.Configuration=='Daily+Weekly')&(B['Horizon (days)']==20)]
chk('Ridge C 20d Rank IC 0.059', round(x['Mean Rank IC'].iloc[0],3)==0.059)

P = panels('S6'); C = P['C']
ac = C[C.Contrast=='Adaptive − Constant gate']
chk('Table 7 / S6 Adaptive vs constant gate all Holm p >= 0.15 (MSE, IC, DA; both inputs/backbones)', ac['Holm p (3 horizons)'].min() >= 0.15 and set(ac.Metric)=={'MSE','Rank IC','DA'} and len(ac)==36, '%.4f' % ac['Holm p (3 horizons)'].min())

P = panels('S10'); Cc = P['C']; x = Cc[Cc['Input specification']=='Price-level']
st = x[x.Model=='Static']; ad = x[x.Model=='Adaptive']
chk('S10 single-run Sharpe Static 0.96/0.91/0.94', list(st['Sharpe ratio, mean'].round(2))==[0.96,0.91,0.94])
chk('S10 single-run Sharpe Adaptive 0.90/0.83/0.85', list(ad['Sharpe ratio, mean'].round(2))==[0.90,0.83,0.85])
chk('S10 Static smaller drawdown (mean) every h', all(st['Max. drawdown, mean'].values > ad['Max. drawdown, mean'].values), str(list(zip(st['Max. drawdown, mean'].round(3), ad['Max. drawdown, mean'].round(3)))))
chk('S10 Static lower drawdown dispersion every h', all(st['Max. drawdown, SD'].values < ad['Max. drawdown, SD'].values))
chk('S10 Static lower Sharpe dispersion every h (wording check)', all(st['Sharpe ratio, SD'].values < ad['Sharpe ratio, SD'].values), str(list(zip(st['Sharpe ratio, SD'].round(3), ad['Sharpe ratio, SD'].round(3)))))

P = panels('S12'); n = sum(len(P[k]) for k in ('A','B'))
chk('S12 414 cells per weighting rule', len(P['A'])==414 and len(P['B'])==414)
x = pd.concat([P['A'],P['B']]); sig = x[(x['Holm p']<0.05)&(x['Loss difference']<0)]
chk('S12 no combined forecast significantly better after Holm', len(sig)==0, 'min Holm %.3f' % x['Holm p'].min())

P = panels('S14'); A, Bp, Cc, Dd, F = P['A'], P['B'], P['C'], P['D'], P['F']
sa = Cc[(Cc.Contrast=='Static − Adaptive')&(~Cc.Setting.str.startswith('T0'))&(Cc.Metric.isin(['MSE','Rank IC']))]
chk('S14 T1-T3 Static-Adaptive MSE/IC Holm >= 0.81', round(sa['Holm p (3 horizons)'].min(),2)>=0.81, '%.4f' % sa['Holm p (3 horizons)'].min())
bd = Dd[(Dd.Contrast=='Static − Adaptive')&(Dd['Input specification']=='Window-relative')&(~Dd.Setting.str.startswith('T0'))]
chk('S14 Adaptive MSE gain bound <= 1.0% every setting', round(bd['Upper bound on Adaptive MSE gain (%)'].max(),1)<=1.0, '%.3f' % bd['Upper bound on Adaptive MSE gain (%)'].max())
al = Cc[(Cc.Contrast=='Adaptive − LSTM')&(Cc.Metric=='MSE')&(Cc.Setting.str[:2].isin(['T1','T3']))]
chk('S14 LSTM lower MSE than Adaptive in all 5 seeds (T1,T3)', (al['Seed wins, first arm']==0).all() and (al['Seed losses, first arm']==5).all() and len(al)==6)
def mse(st, m): return A[(A.Setting.str.startswith(st))&(A.Model==m)].set_index('Horizon (days)')['Test MSE, mean']
for m, lo, hi in (('Static',1.5,1.9),('Adaptive',0.8,2.1),('LSTM',0.4,0.9)):
    ch = (1 - mse('T3',m)/mse('T0',m))*100
    chk('S14 T3 lowers %s MSE by %.1f-%.1f%%' % (m,lo,hi), round(ch.min(),1)==lo and round(ch.max(),1)==hi, rng(ch))
t3 = bd[bd.Setting.str.startswith('T3')]['MSE difference (% of second arm)'].abs()
chk('S14 Static/Adaptive differ <= 0.3% under T3', round(t3.max(),1)<=0.3, rng(t3))
ag = F[F.Model.isin(['Static','Adaptive'])]
v0 = ag[ag.Setting.str.startswith('T0')]['Validation loss change (%)']; v3 = ag[ag.Setting.str.startswith('T3')]['Validation loss change (%)']
chk('S14 val-loss rise 3-33% (T0) -> 2-5% (T3)', round(v0.min())==3 and round(v0.max())==33 and round(v3.min())==2 and round(v3.max())==5, rng(v0)+' / '+rng(v3))
l3 = A[(A.Setting.str.startswith('T3'))&(A.Model=='LSTM')]['R² vs Global Train-Mean, mean']*100
chk('S14 LSTM T3 R2 0.00/+0.03/+0.25', list(l3.round(2))==[0.0,0.03,0.25], str(list(l3.round(3))))
e3 = Bp[(Bp.Setting.str.startswith('T3'))&(Bp.Model=='LSTM')]['R² vs Global Train-Mean']*100
chk('S14 LSTM T3 ensemble R2 +0.07/+0.09/+0.32', list(e3.round(2))==[0.07,0.09,0.32], str(list(e3.round(3))))
r3 = A[(A.Setting.str.startswith('T3'))&(A.Model.isin(['Static','Adaptive']))]['R² vs Global Train-Mean, mean']*100
r0 = A[(A.Setting.str.startswith('T0'))&(A.Model.isin(['Static','Adaptive']))]['R² vs Global Train-Mean, mean']*100
chk('S14 Static/Adaptive R2 -1.2..-0.6 (T3), -3.1..-1.5 (T0)', (round(r3.min(),1),round(r3.max(),1),round(r0.min(),1),round(r0.max(),1))==(-1.2,-0.6,-3.1,-1.5), rng(r3)+' / '+rng(r0))
chk('S14 within 1.3% of training mean under T3', round(-r3.min(),1)<=1.3)
ic = A[A['Horizon (days)']==5]
t0 = ic[ic.Setting.str.startswith('T0')]['Mean Rank IC, mean']; t2 = ic[ic.Setting.str.startswith('T2')]['Mean Rank IC, mean']
t3s = A[(A.Setting.str.startswith('T3'))&(A.Model.isin(['Static','Adaptive']))]['Mean Rank IC, mean']
chk('S14 5d Rank IC 0.010-0.014 (T0) -> 0.027-0.032 (T2)', (round(t0.min(),3),round(t0.max(),3),round(t2.min(),3),round(t2.max(),3))==(0.010,0.014,0.027,0.032), rng(t0)+' / '+rng(t2))
chk('S14 T3 Rank IC Static/Adaptive 0.010-0.026', (round(t3s.min(),3),round(t3s.max(),3))==(0.010,0.026), rng(t3s))

P = panels('S15'); Bq, Cq = P['B'], P['C']
pc = Cq[(Cq.Window.str.startswith('Pooled'))&(Cq['Input specification']=='Window-relative')]
chk('S15 pooled C Static-Adaptive Holm >= 0.89', round(pc['Holm p'].min(),2)>=0.89, '%.4f'%pc['Holm p'].min())
m = pc[pc.Metric=='MSE']['Upper bound on Adaptive MSE gain (%)']
chk('S15 MSE bounds 0.56/0.48/0.37%', list(m.round(2))==[0.56,0.48,0.37], str(list(m.round(3))))
ri = pc[pc.Metric=='Rank IC']['Upper bound on Adaptive gain (metric units)']
chk('S15 Rank IC bounds 0.013-0.018', (round(ri.min(),3),round(ri.max(),3))==(0.013,0.018), rng(ri))
pb = Bq[(Bq.Window.str.startswith('Pooled'))]
cw = pb[(pb['Input specification']=='Window-relative')&(pb.Model.isin(['Static','Adaptive']))]
chk('S15 pooled Rank IC 0.026-0.051, NW p 0.010-0.048', (round(cw['Mean Rank IC'].min(),3),round(cw['Mean Rank IC'].max(),3),round(cw['NW p (two-sided)'].min(),3),round(cw['NW p (two-sided)'].max(),3))==(0.026,0.051,0.010,0.048), rng(cw['Mean Rank IC'])+' '+rng(cw['NW p (two-sided)']))
chk('S15 smallest Holm 0.059 (NW), 0.043 (MBB)', round(cw['NW Holm p'].min(),3)==0.059 and round(cw['MBB Holm p'].min(),3)==0.043)
rd = pb[(pb['Input specification']=='Window-relative')&(pb.Model=='Ridge')]['Mean Rank IC']
chk('S15 Ridge pooled Rank IC 0.033-0.062', (round(rd.min(),3),round(rd.max(),3))==(0.033,0.062), rng(rd))
po = pb[(pb['Input specification']=='Price-level')]
chk('S15 price-level pooled |Rank IC| < 0.012 (Static, Adaptive)', po[po.Model.isin(['Static','Adaptive'])]['Mean Rank IC'].abs().max() < 0.012)
chk('S15 price-level pooled |Rank IC| < 0.012 incl. Ridge (wording check)', po['Mean Rank IC'].abs().max() < 0.012, 'Ridge %s' % rng(po[po.Model=='Ridge']['Mean Rank IC']))
po2 = po[po.Model.isin(['Static','Adaptive'])]
chk('S15 price-level pooled Holm = 1.00', (po2['NW Holm p'].round(2)==1).all())
pm = Cq[(Cq.Window.str.startswith('Pooled'))&(Cq['Input specification']=='Price-level')&(Cq.Metric=='MSE')]
chk('S15 price-level pooled MSE favors Static, Holm 0.061-0.064', (pm.Difference<0).all() and (round(pm['Holm p'].min(),3),round(pm['Holm p'].max(),3))==(0.061,0.064), rng(pm['Holm p']))

P = panels('S16'); Cs, Ds = P['C'], P['D']
e2 = Cs[Cs.Family=='Confirmatory (E2)']
chk('S16 E2 CW NW Holm <= 0.007 (Static, Adaptive)', round(e2['NW Holm p'].max(),3)<=0.007 and len(e2)==6, '%.4f' % e2['NW Holm p'].max())
chk('S16 E2 MBB agrees (Holm < 0.05)', (e2['MBB Holm p']<0.05).all())
ls = Cs[(Cs.Family=='Confirmatory (LSTM)')&(Cs.Test.str.startswith('Clark'))]
chk('S16 LSTM CW Holm 0.015/0.015/<0.001', [round(v,3) for v in ls['NW Holm p']][:2]==[0.015,0.015] and ls['NW Holm p'].iloc[2]<0.001 and (ls['MBB Holm p']<0.05).all(), str(list(ls['NW Holm p'].round(4))))
e4 = Ds[Ds.Family=='Confirmatory (E4)']
chk('S16 Static-Adaptive Holm >= 0.70', round(e4['Holm p'].min(),2)>=0.70, '%.4f' % e4['Holm p'].min())
e3 = Ds[Ds.Family=='Confirmatory (E3)']
a10 = e3[(e3.Model=='Adaptive')&(e3['Horizon (days)']==10)].iloc[0]
chk('S16 T3-T0 Adaptive 10d -3.7%, Holm 0.008', round(a10['Difference (% of reference MSE)'],1)==-3.7 and round(a10['Holm p'],3)==0.008)
chk('S16 T3 lowers MSE in 5 of 6 pairs', (e3.Difference<0).sum()==5)
e1 = Cs[Cs.Family=='Secondary (E1)']; d0 = Cs[Cs.Family=='Descriptive']
chk('S16 T3 pooled Rank IC 0.031-0.075, Holm <= 0.041', (round(e1['Mean statistic'].min(),3),round(e1['Mean statistic'].max(),3))==(0.031,0.075) and round(e1['NW Holm p'].max(),3)<=0.041, rng(e1['Mean statistic'])+' '+rng(e1['NW Holm p']))
chk('S16 T0 pooled Rank IC 0.031-0.065', (round(d0['Mean statistic'].min(),3),round(d0['Mean statistic'].max(),3))==(0.031,0.065), rng(d0['Mean statistic']))

P = panels('S17'); C7, E7 = P['C'], P['E']
w = C7[C7.Arm=='Static-R100']['Mean |weight − 0.25|, mean']; w0 = C7[C7.Arm=='Static']['Mean |weight − 0.25|, mean']
chk('S17 Static-R100 weight MAD 0.027-0.049 vs 0.0003-0.0007', (round(w.min(),3),round(w.max(),3),round(w0.min(),4),round(w0.max(),4))==(0.027,0.049,0.0003,0.0007), rng(w)+' / '+rng(w0))
sf = E7[(E7.Contrast=='Static-R100 − Fixed')&(E7.Metric=='MSE')]
chk('S17 Static-R100 vs Fixed MSE -0.12/+0.30/+0.11%, Holm >= 0.34', list(sf['Difference (% of reference MSE)'].round(2))==[-0.12,0.30,0.11] and round(sf['Holm p (3 horizons)'].min(),2)>=0.34, str(list(sf['Difference (% of reference MSE)'].round(3)))+' '+rng(sf['Holm p (3 horizons)']))
chk('S17 bound caps MSE improvement at 0.5%', round(sf['Upper bound on MSE gain of first arm (%)'].max(),1)<=0.5, rng(sf['Upper bound on MSE gain of first arm (%)']))
sd = C7[(C7['Horizon (days)']==5)&(C7.Branch=='Daily')].set_index('Arm')['Within-model SD of weights, mean']
chk('S17 5d Daily within-model SD 0.022 -> 0.0002', round(sd['Adaptive'],3)==0.022 and round(sd['Adaptive-R100'],4)==0.0002, '%.5f %.5f' % (sd['Adaptive'], sd['Adaptive-R100']))
sa = E7[E7.Contrast=='Static-R100 − Adaptive-R100']
chk('S17 Static-R100 vs Adaptive-R100 Holm >= 0.78', round(sa['Holm p (3 horizons)'].min(),2)>=0.78, '%.4f' % sa['Holm p (3 horizons)'].min())

P = panels('S18'); D8, E8, B8 = P['D'], P['E'], P['B']
x = E8[E8.Contrast=='Static-CS − Adaptive-CS']
chk('S18 Static-CS vs Adaptive-CS Holm = 1.00', (x['Holm p (3 horizons)'].round(2)==1).all())
nn = B8[B8.Arm.isin(['Static-CS','Adaptive-CS','LSTM-CS'])]
top = nn.loc[nn.groupby('Horizon (days)')['Mean Rank IC, mean'].idxmax(),'Arm']
chk('S18 LSTM-CS highest Rank IC (neural CS arms, every h)', (top=='LSTM-CS').all(), str(list(top)))
chk('Letter: S18 no significant ranking signal for any model', (D8['NW Holm p']>=0.05).all() and (D8['MBB Holm p']>=0.05).all(), '%.3f' % D8['NW Holm p'].min())

for ok, n, d in res: print('PASS' if ok else 'FAIL', n, ('['+d+']') if d else '')
print(sum(o for o,_,_ in res), '/', len(res))
