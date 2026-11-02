"""Report additional event-localization sensitivity without changing the recorded gate."""
from core import *
def main():
 d=pd.read_csv(OUT/'precision_window_details.csv');rows=[]
 for (t,r),g in d.groupby(['ticker','resolution']):
  if g.max_standardized_window.max()==0:continue
  for scope,k in [('top1pct',max(1,int(np.ceil(len(g)*.01)))),('top20',20)]:
   high=g.nlargest(k,'max_standardized_window')
   for loc in ['source','anchor']:
    col=f'near_action_or_step_{loc}_pm5';base=float(g[col].mean());rate=float(high[col].mean());rows.append(dict(ticker=t,resolution=r,tail=scope,location=loc,n_tail=len(high),n_tail_near=int(high[col].sum()),baseline_near_fraction=base,tail_near_fraction=rate,enrichment_ratio=rate/base if base else None,note='Overlapping windows repeat underlying dates; tail counts are not independent event observations'))
 csvout('precision_tail_action_sensitivity.csv',rows)
if __name__=='__main__':main()
