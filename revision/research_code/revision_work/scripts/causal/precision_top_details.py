"""Top-window source details and exact nearest action types."""
from core import *
from build_and_audit import load_prices,adjusted,windows,transform

def main():
 w=pd.read_csv(OUT/'precision_window_details.csv');topstd=w.nlargest(20,'max_standardized_window');topraw=w.nlargest(20,'max_raw_window');out=[]
 for ranking,top in [('standardized',topstd),('raw',topraw)]:
  for t,g in top.groupby('ticker'):
   old=load_prices(t);new=load_prices(t,True);na=adjusted(new);a=pd.to_datetime(g.anchor_date).to_numpy(dtype='datetime64[ns]');ow,_,_,ref,di=windows(old,a);nw,_,_,nr,ni=windows(na,a)
   for j,(_,row) in enumerate(g.iterrows()):
    r=row.resolution;k=0 if r=='daily' else 1;x=ow[k][j];nx=nw[k][j];err=np.abs(transform(x[None],ref[j:j+1])[0]-transform(nx[None],nr[j:j+1])[0]);pos,c=np.unravel_index(np.argmax(err),err.shape) if ranking=='raw' else (int(row.window_position),CHANNELS.index(row.channel))
    if r=='daily':start=end=old.datetime.iloc[di[j]-89+pos]
    else:
     weekly=builder._reconstruct_weekly_from_daily(old);wi=np.searchsorted(weekly.available_date.to_numpy(),a[j],side='right')-1;bar=weekly.iloc[wi-25+pos];start=bar.week_start;end=bar.available_date
    si=int(np.searchsorted(new.datetime.to_numpy(),np.datetime64(start)));ei=int(np.searchsorted(new.datetime.to_numpy(),np.datetime64(end)));events=[]
    for typ,mask in [('dividend',new.Dividends!=0),('split',new['Stock Splits']!=0),('ratio_step',new.datetime.isin(pd.to_datetime({'CWEN':['2026-06-01'],'NEE':['2026-06-05']}.get(t,[]))))]:
     for q in np.flatnonzero(mask):
      source_distance=max(si-q,0,q-ei);anchor_distance=abs(ni[j]-q)
      if min(source_distance,anchor_distance)<=5:events.append(f'{typ}:{new.datetime.iloc[q].date()} (source {source_distance}, anchor {anchor_distance} sessions)')
    d=row.to_dict();d.update(ranking=ranking,channel=CHANNELS[c],window_position=int(pos),source_start=str(pd.Timestamp(start).date()),source_end=str(pd.Timestamp(end).date()),original_source_value=float(x[pos,c]),new_adjusted_source_value=float(nx[pos,c]),raw_difference_at_listed_element=float(err[pos,c]),nearby_events='; '.join(events) or 'none within +/-5 sessions',new_vendor_close_at_source_end=float(new.set_index('datetime').loc[end,'close']))
    out.append(d)
 csvout('precision_top20_raw_and_standardized.csv',out)
if __name__=='__main__':main()
