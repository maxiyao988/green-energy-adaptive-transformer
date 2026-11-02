"""CSV storage precision using only price rows on/before the revision cutoff."""
from core import *
import csv,decimal

def main():
 rows=[]
 for t in common.FROZEN_TICKERS:
  for snapshot,path,datecol,cols in [('ORIGINAL',ROOT/f'dataset/finance/panel_raw/{t}_daily.csv','datetime',CHANNELS[:4]),('NEW',ROOT/f'data_raw_revision/prices/{t}.csv','Date',['Open','High','Low','Close','Adj Close'])]:
   with path.open() as f:values=[r for r in csv.DictReader(f) if r[datecol][:10]<=CUTOFF]
   for c in cols:
    texts=[r[c] for r in values];a=np.array(texts,float);res=np.abs(a-a.astype(np.float32).astype(float));dec=[max(0,-decimal.Decimal(v).as_tuple().exponent) for v in texts]
    rows.append(dict(ticker=t,snapshot=snapshot,channel=c,n_rows=len(a),max_date=max(r[datecol][:10] for r in values),decimal_places_min=min(dec),decimal_places_median=float(np.median(dec)),decimal_places_max=max(dec),fraction_exact_float32=float((res==0).mean()),max_float32_rounding_residual=float(res.max()),storage='CSV decimal text parsed float64'))
 csvout('snapshot_csv_precision_through_cutoff.csv',rows)
if __name__=='__main__':main()
