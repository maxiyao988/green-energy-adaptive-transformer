"""Validate final prediction files, recovery gate, report links and output hashes."""
from step3_common import *
import re

def main():
 manifest=json.loads((OUT/'STEP3_MANIFEST.json').read_text());checks=[]
 def check(name,value):checks.append(dict(check=name,passed=bool(value)))
 for item in manifest['files']:check('hash '+item['path'],sha(ROOT/item['path'])==item['sha256'])
 check('protected prior inputs unchanged',manifest['integrity']['passed'])
 a=pd.read_csv(OUT/'task_a_comparison.csv');b=pd.read_csv(OUT/'checkpoint_replay.csv');files=pd.read_csv(OUT/'prediction_file_manifest.csv')
 eligible=a[a.classification.isin(['EXACT','NEAR'])]
 check('all 3 requested recovery attempts recorded',len(a)==3 and set(a.horizon)=={5,10,20})
 check('eligible replay cell count',len(b)==33+len(eligible))
 check('exactly three files per replay cell',len(files)==3*len(b))
 check('no repeated cell/split',not files.duplicated(['mechanism','horizon','seed','split']).any())
 for _,r in a.iterrows():
  accepted=((b.mechanism=='adaptive')&(b.seed==42)&(b.horizon==r.horizon)).sum()
  check(f'gate recovered h{r.horizon}',accepted==int(r.classification in ['EXACT','NEAR']))
 for _,r in files.iterrows():
  d=pd.read_parquet(ROOT/r.path)
  check('row count '+r.path,len(d)==COUNTS[r.split])
  check('unique stock/date '+r.path,not d.duplicated(['ticker','anchor_date']).any())
  check('one exact split '+r.path,d.split.eq(r.split).all())
  check('finite predictions '+r.path,np.isfinite(d[['y_true','y_pred']].to_numpy()).all())
  check('17 tickers each date '+r.path,d.groupby('anchor_date').ticker.nunique().eq(17).all())
 curves=pd.read_csv(OUT/'learning_curves.csv');means=pd.read_csv(OUT/'mean_shift_table.csv')
 check('history run uniqueness',not curves.duplicated(['cell_id','epoch']).any())
 check('Huber loss explicitly stated',curves.loss_type.eq('Huber').all() and curves.huber_delta.eq(1).all())
 check('54 pooled and per-ticker mean rows',len(means)==54 and means.groupby('horizon').ticker.nunique().eq(18).all())
 report=(OUT/'REPORT_STEP3.md').read_text()
 check('blocking section at top',report.startswith('# BLOCKING\n'))
 for target_path in re.findall(r'\]\(([^)]+)\)',report):
  if target_path.startswith(('http://','https://')):continue
  p=target_path.rsplit(':',1)[0] if re.search(r':\d+$',target_path) else target_path
  p=Path(p);p=p if p.is_absolute() else OUT/p
  check('report link '+target_path,p.exists())
 result=dict(passed=all(c['passed'] for c in checks),n_checks=len(checks),n_failed=sum(not c['passed'] for c in checks),checks=checks)
 write_json(OUT/'VERIFY_STEP3.json',result)
 print('Step 3 verification:',result['n_checks'],'checks,',result['n_failed'],'failures')
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
