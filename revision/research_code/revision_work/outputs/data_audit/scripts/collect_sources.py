"""Fetch issuer submissions, EDGAR full-text searches and observed primary URLs; serial <=3.34 requests/s."""
import sys,os,certifi,json,urllib.parse
sys.path.insert(0,str(__import__('pathlib').Path(__file__).resolve().parent))
from audit_io import *
os.environ['SSL_CERT_FILE']=certifi.where()
tickers=['AES','BEP','BLDP','BLNK','CSIQ','CWEN','DQ','ENPH','FCEL','FSLR','HASI','JKS','NEE','ORA','PLUG','RUN','SEDG','BEPC']
mapping=json.loads((OUT/'sources/company_tickers_retry.json').read_text());lookup={v['ticker']:v for v in mapping.values()}
records=[]
for t in tickers:
 v=lookup.get(t)
 if v is None:records.append(dict(ticker=t,status='NOT FOUND'));continue
 cik=str(v['cik_str']).zfill(10)
 if t=='FCEL':j=json.loads((OUT/'sources/submissions_FCEL_retry.json').read_text())
 else:
  status,body,_=fetch('https://data.sec.gov/submissions/CIK'+cik+'.json','submissions_'+t+'.json')
  if status!=200:continue
  j=json.loads(body)
 records.append(dict(ticker=t,cik=cik,name=j['name'],former_names=json.dumps(j.get('formerNames',[])),tickers=json.dumps(j.get('tickers',[])),status='FOUND'))
 # Historical submissions files are URLs supplied by the submissions endpoint, not guessed accession URLs.
 if t in ['BEP','BEPC','BLNK','DQ','NEE','FCEL','CWEN','ORA']:
  for f in j['filings'].get('files',[]):
   if f['filingTo']>='2015-08-05' and f['filingFrom']<='2020-12-31':fetch('https://data.sec.gov/submissions/'+f['name'],'historical_'+t+'_'+f['name'])
 q='"reverse stock split" OR "special distribution" OR "ticker symbol" OR "ADS ratio" OR "stock split"'
 pars=dict(q=q,ciks=cik,dateRange='custom',startdt='2015-08-05',enddt='2026-07-07',count='100')
 fetch('https://efts.sec.gov/LATEST/search-index?'+urllib.parse.urlencode(pars),'efts_identity_'+t+'.json')
rows('issuer_submissions_inventory.csv',records)
urls={
 'fcel2015':'https://www.sec.gov/Archives/edgar/data/886128/000088612815000023/exhibit99112-3x15.htm',
 'fcel2019':'https://www.sec.gov/Archives/edgar/data/886128/000156459019017009/fcel-ex991_8.htm',
 'fcel2024':'https://www.sec.gov/Archives/edgar/data/886128/000110465924115093/tm2427658d1_ex99-1.htm',
 'blnk2017':'https://www.sec.gov/Archives/edgar/data/1429764/000149315219004595/form10-k.htm',
 'blnkprospectus':'https://www.sec.gov/Archives/edgar/data/1429764/000149315218002271/form-424b4.htm',
 'dq2020':'https://www.sec.gov/Archives/edgar/data/1477641/000110465920117962/tm2034256d1_ex99-1.htm',
 'nee2020':'https://www.sec.gov/Archives/edgar/data/37634/000075330820000186/nee10q3q2020.htm',
 'bep2020distribution':'https://www.sec.gov/Archives/edgar/data/1533232/000119312520197159/d37180dex991.htm',
 'bep2020split':'https://www.sec.gov/Archives/edgar/data/1791863/000117184320007534/exh_991.htm',
 'bep2026div':'https://www.sec.gov/Archives/edgar/data/1533232/000117184326000527/exh_991.htm',
 'aes2025div':'https://www.sec.gov/Archives/edgar/data/874761/000087476125000013/aes-20241231.htm',
 'nee2026div':'https://www.sec.gov/Archives/edgar/data/37634/000075330826000031/nee-20260331.htm',
 'cwen2026identity':'https://www.sec.gov/Archives/edgar/data/1567683/000162828026053421/cwen-20260630.htm',
}
for label,url in urls.items():fetch(url,label+'.html')
rows('collection_http_requests.csv',__import__('audit_io')._requests)
