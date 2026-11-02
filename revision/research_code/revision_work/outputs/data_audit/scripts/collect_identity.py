"""Observed identity sources and issuer dividend calendar; serial requests with named UA."""
import sys,os,certifi,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
os.environ['SSL_CERT_FILE']=certifi.where()
urls={
 'cwen2018rename':'https://www.sec.gov/Archives/edgar/data/1567683/000110465918055334/a18-24111_1ex99d1.htm',
 'cwen2015classes':'https://www.sec.gov/Archives/edgar/data/1567683/000110465915038447/a15-11446_18k.htm',
 'hasi2024identity':'https://www.sec.gov/Archives/edgar/data/1561894/000119312524174717/d822715d8k.htm',
 'bep2016name':'https://www.sec.gov/Archives/edgar/data/1533232/000119907317000163/brep20f.htm',
 'blnk2017name':'https://www.sec.gov/Archives/edgar/data/1429764/000149315217013634/form10-q.htm',
 'bep2020completed':'https://www.sec.gov/Archives/edgar/data/1533232/000153323220000013/bepq22020-ex991.htm',
 'nee2026calendar':'https://www.investor.nexteraenergy.com/stock-information/dividend-history',
 'nee2026declaration':'https://www.investor.nexteraenergy.com/news-and-events/news-releases/2026/02-13-2026-163308214'}
for label,url in urls.items():fetch(url,label+'.html')
# SEC-provided primary-document metadata resolves the DQ 2020 annual-report URL.
j=json.loads((OUT/'sources/submissions_DQ.json').read_text());r=j['filings']['recent'];i=next(i for i,(f,d) in enumerate(zip(r['form'],r['filingDate'])) if f=='20-F' and d.startswith('2021'))
acc=r['accessionNumber'][i];doc=r['primaryDocument'][i];url='https://www.sec.gov/Archives/edgar/data/'+str(int(j['cik']))+'/'+acc.replace('-','')+'/'+doc
fetch(url,'dq2020annual.html');js('sources/dq2020annual_resolution.json',dict(source='submissions_DQ.json',accession=acc,primaryDocument=doc,form=r['form'][i],filingDate=r['filingDate'][i],url=url))
rows('identity_http_requests.csv',__import__('audit_io')._requests)
