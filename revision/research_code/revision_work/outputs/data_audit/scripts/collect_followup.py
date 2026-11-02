"""Additional issuer filings resolved from SEC submissions metadata and literal exhibit hrefs."""
import os,sys,json,re,urllib.parse,certifi
from html.parser import HTMLParser
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
os.environ['SSL_CERT_FILE']=certifi.where()
class Links(HTMLParser):
 def __init__(self):super().__init__();self.links=[]
 def handle_starttag(self,tag,attrs):
  if tag=='a':
   d=dict(attrs)
   if 'href' in d:self.links.append(d['href'])
def primary(t,acc,label,exhibits=False):
 j=json.loads((OUT/f'sources/submissions_{t}.json').read_text());r=j['filings']['recent'];i=r['accessionNumber'].index(acc);document=r['primaryDocument'][i]
 # Canonical URL assembled only from observed CIK/accession/primaryDocument fields; verified HTTP status retained.
 url='https://www.sec.gov/Archives/edgar/data/'+str(int(j['cik']))+'/'+acc.replace('-','')+'/'+document
 status,body,m=fetch(url,label+'.html')
 js('sources/'+label+'_resolution.json',dict(source_json=f'sources/submissions_{t}.json',cik=j['cik'],accession=acc,primaryDocument=document,form=r['form'][i],filingDate=r['filingDate'][i],url=url,url_basis='SEC-supplied metadata fields, canonical EDGAR location, HTTP verified'))
 if status==200 and exhibits:
  h=Links();h.feed(body.decode(errors='replace'));links=sorted(set(h.links))
  for k,link in enumerate([v for v in links if ('99' in v.lower() or 'ex' in v.lower()) and '.htm' in v.lower() and not v.startswith('#')][:5]):fetch(urllib.parse.urljoin(url,link),label+'_exhibit_'+str(k)+'.html')
primary('HASI','0001561894-26-000007','hasi2026div')
primary('CWEN','0001628280-26-032303','cwen2026div',True)
primary('CWEN','0001104659-26-053557','cwen2026conversion',True)
fetch('https://www.sec.gov/Archives/edgar/data/1481513/000110465923103745/tm2326937d2_ex99-1.htm','jks2023div.html')
fetch('https://www.sec.gov/Archives/edgar/data/1296445/000143774916025890/ex99-1.htm','ora2016div.html')
# Search dated identity changes and issuer dividend terms, using SEC full-text search.
queries=[('CWEN','"Clearway" "2018"','2018-08-01','2018-10-01'),('CWEN','"Class C" "May 14, 2015"','2015-01-01','2016-12-31'),('BLNK','"name" "2017"','2017-08-01','2017-09-30'),('BEP','"name" "2016"','2016-04-01','2016-06-01'),('HASI','"name" "2024"','2024-06-01','2024-08-01'),('HASI','"0.425"','2026-01-01','2026-05-01'),('CWEN','"dividend" "2026"','2026-05-01','2026-06-15'),('NEE','"0.6232"','2026-02-01','2026-03-15')]
for k,(t,q,start,end) in enumerate(queries):
 j=json.loads((OUT/f'sources/submissions_{t}.json').read_text());pars=dict(q=q,ciks=j['cik'],dateRange='custom',startdt=start,enddt=end)
 fetch('https://efts.sec.gov/LATEST/search-index?'+urllib.parse.urlencode(pars),'efts_followup_'+str(k)+'_'+t+'.json')
rows('followup_http_requests.csv',__import__('audit_io')._requests)
