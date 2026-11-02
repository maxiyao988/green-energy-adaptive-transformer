"""Finish the read-only inventory after a resolved datetime-name collision; exclusive new files only."""
import sys,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit_io import *
import pandas as pd
http=pd.read_csv(OUT/'http_request_inventory.csv').to_dict('records');stamps=[datetime.fromisoformat(r['requested_utc']).timestamp() for r in http];peak=max(sum(0<=x-y<1 for x in stamps) for y in stamps)
js('http_compliance.json',dict(user_agent=UA,n_requests=len(http),n_200=sum(r['status']==200 for r in http),initial_tls_failures=sum(r['status']==0 for r in http),non_200_after_certificate_fix=[r for r in http if r['status'] not in [0,200]],peak_requests_started_in_any_rolling_second=peak,max_requests_per_second_allowed=5,all_user_agents_match=all(r['user_agent']==UA for r in http),tls_resolution='initial three urllib requests failed certificate validation; retried with certifi CA bundle, verification enabled'))
cmd=['rg','-n','-i','3407|seed.*(rationale|chosen|choose|why|reason)|种子.*(原因|选择)','docs','project_progress.md'];r=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,check=False);text('F_search_matches.txt',r.stdout);js('F_search_command.json',dict(command=cmd,exit_code=r.returncode,scope=['docs/','project_progress.md'],fullrun_accessed=False))
a=pd.read_csv(OUT/'A_event_verification.csv');rows('A_event_status_counts.csv',a.groupby('status',as_index=False).agg(n_events=('ticker','size')))
# Complement the earlier exact quotation set with the Static source and non-selection policy.
quotes=pd.read_csv(OUT/'F_seed_provenance_quotes.csv').to_dict('records');doc='docs/experiment4_multiseed_protocol.md';lines=(ROOT/doc).read_text().splitlines()
for n in [87,89,105]:quotes.append(dict(path=doc,line=n,quote=lines[n-1],scope='pre-specified non-selection and Static seed42 evidence provenance'))
rows('F_seed_provenance_quotes_complete.csv',quotes)
rows('execution_attempts.csv',[
 dict(component='C/D',attempt='compute_checks.py',outcome='resolved assertion on object Timestamp vs datetime64 storage',resolution='compute_checks_v2.py canonicalizes dates only for identity comparison; original arrays and labels unchanged'),
 dict(component='HTTP',attempt='initial urllib3 requests',outcome='certificate verification failed, no SEC body downloaded',resolution='certifi CA bundle; TLS verification remains enabled; retries in separate exclusive files'),
 dict(component='A/B metadata',attempt='assemble_inventory.py',outcome='tables completed, then datetime namespace shadowing failed HTTP-summary assembly',resolution='finish_inventory.py completed remaining files without rewriting any table'),
 dict(component='E',attempt='initial I0 smoke',outcome='EXACT; imported original briefly assigned OMP/MKL=2 during initialization',resolution='definitive v2 clamps environment writes to1 before original import; repeated only I0 on the same causal checkpoint, again EXACT')])
print('Inventory complete',len(http),'requests, peak',peak,'per rolling second',flush=True)
