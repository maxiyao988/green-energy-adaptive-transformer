"""Step22A standard-library exclusive I/O; no numerical or training imports."""
import os,sys,json,hashlib,datetime,time
from pathlib import Path
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
FR=ROOT/'revision_work/outputs/frozen'
SEEDS=[0,1,21,42,3407]
HS=[5,10,20]
VARIANTS=['static','adaptive','constant_gate']
GRID=[1e-4,1e-3,1e-2]
COUNTS={'train':30634,'val':6290,'test':6647}
WEIGHTS=['daily_w_5','daily_w_10','daily_w_20','daily_w_30','weekly_w_2','weekly_w_4','weekly_w_8','weekly_w_13']
ENV_KEYS=['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','OMP_DYNAMIC','MKL_DYNAMIC']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def rel(p):return str(Path(p).resolve().relative_to(ROOT))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def target(p):
 p=Path(p).absolute()
 assert p.is_relative_to(OUT) and p.resolve()==p,p
 p.parent.mkdir(parents=True,exist_ok=True)
 return p
def text(p,s):
 with target(p).open('x') as f:f.write(s)
def js(p,d):text(p,json.dumps(d,indent=2,allow_nan=False,default=lambda x:x.item() if hasattr(x,'item') else str(x))+'\n')
def csv(p,d):
 with target(p).open('x',newline='') as f:d.to_csv(f,index=False,float_format='%.17g',na_rep='N/A')
def check_protocol():
 m=json.loads((OUT/'RNG_MANIFEST.json').read_text());assert sha(OUT/'PROTOCOL_STEP22A.md')==m['protocol_sha256'];return m['protocol_sha256']
def profile_env():
 e=dict(os.environ)
 for k in ENV_KEYS[:5]:e.pop(k,None)
 e.update(OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
 return e
def bindings():return json.loads((OUT/'BINDINGS.json').read_text())
def best_completion(folder):
 files=list(Path(folder).glob('attempt_*/COMPLETE.json'));assert len(files)==1,(folder,files)
 p=files[0];return p,json.loads(p.read_text())
def prepare_bindings():
 check_protocol();registry=json.loads((FR/'backbone_registry.json').read_text());items=[];protected={}
 def protect(p,expected=None):
  p=ROOT/p if not Path(p).is_absolute() else Path(p)
  h=sha(p)
  if expected:assert h==expected,p
  protected[rel(p)]=h;return h
 protect(FR/'backbone_registry.json')
 for b in registry:
  assert b['seed'] in SEEDS and b['horizon'] in HS
  protect(b['checkpoint'],b['checkpoint_sha256'])
  cp,cj=best_completion(FR/'backbones'/b['id']/'cache');protect(cp)
  assert cj['validated'] and cj['checkpoint_sha256']==b['checkpoint_sha256']
  artifacts={Path(a['path']).name:a for a in cj['artifacts']}
  item={**{k:b[k] for k in ['id','backbone','input_set','mechanism','horizon','seed','checkpoint','checkpoint_sha256','provenance']},'cache':{},'old':{}}
  for sp in COUNTS:
   a=artifacts[sp+'_experts.npy'];protect(a['path'],a['sha256'])
   k=artifacts['backbone_'+sp+'.parquet'];protect(k['path'],k['sha256'])
   item['cache'][sp]=dict(experts=a['path'],experts_sha256=a['sha256'],keys=k['path'],keys_sha256=k['sha256'])
  mu=artifacts['train_mean_experts.npy'];protect(mu['path'],mu['sha256']);item['mean']=mu
  for v in ['fixed']+VARIANTS:
   p,j=best_completion(FR/'backbones'/b['id']/'head_frozen'/v);protect(p)
   ars={Path(a['path']).name:a for a in j['artifacts']}
   checkpoint=ars['best_routing_head.pt'];protect(checkpoint['path'],checkpoint['sha256'])
   item['old'][v]=dict(checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],predictions={})
   for sp in ['val','test']:
    a=ars[sp+'.parquet'];protect(a['path'],a['sha256']);item['old'][v]['predictions'][sp]=a
  items.append(item)
 assert len(items)==60
 js(OUT/'BINDINGS.json',items)
 js(OUT/'PROTECTED_INPUTS.json',protected)
 return items
def job_id(b,v,lr):return f"{b['id']}_{v}_lr{lr:.0e}"
def job_path(b,v,lr):return OUT/'fits'/job_id(b,v,lr)
def verify_job(p):
 j=json.loads(Path(p).read_text())
 for a in j['artifacts']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
 return j
def artifact(p):return dict(path=rel(p),sha256=sha(p))
