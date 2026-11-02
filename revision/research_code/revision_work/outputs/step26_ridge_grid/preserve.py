"""Opaque streaming hashes only; no deserialization of TEST data."""
from pathlib import Path
import hashlib,json,time,os
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 start=time.time();n=0;size=0
 with (OUT/'PREEXISTING_SHA256.jsonl').open('x') as f:
  for folder,dirs,files in os.walk(ROOT):
   dirs[:]=sorted(d for d in dirs if Path(folder)/d!=OUT)
   for name in sorted(files):
    p=Path(folder)/name
    if p.is_symlink():
     f.write(json.dumps(dict(path=str(p.relative_to(ROOT)),symlink=os.readlink(p)))+'\n');continue
    s=p.stat(); digest=sha(p);s2=p.stat()
    assert (s.st_size,s.st_mtime_ns)==(s2.st_size,s2.st_mtime_ns),str(p)
    f.write(json.dumps(dict(path=str(p.relative_to(ROOT)),size=s.st_size,mtime_ns=s.st_mtime_ns,sha256=digest))+'\n');n+=1;size+=s.st_size
    if n%5000==0:f.flush();print(n,size,round(time.time()-start,1),flush=True)
 with (OUT/'SNAPSHOT_COMPLETE.json').open('x') as f:json.dump(dict(n_files=n,total_bytes=size,seconds=time.time()-start),f,indent=2)
 print('SNAPSHOT_COMPLETE',n,size,time.time()-start,flush=True)
if __name__=='__main__':main()
