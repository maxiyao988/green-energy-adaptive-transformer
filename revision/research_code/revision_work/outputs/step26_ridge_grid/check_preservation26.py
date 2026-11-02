"""Rehash all pre-existing files; report user-side relocations without altering sources."""
from preserve import *
def main():
 start=time.time();n=0;issues=[];oldpaths=set();newpaths=[]
 with (OUT/'PREEXISTING_VERIFICATION.jsonl').open('x') as out:
  with (OUT/'PREEXISTING_SHA256.jsonl').open() as f:
   for line in f:
    r=json.loads(line);p=ROOT/r['path'];oldpaths.add(r['path']);n+=1
    if not p.exists():z=dict(path=r['path'],status='MISSING',sha256_before=r.get('sha256'))
    elif 'symlink' in r:z=dict(path=r['path'],status='UNCHANGED' if p.is_symlink() and os.readlink(p)==r['symlink'] else 'CHANGED_SYMLINK')
    else:
     actual=sha(p);z=dict(path=r['path'],status='UNCHANGED' if actual==r['sha256'] else 'CHANGED_BYTES',sha256_before=r['sha256'],sha256_after=actual)
    out.write(json.dumps(z)+'\n')
    if z['status']!='UNCHANGED':issues.append(z)
  for folder,dirs,files in os.walk(ROOT):
   dirs[:]=[d for d in dirs if Path(folder)/d!=OUT]
   for name in files:
    p=Path(folder)/name;rel=str(p.relative_to(ROOT))
    if rel not in oldpaths:newpaths.append(dict(path=rel,sha256=sha(p) if not p.is_symlink() else None))
 with (OUT/'PREEXISTING_PRESERVATION_CHECK.json').open('x') as f:json.dump(dict(n_records=n,all_preexisting_paths_and_bytes_unchanged=not issues,issues=issues,new_external_paths=newpaths,seconds=time.time()-start,assistant_writes='restricted to step26 output; input folder relocation was user-initiated during this session'),f,indent=2)
 print('PRESERVATION_CHECK',n,'issues',len(issues),'external_new',len(newpaths),time.time()-start,flush=True)
if __name__=='__main__':main()
