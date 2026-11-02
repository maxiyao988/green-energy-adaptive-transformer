"""Publish a stable inventory after all output logs have closed; no redirected log."""
from core import *
def main():
 previous=json.loads((OUT/'MANIFEST_STEP4.json').read_text());previous.pop('files');previous.pop('new_source');previous['completed_utc']=utc();previous['supersedes']='MANIFEST_STEP4.json (that inventory preceded the final line of final_manifest.log)'
 previous['files']=[dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.rglob('*')) if p.is_file()]
 previous['new_source']=[dict(path=rel(p),sha256=sha(p)) for p in sorted((ROOT/'revision_work/scripts/causal').glob('*.py'))]
 previous['authoritative_report']=rel(OUT/'REPORT_STEP4_INTERIM_CORRECTED.md')
 jsonout('MANIFEST_STEP4_VERIFIED.json',previous)
 for row in previous['files']+previous['new_source']:assert sha(ROOT/row['path'])==row['sha256'],row['path']
 print('Verified stable artifact hashes:',len(previous['files']),'outputs and',len(previous['new_source']),'source files')
 print('Stage 3a complete: 15 successful cells / 45 prediction files; h5 IMPROVED, h10 IMPROVED, h20 REVERSED; Stage3b/3c NOT RUN.')
if __name__=='__main__':main()
