"""Final artifact inventory after the interim report; exclusive creation only."""
from core import *
import platform,importlib.metadata

def main():
 assert (OUT/'REPORT_STEP4_INTERIM.md').is_file()
 assert json.loads((OUT/'protected_inputs_verification.json').read_text())['passed']
 files=[]
 for p in sorted(OUT.rglob('*')):
  if p.is_file():files.append(dict(path=rel(p),bytes=p.stat().st_size,sha256=sha(p)))
 source=[dict(path=rel(p),sha256=sha(p)) for p in sorted((ROOT/'revision_work/scripts/causal').glob('*.py'))]
 versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn','matplotlib']}
 import pyarrow
 versions['pyarrow']=pyarrow.__version__
 jsonout('MANIFEST_STEP4.json',dict(completed_utc=utc(),python=sys.version,platform=platform.platform(),versions=versions,cutoff=CUTOFF,training_snapshot='ORIGINAL adjusted July snapshot only',original_invariance_test='FAILED at 1e-6',documented_tolerance=1e-4,precision_gate=json.loads((OUT/'precision_gate_decision.json').read_text()),successful_runs=15,aborted_static_attempts=2,stage3b='NOT RUN',stage3c='NOT RUN',files=files,new_source=source))
 print('MANIFEST COMPLETE',len(files),'outputs',len(source),'source files')
if __name__=='__main__':main()
