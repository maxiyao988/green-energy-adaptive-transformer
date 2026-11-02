"""Seal the immutable Step22F protocol and imported-code/RNG provenance."""
import sys,csv,platform,ast,importlib.metadata
sys.dont_write_bytecode=True
import io_g as io

def main():
    assert not (io.OUT/'RNG_MANIFEST.json').exists(),'Manifest immutable'
    assert not (io.OUT/'fits').exists(),'No training may precede registration'
    for name in ['DATA_GATE.json','PRERUN_REVIEW.json','GUARD_REVIEW.json','DATA_GATE.json']:assert io.read(io.OUT/name)['passed']
    source={};seen=set()
    def include(path,expected=None):
        path=io.Path(path).resolve();actual=io.sha(path);assert expected is None or actual==expected,'Original source changed: '+str(path)
        key=io.rel(path)
        if key in source:assert source[key]==actual
        source[key]=actual
    def manifest(path):
        path=io.Path(path).resolve()
        if path in seen:return
        seen.add(path);include(path);m=io.read(path)
        for r in m.get('implementation',[])+m.get('source_manifests',[]):
            p=io.ROOT/r['path'];include(p,r['sha256'])
            if p.name=='RNG_MANIFEST.json':manifest(p)
    manifest(io.F/'RNG_MANIFEST.json');manifest(io.B2RUN/'RNG_MANIFEST.json')
    with (io.OUT/'REUSED_MODULES.csv').open(newline='') as f:
        for r in csv.DictReader(f):
            p=io.ROOT/r['path']
            if not p.is_relative_to(io.OUT):include(p,r['sha256'])
    for p in [io.TENSOR_BASE/'MANIFEST_STEP4_VERIFIED.json',io.ROOT/'revision_work/outputs/analysis16/registry.csv',io.ROOT/'revision_work/outputs/analysis16/prediction_checks.csv',io.ROOT/'revision_work/outputs/analysis16/protected_label_sources.csv',io.OLD/'T0_SOURCES.json']:
        include(p)
    for h in io.HS:include(io.ROOT/f'revision_work/outputs/analysis16/labels/W3_h{h}.parquet')
    all_modules=[dict(path=p,sha256=h,role='reused Python source; includes AST-only imports') for p,h in sorted(source.items()) if p.endswith('.py')]
    with (io.OUT/'REUSED_MODULES_FINAL.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','sha256','role']);w.writeheader();w.writerows(all_modules)
    for p in io.OUT.glob('*.py'):ast.parse(p.read_text(),str(p))
    versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scipy']}
    sys.path.append(str(io.ROOT/'revision_work/outputs/_deps_pyarrow15'));import pyarrow;versions['pyarrow']=pyarrow.__version__
    implementations=[io.artifact(p) for p in sorted(io.OUT.iterdir()) if p.is_file() and p.name!='RNG_MANIFEST.json']
    io.js(io.OUT/'RNG_MANIFEST.json',dict(protocol_sha256=io.sha(io.OUT/'PROTOCOL_STEP22G.md'),registered_utc=io.utc(),entrypoint='run_g_v2.py',worker='worker_g.py',analysis='analyze_g_v2.py',independent_validation='validate_g_v2.py',
      training_seeds=io.SEEDS,loader_generator_seed=42,bootstrap_seed_order=io.BOOT_SEEDS,bootstrap_base_seed=20261019,bootstrap_seed_sequence=[20261019,'h','L=h',1],draw_order='paired seed indices first, then overlapping noncircular date blocks; exact Step16 AST',B=20000,L='h',n_workers=3,torch_intra=12,torch_inter=12,OMP_NUM_THREADS=None,OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',
      initialization_and_training_rng='original Step22B2 T3 Python/NumPy/Torch run-seed RNG; formal DataLoader conventions unchanged',synthetic_qa_rng_seed=947,target='float64 date-demeaned, float32 model target, no scaling',sigma_cs=io.read(io.OUT/'SIGMAS_CS.json')['values'],long_short='top5 +0.2, bottom5 -0.2; gross200%; 10bp full L1 turnover; excludes borrowing costs/cash interest; descriptive only',
      package_versions=versions,python=sys.version,platform=platform.platform(),machine=platform.machine(),execution_status_at_registration='No Step22G training, optimizer steps, research TEST deserialization or research metrics; assistant launches after sealing.',
      implementation=implementations,source_manifests=[dict(path=p,sha256=h) for p,h in sorted(source.items())],reused_python_source_count=len(all_modules),reused_python_source_index=io.rel(io.OUT/'REUSED_MODULES_FINAL.csv')))
    print('SEALED STEP22G PROTOCOL:',io.check_protocol())
    print('Implementation files:',len(implementations),'reused source/metadata files:',len(source),'reused Python modules:',len(all_modules))
    print('Training remains unstarted.')
if __name__=='__main__':main()
