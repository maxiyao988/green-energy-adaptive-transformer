"""Seal protocol/implementation/provenance before user-launched training."""
import sys,platform,importlib.metadata,csv
sys.dont_write_bytecode=True
import io_e as io

def main():
    assert not (io.OUT/'RNG_MANIFEST.json').exists(),'Manifest is immutable'
    assert not (io.OUT/'fits').exists() and not (io.OUT/'ALL_TRAINING_FINISHED.json').exists()
    for name in ['DATA_GATE.json','SIGMA_IDENTITY_GATE.json','PRERUN_REVIEW.json','FINAL_CODE_REVIEW.json','GUARD_REVIEW.json','ARCHIVED_SOURCE_GATE.json']:
        assert io.read(io.OUT/name)['passed'],name
    registered={}
    def include(path,expected=None):
        path=io.Path(path).resolve();actual=io.sha(path)
        assert expected is None or actual==expected,'Existing source changed: '+str(path)
        rel=io.rel(path)
        if rel in registered:assert registered[rel]==actual
        registered[rel]=actual
    # Recursive fixed protocol chain and all previously registered code/metadata.
    seen=set()
    def manifest(path):
        path=io.Path(path).resolve()
        if path in seen:return
        seen.add(path);include(path);m=io.read(path)
        for row in m.get('implementation',[])+m.get('source_manifests',[]):
            q=io.ROOT/row['path'];include(q,row['sha256'])
            if q.name=='RNG_MANIFEST.json':manifest(q)
    manifest(io.D/'RNG_MANIFEST.json');manifest(io.OLD/'RNG_MANIFEST.json');manifest(io.ROOT/'revision_work/outputs/step22b2/RNG_MANIFEST.json')
    with (io.OUT/'REUSED_MODULES.csv').open(newline='') as f:
        for row in csv.DictReader(f):
            q=io.ROOT/row['path']
            if not q.is_relative_to(io.OUT):include(q,row['sha256'])
    for q in [io.B2/'scripts/b2_worker.py',io.B2/'scripts/b2_data.py',io.B2/'tensor_manifest.json',io.B2/'design.json',io.D/'SIGMAS.json',io.D/'TRAIN_TARGET_SD.csv',io.D/'DATA_GATE.json',io.D/'data_hash_checks.csv',io.D/'existing_rolling_identity_checks.csv',io.D/'ANALYSIS_COMPLETE.json',io.ROOT/'revision_work/outputs/step22b2/ANALYSIS_COMPLETE.json',io.ROOT/'revision_work/outputs/analysis16/registry.csv',io.ROOT/'revision_work/outputs/analysis16/prediction_checks.csv',io.ROOT/'revision_work/outputs/analysis16/protected_label_sources.csv',io.OLD/'T0_SOURCES.json']:
        include(q)
    for root,suffix in [(io.D,'lstm_prediction_files.csv'),(io.ROOT/'revision_work/outputs/step22b2','prediction_files.csv')]:
        m=io.read(root/'ANALYSIS_COMPLETE.json');row=next(a for a in m['artifacts'] if a['path'].endswith('/'+suffix));include(io.ROOT/row['path'],row['sha256'])
    for row in io.read(io.D/'ANALYSIS_COMPLETE.json')['artifacts']:
        if row['path'].endswith('/protected_sources.csv'):include(io.ROOT/row['path'],row['sha256'])
    module_rows=[dict(path=p,sha256=h,role='reused source') for p,h in sorted(registered.items()) if p.endswith('.py')]
    with (io.OUT/'REUSED_MODULES_FINAL.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['path','sha256','role']);w.writeheader();w.writerows(module_rows)
    implementations=[io.artifact(q) for q in sorted(io.OUT.iterdir()) if q.is_file() and q.name!='RNG_MANIFEST.json']
    versions={}
    for name in ['numpy','pandas','torch','scipy']:
        versions[name]=importlib.metadata.version(name)
    sys.path.append(str(io.ROOT/'revision_work/outputs/_deps_pyarrow15'))
    import pyarrow
    versions['pyarrow']=pyarrow.__version__
    m=dict(protocol_sha256=io.sha(io.OUT/'PROTOCOL_STEP22E.md'),registered_utc=io.utc(),
      entrypoint='run_e_v5.py',worker='worker_e_v2.py',analysis='analyze_e_v3.py',independent_validation='validate_e_v2.py',
      training_seeds=io.SEEDS,loader_generator_seed=42,bootstrap_base_seed=20261022,B=20000,L='h',
      bootstrap_date_seed_sequence=[20261022,'h','window_code=1/2/3',0],bootstrap_paired_seed_sequence=[20261022,'h','window_code=1/2/3',1],
      date_blocks_shared_across_models_and_tests=True,paired_seeds_independent_across_windows=True,
      synthetic_qa_rng_seed=947,n_workers=3,torch_intra=12,torch_inter=12,OMP_NUM_THREADS=None,OMP_DYNAMIC='FALSE',MKL_DYNAMIC='FALSE',
      exact_sigmas=io.read(io.OUT/'SIGMAS.json')['values'],package_versions=versions,python=sys.version,platform=platform.platform(),machine=platform.machine(),
      execution_status_at_registration='No research training, optimizer steps, TEST prediction/label deserialization or research TEST evaluation; synthetic QA only. User will run the terminal command.',
      reused_module_index=io.rel(io.OUT/'REUSED_MODULES_FINAL.csv'),implementation=implementations,
      source_manifests=[dict(path=p,sha256=h) for p,h in sorted(registered.items())])
    io.js(io.OUT/'RNG_MANIFEST.json',m);assert io.check_protocol()==m['protocol_sha256']
    print('SEALED PROTOCOL SHA256:',m['protocol_sha256'])
    print('Implementation files:',len(implementations),'reused code/metadata:',len(registered),'reused Python modules:',len(module_rows))
    print('No training launched.')
if __name__=='__main__':main()
