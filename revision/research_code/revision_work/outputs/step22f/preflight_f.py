"""Each launch rechecks immutable source bytes without parsing TEST."""
import sys,csv
sys.dont_write_bytecode=True
import io_f as io

def main():
    start=io.utc();io.check_protocol();n=0
    def rows(path):
        with path.open(newline='') as f:return list(csv.DictReader(f))
    for r in rows(io.OUT/'tensor_hash_checks.csv'):
        assert io.sha(io.ROOT/r['path'])==r['expected_sha256'];n+=1
    for r in rows(io.OUT/'REFERENCE_SOURCES.csv'):
        for key in ['source','canonical','complete','history','static_logits']:
            if r[key]:assert io.sha(io.ROOT/r[key])==r[key+'_sha256'];n+=1
    for r in rows(io.OUT/'trainval_identity_checks.csv'):assert io.sha(io.ROOT/r['source'])==r['sha256'];n+=1
    assert io.read(io.OUT/'DATA_GATE.json')['passed'] and io.read(io.OUT/'PRERUN_REVIEW.json')['passed']
    io.js(io.OUT/'preflight'/(io.stamp()+'.json'),dict(protocol_sha256=io.check_protocol(),started_utc=start,ended_utc=io.utc(),passed=True,opaque_hashes=n,test_deserialized=False))
    print('PREFLIGHT PASS:',n,'opaque source hashes; no TEST parsing.',flush=True)
if __name__=='__main__':main()
