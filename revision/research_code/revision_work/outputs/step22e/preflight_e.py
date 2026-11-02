"""Each launch rechecks frozen bytes; pure SHA channel, no TEST deserialization."""
import sys,csv
sys.dont_write_bytecode=True
import io_e as io

def main():
    started=io.utc();io.check_protocol();n=0
    def rows(p):
        with p.open(newline='') as f:return list(csv.DictReader(f))
    for r in rows(io.OUT/'data_hash_checks.csv'):
        assert io.sha(io.ROOT/r['path'])==r['expected_sha256'];n+=1
    for r in rows(io.OUT/'existing_rolling_identity_checks.csv'):
        assert io.sha(io.ROOT/r['complete'])==r['complete_sha256'];n+=1
    for r in rows(io.OUT/'ARCHIVED_SOURCE_INVENTORY.csv'):
        for key in ['predictions','fit_complete','history','static_logits']:
            if not r[key]:continue
            expected=r['sha256'] if key=='predictions' else r[key+'_sha256']
            assert io.sha(io.ROOT/r[key])==expected,(key,r);n+=1
    assert io.read(io.OUT/'SIGMAS.json')['values']==io.read(io.D/'SIGMAS.json')['values']
    assert io.read(io.OUT/'ARCHIVED_SOURCE_GATE.json')['passed'] and io.read(io.OUT/'PRERUN_REVIEW.json')['passed']
    io.js(io.OUT/'preflight'/(io.stamp()+'.json'),dict(protocol_sha256=io.check_protocol(),started_utc=started,ended_utc=io.utc(),passed=True,opaque_hashes=n,test_deserialized=False))
    print('PREFLIGHT PASS:',n,'opaque hashes; TEST contents remain locked.',flush=True)
if __name__=='__main__':main()
