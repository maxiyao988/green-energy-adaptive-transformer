"""Opaque launch gate and fixed target-sigma identity."""
import sys,csv
sys.dont_write_bytecode=True
import io_g as io
def main():
    io.check_protocol();n=0
    with (io.OUT/'tensor_hash_checks.csv').open(newline='') as f:
        for r in csv.DictReader(f):assert io.sha(io.ROOT/r['path'])==r['expected_sha256'];n+=1
    assert io.read(io.OUT/'DATA_GATE.json')['passed'] and io.read(io.OUT/'PRERUN_REVIEW.json')['passed']
    io.js(io.OUT/'preflight'/(io.stamp()+'.json'),dict(protocol_sha256=io.check_protocol(),utc=io.utc(),passed=True,opaque_hashes=n,test_deserialized=False))
    print('PREFLIGHT PASS; TEST contents locked.',flush=True)
if __name__=='__main__':main()
