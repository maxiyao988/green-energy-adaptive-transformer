"""Verify reviewed source hashes before starting the B3-gated B5 supervisor."""
from b5_io import *
review=json.loads((EXP/'REVIEW_MANIFEST.json').read_text())
for a in review['files']:
 assert sha(ROOT/a['path'])==a['sha256'],a['path']
from b5_queue_v2 import main
if __name__=='__main__':main()
