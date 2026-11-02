"""Reviewed B5 restart: explicit-root B3 audit; old logs/scripts retained."""
from b5_io import *
review=json.loads((EXP/'REVIEW_MANIFEST_V2.json').read_text())
for a in review['files']:
 assert sha(ROOT/a['path'])==a['sha256'],a['path']
from b5_queue_v3 import main
if __name__=='__main__':main()
