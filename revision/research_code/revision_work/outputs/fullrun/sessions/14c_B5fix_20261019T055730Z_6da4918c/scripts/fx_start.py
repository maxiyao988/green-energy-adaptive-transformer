"""Verify reviewed B5FIX sources before starting its exclusive scheduler."""
from fx_io import *
r=json.loads((EXP/'REVIEW_MANIFEST.json').read_text())
for a in r['files']:assert sha(ROOT/a['path'])==a['sha256'],a['path']
from fx_queue import main
if __name__=='__main__':main()
