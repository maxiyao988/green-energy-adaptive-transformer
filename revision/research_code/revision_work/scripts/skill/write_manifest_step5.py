"""Write MANIFEST_STEP5.csv (path, bytes, SHA-256) for every file under the Step 5 output root."""
from skill_common import *

rows = [dict(path=rel(p), bytes=p.stat().st_size, sha256=sha256(p)) for p in sorted(SKILL_OUT.rglob("*")) if p.is_file()]
write_csv(SKILL_OUT / "MANIFEST_STEP5.csv", pd.DataFrame(rows))
print(f"{len(rows)} files")
