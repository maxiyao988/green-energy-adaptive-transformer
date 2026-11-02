"""Finalize the report with signed logging differences and precise location wording."""
from pathlib import Path
p=Path(__file__).with_name('build_report_final.py')
s=p.read_text()
s=s.replace('Repository paths are relative to the repository root; runtime library paths are absolute.','Paths within the repository (including its .venv library sources) are relative to the repository root; external paths are absolute.')
s=s.replace('### h=20 by calendar year within each partition','### h=20 by anchor-date calendar year within each partition')
s=s.replace('**(A) directly explains the principal scale of the gap; (B) is a verified additional logging contribution.**','**(A) explains the principal scale of the gap; (B) is a verified logging asymmetry whose net effect varies by cell.**')
s=s.replace("}/20 cells, by {fmt(min(train_reduction))}–{fmt(max(train_reduction))}, confirming a combined mode/timing contribution without isolating dropout.","}/20 cells; the signed logged-minus-eval differences span {fmt(min(train_reduction))} to {fmt(max(train_reduction))}, measuring the combined mode/timing effect without isolating dropout.")
s=s.replace('while logging adds a smaller observed contribution; no dropout-only causal magnitude is claimed.','while the logging difference is smaller and can have either sign; no dropout-only causal magnitude is claimed.')
exec(compile(s,str(p),'exec'),{'__file__':str(p),'__name__':'__main__'})
