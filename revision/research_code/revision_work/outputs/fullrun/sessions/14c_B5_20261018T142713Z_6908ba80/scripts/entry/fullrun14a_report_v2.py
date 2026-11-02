import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from b5_report import phase_report
phase_report(Path(sys.argv[1]))
