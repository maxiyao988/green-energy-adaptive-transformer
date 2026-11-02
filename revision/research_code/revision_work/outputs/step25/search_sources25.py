"""Stored-result retrieval only. No project imports, prediction evaluation or statistical calculation."""
import sys,csv,json,re,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path.cwd();OUT=ROOT/'revision_work/outputs/step25';seen={};inventory=[];linehits=[];result=[]
def read(p):
 p=Path(p);p=p if p.is_absolute() else ROOT/p;b=p.read_bytes();seen[str(p.relative_to(ROOT))]=hashlib.sha256(b).hexdigest();return b
def table(p):return list(csv.DictReader(read(p).decode('utf-8-sig').splitlines()))
def save(name,rows):
 with (OUT/name).open('x',newline='') as f:
  fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
scopes=[ROOT/'revision_work/outputs/skill_analysis/a3_original',ROOT/'revision_work/outputs/input_sensitivity/step12a/part2_causal_a3',ROOT/'revision_work/outputs/analysis16',ROOT/'revision_work/outputs/analysis16/clarified',ROOT/'dataset/audit/experiment4_inference',ROOT/'revision_work/outputs/manuscript18']
patterns=re.compile(r'\bHAC\b|hac_|Bartlett|Diebold|\bDM\b|\bDM[_-]|\b2h\b|block.length|SENSITIVITY',re.I)
for scope in scopes:
 for p in sorted(scope.glob('*.csv')):
  if p.name in ['a3_bootstrap_replicates.csv','experiment4_block_bootstrap_replicates.csv','a3_seed_date_differences.csv','experiment4_loss_diff_by_seed_date.csv','registry.csv','protected_prediction_sources.csv','prediction_checks.csv','ARTIFACTS.csv','ARTIFACTS_FINAL.csv'] or p.name.startswith('MANIFEST'):continue
  text=read(p).decode('utf-8-sig');lines=text.splitlines();matched=[]
  for i,line in enumerate(lines,1):
   if patterns.search(line):linehits.append(dict(path=str(p.relative_to(ROOT)),line=i,text=line));matched.append(i)
  inventory.append(dict(path=str(p.relative_to(ROOT)),kind='stored CSV',header=lines[0] if lines else '',lines_including_header=len(lines),matching_lines=len(matched),sha256=seen[str(p.relative_to(ROOT))]))
docs=['docs/experiment4_inference_protocol.md','revision_work/scripts/skill/a3_hierarchical.py','scripts/python/panel_pathformer_experiment4_inference.py','revision_work/outputs/skill_analysis/REPORT_STEP5.md','revision_work/outputs/input_sensitivity/step12a/REPORT_STEP12A.md','revision_work/outputs/input_sensitivity/step12a/REPORT_STEP12A_PART1.md','revision_work/outputs/fullrun/REPORT_STEP14A.md','revision_work/outputs/repro_threads/REPORT_STEP3B.md','revision_work/outputs/analysis16/REPORT_STEP16_FINAL.md','revision_work/scripts/analysis16/statistics16.py','revision_work/scripts/analysis16/statistics16_v2.py','revision_work/scripts/analysis16/statistics16_v3.py','revision_work/outputs/skill_analysis/a3_original/a3_run_info.json','revision_work/outputs/input_sensitivity/step12a/part2_causal_a3/a3_run_info.json']
profre=re.compile(r'12/12|3/1|Bartlett|Diebold|\bDM\b|hac_|HAC_|2\s*\*\s*h|SENSITIVITY|Adaptive\s*[−-]\s*Static|Static\s*[−-]\s*Adaptive')
for name in docs:
 p=ROOT/name
 if not p.exists():inventory.append(dict(path=name,kind='named code/document',status='NOT FOUND'));continue
 text=read(p).decode('utf-8-sig');lines=text.splitlines();n=0
 for i,line in enumerate(lines,1):
  if profre.search(line):linehits.append(dict(path=name,line=i,text=line));n+=1
 inventory.append(dict(path=name,kind='provenance document/code (not result)',lines_including_header=len(lines),matching_lines=n,sha256=seen[name]))
# Copy exact stored result rows and their native signs. Never negate estimates or intervals.
for inp,base,profile,step in [('O','revision_work/outputs/skill_analysis/a3_original','12/12 archived prediction family','Step5 / original Experiment4 inference'),('C','revision_work/outputs/input_sensitivity/step12a/part2_causal_a3','3/1, OMP=3','Step12A part2 / Stage3a-3b')]:
 for name,method in [('a3_date_only_mbb.csv','date_only'),('a3_joint_seed_date_bootstrap.csv','joint_seed_date')]:
  path=base+'/'+name
  for r in table(path):
   if r['metric']=='loss':result.append(dict(input_set=inp,source_step=step,prediction_profile=profile,native_contrast='Adaptive-minus-Static',method=method,source=path,**r))
 seeds=table(base+'/a3_seed_level_tests.csv');print('SEED SOURCE',inp,[{k:v for k,v in r.items() if k in ['h','metric','seeds']} for r in seeds if r.get('metric')=='loss'])
 # Date/key retrieval from stored contrast rows, not recomputation of those contrasts.
 p=base+'/a3_seed_date_differences.csv';r=table(p);dates=sorted({x['anchor_date'] for x in r});inventory.append(dict(path=p,kind='stored contrast key/date evidence',observed_first_date=dates[0],observed_last_date=dates[-1],observed_seed_labels=','.join(sorted({x['seed'] for x in r})),sha256=seen[p]))
save('STORED_A3_LOSS_ROWS_NATIVE_ORIENTATION.csv',result)
target=[]
for name in ['A_paired_tests.csv','C_paired_tests.csv']:
 path='revision_work/outputs/analysis16/'+name
 for r in table(path):
  if r.get('contrast')=='static−adaptive' and r.get('metric')=='squared_loss' and r.get('window')=='W3':target.append(dict(source=path,**r))
save('ANALYSIS16_TARGET_ROWS_AS_STORED.csv',target)
for r in target:print('ANALYSIS16',r['source'],r['input_set'],r['h'],r['method'],'L',r['L'],'nSeeds',r['n_seeds'])
# Confirm exact native-orientation Step18 references by copying rather than rescoring.
p='revision_work/outputs/manuscript18/T4_native_orientation_references.csv';native=[r for r in table(p) if r.get('metric')=='loss'];save('STEP18_NATIVE_LOSS_REFERENCE_ROWS_AS_STORED.csv',native)
# Preserve all workbook hits, and distinguish irrelevant NW / literal DM-in-README matches.
raw=table(OUT/'WORKBOOK_CELL_HITS_RAW.csv');classified=[]
for r in raw:
 text=(r['block']+' '+r['column_header']+' '+r['cell_text']+' '+r['source_cell_text']).lower();why=''
 if 'Newey' in r['matched_terms']:
  if 'e3_confirmatory' in text or 'e4_confirmatory' in text:why='无关：W1/W2 T3对比中NW字段占位或说明；非W3五种子同一损失差HAC结果'
  elif r['sheet']=='Table S12':why='无关：Step17缩减预测的Clark–West/NW参考，不是Static−Adaptive平方损失差HAC'
  elif 'cw' in text or 'clark' in text:why='无关：Newey–West用于Clark–West（或与Rank IC并列），不是目标损失差HAC'
  elif 'rank' in text or 'ic' in text:why='无关：Newey–West用于Rank IC，不是目标损失差HAC'
  else:why='无关：其他窗口/设置的NW字段或方法说明，非目标W3损失差HAC'
 elif 'DM' in r['matched_terms'] and r['standalone_DM']=='False':why='无关：DM字面子串仅来自README，不是DM检验'
 elif 'sensitivity' in r['matched_terms']:why='无关：冻结专家后重训head的敏感性，或组合成本/风险厌恶敏感性；不是MBB L=2h'
 else:why='待核对'
 classified.append(dict(**r,relevance='无关',reason=why))
save('WORKBOOK_CELL_HITS_CLASSIFIED.csv',classified);save('STORED_SEARCH_LINE_HITS.csv',linehits);save('STORED_SOURCE_INVENTORY.csv',inventory)
hashes=[]
for p,sha in seen.items():now=hashlib.sha256((ROOT/p).read_bytes()).hexdigest();hashes.append(dict(path=p,sha256_before=sha,sha256_after=now,unchanged=sha==now))
assert all(r['unchanged'] for r in hashes);save('SOURCE_HASHES.csv',hashes)
print('SOURCE inventory',len(inventory),'linehits',len(linehits),'classified workbook hits',len(classified),'sources unchanged',len(hashes))
