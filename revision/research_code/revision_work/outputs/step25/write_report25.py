"""Compose a search report using copied metadata/results only; no statistics are computed."""
import sys,csv,json,hashlib,datetime
from pathlib import Path
sys.dont_write_bytecode=True
R=Path.cwd();O=R/'revision_work/outputs/step25'
def rows(n):return list(csv.DictReader((O/n).open()))
def save(n,rs):
 with (O/n).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
manifest=json.loads((O/'WORKBOOK_SEARCH_MANIFEST.json').read_text());book=R/manifest['workbook'];assert hashlib.sha256(book.read_bytes()).hexdigest()==manifest['sha256_before']
baseO='revision_work/outputs/skill_analysis/a3_original/';baseC='revision_work/outputs/input_sensitivity/step12a/part2_causal_a3/';oldhac='dataset/audit/experiment4_inference/experiment4_hac_summary.csv';matrix=[]
for a,title in [('A','MBB L=2h'),('B','HAC/Bartlett'),('C','DM-style HAC')]:
 for inp,base,prof,step in [('O',baseO,'12/12 archived O prediction family','Step5 / archived Experiment4; Step12A/14a provenance'),('C',baseC,'3/1, OMP=3','Step12A part2 / Stage3a-3b')]:
  src=base+'a3_date_only_mbb.csv'
  if a=='A':src+='; '+base+'a3_joint_seed_date_bootstrap.csv'
  elif inp=='O':src=oldhac+'; '+src
  matrix.append(dict(analysis=a,analysis_name=title,input_set=inp,storage_status='STORED_NOT_IN_SUPPLEMENT',supplement_sheet_cells='NONE',stored_sources=src,prediction_thread_profile=prof,source_step=step,window='W3; stored391-date range2024-11-12 to2026-06-05',seeds='0,1,21,42,3407',native_orientation='Adaptive-minus-Static; not literal requested Static-minus-Adaptive',uniform12_prediction_profile_matches=inp=='O',requested_native_sign_matches=False,uniform12_C_version='NOT_STORED' if inp=='C' else 'not applicable',DM_qualification='Same HAC mean-loss-differential statistic, not a separate independent/classical DM output' if a=='C' else '',strict_scope_note='Historical result exists; not present in supplement. Reverse native sign explicitly retained; C additionally has wrong3/1 prediction profile.'))
save('SENSITIVITY_3X2_CLASSIFICATION.csv',matrix)
# All listed values are stored strings copied exactly, without sign conversion or evaluation.
stored=rows('STORED_A3_LOSS_ROWS_NATIVE_ORIENTATION.csv');sensitivity=[r for r in stored if r['specification']=='SENSITIVITY'];hac=[r for r in stored if r['method']=='date_only' and r['specification']=='PRIMARY'];save('MBB_L2H_ROWS_COPIED_AS_STORED.csv',sensitivity)
save('HAC_DM_ROWS_COPIED_AS_STORED.csv',[{k:r.get(k,'') for k in ['input_set','source_step','prediction_profile','native_contrast','h','n_dates','n_seeds','hac_se','hac_z','hac_p_two_sided','source']} for r in hac])
hits=rows('WORKBOOK_CELL_HITS_CLASSIFIED.csv');coverage=rows('WORKBOOK_SHEET_COVERAGE.csv');num=rows('WORKBOOK_STRUCTURED_BLOCK_LABEL_ROWS.csv');assert not any(x['is_label_L_equals_2h']=='True' for x in num)
text='''# REPORT STEP25 — Sensitivity-result read-only search

**结论：(iii) 补充表里没有，或口径不一致。**

## 1. 检索对象与判定口径

使用 `revision_work/outputs/step23b/Supplementary_Tables_S1-S18_v2.xlsx`；文件存在，未回退到Step23 v1。只读取XLSX压缩包中的XML，没有调用Excel计算引擎、计算公式、训练、重算指标或检验，也没有保存/修改工作簿。所有新文件均在`step25/`以exclusive creation创建。

检索全部19个sheet（包括README和所有表），逐个访问524,196个存储单元格，包括存储的空白/格式单元格；未存储的空白格没有可检索文字。大小写不敏感检索HAC、Bartlett、Diebold、DM、2h、L=2h、L = 2h、block length、sensitivity；另外检索Newey–West/NW字段。公式如有则只检索公式文字及已缓存内容，不执行公式。

“存储状态”与“是否符合正文版本”分开列明：下表的STORED_NOT_IN_SUPPLEMENT表示能找到该项配对诊断的历史存储，不表示已经满足12/12和原始差值方向要求。**六格均不是PRESENT_IN_SUPPLEMENT。C的12/12目标版本A/B/C均NOT_STORED；O有对应归档预测族的存储，但历史原生方向均为Adaptive−Static，而非正文指定的Static−Adaptive。** 本任务没有对估计值、区间或z值换号，也没有据此生成一份新的结果。

## 2. 三项分析 × 两种输入

| 分析 | O：Price-level inputs | C：Window-relative inputs |
|---|---|---|
| A：MBB L=2h | **STORED_NOT_IN_SUPPLEMENT**。12/12归档O预测族，W3，五种子；Step5。文件：`revision_work/outputs/skill_analysis/a3_original/a3_date_only_mbb.csv`及`a3_joint_seed_date_bootstrap.csv`，筛选`metric=loss, specification=SENSITIVITY`，h=5/10/20、L=10/20/40。原生方向Adaptive−Static。 | **STORED_NOT_IN_SUPPLEMENT（仅历史3/1）**。3/1、OMP=3，W3，五种子；Step12A part2。文件：`revision_work/outputs/input_sensitivity/step12a/part2_causal_a3/a3_date_only_mbb.csv`及`a3_joint_seed_date_bootstrap.csv`，同样筛选loss/SENSITIVITY。原生方向Adaptive−Static。**12/12版本：NOT_STORED。** |
| B：HAC/Bartlett | **STORED_NOT_IN_SUPPLEMENT**。12/12归档O预测族，W3，五种子；原Experiment4及Step5复现。文件：`dataset/audit/experiment4_inference/experiment4_hac_summary.csv`；复现字段在`revision_work/outputs/skill_analysis/a3_original/a3_date_only_mbb.csv`的`hac_se,hac_z,hac_p_two_sided`。Bartlett、最大lag=h−1。原生方向Adaptive−Static。 | **STORED_NOT_IN_SUPPLEMENT（仅历史3/1）**。Step12A part2；文件：`revision_work/outputs/input_sensitivity/step12a/part2_causal_a3/a3_date_only_mbb.csv`，`metric=loss`行的`hac_se,hac_z,hac_p_two_sided`。Bartlett、lag=h−1；W3、五种子；原生方向Adaptive−Static。**12/12版本：NOT_STORED。** |
| C：DM | **STORED_NOT_IN_SUPPLEMENT（DM-style HAC，同B）**。12/12归档O预测族；对应文件仍是`dataset/audit/experiment4_inference/experiment4_hac_summary.csv`及`revision_work/outputs/skill_analysis/a3_original/a3_date_only_mbb.csv`的上述HAC字段。原协议明确使用“DM-style HAC secondary diagnostic”称呼同一统计量，不是独立第三份DM结果。W3、五种子；原生方向Adaptive−Static。 | **STORED_NOT_IN_SUPPLEMENT（仅历史3/1；DM-style HAC，同B）**。Step12A复用同一Step5 A3过程，文件：`revision_work/outputs/input_sensitivity/step12a/part2_causal_a3/a3_date_only_mbb.csv`的HAC字段。没有单独的DM文件/列；这里的存储判断仅指原协议对同一HAC诊断的命名。W3、五种子；原生方向Adaptive−Static。**12/12版本：NOT_STORED。** |

以上各路径的完整机器可读判定见`SENSITIVITY_3X2_CLASSIFICATION.csv`。若把“口径一致”进一步限定为**存储数值本身已经是Static−Adaptive方向**，上述历史A3/HAC文件均不满足这一原始方向条件；它们只是同一配对损失差的逆向存储，不能原样抄成Static−Adaptive。analysis16虽有正确方向的12/12 W3比较，但仅有L=h的MBB结果，不能据此填补A/B/C。

## 3. 工作簿命中与无关项

总计646个命中单元格：33个sensitivity、11个DM字面子串、602个额外NW相关命中。HAC、Bartlett、Diebold、独立词DM、2h、L=2h、L = 2h和block length均无命中。11个DM子串都出现在“README”文字中，不能作为DM检验已报告的证据。

为排除“没有写L=2h文字，但数值表内有L=10/20/40”的情况，还逐行检索了同时含h/horizon与L标签的3,603行；没有L标签等于2h的行。这里只比较标签，不计算统计量。记录见`WORKBOOK_STRUCTURED_BLOCK_LABEL_ROWS.csv`。

**每个命中的sheet、单元格、行标签、列标题、完整文字、片段及无关原因都在`WORKBOOK_CELL_HITS_CLASSIFIED.csv`。** 原始未分类记录在`WORKBOOK_CELL_HITS_RAW.csv`；覆盖清单在`WORKBOOK_SHEET_COVERAGE.csv`。代表性位置如下：

| Sheet / 单元格 | 存储内容或上下文 | 判定 |
|---|---|---|
| Table S3!H7 / J7 | `cw p nw vs train mean` / `rank ic nw t` | **无关**：Clark–West / Rank IC，不是Static−Adaptive损失差HAC。 |
| Table S8!Q9 / R9 | `cw nw t` / `p nw` | **无关**：滚动窗口CW参考，不是目标均值损失差HAC。 |
| Table S12!AM7 / AO7 | `cw p nw raw` / `cw p nw holm Step16` | **无关**：Step17缩减预测的CW参考。 |
| Table S15!A130 / G132 | `Rank IC inference,NW and MBB` / `nw se` | **无关**：池化窗口Rank IC的NW，不是目标损失差HAC。 |
| Table S16!F197 / G197 | `nw se` / `nw t`，E2 CW块 | **无关**：W1/W2 held-out CW。其他E3/E4块中的NW字段为占位/其他窗口说明，也不构成目标HAC结果。 |
| Table S18!E49 / F49 / G49 | `nw se` / `nw t` / `nw lag` | **无关**：EXPLORATORY CS目标Rank IC，不是本任务的原目标。 |
| README!B15；Table S11!A1 | Cost and risk-aversion sensitivity | **无关**：组合成本/风险厌恶敏感性。 |
| Table S6!A1401；Table S13!A135 | experts frozen, head retrained sensitivity | **无关**：冻结表示的head-retrained敏感性，不是L=2h。 |
| Table S5!A3等11格 | “Notes and definitions: README...” | **无关**：DM只是README中的子串。 |

所有602个NW命中均单独标注为“无关”，并保留各自CW、Rank IC、其他窗口或占位字段的原因。没有把这些命中合并计作B或C已经出现在补充材料。

## 4. 存储来源核对

### Step5 / a3_original：有历史O的三项诊断

- `a3_date_only_mbb.csv`与`a3_joint_seed_date_bootstrap.csv`包含loss的PRIMARY及SENSITIVITY行；后者的L依次为10/20/40，均有`n_seeds=5,n_dates=391,B=20000`。
- `a3_date_only_mbb.csv`在loss行保存`hac_se,hac_z,hac_p_two_sided`；原Experiment4的`experiment4_hac_summary.csv`另有`max_lag,long_run_variance,se_mean,z_stat,p_value_raw,p_value_holm`。存在存储结果，不只是脚本里写了一个方法。
- `a3_seed_level_tests.csv`原样列出`0,1,21,42,3407`；已存储损失差表的日期范围为2024-11-12至2026-06-05，即W3。`a3_run_info.json`及`REPORT_STEP5.md:228–230`记载原归档MBB/HAC已复现；本任务没有重新运行该复现。
- O线程标签指**预测的训练/数值版本**，不指MBB检索脚本的线程。`REPORT_STEP14A.md:40`说明正文O Static/Adaptive沿用归档formal预测；`REPORT_STEP12A.md:17–39`与`REPORT_STEP12A_PART1.md:46–60`保存所测归档单元的12/12 EXACT证据。这里据这些既有证据将O归为12/12兼容归档预测族，不声称本次重新认证了每个历史进程的完整遥测。

### Step12A：C有存储，但仅3/1

`revision_work/outputs/input_sensitivity/step12a/REPORT_STEP12A.md:13`明确写明C来自Stage3a seed42与Stage3b其他四种子，使用3/1、OMP=3，并非12/12。Part1的12/12复现不能拿来给Part2的C改标签。

`part2_causal_a3/a3_date_only_mbb.csv`及`a3_joint_seed_date_bootstrap.csv`确有loss/SENSITIVITY行和HAC字段；种子及391日期均与W3对应。它们支持“历史结果已存储”，不支持“正文12/12 C版本已有这三项结果”。`part2_causal_7b`是对**种子集成预测**的重采样，不与本任务“股票均值→五种子均值”的原损失差混用。

### analysis16：正文版本有L=h，不能替代L=2h或HAC/DM

只读筛选`A_paired_tests.csv`及`C_paired_tests.csv`中`window=W3, contrast=static−adaptive, metric=squared_loss`的已存储行。O/C都存在`date_only`及`joint_seed_date`，五种子、391日期，但L均为h（5/10/20）；没有L=2h行，也没有同一损失差的HAC/DM统计列。两文件中的W3比较是重复视图，不算两次独立分析。

筛选原行见`ANALYSIS16_TARGET_ROWS_AS_STORED.csv`。其他CW/Rank IC的Newey–West统计量不替代损失差HAC。analysis16中C使用的是14a的12/12预测版本，与Step12A的旧3/1结果区分明确。

### Step18 T4_native_orientation_references：有引用，不等于已进入补充工作簿

`revision_work/outputs/manuscript18/T4_native_orientation_references.csv`保留O归档及C旧3/1的PRIMARY/SENSITIVITY引用；其C的`profile`字段直接写明“C Stage3a/3b 3/1; NOT uniform12”。`contrast`字段明确为“Adaptive−Static (native stored orientation)”。该CSV没有HAC或DM结果列，不能因为有T4提取文件就视为工作簿已经包含三项诊断。

其loss行原样复制到`STEP18_NATIVE_LOSS_REFERENCE_ROWS_AS_STORED.csv`；未换号、未重算。

### B与C不是两份独立检验

`docs/experiment4_inference_protocol.md:366–433`规定同一391日期、先跨股票和五个固定种子聚合的均值损失差，使用Bartlett权重、lag=h−1。随后原文在433–436行明确：

> Call this a "HAC mean-loss-differential test" or "DM-style HAC secondary diagnostic". Do NOT present it as a literal classical single-series Diebold-Mariano setup without qualification, because the loss differential has first been aggregated across the stock panel and five fixed seeds.

因此B/HAC与C/DM-style在原协议中指同一个次要诊断；没有发现另一个独立经典DM文件、列或另一套p值。`scripts/python/panel_pathformer_experiment4_inference.py:681–730`及`revision_work/scripts/skill/a3_hierarchical.py:103–114`是相应存储来源的代码证据，仅被读取，没有执行。

## 5. 文件与只读核验

- `SENSITIVITY_3X2_CLASSIFICATION.csv`：六格存储、线程、步骤、方向及严格口径说明。
- `WORKBOOK_CELL_HITS_CLASSIFIED.csv`：全部646个命中及行/列上下文、无关标注。
- `WORKBOOK_SHEET_COVERAGE.csv`、`WORKBOOK_SEARCH_MANIFEST.json`：逐sheet覆盖、精确工作簿路径及检索前后SHA-256。
- `MBB_L2H_ROWS_COPIED_AS_STORED.csv`、`HAC_DM_ROWS_COPIED_AS_STORED.csv`：历史原行/字段摘录；所有数值字符串按存储复制，不生成换号版本。
- `STORED_SOURCE_INVENTORY.csv`、`STORED_SEARCH_LINE_HITS.csv`、`SOURCE_HASHES.csv`：已检查的存储结果、文件/行证据及读取前后哈希。

所有被核验的既有源文件和工作簿SHA-256均未变化。本报告没有提出或执行新增计算。
'''
with (O/'REPORT_STEP25_SENSITIVITY_SEARCH.md').open('x') as f:f.write(text)
# Final immutable output inventory and workbook verification.
assert hashlib.sha256(book.read_bytes()).hexdigest()==manifest['sha256_before']
hashes=rows('SOURCE_HASHES.csv');assert all(hashlib.sha256((R/r['path']).read_bytes()).hexdigest()==r['sha256_before'] for r in hashes)
save('STEP25_OUTPUT_MANIFEST.csv',[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(O.iterdir()) if p.is_file()]);print('REPORT CREATED; 646 workbook hits; 6 historical stored-not-in-supplement cells; C uniform12 target diagnostics NOT_STORED; existing sources unchanged')
