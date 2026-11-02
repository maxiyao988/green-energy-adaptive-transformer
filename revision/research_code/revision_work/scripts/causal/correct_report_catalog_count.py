"""Preserve the exclusively-created report and issue a corrected copy for one count typo."""
from core import *
def main():
 original=OUT/'REPORT_STEP4_INTERIM.md';d=pd.read_csv(OUT/'prior_corporate_support_catalog.csv');text=original.read_text();old='all 14 prior corporate-action support CSVs';new=f'all {len(d)} prior corporate-action support CSVs';assert old in text
 corrected=text.replace(old,new);note='Catalog-count correction: this copy corrects a single prose typo (13 support CSVs, not 14; the header was mistakenly counted). All 13 files were read. Numerical results, checks, classification and training are unchanged. The requested original report is preserved because outputs use exclusive creation.\n\n'
 corrected=corrected.replace('# BLOCKING\n\n','# BLOCKING\n\n'+note,1)
 with target(OUT/'REPORT_STEP4_INTERIM_CORRECTED.md').open('x') as f:f.write(corrected)
 jsonout('report_catalog_count_erratum.json',dict(completed_utc=utc(),original_report=rel(original),original_sha256=sha(original),corrected_report=rel(OUT/'REPORT_STEP4_INTERIM_CORRECTED.md'),corrected_sha256=sha(OUT/'REPORT_STEP4_INTERIM_CORRECTED.md'),correction='13 support CSVs; original prose incorrectly says 14. Catalog and all numeric results unchanged.',actual_support_csv_count=len(d)))
 print('Corrected catalog count:',len(d))
if __name__=='__main__':main()
