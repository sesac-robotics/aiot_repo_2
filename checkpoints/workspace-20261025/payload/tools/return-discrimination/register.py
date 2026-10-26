import json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/return-discrimination-ensemble-20261024';P=B/'output/mild-return-calibrated-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads(p.read_text());v=read(O/'verification.json');assert v['AI_development_error_detected']==4 and v['Test_source_overlap']==0 and v['production_replacement'] is False
f=B/'output/CURRENT_FIVE_EXERCISE_RUN.json';d=read(f);assert Path(d['run'])==P;d.update(parent_run=str(P),run=str(O),status=v['status'],report=str(O/'RESULTS.txt'),all_five_reliable=False,production_replacement=False);f.write_text(json.dumps(d,ensure_ascii=False,indent=2));print('CURRENT_EXPERIMENT_REGISTERED')
