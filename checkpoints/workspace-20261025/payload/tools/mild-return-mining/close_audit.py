import json,sys
from pathlib import Path
B=Path(__file__).resolve().parents[2];O=B/'output/mild-return-calibrated-20261024';P=B/'output/return-root-cause-v3-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
read=lambda p:json.loads(p.read_text());a=read(P/'comparison.frozen.json');b=read(O/'comparison.frozen.json');old=next(h for h in read(P/'rule-training.json')['heads'] if h['key']==['squats','standing_height_return','height_upper_reversal_v1']);new=next(h for h in read(O/'rule-training.json')['heads'] if h['key']==old['key']);assert {p['id']:p['target'] for p in old['validation_predictions']}=={p['id']:p['target'] for p in new['validation_predictions']}
lineage=list((O/'target-lineage').glob('*.json'));assert lineage
v=read(O/'verification.json');v['previous_same_bank_rule_development']=old['fixed_development'];v['current_same_bank_rule_development']=new['fixed_development'];v['same_Test_squat_rule_previous_normal_false_alarms']=round(a['same_bank_by_exercise']['squats']['rule_base']['normal_false_alarm']*15);v['same_Test_squat_rule_current_normal_false_alarms']=round(b['same_bank_by_exercise']['squats']['rule_base']['normal_false_alarm']*15);v['target_lineage_audit_passed']=True;(O/'verification.json').write_text(json.dumps(v,indent=2))
p=O/'RESULTS.txt';t=p.read_text();t+='\n동일 오류6개 기준 이전 동일 특징 Rule 검출1/6 → 재학습3/6, 정상 오판0/13 → 1/13. 동일 Test 스쿼트 전체 Rule 정상 오판6/15 → 8/15로 악화했습니다. AI 검출 개선 및 정상 오판 억제 목표 미달입니다.\n팀 영상의 기존 라벨 대비 입력 타깃 계보 감사도 통과했습니다.\n';p.write_text(t)
files=[O/'score-candidates.joblib',O/'score-inputs.frozen.json',O/'test-inputs.frozen.json',O/'score-training.json',O/'rules.frozen.json',O/'comparison.frozen.json',O/'verification.json',O/'RESULTS.txt',B/'tools/error-enrichment/models.py',*sorted((B/'tools/mild-return-mining').glob('*.py'))]
(O/'artifact-hashes.frozen.json').write_text(json.dumps({str(f):sha(f) for f in files},indent=2));print('AUDIT_CLOSED: actual Rule gain with FP regression; AI target not met')
