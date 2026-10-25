"""Frozen deployed heads on recovered people, compared with out-of-person candidates."""
import json,sys
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;B=O.parents[1];sys.path.insert(0,str(B/'output/live-webcam-20261023'));from checklist_comparison import models
heads,_=models();report=json.loads((O/'comparison.json').read_text());rows={r['id']:r for r in json.loads((O/'feature-rows.json').read_text())};results={}
for key,part in report['items'].items():
 head,path,sha=heads[key];training=set(head.get('training_people',[]));pred=[]
 for p in part['predictions']:
  r=rows[p['id']]
  if r['origin']!='recovered':continue
  assert r['person'] not in training and r['person'].removeprefix('rehab26_') not in training
  # Before coverage uses the pre-change observation gate and the unchanged head.
  old=r['before'].get(key);a=r['after'][key];before=float(np.clip(head['model'].predict([old['x']])[0],0,2)) if old else None
  pred.append(dict(id=r['id'],person=r['person'],target=p['target'],deployed_ai_before=before,candidate_ai_after=p['after_ai'],old_head_new_gate_ai=float(np.clip(head['model'].predict([a['x']])[0],0,2))))
 common=[p for p in pred if p['deployed_ai_before'] is not None]
 results[key]=dict(cases=pred,common_n=len(common),deployed_MAE=float(np.mean([abs(p['deployed_ai_before']-p['target']) for p in common])) if common else None,candidate_MAE=float(np.mean([abs(p['candidate_ai_after']-p['target']) for p in common])) if common else None,training_people=sorted(training),new_label_classes=sorted(set(p['target'] for p in pred)),limitation='회수한 사례가 모두 정상 관측 라벨이므로 오류 검출 개선을 의미하지 않음. 후보는 사람별 교차검증의 해당 사람 제외 모델. 최종 독립 시험 아님.')
(O/'deployed-reference.json').write_text(json.dumps(results,ensure_ascii=False,indent=2));print(json.dumps({k:{kk:vv for kk,vv in v.items() if kk!='cases'} for k,v in results.items()},ensure_ascii=False))
