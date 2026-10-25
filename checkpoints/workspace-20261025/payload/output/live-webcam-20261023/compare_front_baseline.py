"""기존 검증 사람의 동일 정면 관절에서 기존 깊이 AI와 후보를 비교한다."""
import sys,json
from pathlib import Path
import cv2,numpy as np
R=Path(__file__).parent;O=R/'front-candidate';report=json.loads((O/'report.json').read_text());meta=json.loads((R/'recordings/20261023-110200-964840/metadata.json').read_text());sys.path.insert(0,str(R/'runtime'/meta['code_model_sha256']))
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
from runtime_snapshot import read_source,digest
engine=ScoringEngine(bundle=R/'runtime'/meta['code_model_sha256']/'artifacts/posture_score');rows=json.loads((O/'training-rows.json').read_text());out=[]
for r in rows:
 if r['split']!='validation':continue
 z=np.load(r['cache_path']);cap=cv2.VideoCapture(r['front_source']);w=int(cap.get(3));h=int(cap.get(4));cap.release()
 ticks=np.rint((z['time_seconds']-z['time_seconds'][0])*10).astype(int)
 xy=np.full((ticks[-1]+1,33,2),np.nan);xy[ticks]=z['image_xy']
 seq=PoseSequence(xy,np.arange(len(xy)),10.,w,h)
 result=engine.score('squats',seq,view='front',complete_attempt=True)
 item=result['items'][0];out.append(dict(id=r['id'],target=r['target'],old_ai=item['ai_grade'],old_rule=item['rule_grade']))
valid=[r for r in out if r['old_ai'] is not None];report['baseline_same_validation']=dict(rows=out,n=len(valid),ai_grade_MAE=float(np.mean([abs(r['old_ai']-r['target']) for r in valid])),rule_observed=sum(r['old_rule'] is not None for r in out))
report['source_preserved']=report['source_preserved'] and digest(read_source())==meta['code_model_sha256']
(O/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(report['baseline_same_validation'])
