"""고정 시각 라벨 이후에만 보존된 팀 모델을 읽어 진단한다. 학습·모델 쓰기 없음."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
P=ROOT/'recordings/20261023-110200-964840'
meta=json.loads((P/'metadata.json').read_text())
labels=json.loads((P/'labels.frozen.json').read_text())
assert hashlib.sha256((P/'labels.frozen.json').read_bytes()).hexdigest()==(P/'labels.frozen.sha256').read_text().strip()
assert hashlib.sha256((P/'blind.avi').read_bytes()).hexdigest()==labels['video_sha256']
snapshot=ROOT/'runtime'/meta['code_model_sha256']
sys.path.insert(0,str(snapshot))
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
from runtime_snapshot import read_source,digest
before=digest(read_source())
e=ScoringEngine(bundle=snapshot/'artifacts/posture_score')
z=np.load(P/'pose.npz');results=[]
def safe(v):
 if isinstance(v,dict):return {k:safe(x) for k,x in v.items()}
 if isinstance(v,(list,tuple,np.ndarray)):return [safe(x) for x in v]
 if isinstance(v,np.generic):return safe(v.item())
 if isinstance(v,float) and not np.isfinite(v):return None
 return v
for rep in labels['repetitions']:
 a,b=rep['start_frame'],rep['end_frame'];ticks=z['frame_numbers'][a:b+1];points=z['image_xy'][a:b+1]
 full=np.full((int(ticks[-1]-ticks[0])+1,33,2),np.nan);full[ticks-ticks[0]]=points
 seq=PoseSequence(full,np.arange(len(full)),10.,int(z['width']),int(z['height']))
 r=e.score('squats',seq,view='front',complete_attempt=True)
 coverage={name:round(float(np.isfinite(points[:,idx]).all(axis=1).mean()),3) for name,idx in [('left_shoulder',11),('right_shoulder',12),('left_hip',23),('right_hip',24),('left_knee',25),('right_knee',26),('left_ankle',27),('right_ankle',28)]}
 results.append(dict(number=rep['number'],start_frame=a,end_frame=b,start_seconds=float(z['timestamps'][a]),end_seconds=float(z['timestamps'][b]),coverage=coverage,all_six_fraction=float(np.isfinite(points[:,[11,12,23,24,25,26]]).all((1,2)).mean()),legacy_result=r))
after=digest(read_source());assert before==after
out=safe(dict(source_unchanged=True,team_digest_before=before,team_digest_after=after,recording_model_digest=meta['code_model_sha256'],comparison_scope='정면 초안과 기존 항목은 정의가 달라 MAE·일치율 계산 금지',repetitions=results))
(P/'legacy-comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for row in out['repetitions']:
 print('REP',row['number'],'six',round(row['all_six_fraction'],3),'coverage',row['coverage'])
 r=row['legacy_result'];print('total',r.get('ai_score'),r.get('rule_score'))
 for i in r['items']:print(json.dumps(i,ensure_ascii=False))
print('source unchanged',before==after)
heads=[]
for key,head in e.model('squats').items():
 if isinstance(key,tuple):heads.append(dict(key=list(key),boundary=safe(head.get('boundary'))))
(P/'legacy-head-boundaries.json').write_text(json.dumps(heads,ensure_ascii=False,indent=2))
print('front boundaries',[h for h in heads if h['key'][-1]=='front'])
