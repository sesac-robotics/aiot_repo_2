"""동일 5회에서 누락·구간·화면 변환 영향을 분리하는 개발 진단."""
import json
from pathlib import Path
import numpy as np
from front_candidate import predict,vector
R=Path(__file__).parent;P=R/'recordings/20261023-110200-964840';z=np.load(P/'pose.npz');labels=json.loads((P/'labels.frozen.json').read_text());out=[]
for rep in labels['repetitions']:
 a,b=rep['start_frame'],rep['end_frame'];xy=z['image_xy'][a:b+1];base=predict(xy);variants={}
 for name,pts in [('left_knee_only',xy.copy()),('right_knee_only',xy.copy()),('scale_1_2',xy*1.2),('translate',xy+np.array([30,20]))]:
  if name=='left_knee_only':pts[:,26]=np.nan
  if name=='right_knee_only':pts[:,25]=np.nan
  try:variants[name]=predict(pts)
  except ValueError as e:variants[name]={'error':str(e)}
 for shift in [-5,5]:
  aa=max(0,a+shift);bb=min(len(z['image_xy'])-1,b+shift)
  try:variants[f'boundary_shift_{shift}']=predict(z['image_xy'][aa:bb+1])
  except ValueError as e:variants[f'boundary_shift_{shift}']={'error':str(e)}
 # 1·2·5회에도 실제 4회의 오른쪽 무릎 누락 패턴을 같은 상대 위치에 적용한다.
 fourth=z['image_xy'][195:251];bad=np.where(~np.isfinite(fourth[:,26]).all(1))[0]
 masked=xy.copy()
 if len(bad):masked[np.minimum(len(xy)-1,np.rint(bad/max(1,len(fourth)-1)*(len(xy)-1)).astype(int)),26]=np.nan
 variants['rep4_missing_pattern']=predict(masked)
 out.append(dict(number=rep['number'],base=base,variants=variants,raw_hip_pixel_range=float(np.percentile(xy[:,[23,24],1].mean(1),95)-np.percentile(xy[:,[23,24],1].mean(1),5))))
(R/'front-candidate/sensitivity.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
for r in out:
 print(r['number'],'range',round(r['base']['features'][0],3),'pixels',round(r['raw_hip_pixel_range'],1),'score',r['base']['ai_assisted_points'], 'variants', {k:v.get('ai_assisted_points',v.get('error')) for k,v in r['variants'].items()})
