from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');O=Path(__file__).parent
sys.path.insert(0,str(B/'tools/shoulder-work'))
from score_items_v4 import score
base=json.loads((O/'verification.json').read_text());F=Path(base['session']);z=np.load(F/'pose.npz');rows=[]
for index,old in enumerate(base['rows']):
    start,end=old['segment_ticks'];extended=end+3
    if index+1<len(base['rows']):assert extended<base['rows'][index+1]['segment_ticks'][0]
    take=(z['frame_numbers']>=start)&(z['frame_numbers']<=extended)
    samples=[(int(t),p) for t,p in zip(z['frame_numbers'][take],z['image_xy'][take])]
    assert samples[-1][0]==extended
    out=score(samples,models=None,width=int(z['width']),height=int(z['height']),trunk_window=[start,end],exercise_confirmed=True)
    lower=out['features']['lower_return'];body=out['features']['frontal_trunk_hold']
    assert lower['status']=='features_ready' and body['status']=='features_ready'
    assert max(lower['return_ticks'])<=extended
    assert all(r['points'] is None and r['grade'] is None for r in out['items'])
    rows.append(dict(number=old['number'],original_ticks=[start,end],actual_extended_end=extended,return_ticks=lower['return_ticks'],body_window_unchanged=body['window_ticks']==[start,end],next_segment_not_used=True,all_scores_held=True))
assert hashlib.sha256((F/'pose.npz').read_bytes()).hexdigest()==base['raw_sha256']
result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),scope='same exposed saved session suffix-timing integration diagnostic; no video quality truth',session=base['session'],source_sha256=base['raw_sha256'],actual_suffix_maximum_slots=3,lower_features_ready=len(rows),body_features_ready=len(rows),rows=rows,models_trained=False,grade_support_enabled=False,production_changed=False,camera_opened=False)
(O/'context-verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False))
