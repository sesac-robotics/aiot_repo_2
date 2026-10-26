from pathlib import Path
import sys,json,ast,threading,hashlib,collections
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid'); L=B/'output/live-webcam-20261023'; O=Path(__file__).parent; S=L/'sessions/1792859978170477383'
d=json.loads((S/'final-state.json').read_text());sys.path[:0]=[str(L),str(L/'runtime'/d['loaded_code_model_sha256'])]
from shoulder_candidate import ShoulderTracker
from app.scoring.engine import ScoringEngine
from app.scoring.sequence import PoseSequence
from live_observation import mask_frame_border
from front_policy import apply_front_policy
from rubric100 import assessment as assess100
from unified_assessment import apply as unify_assessment
ENGINE=ScoringEngine();MODEL_LOCK=threading.Lock()
source=ast.parse((L/'server.py').read_text());code=ast.Module(body=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ('safe','score_samples')],type_ignores=[]);exec(compile(code,'saved-score-functions','exec'))
p=np.load(S/'pose.npz');t=ShoulderTracker();segments=[];phases=collections.Counter()
for tick,xy in zip(p['frame_numbers'],p['image_xy']):
 seg=t.push(int(tick),xy,int(p['width']),int(p['height']));phases[t.phase]+=1
 if seg:segments.append(seg)
rows=[]
for seg,r in zip(segments,d['repetitions']):
 fresh=score_samples(seg,int(p['width']),int(p['height']),'shoulder','front')['score_100'];saved=r['score_100'];items=saved['items'];rows.append(dict(number=r['number'],start_tick=seg[0][0],end_tick=seg[-1][0],duration=r['duration'],samples=len(seg),score_exact_match=fresh==saved,ai=saved['ai']['earned_subtotal'],ai_max=saved['ai']['assessed_weight'],rule=saved['rule']['total'],items=items,source_items=r['items'],legacy_ai=r['ai']['earned'],item_sum=round(sum(i['ai_points'] for i in items if i['ai_points'] is not None),1)))
model=ENGINE.model('shoulder'); print('MODEL_KEYS',list(model)); support=json.loads((ENGINE.bundle/'training_support.json').read_text())['exercises']['shoulder']
report=dict(session=str(S),samples=len(p['frame_numbers']),seconds=float((p['frame_numbers'][-1]-p['frame_numbers'][0])/10),saved_count=d['movement']['count'],replayed_count=t.count,scored_rows=len(d['repetitions']),all_scores_exact_match=all(r['score_exact_match'] for r in rows),nonoverlap=all(a[-1][0]<b[0][0] for a,b in zip(segments,segments[1:])),ai_mean=float(np.mean([r['ai'] for r in rows])),rule_mean=float(np.mean([r['rule'] for r in rows])),phase_counts=dict(phases),training_support=support,rows=rows,video_present=bool(list(S.glob('*.mp4'))),sha256={n:hashlib.sha256((S/n).read_bytes()).hexdigest() for n in ['pose.npz','state.json','final-state.json']})
(O/'analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ['rows','phase_counts','sha256','training_support']},ensure_ascii=False));print(json.dumps([{k:v for k,v in r.items() if k not in ['items','source_items']} for r in rows],ensure_ascii=False));print(json.dumps(support,ensure_ascii=False))
