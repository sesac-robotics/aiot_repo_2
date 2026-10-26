from pathlib import Path
import json,datetime,hashlib
B=Path('/home/jayden/SeSACPAI-Mid');R=B/'output/shoulder-work-20261024/full-review-v1';O=B/'output/shoulder-work-20261024/normal-reference-v2';O.mkdir(exist_ok=True)
inv={r['id']:r for r in json.loads((R/'inventory.frozen.json').read_text())['sources']};labels=json.loads((R/'dense-visual-labels.json').read_text());plan=[]
for l in labels:
 s=inv[l['id']]
 if s['split']=='test' or l['movement_protocol']!='shoulder_press':continue
 if l['lower_return_grade']!=2 and l['frontal_trunk_grade']!=2:continue
 if l['lower_return_grade']==2 and l['observed_events']:a,b=l['observed_events'][0]
 else:a,b=l['reviewed_windows'][0]
 target_start=max(0,a-3);target_end=b+.2;start=max(0,target_start-3)
 plan.append(dict(id=s['id'],path=s['path'],source_sha256=s['source_sha256'],person=s['person'],recording_group=s['recording_group'],split=s['split'],view=s['view'],return_event=[a,b] if l['lower_return_grade']==2 else None,reviewed_windows=l['reviewed_windows'],target_context=[target_start,target_end],past_warmup_seconds=3,extract_start=start,extract_end=target_end,lower_grade=l['lower_return_grade'],trunk_grade=l['frontal_trunk_grade']))
p=O/'plan.frozen.json';assert not p.exists();p.write_text(json.dumps({'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'source-and-time-matched-normal-reference-observations','parent':'normal-reference-v1','fixed_setting':'nu=.05 gamma=scale','not_a_grade_model':True,'source_roles_unchanged':True,'no_test_inference_or_fit':True,'labels_sha256':hashlib.sha256((R/'dense-visual-labels.json').read_bytes()).hexdigest(),'rows':plan},ensure_ascii=False,indent=2)+'\n');print(len(plan))
