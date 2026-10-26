import sys,json,hashlib,datetime
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');app=B/'output/live-webcam-20261023';sys.path.insert(0,str(app))
from movement import MovementTracker
from live_observation import mask_frame_border,movement_observation
from runtime_snapshot import read_source,digest
p=app/'sessions/1792857230761398358';s=json.loads((p/'final-state.json').read_text());z=np.load(p/'pose.npz');xy=z['image_xy'];ticks=z['frame_numbers'];w=int(z['width']);h=int(z['height']);masked=mask_frame_border(xy,w,h);t=MovementTracker('squats','front');segments=[]
for tick,pts in zip(ticks,xy):
 seg=t.push(int(tick),pts,width=w,height=h)
 if seg:segments.append({'start_tick':seg[0][0],'end_tick':seg[-1][0],'start_seconds':(seg[0][0]-int(ticks[0]))/10,'end_seconds':(seg[-1][0]-int(ticks[0]))/10,'samples':len(seg),'duration':(seg[-1][0]-seg[0][0])/10,'required_observation_fraction':sum(movement_observation(a,w,h,'squats','front')['ready']for _,a in seg)/len(seg)})
rows=[]
for r in s['repetitions']:
 a=r['score_100'];rows.append({'number':r['number'],'duration':r['duration'],'ai':a['ai'],'rule':a['rule'],'items':[{k:i.get(k)for k in ['id','ai_points','rule_points','weight','reasons']}for i in a['items']],'depth_features':r['front_candidate']['features'],'coaching':r['coaching']['messages']})
source=read_source();runtime=app/'runtime'/s['loaded_code_model_sha256'];fixed={name:(runtime/name).read_bytes()for name in source}
prev=json.loads((app/'sessions/1792846159556060119/final-state.json').read_text())
result={'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'session':str(p),'samples':len(ticks),'seconds':(int(ticks[-1])-int(ticks[0]))/10,'exercise':s['exercise'],'view':s['view'],'stored_count':s['movement']['count'],'saved_rows':len(rows),'replay_count':t.count,'segments':segments,'uncertain_attempts':t.uncertain_attempts,'observation':{'tracker_required_ready_frames':sum(movement_observation(a,w,h,'squats','front')['ready']for a in xy),'denominator':len(xy),'joints':{str(j):{'raw_finite_frames':int(np.isfinite(xy[:,j]).all(1).sum()),'inside_border_frames':int(np.isfinite(masked[:,j]).all(1).sum())}for j in [11,12,23,24,25,26]}},'scores':rows,'runtime_integrity':{'loaded':s['loaded_code_model_sha256'],'current_source':digest(source),'fixed_snapshot':digest(fixed),'initial_squat_loaded':prev['loaded_code_model_sha256'],'standing_state':s['standing_state_model_sha256'],'first_test_standing_state':prev.get('standing_state_model_sha256')},'original_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in p.iterdir()if f.is_file()},'limitations':['영상 없음: 실제 수행 횟수/깊이/복귀 정답 확인 불가','관측 중단을 실제 복귀 실패로 라벨링하지 않음','원시 재생 일치는 정확도 검증이 아님','학습 데이터 혼입 여부는 모델 담당 원본 감사와 구분']}
out=B/'project-documents/squat-live-20261025/session-1792857230761398358';(out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in ['scores','original_hashes']},ensure_ascii=False,indent=2));print('scores',json.dumps(rows,ensure_ascii=False)[:3000])
