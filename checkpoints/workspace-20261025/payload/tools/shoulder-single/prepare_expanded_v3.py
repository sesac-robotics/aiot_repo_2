from pathlib import Path
import json,hashlib,sys,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-expanded-train-v3-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
I=json.loads((B/'output/shoulder-data-inventory-20261025/INVENTORY.json').read_text());sources=[]
for r in I['team_sources']:
 if r['current_frozen_role']!='train':continue
 sources.append(dict(id=r['id'],path=r['path'],person='r2d2_'+r['person'],source_sha256=r['source_sha256'],split='train',family='team',posture='seated',declared_view='front',group='r2d2_'+r['person']))
reg={r['id']:r for r in json.loads((B/'output/three-exercise-comparison-20261022/youtube_candidates.json').read_text())['records']}
for sid in ['gpAwfWZBd1o','vlFGTI5JzjI']:
 r=reg[sid];assert r['evaluation_role']=='development_train';sources.append(dict(id='youtube_'+sid,path=r['video_path'],person=r['channel_id'],source_sha256=r['video_sha256'],split='train',family='external',posture='seated',declared_view='mixed_requires_window_review',group=r['channel_id']))
for r in sources:
 assert hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['source_sha256'];r['sources']=[r['source_sha256']];c=cv2.VideoCapture(r['path']);fps=c.get(5);count=int(c.get(7));ok,fr=c.read();assert ok;r.update(fps=fps,frame_count=count,width=fr.shape[1],height=fr.shape[0],duration=count/fps);c.release()
assert_training_allowed(sources);p=O/'source-plan.frozen.json';assert not p.exists();p.write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n');print([(r['id'],round(r['duration'],2),r['width'],r['height']) for r in sources])
