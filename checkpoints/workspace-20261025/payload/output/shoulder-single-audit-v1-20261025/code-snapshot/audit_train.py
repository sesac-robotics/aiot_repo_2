"""Audit deployed shoulder Train features against original source caches; no label inference."""
from pathlib import Path
import sys,json,hashlib,collections
import numpy as np,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/shoulder-single-audit-v1-20261025';L=B/'output/live-webcam-20261023';R=L/'runtime/c76334937f21ef24a27aa1c1687de6c9577bf6249e5cac606151a328893746f7';sys.path[:0]=[str(R),str(B/'tools/shoulder-work'),str(B/'tools/error-enrichment')]
from app.scoring import features
from app.scoring.sequence import PoseSequence
from app.scoring.engine import ScoringEngine
from observation_features_v4 import dense_samples,quality
from split_guard import assert_training_allowed
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=B/'output/five-exercise-recovery-20261022/held_round2/candidate_input_manifest.json';allrows=json.loads(manifest.read_text());rows=[r for r in allrows if r['exercise']=='shoulder' and r['split']=='train'];assert_training_allowed(rows)
inv=json.loads((B/'output/shoulder-work-20261024/full-review-v1/inventory.frozen.json').read_text())['sources'];inventory={r['source_sha256']:r for r in inv};hashes={h for r in rows for h in r['sources']};cache=collections.defaultdict(list)
for p in (B/'output').rglob('primary.json'):
 if any(v in p.parts for v in ['code-snapshot','snapshots','before']):continue
 try:
  d=json.loads(p.read_text());c=d.get('config',{});h=c.get('source_sha256')
  if h in hashes and c.get('sample_hz')==10 and p.with_suffix('.npz').exists():cache[h].append((p,d))
 except (ValueError,OSError):pass
model=ScoringEngine().model('shoulder');out=[]
for row in rows:
 candidates=[]
 for h in row['sources']:
  if h not in inventory:continue
  src=inventory[h];cap=cv2.VideoCapture(src['path']);w=int(cap.get(3));height=int(cap.get(4));cap.release()
  for p,d in cache[h]:
   if d.get('id')!=row['id']:continue
   z=np.load(p.with_suffix('.npz'));fs=z['frame_numbers'];xy=z['image_xy']
   if str(z['fingerprint'])!=d.get('fingerprint'):continue
   fps=d['config']['segment']['fps'];seq=PoseSequence(xy,fs, float(fps),w,height)
   try:notes=features.describe('shoulder',seq,model,view='front' if row['items'][0]['bank']=='front' else 'front_oblique')
   except ValueError:continue
   diffs=[]
   for i in [1,3]:
    old=row['items'][i].get('feature');new=notes[i].get('feature')
    if old is not None and new is not None:diffs.append(float(np.max(np.abs(np.array(old)-new))))
   if not diffs:continue
   candidates.append((max(diffs),p,d,z,src,w,height,notes))
 if not candidates:out.append({'id':row['id'],'status':'cache_or_canonical_source_unresolved','person':row['person'],'sources':row['sources']});continue
 err,p,d,z,src,w,height,notes=min(candidates,key=lambda r:r[0]);fps=d['config']['segment']['fps'];times=z['frame_numbers']/fps
 # Keep nearest actual source frame per nominal slot, with every absent slot NaN.
 slots={}
 for tm,x in zip(times,z['image_xy']):
  tick=round(float(tm)*10);distance=abs(tm-tick/10)
  if tick not in slots or distance<slots[tick][0]:slots[tick]=(distance,x)
 ticks,xy=dense_samples([(k,v[1]) for k,v in sorted(slots.items())],w,height);q=quality(ticks,xy)
 rec=dict(id=row['id'],person=row['person'],split='train',sources=row['sources'],source_id=src['id'],source_path=src['path'],source_sha256=src['source_sha256'],canonical_split=src['split'],view=src['view'],cache=str(p.with_suffix('.npz')),cache_fingerprint=d['fingerprint'],cache_sha256=sha(p.with_suffix('.npz')),source_fps=fps,width=w,height=height,time_range=times[[0,-1]].tolist(),feature_max_abs_difference=err,status='exact_feature_provenance' if err<1e-8 else 'feature_version_or_cache_mismatch',quality=q,items=[])
 for i in [1,3]:
  note=row['items'][i];f=note.get('feature');rec['items'].append(dict(item=note['item'],old_grade=row['levels'][i],old_eligible=note['eligible'],bank=note['bank'],feature=f,tail_feature_extent_seconds=min(5,max(2,round(len(times)*.15)))/10 if i==1 else None))
 out.append(rec)
result={'selection':'all86_deployed_should er_Train_manifest_rows'.replace('should er','shoulder'),'rows':out,'status_counts':dict(collections.Counter(r['status'] for r in out)),'quality_flagged_rows':sum(bool(r.get('quality',{}).get('flags')) for r in out),'source_role_conflicts':[r['id'] for r in out if r.get('canonical_split','train')!='train'],'manifest_sha256':sha(manifest),'no_fit':True,'no_grade_inferred':True}
(O/'train-provenance-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='rows'},ensure_ascii=False))
