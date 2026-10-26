"""Resolve inherited Train annotation aliases using source hash + exact interval receipts."""
from pathlib import Path
import json,sys,hashlib,collections,numpy as np,cv2
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025/alias-v3';O.mkdir(exist_ok=True);P=B/'output/four-exercise-review-train-v1-20261025/reconciled-v2';sys.path.insert(0,str(B/'tools/error-enrichment'));from split_guard import assert_training_allowed
rows=json.loads((P/'score-inputs.frozen.json').read_text());byid={r['id']:r for r in rows};assert_training_allowed(rows);old=json.loads((B/'output/four-exercise-review-train-v1-20261025/all-Train-input-audit.json').read_text());unresolved={r['id'] for r in old if any(f.startswith('unresolved') for f in r['flags'])};ann={};receipts=[]
def add(path,field=None):
 a=json.loads(path.read_text());a=a[field] if field else a
 for d in a:
  if d.get('id') in byid and d.get('split','train')=='train':ann[d['id']]=dict(d,annotation_file=str(path))
for folder in ['paired-failure-mining-20261024','return-failure-expansion-20261024','return-label-expansion-v3-20261024']:
 add(B/'output'/folder/'labels-v2.frozen.json','reviewed')
add(B/'data/error-enrichment-20261024/annotations/segments.frozen.json')
add(B/'data/error-enrichment-20261024/annotations/rehab-segments.frozen.json')
add(B/'output/ai-rule-paired-20261024/reviewed-intake-v3.frozen.json','labels')
add(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json','labels')
# Reviewed team IDs are stored without the r2d2 prefix.
team=json.loads((B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json').read_text())['labels'];ann.update({'r2d2_'+d['id']:dict(d,annotation_file=str(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')) for d in team if 'r2d2_'+d['id'] in byid})
# Additional reviewed YTW intervals encode milliseconds in immutable IDs; link their intake and existing source.
newsrc={r['review_id']:r for r in json.loads((B/'output/label-recheck-20261024/new-intake.json').read_text())}
for r in rows:
 if r['id'].startswith('recheck_'):
  _,sid,a,b=r['id'].split('_');d=newsrc[sid];ann[r['id']]=dict(d,id=r['id'],start=int(a)/1000,end=int(b)/1000,annotation_file=str(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')+';ID_interval_and_new-intake.json')
knownsha={h for r in rows for h in r['sources']};index=collections.defaultdict(list)
for p in (B/'output').rglob('*.npz'):
 if p.name not in ['primary.npz','side.npz','front.npz'] or any(s in str(p) for s in ['/snapshots/','/before/','/team_install','/team_share','/code-snapshot/']):continue
 m=p.with_suffix('.json')
 if not m.exists():continue
 try:
  d=json.loads(m.read_text());cfg=d['config'];sha=cfg['source_sha256']
  if sha not in knownsha:continue
  if hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()!=d['fingerprint']:continue
  index[sha].append((p,d))
 except (KeyError,ValueError):continue
srcs=collections.defaultdict(list)
for l in (B/'project-documents/project-handbook/data-audit-20261025/catalogs/held-media.jsonl').open():
 d=json.loads(l)
 if d.get('sha256') in knownsha and d.get('exists'):srcs[d['sha256']].append(d['path'])
source_cache={};out=[];failed=[]
for r in rows:
 rid=r['id'];d=ann.get(rid);candidates=[v for h in r['sources'] for v in index[h]];aliases={rid,rid.removesuffix('_held3_score'),rid.removesuffix('_held3_score').removesuffix('_held3')}
 if d:aliases|={d.get('review_id',''),d.get('source_id','')}
 exact=[v for v in candidates if v[0].parent.name in aliases]
 if r['family']=='R2D2' and d and d.get('source_id'):exact=[v for v in exact if v[0].parent.name==d['source_id']]
 if not exact:
  failed.append(dict(id=rid,exercise=r['exercise'],reason='no_exact_alias_cache',candidate_paths=[str(p) for p,m in candidates]));continue
 # Prefer the source path used by the annotation's originating experiment, then newest valid cache.
 folder=Path(d['annotation_file']).parent.name if d else ''
 exact.sort(key=lambda v:(folder in str(v[0]) if folder else False,v[0].stat().st_mtime_ns),reverse=True);chosen=None
 for p,m in exact:
  z=np.load(p);cfg=m['config'];sha=cfg['source_sha256'];fps=float(cfg['segment']['fps']);fs=z['frame_numbers'];a=float(d['start']) if d and 'start' in d else float(fs[0]/fps);b=float(d['end']) if d and 'end' in d else float(fs[-1]/fps+1/fps);ids=np.flatnonzero((fs/fps>=a-1e-6)&(fs/fps<b-1e-6))
  if len(ids)<5 or str(z['fingerprint'])!=m['fingerprint']:continue
  path=d.get('path') if d else None
  if not path or not Path(path).is_file():path=next((q for q in srcs[sha] if Path(q).suffix.lower() in ['.mp4','.mkv','.mov','.webm','.avi'] and Path(q).exists()),None)
  if not path:continue
  if sha not in source_cache:
   realsha=hashlib.sha256(Path(path).read_bytes()).hexdigest()
   if realsha!=sha:
    path=next((q for q in srcs[sha] if Path(q).suffix.lower() in ['.mp4','.mkv','.mov','.webm','.avi'] and Path(q).exists() and hashlib.sha256(Path(q).read_bytes()).hexdigest()==sha),None)
    if path is None:continue
   cap=cv2.VideoCapture(path);ok,im=cap.read();real_fps=cap.get(5);count=int(cap.get(7));cap.release()
   if not ok or abs(real_fps-fps)>.05:continue
   source_cache[sha]=dict(path=path,fps=real_fps,width=im.shape[1],height=im.shape[0],frame_count=count)
  src=source_cache[sha];chosen=dict(id=rid,exercise=r['exercise'],person=r['person'],group=r['group'],family=r['family'],split='train',sources=r['sources'],source_sha256=sha,source_path=src['path'],fps=fps,width=src['width'],height=src['height'],frame_count=src['frame_count'],window_seconds=[a,b],cache_path=str(p),cache_fingerprint=m['fingerprint'],cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),cache_indices=ids.tolist(),actual_native_frames=fs[ids].tolist(),annotation_file=d.get('annotation_file') if d else 'legacy_extraction_config_exact_alias',previously_unresolved=rid in unresolved,cache_context_frames=fs.tolist());break
 if chosen:out.append(chosen)
 else:failed.append(dict(id=rid,exercise=r['exercise'],reason='alias_found_but_window_source_or_cache_not_admitted'))
assert_training_allowed(out)
for name,value in [('resolved-inputs.frozen.json',out),('unresolved-inputs.json',failed),('resolution-summary.json',dict(total_candidates=len(rows),linked=len(out),old_unresolved=len(unresolved),old_unresolved_now_linked=sum(x['previously_unresolved'] for x in out),remaining_old_unresolved=sorted(unresolved-{x['id'] for x in out}),linked_by_exercise=dict(collections.Counter(x['exercise'] for x in out)),source_hashes_verified=len(source_cache)))]:
 p=O/name;assert not p.exists();p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
print(json.loads((O/'resolution-summary.json').read_text()))
