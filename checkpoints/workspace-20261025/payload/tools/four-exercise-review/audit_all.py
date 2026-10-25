from pathlib import Path
import json,sys,collections,hashlib,numpy as np
sys.dont_write_bytecode=True
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-review-train-v1-20261025';sys.path.insert(0,str(B/'tools/error-enrichment'));from cache_sequences import Resolver;from split_guard import assert_training_allowed
rows=[r for r in json.loads((B/'output/squat-fixed-reference-combined-v3-20261024/score-inputs.frozen.json').read_text()) if r['exercise']!='squats' and r['split']=='train'];assert_training_allowed(rows);res=Resolver();audit=[]
for r in rows:
 d={k:r[k] for k in ['id','exercise','person','group','family','sources','split']};d['items']=[dict(item=x['item'],bank=x['bank'],grade=g,eligible=x['eligible'],holds=x.get('holds',[])) for x,g in zip(r['items'],r['levels'])];flags=[]
 try:
  pairs=res.resolve(r);assert pairs,'no verified original cache';seq,p=pairs[0];z=np.load(p);m=json.loads(p.with_suffix('.json').read_text());assert m['config']['source_sha256'] in r['sources'];assert str(z['fingerprint'])==m['fingerprint'];d['cache_path']=str(p);d['cache_fingerprint']=m['fingerprint'];d['cache_observations']=len(seq)
  face=seq[:,[0,2,5,7,8]];sh=seq[:,[11,12]];facecenter=np.nanmedian(face,axis=1);w=np.linalg.norm(sh[:,0]-sh[:,1],axis=1);dist=np.linalg.norm(sh.mean(axis=1)-facecenter,axis=1)/np.maximum(w,1);good=np.isfinite(sh).all(axis=(1,2));fraction=float(np.mean(good & (dist<.45)))
  d['shoulders_near_face_fraction']=fraction
  if fraction>.15:flags.append('possible_face_shoulder_misconnection_needs_visual_confirmation')
  if len(seq)<8:flags.append('few_native_observations')
  for item in d['items']:
   if item['eligible'] and item['grade'] is None:flags.append('eligible_without_grade:'+item['item'])
  if d['exercise'] in ['pushups','shoulder']:
   arm=np.isfinite(seq[:,[11,12,13,14,15,16]]).all(axis=(1,2));d['bilateral_arm_observed_fraction']=float(arm.mean())
 except Exception as e:flags.append('unresolved_cache:'+str(e))
 d['flags']=sorted(set(flags));audit.append(d)
(O/'all-Train-input-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n');print(collections.Counter(r['exercise'] for r in audit));print(collections.Counter(f.split(':')[0] for r in audit for f in r['flags']))
