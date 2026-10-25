import json,sys,copy,datetime,math
from pathlib import Path
from fractions import Fraction
import numpy as np
from cycle_features import extract,VERSION
B=Path(__file__).resolve().parents[2];P=B/'output/return-label-enriched-v4-20261024';O=B/'output/return-cycle-aligned-v1-20261024';O.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
read=lambda p:json.loads(p.read_text());bank='observed_elbow_cycle_v1';inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};team={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};plans={r['id']:r for r in read(B/'output/pushup-root-cause-20261024/crop-plan-v2.frozen.json')};external={}
for folder,label_file in [('paired-failure-mining-20261024','labels-v2.frozen.json'),('return-failure-expansion-20261024','labels-v2.frozen.json'),('return-label-expansion-v3-20261024','labels-v2.frozen.json')]:
 for d in read(B/'output'/folder/label_file)['reviewed']:
  if d.get('exercise')=='pushups' and d.get('train_eligible') and d['grades'][2] is not None:external[d['id']]=(d,B/'output'/folder/'observations'/d['review_id']/'primary.npz')
audit=[]
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),policy_sha256=sha(POLICY),bank=bank,feature_code_sha256=sha(B/'tools/return-label-expansion-v3/cycle_features.py'),target_scope='R2D2 normal-return per-interval drafts and explicitly reviewed complete external pushup intervals only; no source-wide error grade copied to guessed first cycles',phase_policy='first observed descent/bottom/ascent/upper reversal; <=0.35s same-source end context; original bottom must precede original end; no later repetition replacing a failed return',original_labels_and_roles_unchanged=True,diagnostic_Test_previously_exposed=True,independent_final_Test_opened=False,Test_not_for_numeric_parameter_selection=True,production_replacement=False)
p=O/'MODEL_SELECTION_PROTOCOL.frozen.json';assert not p.exists();p.write_text(json.dumps(protocol,indent=2))
for name in ['score-inputs.frozen.json','test-inputs.frozen.json']:
 rows=read(P/name)
 for r in rows:
  if r['exercise']!='pushups':continue
  available=[(i,g) for i,g in zip(r['items'],r['levels']) if i['item']=='return_extension' and g is not None];targets={g for i,g in available};assert len(targets)<=1
  if not targets:continue
  g=next(iter(targets));caches=[];reason='label_scope_not_verified_complete_interval'
  if r['family']=='R2D2' and g==2:
   d=team[r['id'][5:]];sid=d['source_id'];src=inv[sid];fps=float(Fraction(src['fps']));a,b=d['start'],d['end'];path=src['copy'];source=src['sha256']
   if sid in plans:
    plan=plans[sid];caches=[B/'output/pushup-root-cause-20261024/observations'/sid/'primary.npz',Path(plan['previous_cache'])]
   else:caches=[B/'output/integrated-five-20261024/team-observations'/sid/'side.npz']
  elif r['id'] in external:
   d,p=external[r['id']];fps=d['fps'];a,b=d['start'],d['end'];path=d['path'];source=d['source_sha256'];caches=[p]
  else:
   r['items'].append(dict(item='return_extension',bank=bank,feature=None,eligible=False,holds=[reason]));r['levels'].append(g);audit.append(dict(id=r['id'],split=r['split'],grade=g,eligible=False,hold=reason));continue
  f=None;cache=None
  for cache in caches:
   if not cache.exists():continue
   z=np.load(cache);meta=read(cache.with_suffix('.json'));assert meta['config']['source_sha256']==source and source in r['sources'];assert str(z['fingerprint'])==meta['fingerprint'];frames=z['frame_numbers'];mask=(frames/fps>=a)&(frames/fps<b+.35);f=extract(z['image_xy'][mask],z['confidence'][mask],frames[mask],math.ceil(b*fps))
   if f is not None:break
  r['items'].append(dict(item='return_extension',bank=bank,feature=f['feature'] if f else None,eligible=f is not None,holds=[] if f else ['first_complete_return_not_observed'],feature_version=VERSION));r['levels'].append(g)
  audit.append(dict(id=r['id'],split=r['split'],grade=g,eligible=f is not None,person=r['person'],source_sha256=source,source_path=path,cache=str(cache),cache_sha256=sha(cache) if cache and cache.exists() else None,start=a,end=b,context_seconds=.35,observed=f,old_tail_feature=next((i['feature'] for i,v in available if i['bank'] in ['front','side_legacy_roi_v1']),None)))
 if name.startswith('score'):assert_training_allowed([r for r in rows if r['split']=='train'])
 p=O/name;assert not p.exists();p.write_text(json.dumps(rows,indent=2,allow_nan=False))
(O/'feature-provenance.json').write_text(json.dumps(audit,indent=2));(O/'original-test-inputs.frozen.json').write_bytes((P/'test-inputs.frozen.json').read_bytes())
s=(B/'tools/return-label-expansion-v3/train_monotone.py').read_text().replace('return-extension-monotone-v1-20261024','return-cycle-aligned-v1-20261024').replace('observed_arm_extension_shared_v1',bank)
(B/'tools/return-label-expansion-v3/train_cycles.py').write_text(s)
print('observed cycle items',[(split,sum(r['eligible'] and r['split']==split for r in audit)) for split in ['train','validation','test']],flush=True)
