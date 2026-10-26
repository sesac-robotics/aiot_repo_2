import json,sys,datetime,importlib.util
from pathlib import Path
from fractions import Fraction
import numpy as np
import features as fixed
B=Path(__file__).resolve().parents[2];P=B/'output/return-cycle-aligned-v1-20261024';O=B/'output/squat-fixed-reference-v1-20261024';sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import POLICY,assert_training_allowed
spec=importlib.util.spec_from_file_location('upper',B/'tools/return-root-cause/features.py');upper=importlib.util.module_from_spec(spec);spec.loader.exec_module(upper)
read=lambda p:json.loads(p.read_text());labs={r['id']:r for r in read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels']};team={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};refs=read(B/'output/squat-side-return-v5-20261024/MODEL_SELECTION_PROTOCOL.frozen.json')['reference_windows'];audit=[];bank='standing_fixed_initial_height_v2'
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),policy_sha256=sha(POLICY),bank=bank,feature_names=fixed.NAMES,feature_code_sha256=sha(B/'tools/squat-fixed-reference/features.py'),hypothesis='Re-normalizing by the shortened projected return leg can cancel genuine standing-height loss. Use initial scale for both support-relative heights.',timing_and_reference_windows_unchanged=True,source_roles_and_labels_unchanged=True,new_failure_labels_this_ablation=0,selection='same Train-only nested group CV and C grid/cutoff rules as initial-context model; no Test numerical selection',diagnostic_development_previously_exposed=True,independent_final_Test_opened=False,production_replacement=False)
p=O/'MODEL_SELECTION_PROTOCOL.frozen.json';assert not p.exists();p.write_text(json.dumps(protocol,indent=2))
for name in ['score-inputs.frozen.json','test-inputs.frozen.json']:
 rows=read(P/name)
 for r in rows:
  prior=next(((i,g) for i,g in zip(r['items'],r['levels']) if i['bank']=='standing_initial_context_v1'),None)
  if prior is None:continue
  old,g=prior;original_end=None
  if r['id'].startswith(('return_squat_','reviewed_return_squat_')):
   d=labs[r['id'].removeprefix('reviewed_').removeprefix('return_')];stem=Path(d['path']).stem;cache=B/'output/squat-return-20261023'/(stem+'-heavy.npz');z=np.load(cache);first,end=(369,378) if 'diag_m4' in stem else (0,2) if 'front_m4' in stem else (0,6);rx=np.load(B/'output/return-recheck-20261024/m4-front-initial-reference.npz')['image_xy'] if 'front_m4' in stem else z['image_xy'][(z['frame_numbers']>=first)&(z['frame_numbers']<=end)];mask=(z['frame_numbers']>=d['first_frame'])&(z['frame_numbers']<=d['last_frame']+round(.4*d['fps']));original_end=d['last_frame'];path=d['path']
  elif r['family']=='R2D2':
   d=team[r['id'][5:]];sid=d['source_id'];first,end=refs.get(sid,[0,.22]);cache=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';z=np.load(cache);t=z['frame_numbers']/float(Fraction(inv[sid]['fps']));rx=z['image_xy'][(t>=first)&(t<=end)];mask=(t>=max(d['start'],end))&(t<d['end']+.4);original_end=d['end']*float(Fraction(inv[sid]['fps']));path=inv[sid]['copy']
  else:continue
  frames=z['frame_numbers'][mask];seq=z['image_xy'][mask];q=rx[:,[11,12]].mean(1);h=rx[:,[23,24]].mean(1);ref=dict(hip_y=float(np.median(h[:,1])),shoulder_y=float(np.median(q[:,1])),center_x=float(np.median(h[:,0])),scale=float(np.median(np.linalg.norm(h-q,axis=1))),torso_x=float(np.median((q-h)[:,0])));base=upper.extract(seq,ref);base=base if base and frames[base['bottom_index']]<=original_end else None;f=fixed.extract(seq,rx,base)
  r['items'].append(dict(item='standing_height_return',bank=bank,feature=f['x'] if f else None,eligible=f is not None and g is not None,holds=[] if f and g is not None else ['reference_or_completed_return_unobserved'],feature_version=fixed.VERSION));r['levels'].append(g)
  audit.append(dict(id=r['id'],person=r['person'],split=r['split'],grade=g,source_path=path,cache=str(cache),cache_sha256=sha(cache),reference_window=[first,end],original_attempt_end_frame=original_end,return_source_frame=int(frames[f['return_index']]) if f else None,bottom_source_frame=int(frames[f['bottom_index']]) if f else None,old_feature=old['feature'],new_feature=f['x'] if f else None,return_leg_ratios=f['observed_return_leg_ratios'] if f else None,eligible=f is not None and g is not None))
 if name.startswith('score'):assert_training_allowed([r for r in rows if r['split']=='train'])
 p=O/name;assert not p.exists();p.write_text(json.dumps(rows,indent=2,allow_nan=False))
(O/'feature-provenance.json').write_text(json.dumps(audit,indent=2));(O/'original-test-inputs.frozen.json').write_bytes((P/'test-inputs.frozen.json').read_bytes())
s=(B/'tools/return-recheck/train_initial_context.py').read_text().replace('output/return-initial-context-20261024','output/squat-fixed-reference-v1-20261024').replace('standing_initial_context_v1',bank).replace('feature_dimensions=10','feature_dimensions=4');(B/'tools/squat-fixed-reference/train.py').write_text(s)
print('PREPARED',len(audit),'eligible',sum(r['eligible'] for r in audit))
for r in audit:
 if r['split']=='validation' and r['grade']==1:print(r['id'],'old',r['old_feature'][:4],'new',r['new_feature'])
