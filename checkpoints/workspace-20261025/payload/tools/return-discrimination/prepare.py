import copy,json,sys,importlib.util,datetime
from pathlib import Path
from fractions import Fraction
import numpy as np
B=Path(__file__).resolve().parents[2];O=B/'output/return-discrimination-v1-20261024';O.mkdir(exist_ok=True);P=B/'output/mild-return-calibrated-20261024';sys.path[:0]=[str(B/'tools/return-discrimination'),str(B/'tools/error-enrichment')]
import features as geom
from common import sha
from split_guard import POLICY,assert_training_allowed
spec=importlib.util.spec_from_file_location('upper',B/'tools/return-root-cause/features.py');upper=importlib.util.module_from_spec(spec);spec.loader.exec_module(upper)
read=lambda p:json.loads(p.read_text());rows=read(P/'score-inputs.frozen.json');test=read(P/'test-inputs.frozen.json');labels=read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels'];byname={r['id']:r for r in labels};team={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};refs=read(B/'output/squat-side-return-v5-20261024/MODEL_SELECTION_PROTOCOL.frozen.json')['reference_windows'];audit=[]
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(P),new_bank='standing_support_extension_v1',feature_names=geom.NAMES,feature_code_sha256=sha(B/'tools/return-discrimination/features.py'),model_code_sha256=sha(B/'tools/error-enrichment/models.py'),new_labels_this_experiment=0,selection='nested participant-group CV; train mild subset same5 IDs; first require recall>=.65 and FP<=.15, maximize mild recall then total recall then minimize FP, balanced group MAE',candidate_models=['positive_deficit','logistic'],C=[.1,1.,10.],outer_folds=4,inner_folds=3,threshold_fit_Train_only=True,Test_feature_only_augmentation='same51rows, original labels/features and source roles retained',production_replacement=False),indent=2))
for r in rows+test:
 if r['exercise']!='squats':continue
 z=None;first=None;end=None;seq=None;rx=None;frames=None
 if r['id'].startswith(('return_squat_','reviewed_return_squat_')):
  name=r['id'].removeprefix('reviewed_').removeprefix('return_');d=byname[name];stem=Path(d['path']).stem;normal=[a for a in labels if a['path']==d['path'] and a['grade']==2];first=369 if 'diag_m4' in stem else min(a['last_frame'] for a in normal)-6;end=first+8
  if d['first_frame']<=end:continue
  cache=B/'output/squat-return-20261023'/(stem+'-heavy.npz');z=np.load(cache);mask=(z['frame_numbers']>=d['first_frame'])&(z['frame_numbers']<=d['last_frame']);frames=z['frame_numbers'][mask];rx=z['image_xy'][(z['frame_numbers']>=first)&(z['frame_numbers']<=end)];seq=z['image_xy'][mask]
 elif r['family']=='R2D2':
  d=team[r['id'][5:]];sid=d['source_id']
  if sid not in refs and sid!='V15':continue
  first,end=refs.get(sid,[0,.22]);cache=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';z=np.load(cache);t=z['frame_numbers']/float(Fraction(inv[sid]['fps']));rx=z['image_xy'][(t>=first)&(t<=end)];mask=(t>=max(d['start'],end))&(t<d['end']);seq=z['image_xy'][mask];frames=z['frame_numbers'][mask]
 if seq is None or len(rx)<3:continue
 q=rx[:,[11,12]].mean(1);h=rx[:,[23,24]].mean(1);ref=dict(hip_y=float(np.median(h[:,1])),shoulder_y=float(np.median(q[:,1])),center_x=float(np.median(h[:,0])),scale=float(np.median(np.linalg.norm(h-q,axis=1))),torso_x=float(np.median((q-h)[:,0])));base=upper.extract(seq,ref);f=geom.extract(seq,rx,base);g=next((g for i,g in zip(r['items'],r['levels']) if i['bank']=='height_upper_reversal_v1'),None)
 r['items'].append(dict(item='standing_height_return',bank='standing_support_extension_v1',feature=f['x'] if f else None,eligible=f is not None,holds=[] if f else ['coherent_reference_support_geometry_unavailable'],feature_version=geom.VERSION));r['levels'].append(g if f else None);audit.append(dict(id=r['id'],split=r['split'],grade=g,eligible=f is not None,cache_sha256=sha(cache),reference_window=[first,end],reference_before_attempt=True,return_source_frame=int(frames[f['return_index']]) if f else None,features=f['x'] if f else None))
assert_training_allowed([r for r in rows if r['split']=='train'])
for n,v in [('score-inputs.frozen.json',rows),('test-inputs.frozen.json',test),('feature-provenance.json',audit)]:
 assert not (O/n).exists();(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
(O/'original-test-inputs.frozen.json').write_bytes((P/'test-inputs.frozen.json').read_bytes());print('PREPARED',len(audit),'new features',[(s,sum(r['eligible'] and r['split']==s for r in audit)) for s in ['train','validation','test']])
