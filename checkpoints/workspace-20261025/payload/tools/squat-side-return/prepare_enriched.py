import copy,datetime,json,sys,os
from pathlib import Path
import cv2,numpy as np
B=Path(__file__).resolve().parents[2];BASE=Path(os.environ.get('SIDE_RETURN_BASE',str(B/'output/squat-side-return-v4-20261024')));D=B/'output/squat-side-return-v3-20261024';O=Path(os.environ.get('SIDE_RETURN_FINAL',str(B/'output/squat-side-return-enriched-20261024')));O.mkdir(exist_ok=True);SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five'),str(B/'five-exercise-comparison')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from team_features import describe,height_features
import three_items
read=lambda p:json.loads(p.read_text());rows=read(BASE/'score-inputs.frozen.json');groups={r['person']:r['group'] for r in rows};audit=[]
for d in read(D/'mining/labels-v2.frozen.json')['reviewed']:
    assert d['split']=='train' and d['train_eligible'] and sha(d['path'])==d['source_sha256']
    cp=D/'observations'/d['review_id']/'primary.npz';z=np.load(cp);meta=read(cp.with_suffix('.json'));assert str(z['fingerprint'])==meta['fingerprint'] and meta['config']['source_sha256']==d['source_sha256']
    t=z['frame_numbers']/d['fps'];cap=cv2.VideoCapture(d['path'],cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,1]);w,h=cap.get(3),cap.get(4);cap.release();x=three_items.mask_image_border(z['image_xy'],w,h);a,e=d['reference_window'];assert e<=d['start'];ref=height_features.reference(x[(t>=a)&(t<=e)])
    notes=describe('squats',x[(t>=d['start'])&(t<d['end'])],'front',d['end']-d['start'],ref,w,h)
    for n,g in zip(notes,d['grades']):
        for k in ['rule_grade','measurement','higher_is_better','proxy_only']:n.pop(k,None)
        if g is None:n.update(eligible=False,feature=None,holds=['criterion_not_manually_labeled'])
        if n['feature'] is not None and not np.isfinite(n['feature']).all():n.update(eligible=False,feature=None,holds=['nonfinite_required_pose'])
    r=dict(id=d['id'],exercise='squats',person=d['person'],group=groups.get(d['person'],d['person']),sources=[d['source_sha256']],recording_groups=[d['person']],split='train',levels=d['grades'],items=notes,family='QEVD',origin='side_return_error_expansion',view='front',variant='squats',label_origin='dense_original_image_visual_draft',reference_total_allowed=False,whole_attempt_observed=True)
    assert_training_allowed([r]);rows.append(r);audit.append(dict(id=r['id'],cache_sha256=sha(cp),eligible_items=[n['item'] for n in notes if n['eligible']],holds={n['item']:n['holds'] for n in notes if not n['eligible']}))
assert len(rows)==974 and len({r['id'] for r in rows})==974
assert_training_allowed([r for r in rows if r['split']=='train'])
changed={('squats','standing_height_return','height_v4'),('squats','descent_range','front')}
for n,v in [('score-inputs.frozen.json',rows),('new-feature-provenance.json',audit)]:
    assert not (O/n).exists();(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
(O/'test-inputs.frozen.json').write_bytes((BASE/'test-inputs.frozen.json').read_bytes())
subset=[]
for r in rows:
    pairs=[(n,g) for n,g in zip(r['items'],r['levels']) if (r['exercise'],n['item'],n['bank']) in changed]
    if pairs:
        s=copy.deepcopy(r);s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs];subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,indent=2,allow_nan=False))
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(B/'output/return-failure-scores-20261024'),side_reference_protocol=str(BASE/'MODEL_SELECTION_PROTOCOL.frozen.json'),side_protocol_sha256=sha(BASE/'MODEL_SELECTION_PROTOCOL.frozen.json'),changed_keys=sorted(changed),source_roles_unchanged=True,policy_sha256=sha(POLICY),labels_sha256=sha(D/'mining/labels-v2.frozen.json'),training_input_sha256=sha(O/'score-inputs.frozen.json'),test_input_sha256=sha(O/'test-inputs.frozen.json'),selection='Train group CV and preexisting fixed development gates; no Test-driven thresholds or selection',new_real_return_failure_segments=0,new_shallow_descent_error_segments=4,production_replacement=False),indent=2))
print('PREPARED',len(rows),'new features',audit)
