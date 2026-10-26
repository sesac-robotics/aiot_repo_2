"""Freeze a Train-only timing fix and newly re-reviewed partial return targets."""
import copy,datetime,json,sys,os
from pathlib import Path
from fractions import Fraction
import numpy as np
B=Path(__file__).resolve().parents[2];O=Path(os.environ.get('RETURN_ROOT_OUTPUT',str(B/'output/return-root-cause-20261024')));O.mkdir(exist_ok=True);OLD=B/'output/squat-side-return-enriched-v2-20261024';SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from team_features import height_features
# Avoid importing the older integrated-five module named features.
import importlib.util
spec=importlib.util.spec_from_file_location('upper_reversal',B/'tools/return-root-cause/features.py');terminal=importlib.util.module_from_spec(spec);spec.loader.exec_module(terminal)
read=lambda p:json.loads(p.read_text());rows=read(OLD/'score-inputs.frozen.json');test=read(OLD/'test-inputs.frozen.json');parent=copy.deepcopy(rows);oldcf={r['id']:r for r in read(B/'output/squat-return-20261023/feature-rows.json')};labels={r['id']:r for r in read(B/'output/squat-return-20261023/labels-v2.frozen.json')['labels']};team={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};refs=read(B/'output/squat-side-return-v5-20261024/MODEL_SELECTION_PROTOCOL.frozen.json')['reference_windows'];groups={r['person']:r['group'] for r in rows};audit=[]
new=[]
for name in ['squat_diag_m4_5_7_r10','squat_diag_m4_5_7_r11']:
    d=labels[name];digest=sha(d['path']);assert d['split']=='train'
    new.append(dict(id='reviewed_return_'+name,exercise='squats',person=d['person'],group=groups[d['person']],sources=[digest],recording_groups=[d['person']],split='train',levels=[],items=[],family='CFRep_reviewed',origin='re_reviewed_return_failure',view='diagonal',whole_attempt_observed=True,reference_total_allowed=False,label_origin='dense_chronological_review_nonexpert',reviewed_label=dict(criterion='standing_height_return',grade=1,frames=[d['first_frame'],d['last_frame']],previous_status='withdrawn_after_pelvis_only_target_audit',reason='Dense review against the prior full standing frame376 confirms shoulder height remains below initial at the upper turnaround and another descent follows. Pelvis height alone must not stand for the two-channel initial standing-height target. Trunk angle alone is not a label.',expert_verified=False)))
rows+=new
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(OLD),target='initial standing hip AND shoulder height after a completed descent; angle alone not target; unchanged existing development/Test grades',grade0='no meaningful recovery in a completed attempt, not stop/cut/hold; none confirmed',grade1='observed partial height return followed by another descent',grade2='observed return to initial standing heights',timing_fix='post-bottom observed upper reversal rather than last-quarter-only window',training_and_selection='unchanged Train group CV only; fixed development reports, no Test tuning',feature_version=terminal.VERSION,feature_implementation_sha256=sha(B/'tools/return-root-cause/features.py'),policy_sha256=sha(POLICY),new_Train_labels=new,original_Test_kept_byte_identical_in='original-test-inputs.frozen.json',new_Test_feature_augmentation='same51IDs/source roles/existing targets; only append the corrected bank for three already-reviewed side diagnostic rows',final_Test_opened=False,production_replacement=False)
assert not (O/'MODEL_SELECTION_PROTOCOL.frozen.json').exists();(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(protocol,indent=2))
for r in rows+test:
    result=None;grade=None;cache=None;frames=None;refxy=None;sequence=None
    if r['family']=='CFRep_reviewed' and (r['id'].startswith('return_') or r['id'].startswith('reviewed_return_')):
        name=r['id'][len('reviewed_return_'):] if r['id'].startswith('reviewed_return_') else r['id'][len('return_'):]
        d=labels[name];stem=Path(d['path']).stem;cache=B/'output/squat-return-20261023'/(stem+'-heavy.npz');z=np.load(cache);mask=(z['frame_numbers']>=d['first_frame'])&(z['frame_numbers']<=d['last_frame']);frames=z['frame_numbers'][mask]
        if name in oldcf:
            ref=oldcf[name]['features']['baseline'];grade=d['grade'];first=18 if stem=='squat_diag_m4_5_7' else min(a['last_frame'] for a in labels.values() if Path(a['path']).stem==stem and a['grade']==2)-6
            refxy=z['image_xy'][(z['frame_numbers']>=first)&(z['frame_numbers']<=first+8)]
        elif r in new:
            refxy=z['image_xy'][(z['frame_numbers']>=369)&(z['frame_numbers']<=378)];ref=height_features.reference(refxy);grade=1
        else:continue
        sequence=z['image_xy'][mask];result=terminal.extract(sequence,ref,int(z['width']),int(z['height']))
    elif r['family']=='R2D2' and r['exercise']=='squats':
        d=team[r['id'][5:]];sid=d['source_id']
        if sid not in refs and sid!='V15':continue
        a,e=refs[sid] if sid in refs else [0,.22];cache=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';z=np.load(cache);t=z['frame_numbers']/float(Fraction(inv[sid]['fps']));refxy=z['image_xy'][(t>=a)&(t<=e)];ref=height_features.reference(refxy);start=max(d['start'],e)
        mask=(t>=start)&(t<d['end']);frames=z['frame_numbers'][mask];grade=next(i['grade'] for i in d['items'] if i['id']=='standing_height_return');sequence=z['image_xy'][mask];result=terminal.extract(sequence,ref,1080,1920)
    if cache is None:continue
    note=dict(item='standing_height_return',bank='height_upper_reversal_v1',feature=result['x'] if result else None,eligible=result is not None,holds=[] if result else ['required_post_bottom_geometry_not_observed'],feature_version=terminal.VERSION)
    r['items'].append(note);r['levels'].append(grade if result else None)
    pose=terminal.with_reference_pose(sequence,refxy,result)
    r['items'].append(dict(item='standing_height_return',bank='height_reference_pose_v1',feature=pose['x'] if pose else None,eligible=pose is not None,holds=[] if pose else ['reference_and_return_coherent_leg_chain_not_observed'],feature_version='height-initial-reference-pose-v1',side=pose['side'] if pose else None));r['levels'].append(grade if pose else None)
    audit.append(dict(id=r['id'],split=r['split'],grade=grade,eligible=result is not None,cache_sha256=sha(cache),return_source_frame=int(frames[result['return_index']]) if result else None,bottom_source_frame=int(frames[result['bottom_index']]) if result else None,features=note['feature']))
assert all(r['items'] and r['items'][0]['eligible'] for r in new)
assert_training_allowed([r for r in rows if r['split']=='train'])
assert not {s for r in rows for s in r['sources']}&{s for r in test for s in r['sources']}
for r,p in zip(rows,parent):assert r['id']==p['id'] and r['split']==p['split'] and r['sources']==p['sources'] and r['levels'][:len(p['levels'])]==p['levels']
for r,p in zip(test,read(OLD/'test-inputs.frozen.json')):assert r['id']==p['id'] and r['levels'][:len(p['levels'])]==p['levels'] and r['sources']==p['sources'] and r['split']==p['split']
for n,v in [('score-inputs.frozen.json',rows),('test-inputs.frozen.json',test),('feature-provenance.json',audit),('new-labels.frozen.json',new)]:
    assert not (O/n).exists();(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
(O/'original-test-inputs.frozen.json').write_bytes((OLD/'test-inputs.frozen.json').read_bytes())
subset=[]
for r in rows:
    pairs=[(n,g) for n,g in zip(r['items'],r['levels']) if n['bank'] in ['height_upper_reversal_v1','height_reference_pose_v1']]
    if pairs:s=copy.deepcopy(r);s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs];subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,indent=2,allow_nan=False))
print('PREPARED',len(rows),'rows;',len(new),'reviewed partial returns; new features',sum(r['eligible'] for r in audit),'Train',sum(r['eligible'] and r['split']=='train' for r in audit),'dev',sum(r['eligible'] and r['split']=='validation' for r in audit),'Test',sum(r['eligible'] and r['split']=='test' for r in audit))
print('M4R09',next(r for r in audit if r['id']=='return_squat_diag_m4_5_7_r09'))
