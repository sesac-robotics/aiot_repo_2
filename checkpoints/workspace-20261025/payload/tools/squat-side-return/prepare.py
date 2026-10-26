"""Recover observed side geometry without changing labels, roles or fixed Test."""
import copy,json,sys,datetime,os
from fractions import Fraction
from pathlib import Path
import numpy as np

B=Path(__file__).resolve().parents[2];O=Path(os.environ.get('SIDE_RETURN_OUTPUT',str(B/'output/squat-side-return-20261024')));O.mkdir(exist_ok=True);OLD=B/'output/return-failure-scores-20261024';SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True)
sys.path[:0]=[str(B/'tools/error-enrichment'),str(B/'tools/integrated-five')]
from common import sha
from split_guard import assert_training_allowed,POLICY
from team_features import describe,height_features
import side_height
read=lambda p:json.loads(p.read_text())
references={'V03':(0,.4),'V08':(2,2.5),'V09':(2.5,3),'V11':(1.8,2.25),'V14':(0,.22),'V16':(0,.4),'V19':(1.5,2)}
proof=B/'output/squat-side-return-20261024'
protocol=dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reference_windows=references,reference_proof_sheets={n:sha(proof/n) for n in ['train-starts.jpg','later-references.jpg']},selection='existing Train-only source-role and view-support deficit, not Test predictions',reference_rule='visually verified upright frames ending before interval start, no automatic first-frame calibration',geometry='same six bilateral height_v4 physical features; side uses fixed-feet guard instead of rejecting expected horizontal hip travel',source_code_sha256={n:sha(B/'tools/integrated-five'/n) for n in ['side_height.py','team_features.py']},Test='parent byte-identical; no reference annotation or new Test feature selection',production_replacement=False)
p=O/'MODEL_SELECTION_PROTOCOL.frozen.json'
assert not p.exists();p.write_text(json.dumps(protocol,indent=2))
labels={r['id']:r for r in read(B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json')['labels']};inv={r['id']:r for r in read(B/'output/r2d2-review-20261024/inventory.json')};policy=read(POLICY)
parent=read(OLD/'score-inputs.frozen.json');rows=copy.deepcopy(parent);audit=[]
for r in rows:
    if r['family']!='R2D2' or r['exercise']!='squats':continue
    d=labels[r['id'][5:]];sid=d['source_id']
    if sid not in references:continue
    assert r['split']==policy['assignments'][sid]=='train'
    assert_training_allowed([r])
    src=inv[sid];cache=B/'output/integrated-five-20261024/team-observations'/sid/'side.npz';z=np.load(cache);t=z['frame_numbers']/float(Fraction(src['fps']));a,e=references[sid];refxy=z['image_xy'][(t>=a)&(t<=e)];ref=side_height.reference(refxy,height_features) if len(refxy)>=3 else None
    seq=z['image_xy'][(t>=d['start'])&(t<d['end'])]
    # Team recordings are rotated into portrait by the decoder.
    w,h=(src['height'],src['width']) if src.get('rotation') else (src['width'],src['height'])
    note=describe('squats',seq,d['view'],d['end']-d['start'],ref,w,h,baseline_verified=True,baseline_end=e,segment_start=d['start'])[1]
    index=next(i for i,n in enumerate(r['items']) if n['item']=='standing_height_return')
    assert r['items'][index]['eligible'] is False and r['levels'][index] is None
    r['items'][index]=note
    target=next(i['grade'] for i in d['items'] if i['id']=='standing_height_return')
    r['levels'][index]=target if note['eligible'] else None
    audit.append(dict(id=r['id'],source=sid,split=r['split'],manual_grade=target,recovered=note['eligible'],holds=note['holds'],reference_window=[a,e],reference_pose_frames=len(refxy),cache_sha256=sha(cache),feature=note['feature']))
assert len(audit)==35
for r,p in zip(rows,parent):
    assert r['id']==p['id'] and r['sources']==p['sources'] and r['split']==p['split']
assert_training_allowed([r for r in rows if r['split']=='train'])
for n,v in [('score-inputs.frozen.json',rows),('feature-provenance.json',audit)]:
    p=O/n;assert not p.exists();p.write_text(json.dumps(v,indent=2,allow_nan=False))
subset=[]
for r in rows:
    pairs=[(n,g) for n,g in zip(r['items'],r['levels']) if n['item']=='standing_height_return' and n['bank']=='height_v4']
    if pairs:
        s=copy.deepcopy(r);s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs];subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,indent=2,allow_nan=False))
(O/'test-inputs.frozen.json').write_bytes((OLD/'test-inputs.frozen.json').read_bytes())
print('RECOVERED',sum(r['recovered'] for r in audit),'of35 Train sides; fixed Test3 sides unchanged; prior all970 source rows retained')
print('HELD',[(r['id'],r['holds'],r['reference_pose_frames']) for r in audit if not r['recovered']])
