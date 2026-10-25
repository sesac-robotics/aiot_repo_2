import copy,json,sys,datetime
from pathlib import Path
B=Path(__file__).resolve().parents[2];OLD=B/'output/return-root-cause-v3-20261024';O=B/'output/mild-return-fit-20261024';SUB=O/'changed-heads-fit';SUB.mkdir(exist_ok=True);sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed,POLICY
read=lambda p:json.loads(p.read_text());rows=read(OLD/'score-inputs.frozen.json');target='return_squat_front_w1_9_3_r08';row=next(r for r in rows if r['id']==target);assert row['split']=='train' and row['person']=='cfrep_w1'
revision=dict(id=target,person=row['person'],split=row['split'],sources=row['sources'],criterion='standing_height_return',old_grade=2,new_grade=1,source_frames=[313,351],reference_frame=67,upper_turnaround_frames=[348,351],next_descent_frames=[354,360,366],normal_comparison='same source r07 and r09 return upright',evidence={n:sha(B/'output/mild-return-mining-20261024'/n) for n in ['w1-f67.jpg','w1-f348.jpg','w1-f351.jpg','w1-f354.jpg']},reason='Original-size chronological review: knees remain slightly flexed and hip/shoulder heights below initial standing at the upper turnaround, then next descent begins without full height recovery. Not a mid-ascent sample or a stopped/cut attempt.',reviewer='AI_assisted_visual_draft_not_expert',original_frozen_labels_preserved=True)
changed=set()
for i,(n,g) in enumerate(zip(row['items'],row['levels'])):
 if n['item']=='standing_height_return' and n['eligible']:
  assert g==2;row['levels'][i]=1;changed.add((row['exercise'],n['item'],n['bank']))
row['label_revision']=revision;assert len(changed)==3;assert_training_allowed([r for r in rows if r['split']=='train'])
for n,v in [('score-inputs.frozen.json',rows),('label-revisions.frozen.json',[revision])]:
 assert not (O/n).exists();(O/n).write_text(json.dumps(v,indent=2,allow_nan=False))
(O/'test-inputs.frozen.json').write_bytes((OLD/'test-inputs.frozen.json').read_bytes())
subset=[]
for r in rows:
 pairs=[(n,g) for n,g in zip(r['items'],r['levels']) if (r['exercise'],n['item'],n['bank']) in changed]
 if pairs:s=copy.deepcopy(r);s['items']=[n for n,g in pairs];s['levels']=[g for n,g in pairs];subset.append(s)
(SUB/'score-inputs.frozen.json').write_text(json.dumps(subset,indent=2,allow_nan=False))
(O/'MODEL_SELECTION_PROTOCOL.frozen.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_run=str(OLD),changed_keys=sorted(changed),Train_label_revision=revision,policy_sha256=sha(POLICY),Train_input_sha256=sha(O/'score-inputs.frozen.json'),Test_input_sha256=sha(O/'test-inputs.frozen.json'),selection='unchanged Train group CV and fixed development gates; no Test tuning',source_Train_Test_roles_unchanged=True,production_replacement=False),indent=2));print('PREPARED',len(rows),'label revision1; heads',len(changed))
