"""Add directly re-reviewed external Train plank states; retain source lineage."""
import json,sys,copy
from pathlib import Path
import numpy as np
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'output/initial-pose-return-v4-20261024';O=B/'output/initial-pose-return-v5-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from common import sha
from split_guard import assert_training_allowed
O.mkdir(exist_ok=True)
assert not (O/'state-labels-v3.frozen.json').exists()
x=json.loads((D/'state-labels-v3.frozen.json').read_text());sources=json.loads((D/'source-intake.frozen.json').read_text())
old=B/'output/five-exercise-comparison-20261021'
catalog={r['id']:r for r in json.loads((old/'labels.json').read_text())}
specs=[('dvids_navy773312_2997_3087',0,'navy773312_core_100.jpg'),('dvids_navy773312_3656_3746',1,'navy773312_core_122.jpg'),('youtube_jonple_error',0,'jonple_error_6_9.jpg'),('youtube_jonple_normal',1,'jonple_normal_12_15.jpg')]
added=[]
for sid,target,sheet_name in specs:
 s=catalog[sid];assert s['split']=='train';cp=old/'heavy'/sid/'side.npz';z=np.load(cp);meta=json.loads(cp.with_suffix('.json').read_text());digest=sha(s['video_path']);assert digest==s['video_sha256']==meta['config']['source_sha256'];assert str(z['fingerprint'])==meta['fingerprint']
 sheet=old/'review'/sheet_name;lineage_path=old/'youtube_segments'/(sid+'_lineage.json');lineage=json.loads(lineage_path.read_text()) if lineage_path.exists() else None
 hashes=[digest]
 if lineage:
  assert sha(lineage['original_video_path'])==lineage['original_sha256'];assert lineage['derived_sha256']==digest;hashes.append(lineage['original_sha256'])
 # Stable interiors only; eight distinct observed frames, no transition labels.
 fps=float(s['segment']['fps']);indices=np.linspace(3,len(z['frame_numbers'])-4,8).round().astype(int)
 for n in indices:
  frame=int(z['frame_numbers'][n]);r=dict(id=f'{sid}_state_{frame}',source_id=sid,exercise='plank',person=s['person'],group=s['person'],split='train',sources=hashes,target=target,phase=None,review_time=frame/fps,source_sha256=digest,source_path=s['video_path'],cache=str(cp),fps=fps,review_sheet=str(sheet),review_sheet_sha256=sha(sheet),review_status='direct_chronological_visual_recheck_nonexpert',not_ready_frame_is_not_repetition_failure=True,overall_quality_grade=None,reason='Visible extended support and aligned trunk' if target else 'Visible collapsed trunk/lower-body support, not a valid initial hold',family='US_Navy' if sid.startswith('dvids') else 'other_reviewed',lineage=lineage)
  x['labels'].append(r);added.append(r)
 sources.append(dict(id=sid,exercise='plank',person=s['person'],person_is_canonical=True,split='train',source=s['video_path'],source_sha256=digest,fps=fps,cache=str(cp),review_start=float(z['frame_numbers'][indices[0]])/fps,review_end=float(z['frame_numbers'][indices[-1]])/fps,external=True))
assert_training_allowed(x['labels'])
# Cross-check existing mixed score inputs, not just a newly assigned split tag.
score=json.loads((B/'output/paired-failure-enriched-five-v2-20261024/score-inputs.frozen.json').read_text())
for sid,_,_ in specs:
 rows=[r for r in score if r['id']==sid];assert len(rows)==1 and rows[0]['split']=='train';assert rows[0]['person']==catalog[sid]['person']
x.update(supersedes=str(D/'state-labels-v3.frozen.json'),revision_reason='Add two independent existing Train performers with directly re-reviewed aligned and collapsed plank states. No Test states or guessed contact labels.',external_additions=len(added))
(O/'state-labels-v3.frozen.json').write_text(json.dumps(x,ensure_ascii=False,indent=2));(O/'source-intake.frozen.json').write_text(json.dumps(sources,indent=2));(O/'external-intake-audit.json').write_text(json.dumps(dict(added_frames=len(added),added_source_segments=4,added_people=2,existing_team_labels_unchanged=True,Test_used=False,source_lineage_preserved=True),indent=2));print('EXTERNAL_STATES',len(added),'TOTAL',len(x['labels']))
