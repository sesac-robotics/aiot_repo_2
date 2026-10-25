from pathlib import Path
import json,hashlib
from manual_labels import ADDITIONAL_STATE_POINTS
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';D=B/'data/error-enrichment-20261024/annotations';rows={r['id']:r for r in json.loads((O/'state-review-manifest.json').read_text())};states=json.loads((D/'standing-states.frozen.json').read_text())
for id,entry in ADDITIONAL_STATE_POINTS.items():
 r=rows[id];sha=hashlib.file_digest(Path(r['path']).open('rb'),'sha256').hexdigest()
 for label,times in entry.items():
  for t in times:states.append(dict(id=id+'_'+str(round(t*r['fps'])),review_id=id,path=r['path'],source_sha256=sha,frame=round(t*r['fps']),fps=r['fps'],person=r['person'],split=r['split'],state=label,target=int(label=='standing'),sheet=r['sheet'],annotation_status='nonexpert_visual_frame_state_draft',failure_label=False))
assert len({r['id'] for r in states})==len(states)
(D/'standing-states-combined.frozen.json').write_text(json.dumps(states,ensure_ascii=False,indent=2));print('combined state frames',len(states),'people',len({r['person'] for r in states}))
# Extra partial-standing anchors are directly visible in the reviewed training sheet.
source=rows['squat_diag_m1_6_7'];digest=hashlib.file_digest(Path(source['path']).open('rb'),'sha256').hexdigest()
for label,fs in [('standing',[440,444,446,452]),('flexed',[499,507,548,555,600,607,650,657])]:
 for f in fs:
  states.append(dict(id='m1_partial_state_'+str(f),review_id=source['id'],path=source['path'],source_sha256=digest,frame=f,fps=source['fps'],person=source['person'],split=source['split'],state=label,target=int(label=='standing'),sheet=str(B/'output/squat-return-20261023/squat_diag_m1_6_7-2.jpg' if f<599 else B/'output/squat-return-20261023/squat_diag_m1_6_7-3.jpg'),annotation_status='nonexpert_visual_frame_state_draft',failure_label=False))
(D/'standing-states-combined.frozen.json').write_text(json.dumps(states,ensure_ascii=False,indent=2))
