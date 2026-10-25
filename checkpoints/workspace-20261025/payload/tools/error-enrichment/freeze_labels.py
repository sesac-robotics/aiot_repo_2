import json,hashlib
from pathlib import Path
from manual_labels import SEGMENTS,STATE_POINTS,QUARANTINE
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';D=B/'data/error-enrichment-20261024';(D/'annotations').mkdir(parents=True,exist_ok=True)
rs={r['review_id']:r for r in json.loads((O/'intake.json').read_text())};out=[]
for k,d in enumerate(SEGMENTS):
 r=rs[d['review_id']];d=dict(d)
 if r['exercise']=='plank':
  # Absence of a visible upper-body event cannot certify whole-body continuity.
  d['visible_upper_interruption']=int(d['levels'][3]==0);d['levels']=[None]*4
 assert 0<=d['start']<d['end']<r['duration'];assert all(g in [None,0,1,2] for g in d['levels']);out.append(dict(d,id='enrich_'+d['review_id']+'_'+str(k).zfill(3),source_id=r['id'],exercise=r['exercise'],person=r['person'],path=r['path'],source_sha256=r['sha256'],fps=r['fps'],split=r['split'],family='MultiView' if r['id'].startswith('subject') else 'QEVD',sheet=str(O/(d['review_id']+'-dense.jpg')),review_method='chronological_4fps_detail_plus_full_source_overview',annotation_status='nonexpert_visual_draft',train_eligible=d['review_id'] not in QUARANTINE,quarantine_reason=QUARANTINE.get(d['review_id']),null_not_filled=True))
p=D/'annotations/segments.frozen.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2));state=[]
for e,d in STATE_POINTS.items():
 r=rs[e]
 for label,times in d.items():
  for t in times:state.append(dict(id=e+'_'+str(round(t*r['fps'])),review_id=e,path=r['path'],source_sha256=r['sha256'],frame=round(t*r['fps']),fps=r['fps'],person=r['person'],split=r['split'],state=label,target=int(label=='standing'),sheet=r['sheet'],detail_sheet=str(O/(e+'-dense.jpg')) if (O/(e+'-dense.jpg')).exists() else None,annotation_status='nonexpert_visual_frame_state_draft',failure_label=False))
(D/'annotations/standing-states.frozen.json').write_text(json.dumps(state,ensure_ascii=False,indent=2));(O/'label-freeze.json').write_text(json.dumps(dict(segments=len(out),posture_state_frames=len(state),quarantine=QUARANTINE,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),state_labels_are_not_failure_labels=True),ensure_ascii=False,indent=2));print('segments',len(out),'statepoints',len(state))
