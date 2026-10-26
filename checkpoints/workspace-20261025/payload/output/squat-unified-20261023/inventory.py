"""Whole local source manifest; metadata candidates are never posture ground truth."""
from pathlib import Path
import json,csv,collections,tarfile,zipfile,re,hashlib
R=Path(__file__).resolve().parents[2];O=Path(__file__).parent
rows=[]
def add(source,identity,path,**kw):rows.append(dict(source=source,id=identity,path=str(path),present=Path(path).exists(),**kw))
q=R/'data/qevd-selected'
for r in csv.DictReader(open(q/'manifests/short_clips.csv')):
 labels=json.loads(r['labels'])
 if any(x.startswith('squats') for x in labels):add('QEVD',r['video_id'],q/r['video_path'],person='qevd_'+r['worker_id'],provider_split=r['split'],provider_labels=labels,role='candidate_needs_item_review',variant='plie_or_nonstandard' if any('plie' in x for x in labels) else 'candidate')
for r in csv.DictReader(open(R/'data/rehab26-squat/manifests/repetitions.csv')):
 if r['exercise_id']=='6':add('REHAB26',f"{r['video_id']}_{r['first_frame']}_{r['last_frame']}",R/'data/rehab26-squat/raw/videos/ex6'/(r['video_id']+'.mp4'),person='rehab26_'+r['person'],start_frame=int(r['first_frame']),end_frame=int(r['last_frame']),camera=r['camera'],orientation=r['person_orientation'],provider_labels=r['error_description'],role='candidate_needs_item_review')
for source,base in [('MultiView',R/'data/A Multi-View Raw Video Dataset of Seven Fitness Ex'),('REHAB24',R/'data/rehab24-squat'),('PoseComparison',R/'data/pose-comparison-selected')]:
 for p in sorted(base.rglob('*.mp4')):
  if source=='REHAB24' or 'squat' in str(p).lower():
   match=re.search(r'subject_(\d+)',p.name);add(source,str(p.relative_to(base)),p,person=(f'multiview_{match[1]}' if match else None),role='candidate_needs_item_review',bytes=p.stat().st_size)
# All archived AI-Hub metadata; no code from downloaded model archives is executed.
a=R/'output/five-exercise-comparison-20261021/aihub_documents/validation_body_01.zip';aihub_counts=collections.Counter();aihub_example=None
if a.exists():
 with zipfile.ZipFile(a) as z:
  for info in z.infolist():
   if not info.filename.endswith('.json'):continue
   d=json.loads(z.read(info));text=json.dumps(d,ensure_ascii=False);aihub_counts['all_json_records']+=1
   if '-3d' not in info.filename:aihub_counts['exercise:'+d.get('type_info',{}).get('exercise','unknown')]+=1
   if '스쿼트' in text or 'squat' in text.lower():
    aihub_counts['squat_metadata_records']+=1;add('AIHub231',info.filename,a,person=None,role='archive_candidate_person_view_and_variant_review_required',metadata=d)
archive=[]
for p in (R/'data/aihub231').rglob('*.tar'):
 with tarfile.open(p,'r:') as t:members=[dict(name=m.name,size=m.size,offset_data=m.offset_data) for m in t if m.isfile()]
 archive.append(dict(path=str(p),bytes=p.stat().st_size,members=members,role='raw/model archive inventory only; no whole archive extraction or model execution'))
# User videos and pose-only sessions have different observable content.
L=R/'output/live-webcam-20261023'
for p in sorted((L/'recordings').glob('*/pose.npz')):add('UserWebcam',p.parent.name,p,person='local_user_1',role='saved_pose_with_explicit_video_recording',videos=[str(x) for x in p.parent.glob('*') if x.suffix in ('.avi','.mp4')])
for p in sorted((L/'sessions').glob('*/pose.npz')):add('UserPoseSession',p.parent.name,p,person='local_user_1',role='pose_only_not_independent_visual_labels')
# External downloaded sources are inventoried from their actual label records, including non-squat Navy.
external=[]
for name in ['labels.json','supplemental_labels.json']:
 p=R/'output/five-exercise-comparison-20261021'/name
 for r in json.loads(p.read_text()):
  if 'navy' in r.get('id','') or r.get('video_path','').startswith(str(R/'output')):
   external.append(dict(id=r.get('id'),exercise=r.get('exercise'),path=r.get('video_path'),person=r.get('person'),label_file=str(p)))
   if r.get('exercise')=='squats' and '/output/' in r.get('video_path',''):add('ExternalVideo',r['id'],r['video_path'],person=r.get('person'),role='existing_visual_draft_needs_new_rubric_audit',label_file=str(p))
# Additional external squat annotations (including later YouTube sources).
for name in ['score_annotations.json','additional_score_annotations.json']:
 p=R/'output/three-exercise-comparison-20261022'/name
 if p.exists():
  for r in json.loads(p.read_text()):
   if r.get('exercise')=='squats' and '/output/' in r.get('video_path',''):
    add('ExternalVideo',r['id'],r['video_path'],person=r.get('person'),role='existing_visual_draft_needs_new_rubric_audit',label_file=str(p))
# Search metadata for all actually downloaded YouTube originals; keyword match
# only places a source in the review queue, never assigns a posture label.
for folder in [R/'output/three-exercise-comparison-20261022/sources',R/'output/five-exercise-comparison-20261021/sources']:
 for p in sorted(folder.glob('*.metadata.json')):
  d=json.loads(p.read_text());video=p.with_name(p.name.replace('.metadata.json','.mp4'));text=(str(d.get('title',''))+' '+str(d.get('description',''))).lower()
  add('YouTube',d.get('id',p.stem),video,person='channel_group:'+str(d.get('channel_id','unknown')),exercise_candidate='squat' if 'squat' in text or '스쿼트' in text else 'other_or_unknown',role='source_metadata_only_not_posture_label',title=d.get('title'),metadata_path=str(p))
# Canonical IDs prevent repeated views/copies from being reported as independent repeats.
summary=dict(source_counts=dict(collections.Counter(r['source'] for r in rows)),stored_counts=dict(collections.Counter(r['source'] for r in rows if r['present'])),aihub_metadata_counts=dict(aihub_counts),rehab26_unique_repetitions=len({(r['person'],r['start_frame'],r['end_frame']) for r in rows if r['source']=='REHAB26'}),external_records=list({r['id']:r for r in external}.values()),full_frame_visual_review_completed=False,all_sources_enumerated=True,independent_final_test=False,notes=['QEVD provider labels are search hints, not four-item scores','AI-Hub person IDs/FPS/variant eligibility must be established before learning','MultiView Exercise-wise and Quality-wise paths may be exact duplicate bytes; paths are not independent people','Navy labels found in current canonical files concern plank, not squat','Existing test exposure including Honza prohibits claims of untouched final validation'])
(O/'source-inventory.json').write_text(json.dumps(dict(summary=summary,records=rows,archives=archive),ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='external_records'},ensure_ascii=False,indent=2))
