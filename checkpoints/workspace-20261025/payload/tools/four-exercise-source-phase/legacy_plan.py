from pathlib import Path
import json,hashlib,cv2
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/four-exercise-source-phase-v2-20261025';P=B/'output/four-exercise-review-train-v1-20261025/reconciled-v2';rows={r['id']:r for r in json.loads((P/'score-inputs.frozen.json').read_text())};remain=json.loads((O/'resolution-summary.json').read_text())['remaining_old_unresolved'];M=B/'output/five-exercise-comparison-20261021';manifest={r['id']:r for r in json.loads((M/'development_manifest.json').read_text())};lineage=[]
for p in [M/'youtube_segment_exports.json',M/'plank_support_enrichment_plan_exports.json',M/'plank_side_export_plan_exports.json']:
 a=json.loads(p.read_text());a=a['records'] if isinstance(a,dict) else a
 for r in a:lineage.append(dict(r,lineage_file=str(p)))
for p in [M/'sources/dvids_640237_side_lineage.json',M/'fao_candidate_lineage.json']:
 d=json.loads(p.read_text());lineage.append(dict(d,lineage_file=str(p)))
plans=[]
for rid in remain:
 r=rows[rid];aliases=[rid,rid.removesuffix('_held3_score'),rid.removesuffix('_held3_score').removesuffix('_held3')];req=next((B/'output/five-exercise-recovery-20261023/requests'/f'{a}.json' for a in aliases if (B/'output/five-exercise-recovery-20261023/requests'/f'{a}.json').exists()),None);rec=manifest.get(rid)
 if req:
  d=json.loads(req.read_text());path=d['video_path'];seg=d['assessment_input'];a=seg['start_frame'];b=seg['end_frame'];view=seg['view'];evidence=str(req)
 elif rec:
  path=rec['video_path'];a=rec['segment']['start_frame'];b=rec['segment']['end_frame'];view=rec['required_views'][0];evidence=str(M/'development_manifest.json')
 else:
  ex=next((v for v in lineage if v.get('id')==rid and v.get('video_path')),None)
  if ex is None:raise ValueError('missing legacy manifest '+rid)
  path=ex['video_path'];a=0;b=ex['frames']-1;view='side';evidence=ex['lineage_file']
 cap=cv2.VideoCapture(path);fps=cap.get(5);count=int(cap.get(7));ok,im=cap.read();cap.release();assert ok;sha=hashlib.sha256(Path(path).read_bytes()).hexdigest();parents=[]
 if sha not in r['sources']:
  for l in lineage:
   child=l.get('video_sha256',l.get('derived_sha256'));parent=l.get('source_sha256',l.get('original_sha256'))
   if child==sha and parent in r['sources']:parents.append(l)
  assert parents,('unverified derived origin',rid,path,sha,r['sources'])
  # Recheck native pixel lineage at start, midpoint, end of this very target window.
  l=parents[0];orig=l.get('source_path',l.get('original_video_path'));assert hashlib.sha256(Path(orig).read_bytes()).hexdigest() in r['sources'];crop=l.get('crop_xyxy',l.get('spatial_crop_xyxy'));offset=l.get('original_start_frame',l.get('start_frame',l.get('source_start_frame',0)));offset=0 if l.get('frame_numbers_preserved') else offset;c1=cv2.VideoCapture(path);c2=cv2.VideoCapture(orig);pixels=[]
  for f in [a,(a+b)//2,b]:
   c1.set(1,f);q,x=c1.read();c2.set(1,f+offset);w,y=c2.read();assert q and w
   if crop:y=y[crop[1]:crop[3],crop[0]:crop[2]]
   same=x.shape==y.shape and bool((x==y).all());pixels.append(dict(clip_frame=f,source_frame=f+offset,exact_pixels_equal=same));assert same,(rid,f)
  c1.release();c2.release();parent_evidence=dict(original_path=orig,original_sha256=hashlib.sha256(Path(orig).read_bytes()).hexdigest(),crop=crop,offset_frames=offset,lineage_file=l['lineage_file'],pixel_checks=pixels)
 else:parent_evidence=None
 plans.append(dict(id=rid,exercise=r['exercise'],person=r['person'],group=r['group'],family=r['family'],split='train',sources=r['sources'],source_path=path,source_sha256=sha,extraction_source_sha256=sha,original_lineage=parent_evidence,fps=fps,width=im.shape[1],height=im.shape[0],frame_count=count,window_seconds=[a/fps,(b+1)/fps],segment=dict(start_frame=a,end_frame=b,fps=fps),required_views=[view],manifest_evidence=evidence,previously_unresolved=True))
p=O/'legacy-extraction-plan.frozen.json';assert not p.exists();p.write_text(json.dumps(plans,ensure_ascii=False,indent=2)+'\n');print('legacy plans',len(plans),'derived lineages',sum(r['original_lineage'] is not None for r in plans))
