from pathlib import Path
import json,hashlib,collections
B=Path('/home/jayden/SeSACPAI-Mid');D=B/'data/r2d2-team-20261023';O=B/'output/r2d2-review-20261024';L=D/'annotations'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
inv=read(L/'sources.json');rows=read(L/'labels.frozen.json')['labels'];clips=read(D/'clip-manifest.json');cmap={c['id']:c for c in clips};assert len(cmap)==len(rows)==len(clips)==281
for s in inv:
 assert sha(Path(s['copy']))==s['sha256']==sha(Path(s['source']))
for r in rows:
 c=cmap[r['id']];assert str(B/r['clip'])==c['path'];assert sha(Path(c['path']))==c['sha256'];assert c['all_frames_decoded'];assert abs(c['actual_duration']-(r['end']-r['start']))<.15
 assert len(r['items'])==4 and all(i['grade'] in (None,0,1,2) for i in r['items']);assert r['observed_points']==sum(i['points'] or 0 for i in r['items'])
 assert (r['total_score'] is not None)==(r['complete_cycle'] and all(i['grade'] is not None for i in r['items']))
 for p in r['evidence_sheets']:assert (B/p).exists()
for s in inv:
 intervals=[(r['start'],r['end']) for r in rows if r['source_id']==s['id']]+[(r['start'],r['end']) for r in read(L/'unscored-intervals.json') if r['source_id']==s['id']];last=0
 for a,b in sorted(intervals):assert abs(a-last)<1e-6;last=b
 assert abs(last-s['duration'])<1e-6
for fold in read(L/'split-manifest.json')['folds']:
 rm={r['id']:r for r in rows};assert not set(fold['train'])&set(fold['validation']);assert all(rm[i]['person']!=fold['held_out_person'] for i in fold['train']);assert all(rm[i]['person']==fold['held_out_person'] for i in fold['validation'])
result=dict(status='passed',source_count=len(inv),clip_count=len(clips),people=4,exercises=5,all_original_copy_hashes_match=True,all_clip_hashes_match=True,all_clips_decoded=True,all_clip_durations_within_150ms=True,all_source_time_accounted=True,no_segment_overlap=True,no_person_overlap_in_folds=True,labels_sha256=sha(L/'labels.frozen.json'),rubrics_sha256=sha(L/'rubrics.json'),annotation_status='non_expert_visual_draft',review_method='chronological_4fps_contact_sheets',training_executed=False,quality_counts=dict(collections.Counter(r['quality'] for r in rows)))
(O/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(D/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps(result,ensure_ascii=False))
