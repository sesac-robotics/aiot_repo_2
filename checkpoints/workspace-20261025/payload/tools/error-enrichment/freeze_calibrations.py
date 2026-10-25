from standing_state import *
rows=read(O/'extra-calibration-candidates.json');states=read(D/'standing-states-hard-reviewed.frozen.json');existing=read(O/'state-review-manifest.json')
for r in rows:
 assert r['id'] not in {v['id'] for v in existing}
 existing.append(dict(r,times=[0]));states.append(dict(id=r['id']+'_calibration0',review_id=r['id'],path=r['path'],source_sha256=r['sha256'],frame=0,fps=r['fps'],person=r['person'],split=r['split'],state='standing',target=1,sheet=r['sheet'],annotation_status='nonexpert_visual_calibration_frame',failure_label=False,review_note_ko='직접 확인한 운동 전 기립 프레임. 이후 복귀 정답을 입력 기준으로 사용하지 않음.'))
(O/'state-review-manifest.json').write_text(json.dumps(existing,ensure_ascii=False,indent=2));(D/'standing-states-final.frozen.json').write_text(json.dumps(states,ensure_ascii=False,indent=2));print('state final annotated',len(states))
