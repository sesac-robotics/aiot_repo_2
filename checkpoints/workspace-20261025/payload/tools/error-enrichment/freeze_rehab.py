from pathlib import Path
import json
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/error-enrichment-20261024';D=B/'data/error-enrichment-20261024/annotations';rs=json.loads((O/'rehab-intake.json').read_text());labels=[]
for r in rs:
 index=int(r['review_id'][1:]);knee=index in [3,7,11,15];trunk=index in [4,8,12,16];levels=[None,2,0 if knee else 2,1 if trunk else 2]
 note='완전한 하강·상승 구간에서 선 높이 복귀가 보인다. 정면 촬영에서 '
 note+=('하단의 무릎 간격이 시작/발 간격에 비해 뚜렷하게 안쪽으로 무너진다. ' if knee else '뚜렷한 무릎 안쪽 붕괴는 보이지 않는다. ')
 note+=('하강·상승 때 몸통이 한쪽으로 일부 기울고 어깨 중심이 골반 중심에서 벗어난다. ' if trunk else '큰 좌우 몸통 쏠림은 보이지 않는다. ')
 note+='재활 시범의 제공 NORMAL/오류 문자열을 점수로 복사하지 않았다. 정면만으로 깊이를 확정하지 않아 하강 범위는 보류.'
 labels.append(dict(id='enrich_'+r['review_id'],review_id=r['review_id'],source_id=r['video_id'],exercise='squats',person=r['person'],path=r['path'],source_sha256=r['source_sha256'],fps=r['fps'],split=r['split'],family='REHAB26',start=r['first_frame']/r['fps'],end=(r['last_frame']+1)/r['fps'],first_frame=r['first_frame'],last_frame=r['last_frame'],view='front',variant='rehabilitation_squat_observable_items_only',levels=levels,evidence_note_ko=note,fault_tags=['knees_inward'] if knee else ['lateral_trunk_lean'] if trunk else [],sheet=str(O/f'rehab-detail-{(index-1)//4+1:02}.jpg'),review_method='12_chronological_source_frames_with_uncropped_overview_and_three_camera_view_check',annotation_status='nonexpert_visual_draft',train_eligible=True,null_not_filled=True))
(D/'rehab-segments.frozen.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2));print('rehab labels',len(labels),'errors',sum(any(v is not None and v<2 for v in r['levels']) for r in labels))
