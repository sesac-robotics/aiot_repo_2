"""영상 직접 재검토 초안을 항목별로 고정한다. 제공자 오류명을 그대로 복사하지 않는다."""
import json,hashlib
from pathlib import Path
O=Path(__file__).parent;rs=json.loads((O/'source-manifest.json').read_text());out=[]
# 실제 두 시점 표본 검토의 판단. 0/1/2는 이 관측 과제의 초안이며 건강·안전 등급 아님.
depths={'Petra':[1,1,1,1,1],'Terka':[1,1,1,0,1],'Vasil':[2,2,2,1,2],'Mima':[2,1,1,1,1]}
body={'Petra':[2,2,0,2,2],'Terka':[2,2,1,2,2],'Vasil':[2,2,0,2,2],'Mima':[2,2,1,2,2]}
for r in rs:
 idx=int(r['id'].split('_')[1])-1
 levels=dict(knee_inward=0 if idx==1 else 2,descent_range=depths[r['person']][idx],standing_return=2,trunk_balance=body[r['person']][idx],heel_support=0 if idx==4 else 2,spine_control=None)
 evidence=dict(knee_inward='하단에서 양 무릎이 발 간격 대비 뚜렷이 안쪽으로 모이는 모습 확인' if idx==1 else '정면 표본에서 뚜렷한 안쪽 모임 신호가 보이지 않음. 발 방향 세밀 판정은 제한',
 descent_range='측면 하단에서 허벅지 수평에 이르지 않는 최소 하강' if levels['descent_range']==0 else '측면 하단에서 허벅지가 수평보다 위인 부분 범위; 개인에게 잘못된 운동이라는 뜻은 아님' if levels['descent_range']==1 else '측면 하단에서 허벅지가 수평 가까이 또는 그 아래에 도달',
 standing_return='종료 표본에서 무릎·엉덩이를 펴 선 자세로 돌아옴',trunk_balance='정면 하단에서 뚜렷한 한쪽 기울임 확인' if levels['trunk_balance']==0 else '정면에서 한쪽 기울임이 보이지만 크기·지속 정도가 부분적' if levels['trunk_balance']==1 else '뚜렷한 좌우 기울임 신호가 보이지 않음',
 heel_support='측면 하단에서 뒤꿈치가 바닥에서 올라가고 앞발로 지지하는 모습 확인' if idx==4 else '정면·측면 표본에서 뚜렷한 뒤꿈치 들림 신호가 보이지 않음',spine_control='표본만으로 허리 분절 정렬·미세 반동을 확정하지 않음')
 out.append(dict(id=r['id'],person=r['person'],split=r['split'],recording_group='ex6_'+r['person'],sources=[v['sha256'] for v in r['views'].values()],levels=levels,evidence=evidence,label_origin='source_aware_direct_visual_draft',rule_generated_label=False,expert_validated=False,whole_normal_approved=False,review_frames_per_view=9,review_method='정면 cam1·측면 cam3 시간순 확대 표본 직접 검토; 연속 재생 아님',reference_goal='하강 범위 등급은 허벅지 수평을 참조한 영상상 범위 기술이며 개인에게 강요하는 깊이 기준 아님',review_evidence=str(O/(r['person']+'-dense.jpg'))))
b=json.dumps(dict(schema='squat-observation-checklist-v2',frozen_before_training=True,records=out),ensure_ascii=False,indent=2).encode();(O/'labels.frozen.json').write_bytes(b);(O/'labels.sha256').write_text(hashlib.sha256(b).hexdigest());print('frozen',len(out))
