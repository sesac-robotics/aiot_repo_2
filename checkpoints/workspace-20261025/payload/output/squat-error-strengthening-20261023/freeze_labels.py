import json,hashlib
from pathlib import Path
O=Path(__file__).resolve().parent
# Written after direct inspection of each of the eight chronological sheets.
reviewed={
'Petra_a10_0':(2,2),'Petra_a10_1':(0,None),'Petra_a10_2':(None,0),
'Petra_a20_0':(2,2),'Petra_a20_1':(0,None),'Petra_a20_2':(None,0),
'Terka_a10_0':(2,2),'Terka_a10_1':(0,None),'Terka_a10_2':(None,1),
'Terka_a20_0':(2,2),'Terka_a20_1':(0,None),'Terka_a20_2':(None,1),
'Vasil_a10_0':(2,2),'Vasil_a10_1':(0,None),'Vasil_a10_2':(None,0),
'Vasil_a20_0':(2,2),'Vasil_a20_1':(0,None),'Vasil_a20_2':(None,0),
'Mima_a10_0':(2,2),'Mima_a10_1':(0,None),'Mima_a10_2':(None,1),
'Mima_a20_0':(2,2),'Mima_a20_1':(0,None),'Mima_a20_2':(None,1),
}
rows=[]
for r in json.loads((O/'manifest.json').read_text()):
 k,t=reviewed[r['id']];reason={}
 if k==0:reason['knee_inward']='하강 표본에서 양 무릎이 발 지지 폭 안쪽으로 모이는 정렬을 직접 관찰. 시작 대비 무릎 간격 감소만을 뜻하지 않음.'
 elif k==2:reason['knee_inward']='근정면의 하강·상승 표본에서 뚜렷한 안쪽 무릎 정렬 오류 없음. 깊이·전체 정상 여부와 별개.'
 if t==0:reason['trunk_balance']='하강 표본에서 시작 대비 어깨·상체가 골반에 대해 한쪽으로 뚜렷하게 기울어짐.'
 elif t==1:reason['trunk_balance']='하강 표본에서 시작 대비 한쪽 기울임이 관찰되지만 뚜렷한 큰 기울임 사례보다 약함. 부분 관측 라벨 초안.'
 elif t==2:reason['trunk_balance']='시작·하강·복귀 표본에서 뚜렷한 좌우 기울임 변화 없음. 전후 숙임 정상 여부와 별개.'
 rows.append(dict(**r,levels={'knee_inward':k,'trunk_balance':t,'standing_return':2,'descent_range':None},evidence_reasons=reason,label_origin='independent_chronological_visual_draft',reviewed=True,model_outputs_used=False,rule_generated=False,expert_validated=False,whole_normal_approved=False,review_method='9 chronological enlarged raw-frame samples per repetition; not continuous playback',criterion_version='front-observation100-consistent-v1',view_scope='near_front_10_or_20_degrees',return_evidence='시작·종료 표본에서 같은 선 자세 높이로 복귀; 새 복귀 오류 라벨은 없음'))
raw=json.dumps(dict(records=rows,frozen_before_training=True,provider_labels_used_only_for_selection=True,expert_validated=False),ensure_ascii=False,indent=2).encode();(O/'labels.frozen.json').write_bytes(raw);(O/'labels.sha256').write_text(hashlib.sha256(raw).hexdigest())
ret=[dict(id=r['id'],person=r['person'],evidence=r['new_evidence'],return_failure=None,decision='not_error_label',reason='중간 구간의 기립 복귀가 관찰되고 마지막 하강은 다음 반복 도중 영상이 끝남. 클립 끝의 낮은 위치를 복귀 실패로 쓰지 않음.' if r['id']!='00188989' else '긴 옷에 가린 골반·무릎과 영상 끝 하강을 구분하기 어려워 복귀 실패 정답으로 만들지 않음.') for r in json.loads((O/'return-candidates.json').read_text())]
(O/'return-review.json').write_text(json.dumps(dict(reviewed=ret,new_confirmed_return_errors=0,used_for_AI_training=False),ensure_ascii=False,indent=2));print('frozen',len(rows))
