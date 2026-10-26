"""Direct visual draft decisions from sheets inspected in this review.
Nulls mean unobservable; provider labels are not converted into grades.
"""
SEGMENTS=[]
def add(e,a,b,y,note,tags=(),view='front',variant=None):
 SEGMENTS.append(dict(review_id=e,start=a,end=b,levels=y,evidence_note_ko=note,fault_tags=list(tags),view=view,variant=variant))
# Press: full lower->upper->lower windows, judged independently of provider bad.
add('E003',8.0,10.75,[1,2,2,2],'머리 위에서 양손을 가깝게 모으고 팔꿈치 굽힘을 남긴 상단. 양팔 동시 움직임과 어깨 부근 하단 복귀는 확인. 정면 몸통의 큰 좌우 이동은 없음.',['overhead_reach_partial','behind_head_lower_path'],variant='standing_press')
add('E003',10.75,13.25,[1,2,2,2],'다시 올린 상단에서도 팔꿈치가 굽혀진 좁은 형태. 머리 뒤쪽으로 내리지만 팔꿈치는 어깨 부근으로 복귀. 이 경로 자체를 하단 미복귀로 바꾸지 않음.',['overhead_reach_partial','behind_head_lower_path'],variant='standing_press')
add('E012',8.25,10.75,[None,2,1,2],'양팔 상단 시점에 일부 차이가 나며 내려올 때도 양쪽 도달이 엇갈린다. 상단 손 일부가 프레임에 걸려 최대 도달은 보류.',['asynchronous_arms','overhead_crop'],variant='standing_press')
add('E013',8.0,11.0,[1,2,2,2],'상단에서 양팔꿈치 굽힘이 상당히 남아 머리 위 완전 도달이 제한된다. 하단 복귀와 양팔 동시 움직임은 구분해 충족 처리.',['overhead_reach_partial'],variant='standing_press')
add('E013',11.0,13.5,[1,2,2,2],'양팔을 올리지만 상단에서도 팔꿈치가 구부러진 부분 프레스. 하단으로 복귀.',['overhead_reach_partial'],variant='standing_press')
add('E016',8.5,9.75,[2,2,2,2],'양팔이 머리 위 거의 펴진 상단과 어깨 부근 하단이 구분된다. 제공자 bad 문자열을 네 항목 실패로 복사하지 않았다.',variant='standing_press')
add('E019',10.0,12.5,[1,2,0,None],'화면 왼쪽 팔을 올릴 때 반대 팔은 낮은 위치에 남고 나중에 오른다. 양팔 동시 프레스 목표의 큰 시차. 역광으로 몸통 기울기 정량 등급은 보류.',['asynchronous_arms','bilateral_target_mismatch'],variant='seated_press')
add('E019',12.5,14.75,[1,2,0,None],'한 팔이 먼저 올라가고 반대 팔은 하단에 남는 비대칭 프레스 반복. 하단 복귀 자체는 보인다.',['asynchronous_arms','bilateral_target_mismatch'],variant='seated_press')
add('E020',8.0,9.75,[1,2,1,2],'양팔 도달 높이와 상승·하강 시점에 차이가 있다. 상단에서 한쪽 팔 굽힘이 남는다.',['overhead_reach_partial','asynchronous_arms'],variant='standing_press')
add('E020',9.75,11.75,[1,2,1,2],'양팔 상단 도달이 엇갈리며 팔꿈치 굽힘이 남는 부분 프레스.',['overhead_reach_partial','asynchronous_arms'],variant='standing_press')
# Squat cycles ending BEFORE the next descent. A cut trailing descent is never failure.
add('E026',.5,2.85,[1,2,None,2],'골반을 조금 내리면서 상체를 앞으로 크게 숙인 뒤 시작 선 높이로 돌아옴. 밝은 역광과 옷 때문에 무릎 간격 세부는 보류.',['shallow_descent','forward_fold_observed'])
add('E029',.5,3.5,[None,2,None,2],'하강 뒤 다시 서는 높이 확인. 어두운 장면의 하단 깊이·무릎 정렬 세부는 보류. 끝 하강은 포함하지 않는다.')
add('E031',.5,3.5,[1,2,None,2],'하단에서 골반이 높게 남는 부분 하강 뒤 선 높이로 복귀. 영상 아래쪽 발은 잘려 무릎·발 간격 판단은 보류.',['shallow_descent','feet_cropped'])
add('E039',.25,3.25,[2,2,None,2],'첫 완전 시도에서 충분한 하강과 원래 골반 높이 복귀. 두 번째 하강 중 영상이 잘리므로 첫 시도만 채택.')
add('E041',.25,1.95,[2,2,2,2],'첫 시도에서 하단과 서기 높이 복귀, 정면 무릎 간격 유지가 구분된다.')
add('E041',2.25,3.95,[2,2,2,2],'두 번째 완전 시도도 선 높이 복귀. 뒤에 잘린 세 번째 하강은 제외.')
# Front pushups: only arm range and return are observable, not sagittal trunk/support.
for e,a,b,depth,note in [
 ('E046',1.0,4.0,2,'가슴을 낮추고 팔을 거의 편 상단으로 복귀. 시작 하단은 제외.'),
 ('E047',.25,2.5,2,'양팔을 굽혀 충분히 낮췄다가 거의 펴는 상단 확인. 하체는 가려짐.'),
 ('E049',1.0,2.75,1,'팔을 조금만 굽혔다가 올라오며 하강 범위가 작다.'),
 ('E049',2.75,4.75,1,'두 번째 시도도 작은 팔 굽힘과 제한된 가슴 하강.'),
 ('E050',.75,2.75,1,'상단에서 내려가지만 양팔 굽힘과 가슴 하강이 부분 범위에 머묾.'),
 ('E051',.25,1.5,2,'팔을 굽혀 낮춘 뒤 팔 펴기 상단으로 복귀. 옷과 정면 시점으로 하체 항목은 보류.'),
 ('E052',.25,2.5,2,'가슴을 바닥 가까이 낮췄다가 팔을 거의 편 상단으로 복귀.'),
 ('E053',.5,2.25,1,'가슴을 충분히 내리기 전 올라오는 작은 팔 굽힘. 다리·골반 정렬은 정면에서 확정하지 않음.'),
 ('E053',2.5,3.75,1,'다음 시도에서도 상단과 부분 하강을 왕복함.'),
 ('E054',.5,2.5,2,'하강에서 팔꿈치가 충분히 굽고 상단에서 팔을 다시 펴는 모습.'),
 ('E056',.25,1.75,2,'가슴을 낮춘 뒤 팔을 펴는 상단으로 복귀. 하체 접촉은 보이지 않아 보류.')]:
 add(e,a,b,[None,depth,2,None],note,['descent_partial'] if depth==1 else [])
# Visible interruption is an OBSERVED event. Intent/premature ending is not inferred.
for e,t in [('E057',1.5),('E058',3.0),('E059',2.25)]:
 add(e,0,4.75,[None,None,None,0],f'약 {t}초부터 전완 지지 형태를 풀고 쉬거나 일어나는 실제 전환이 보인다. 조기 종료 의도나 타이머 실패는 추정하지 않음.',['visible_upper_support_interruption'],variant='front_visible_interruption')
for e in ['E060','E061']:
 add(e,0,4.5,[None,None,None,2],'관찰 구간에서 전완·상체 지지의 뚜렷한 중단은 보이지 않는다. 보이지 않는 골반·무릎 접지 및 몸통 일직선 정답은 부여하지 않음.',variant='front_visible_interruption')
# QEVD hinged YTW remains separate from upright team YTW; frame projection only.
for e,a,b,y,note in [
 ('E069',0,3.25,[1,2,1,2],'숙인 몸통은 유지되며 곧게 편 T는 구분된다. 첫 팔 올림은 팔꿈치가 굽은 부분 형태이며 W도 일부 굽힘 형태만 보인다.'),
 ('E070',0,3.75,[1,2,1,2],'몸통을 숙여 유지. 첫 대각선 팔 올림에서 굽힘이 남으며 T는 구분되고 마지막 W는 부분 형태.'),
 ('E071',0,3.0,[1,2,0,2],'대각선 팔 올림은 굽힘이 남고 T는 보이나 양손이 팔꿈치 위로 올라온 양측 W는 만들지 않고 팔을 내린다.'),
 ('E072',0,3.5,[1,2,2,2],'Y 도달은 부분적. 팔을 옆으로 편 T와 양팔을 굽혀 손을 올린 W는 구분된다. 숙인 몸통 유지.'),
 ('E073',0,2.25,[0,2,0,2],'첫 전체 시도는 팔을 옆으로 편 T 뒤 곧바로 내림. 대각선 Y와 양측 W 단계가 없다.'),
 ('E074',0,3.25,[1,2,1,2],'첫 올림은 굽힌 팔의 부분 형태. T는 구분되며 W는 지나가는 부분 굽힘 형태. 숙인 몸통 유지.')]:
 add(e,a,b,y,note,['Y_shape_partial' if y[0]==1 else 'Y_stage_missing']+(['W_stage_missing'] if y[2]==0 else []),variant='hinged_ytw')
# Possible common recording setting: conservative review flag, not identity assertion.
QUARANTINE={'E069':'E074와 동일한 문·수납장 촬영 배경이 보여 촬영 그룹 누출 가능성 검토 전 신규 학습에서 제외','E074':'E069와 공통 촬영 배경. 다른 worker ID라도 독립 사람·촬영 검증으로 단정하지 않음'}
# Direct single-frame posture-state anchors. Flexed is NOT a failed repetition label.
STATE_POINTS={
'E026':{'standing':[0,.5,2.73,3.0],'flexed':[1.5,1.73,3.73,4.0]},
'E027':{'standing':[0,.57,3.3],'flexed':[1.63,2.2,4.93]},
'E029':{'standing':[0,.5,3.0,3.5],'flexed':[1.5,1.73,2.0,4.5]},
'E030':{'standing':[0,.53,3.3],'flexed':[1.63,2.2,4.4]},
'E031':{'standing':[0,.5,2.73,3.0],'flexed':[1.5,1.73,2.0,4.5]},
'E032':{'standing':[0,.57,3.33],'flexed':[1.67,2.23,4.43]},
'E033':{'standing':[0,.57,3.4],'flexed':[1.7,2.27,5.1]},
'E034':{'standing':[0,.53,2.73],'flexed':[1.07,1.63,4.37]},
'E035':{'standing':[0,.57,3.3],'flexed':[1.67,2.2,4.43]},
'E037':{'standing':[0,.53,3.3],'flexed':[1.63,2.2,4.4]},
'E039':{'standing':[0,.27,3.0,3.27],'flexed':[1.27,1.5,1.73,4.27,4.5]},
'E040':{'standing':[0,.53,3.23],'flexed':[1.63,2.17,4.33]},
'E041':{'standing':[0,.27,1.73,2.0,3.73,4.0],'flexed':[1.0,1.27,2.73,3.0,4.5]},
'E042':{'standing':[0,.57,2.8,4.47],'flexed':[1.67,3.37,5.03]},
'E043':{'standing':[0,.53,2.63,3.17],'flexed':[1.6,2.1,4.2,4.73]}}
ADDITIONAL_STATE_POINTS={
'V01':{'standing':[0,.5,3.0],'flexed':[1,1.5,2,2.5]},
'V02':{'standing':[0,3.0,3.5],'flexed':[1,1.5,2,2.5]},
'V06':{'standing':[0,2.5,3.0],'flexed':[1,1.5,2,4]},
'V07':{'standing':[0,.5,3,3.5],'flexed':[1.5,2,2.5]},
'V10':{'standing':[.5,2.5,3,3.5],'flexed':[1,1.5,2,4]},
'V12':{'standing':[0,.5,3.5,4],'flexed':[1.5,2,2.5,3]},
'V13':{'standing':[0,2,2.5],'flexed':[.5,1,1.5,3,3.5]},
'V17':{'standing':[0,.5,3.5,4],'flexed':[1.5,2,2.5,3]},
'V18':{'standing':[0,.5,1,1.5,2,2.5],'flexed':[3.5,4]},
'squat_diag_m1_6_7':{'standing':[0,.5,2,3.5],'flexed':[1,1.5,2.5,3,4]},
'squat_diag_m4_5_7':{'standing':[.5,2,3.5],'flexed':[0,1.5,2.5,3,4]},
'squat_diag_w1_9_3':{'standing':[0,2,3.5],'flexed':[1,1.5,2.5,3,4]},
'squat_diag_w2_5_7':{'standing':[0,.5,1,2.5,4],'flexed':[1.5,2,3,3.5]},
'squat_front_m3_9_3':{'standing':[0,1.5,3],'flexed':[1,2,2.5,3.5,4]},
'squat_diag_m2_8_6':{'standing':[0,.5,1,1.5,3.5],'flexed':[2,2.5,3,4]}}
