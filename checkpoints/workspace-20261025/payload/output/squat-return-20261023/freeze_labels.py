from pathlib import Path
import json,hashlib,collections
O=Path(__file__).resolve().parent;rows=json.loads((O/'cfrep-manifest.json').read_text());labels=[]
for r in rows:
 name=Path(r['path']).stem;num=int(r['id'].split('_r')[-1]);grade=2;reason='시간순 원본 표본에서 하강 후 시작 높이의 선 자세로 복귀함. 제공자 rep/no-rep를 복귀 라벨로 복사하지 않음.'
 if name=='squat_front_m1_9_4' and num==11:grade=1;reason='하강 후 올라오지만 선 높이에 이르기 전에 다음 하강으로 전환함. 영상 끝 잘림이 아닌 중간 구간의 부분 복귀.'
 if name=='squat_diag_m1_6_7' and num in [10,11,12,13]:grade=1;reason='초기 기립 기준보다 낮은 굽힌 자세로 상승을 끝냄. 후속 하강 또는 종료 직후 표본까지 확인한 부분 복귀.'
 if name=='squat_diag_m4_5_7' and num in [10,11]:grade=1;reason='이전 정상 기립보다 낮고 굽힌 위치에서 상승 후 다시 하강함. 뒤의 완전 기립 구간과 구분.'
 if name=='squat_diag_w2_5_7' and num in range(6,12):grade=1;reason='초기 기립보다 낮은 상단 위치에서 다음 하강으로 전환. 마지막 완전 기립 복귀와 구분되는 부분 복귀.'
 if name=='squat_diag_m4_5_7' and num<=5:grade=None;reason='영상 첫 장면이 하강 상태라 최초 서기 기준 확보 전의 반복. 이 구간은 AI 학습·평가에서 제외.'
 if name=='squat_front_m4_6_6' and num in range(7,11):grade=None;reason='무효 반복이지만 몸통 숙임과 골반 높이 미복귀를 표본만으로 확실히 구분하기 어려움. 복귀 오류로 임의 변환하지 않음.'
 labels.append(dict(**r,grade=grade,reason=reason,review_method='원본 시간순 9프레임 표본 직접 검토',expert_validated=False))
d=dict(target='standing_return',policy='정면 관측 서기 높이 복귀 0/1/2, 배점20 유지',label_values={'0':'거의 복귀하지 못함 — 이번 확보 자료 없음','1':'명확한 부분 복귀','2':'시작 선 높이로 복귀'},labels=labels,split_policy='동일 사람의 모든 시점·원본은 같은 역할. w2 고정 검증, m2/w4 미열람 최종 예비.',independent_expert_review=False)
p=O/'labels.frozen.json';p.write_text(json.dumps(d,ensure_ascii=False,indent=2));(O/'labels.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'\n');print(collections.Counter(r['grade'] for r in labels));print('rows',len(labels))
