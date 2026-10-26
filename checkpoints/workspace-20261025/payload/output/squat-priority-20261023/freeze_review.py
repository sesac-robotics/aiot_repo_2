from pathlib import Path
import json,hashlib
O=Path(__file__).parent
first=json.loads((O/'preparation.json').read_text())['selected'];ranked=json.loads((O/'ranked-preparation.json').read_text())
a={0:(0,62),1:(21,63),3:(20,62),5:(21,85),7:(24,73),8:(20,83),10:(21,85),12:(21,85),13:(21,62),14:(21,63),15:(21,64),17:(21,86),18:(6,86),19:(20,82),20:(21,63),21:(0,60),23:(21,63),24:(20,83),28:(0,42),29:(21,62),30:(21,62),31:(0,61)}
b={2:(0,40),3:(20,61),14:(0,41),15:(20,104),16:(0,66),18:(0,46),22:(0,82),24:(0,41),26:(21,63),28:(0,40),29:(0,62),30:(22,66),31:(0,41)}
ka={13,15,17,18,20,21,23,29,30,31};kb={14,15,16,18,22,24,26,28,29,30,31}
records=[];labels=[]
for source,items,approved,kneeapproved in [('provider_motion_selection',first,a,ka),('pose_hypothesis_selection',ranked,b,kb)]:
 for i,r in enumerate(items):
  decision=dict(id=r['id'],person=r['person'],path=r['path'],selection_source=source,review_frames=r['review_frames'],evidence=r['sheet'],direct_sampled_raw_frame_review=True,continuous_video_review=False,expert_validated=False,model_or_rule_output_used_as_target=False)
  if i not in approved:
   reason=('점프 스쿼트 변형으로 기본 맨몸 스쿼트 평가에서 제외' if source=='provider_motion_selection' and i==22 else '한발 지지·무릎 들어올리기 변형으로 기본 양발 스쿼트 오류와 구분' if source=='pose_hypothesis_selection' and i in (5,6) else '몸통·무릎이 잘리거나 저해상도·노출·깊은 전경으로 좌우 이탈을 확정하기 어려움; 성긴 표본에서 완결 반복을 확정 못한 경우 포함')
   decision.update(action='exclude_or_uncertain',reason=reason,levels=None)
  else:
   start,end=approved[i];knee=2 if i in kneeapproved else None
   decision.update(action='label_observed_controls',frames=[start,end],levels={'trunk_balance':2,'knee_inward':knee,'standing_return':None},reason='지정 구간의 하강·상승 표본에서 뚜렷한 몸통 좌우 이탈 신호 없음. 무릎을 라벨링한 경우 양 무릎에 뚜렷한 안쪽 이동이 보이지 않음. 정면 상단 높이만으로 실제 신전 각도 정상·오류를 확정하지 않음. 무릎 전방 이동·자연스러운 전경·좁은 스탠스 자체를 오류로 쓰지 않음.')
   labels.append(dict(id='priority_'+r['id'],person=r['person'],split='train',path=r['path'],view='front',frames=[start,end],fps=r['fps'],trunk_balance=2,knee_inward=knee,return_extension=None,knee_foot=None,descent_range=None,original100_levels=[None,None,None,2],source_sha256=hashlib.sha256(Path(r['path']).read_bytes()).hexdigest(),review=decision))
  records.append(decision)
result=dict(version='priority-visual-draft-v1',rubric_version='squat-core100-observation-v3',expert_validated=False,frozen_before_training=True,labels=labels,decisions=records,new_confirmed_errors={'return_extension':0,'knee_inward':0,'trunk_balance':0},purpose='새 오류를 억지로 만들지 않고 오류 후보였던 정상·미관측 사례를 대조군으로 보강. 보조 무릎 간격 라벨은 원래 무릎·발 합성 30점에 자동 사용하지 않음.')
(O/'labels.frozen.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(O/'labels.sha256').write_text(hashlib.sha256((O/'labels.frozen.json').read_bytes()).hexdigest());print(len(records),len(labels),sum(x['knee_inward'] is not None for x in labels))
