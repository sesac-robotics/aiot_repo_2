"""Item-level visual-review overlays. Original annotations remain immutable.
This review was prompted by development failure analysis and is not blind validation.
"""
from common import *
import copy
rows=read(O/'score-inputs.frozen.json');before=copy.deepcopy(rows);changes=[]
for r in rows:
 team=r['id'].startswith(('r2d2_V23_','r2d2_V27_','r2d2_V44_')) and r['levels'][3]==1
 rehab=r['id']=='enrich_R008'
 if not (team or rehab):continue
 reason=('원본 시간순 장면을 다시 확인하면 한 팔 올림·어깨 높이 차이와 고개 기울임이 주로 보인다. 이 관찰만으로 어깨 중심과 골반 중심 사이의 몸통 좌우 기울임을 1등급으로 확정하기 어려워 해당 항목만 보류한다. 양팔 시차 및 Y/T/W 오류 라벨은 유지한다.' if team else '정면 확대 장면에서 고개 움직임은 보이나 몸통 자체의 뚜렷한 좌우 기울임은 확정하기 어렵다. 제공자의 ERR_TORSO_TO_SIDE 문자열만으로 프로젝트 몸통 점수를 확정하지 않아 이 항목을 보류한다. 원래 제공자 라벨은 수정하지 않는다.')
 item=r['items'][3]['item'];indices=[i for i,n in enumerate(r['items']) if n['item']==item]
 for i in indices:r['levels'][i]=None;r['items'][i].update(eligible=False,feature=None,holds=r['items'][i].get('holds',[])+['visual_review_trunk_vs_head_or_shoulder_motion_ambiguous'])
 r['reference_total_allowed']=False
 changes.append(dict(id=r['id'],person=r['person'],split=r['split'],criterion=item,old_grade=1,new_grade=None,reason_ko=reason,affected_item_indices=indices,evidence_sheets=([str(B/'output/r2d2-review-20261024'/f"{r['id'][5:8]}-dense-{j:02}.jpg") for j in [1,2]] if team else [str(O/'rehab-detail-02.jpg')]),review_prompted_by_development_failure_analysis=True,independent_blind_review=False))
save('score-inputs-before-visual-revision.frozen.json',before);save('score-inputs.frozen.json',rows);save('label-revisions.frozen.json',dict(revisions=changes,original_files_modified=False,uncertain_grade_not_replaced_with_normal=True,revised_validation_not_an_independent_accuracy_claim=True));print('VISUAL_ITEM_REVISIONS',len(changes),flush=True)
