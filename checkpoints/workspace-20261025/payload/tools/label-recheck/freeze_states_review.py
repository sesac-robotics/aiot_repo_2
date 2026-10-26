from pathlib import Path
import json,hashlib
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-recheck-20261024';rows=json.loads((B/'data/error-enrichment-20261024/annotations/standing-states-final.frozen.json').read_text());holds={15:'motion_blur_and_standing_transition',49:'motion_blur_and_standing_transition',108:'early_descent_transition_not_clear_standing',153:'next_descent_transition_not_clear_standing',184:'relaxed_or_flexed_knee_not_clear_full_standing_calibration',185:'relaxed_or_flexed_knee_not_clear_full_standing_calibration',186:'relaxed_or_flexed_knee_not_clear_full_standing_calibration',193:'rising_transition_not_clear_standing',201:'relaxed_or_flexed_knee_ambiguous',203:'relaxed_or_flexed_knee_ambiguous',265:'relaxed_or_flexed_knee_ambiguous'}
revisions=[];clear=[]
for index,r in enumerate(rows):
 if index in holds:revisions.append(dict(id=r['id'],original_state=r['state'],original_target=r['target'],new_target=None,reason=holds[index],review_index=index,person=r['person'],sheet=str(O/f'state-audit-{index//30+1:02}.jpg')))
 else:clear.append(r)
(O/'standing-states-reviewed.frozen.json').write_text(json.dumps(clear,ensure_ascii=False,indent=2));(O/'state-review-receipt.json').write_text(json.dumps(dict(total_reviewed=len(rows),clear_retained=len(clear),ambiguous_held=len(revisions),revisions=revisions,review_method='Exact source frame visual reinspection; no score predictions displayed.',not_repetition_failure_labels=True,expert_verified=False,original_files_modified=False),ensure_ascii=False,indent=2));print(len(clear),len(revisions))
