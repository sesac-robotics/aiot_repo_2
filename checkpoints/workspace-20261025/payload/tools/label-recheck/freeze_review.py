"""Versioned annotation corrections; fixed source partition is never reshuffled."""
from pathlib import Path
import copy,hashlib,json,collections
B=Path('/home/jayden/SeSACPAI-Mid');O=B/'output/label-recheck-20261024';D=B/'data/r2d2-team-20261023/annotations'
def digest(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def main():
 original=json.loads((D/'labels.frozen.json').read_text());rows=copy.deepcopy(original['labels']);by={r['id']:r for r in rows}
 previous=json.loads((B/'output/error-enrichment-20261024/label-revisions.frozen.json').read_text())['revisions']
 for revision in previous:
  if not revision['id'].startswith('r2d2_'):continue
  r=by[revision['id'][5:]];r['items'][3].update(grade=None,points=None,training_mask=False)
  r['evidence_note_ko']+=' 재검토: 고개·어깨 높이 변화와 몸통 기울임을 분리하기 어려워 몸통만 보류.'
 # The W at 14s ends the preceding attempt. The next W appears at ~24s.
 ids=['V38_014000_020750','V38_020750_025750'];merged=copy.deepcopy(by[ids[0]])
 merged.update(id='V38_014000_025750',start=14.,end=25.75,complete_cycle=True,
               replaces=ids,clip='data/r2d2-team-20261023/clips/ytw/V38_014000_025750.mp4',
               evidence_note_ko='14초 W에서 올린 팔을 약 15~23초까지 유지하며 Y와 머리 위로 모이는 형태가 혼재하고 약 24초에 W로 돌아온다. 중간에 곧게 편 T는 없다. 두 개의 독립 반복으로 나누지 않고 한 구간으로 합친다. Y 등급은 혼재해 보류하고, T 누락과 W 도달은 각각 관찰한다.')
 for item,g in zip(merged['items'],[None,0,2,2]):item.update(grade=g,points=None if g is None else item['weight']*g/2,training_mask=g is not None)
 rows=[r for r in rows if r['id'] not in ids]+[merged];rows.sort(key=lambda r:(r['source_id'],r['start']))
 for r in rows:
  grades=[i['grade'] for i in r['items']];r['observed_max']=sum(i['weight'] for i in r['items'] if i['grade'] is not None);r['observed_points']=sum(i['points'] for i in r['items'] if i['points'] is not None)
  r['total_score']=r['observed_points'] if r['observed_max']==100 and r['complete_cycle'] else None
  r['quality']='clear_fault' if 0 in grades else 'partial_or_fault' if 1 in grades else 'insufficient_observation' if None in grades or not r['complete_cycle'] else 'observed_criteria_met'
  r['annotation_status']='sampled_second_visual_review_non_expert';r['expert_verified']=False
 atlas=json.loads((O/'atlas-manifest.json').read_text());sheets=sorted({r['sheet'] for r in atlas})
 receipt=dict(original_sources_reviewed=57,original_segments_sampled=281,frames_per_segment=9,
              review_method='Prior grades and predictions hidden in new contact sheets; same reviewer, not independent or blind.',
              limitations=['Nine sampled 4fps frames per original segment; not continuous playback of every frame.','No expert ground truth certification.'],
              sheets=[dict(path=p,sha256=digest(Path(p))) for p in sheets],
              detailed_confirmation_sheets=[str(B/'output/r2d2-review-20261024'/n) for n in ['V02-dense-01.jpg','V18-dense-01.jpg','V18-dense-02.jpg','V36-dense-01.jpg','V38-dense-02.jpg','V38-dense-03.jpg','V27-dense-02.jpg']],
              findings=[dict(action='merge_attempt_boundary_and_hold_ambiguous_Y_grade',replaces=ids,replacement=merged['id'],source_id='V38',split='train',test_labels_changed=False),
                        dict(action='suspected_shoulder_lower_return_issue_dismissed_after_dense_review',id='V27_020000_022150',grade_retained=2,reason='At 21.75–22.0s both elbows have returned to shoulder level.'),
                        dict(action='suspected_multiple_pushups_dismissed_after_dense_review',id='V36_004250_009250',reason='4fps chronological frames show one descent, prolonged bottom hold, then return.')],
              preserved_prior_item_holds=previous,original_labels_sha256=digest(D/'labels.frozen.json'),original_files_modified=False)
 (O/'team-labels-reviewed.frozen.json').write_text(json.dumps(dict(version='r2d2-second-visual-review-v2',labels=rows),ensure_ascii=False,indent=2))
 (O/'visual-review-receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
 print('REVIEWED_LABELS',len(rows),'original',len(original['labels']),'sheets',len(sheets))
if __name__=='__main__':main()
