"""Freeze reviewed decisions; verify protected splits and the unchanged comparison."""
import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

B=Path(__file__).resolve().parents[2]
R=B/'output/squat-pose-alignment-v5-20261024'
P=B/'output/squat-fixed-reference-combined-v3-20261024'
sys.path.insert(0,str(B/'tools/error-enrichment'))
from split_guard import POLICY,assert_training_allowed

def read(p):return json.loads(Path(p).read_text())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,value):
    p=R/name;assert not p.exists(),p
    p.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    sources=read(R/'remaining-Train/intake.frozen.json');assert len(sources)==9
    assert_training_allowed(sources)
    reasons={
      '005':'영상이이미굽힌상태에서시작. 상단에서도무릎/골반굽힘이남는반복의심을전체5Hz추가검토했으나운동전기립기준이원본에없음.',
      '007':'여러상단에서다리를펴고서있는표본이있음. 하강몸통전경과모자이크를복귀실패로복사안함.',
      '008':'2/6/10/13초등여러상단기립표본. 얕은깊이/하강자세와복귀목표를분리,새오류확정없음.',
      '009':'초기부터몸통전경이있고후반다리펴는상단이다수. 몸통정렬/깊이태그를서기높이실패로삼지않음.',
      '016':'초기몸통/무릎자세와후반상단기립을분리. 영상표본만으로미세높이결손확정하지않음.',
      '018':'반복상단에서기립표본이보임. 운동깊이/발동작과복귀실패를혼용하지않음.',
      '019':'초기쪼그린상태뒤여러기립상단. 마지막종료와바닥접촉장면을복귀실패로라벨링하지않음.',
      '023':'1/3/8/10/12초등기립상단표본. 프레임간하강중인모습을미세복귀실패로추정하지않음.',
      '026':'2/4/7/9/11초등상단기립표본. 모자이크/하강몸통움직임은실패확정근거아님.'}
    decisions=[]
    test=read(P/'test-inputs.frozen.json')
    test_hashes={x for row in test for x in row['sources']}
    assert not test_hashes & {x['source_sha256'] for x in sources}
    for row in sources:
        assert sha(row['path'])==row['source_sha256']
        screen=R/'remaining-Train'/(row['id']+'-screen.jpg');num=row['id'].split('_')[1]
        decisions.append(dict(row,screen=str(screen),screen_sha256=sha(screen),screening_hz=1,
            decision='held_partial_return_candidate_missing_initial_ready' if num=='005' else 'no_failure_confirmed_in_screen',
            training_grade=None,proposed_grade=1 if num=='005' else None,
            reason_ko=reasons[num],expert_verified=False,new_training_used=False))
    save('review-decisions.frozen.json',decisions)
    src=next(x for x in sources if '005' in x['id'])
    pages=read(R/'remaining-Train/005-dense-manifest.frozen.json')
    assert len(pages)==5 and sum(len(p['frames']) for p in pages)==154
    save('partial-return-draft.frozen.json',dict(id=src['id']+'_upper_turn_draft',
        source_sha256=src['source_sha256'],person=src['person'],split='train',
        observed_behavior='repeated_upper_turn_with_residual_knee_and_hip_flexion',
        upper_turn_windows_seconds=[[2.2,2.6],[4.0,4.4],[5.8,6.2],[9.0,9.4]],
        proposed_grade=1,training_grade=None,
        criterion='standing_height_return',expert_verified=False,
        whole_source_screened_at_5Hz=True,all_frames_reviewed=False,
        evidence=[dict(path=p['path'],sha256=sha(p['path']),frames=p['frames']) for p in pages],
        hold_reasons=['no_pre_exercise_standing_reference_in_source',
            'cannot_use_later_or_other_recording_ready_as_earlier_reference',
            'AI_assisted_visual_grade_proposal_requires_target_adjudication'],
        fitted=False,provider_bad_not_copied_to_target=True))
    # Verify the same 163 paired target decisions and their source roles.
    for name,digest in read(P/'artifact-hashes.frozen.json').items():assert sha(P/name)==digest,name
    assert sha(P/'new-head.joblib')==read(R/'protocol.frozen.json')['model_sha256']
    paired=read(P/'comparison.frozen.json')['paired_predictions']
    assert len(paired)==len({(x['id'],x['criterion']) for x in paired})==163
    assert sha(POLICY)=='7aded6fa543285e7e8926b690317c2cc53e1bc96c7f9ca1c181f3077be23f809'
    lp=B/'output/label-recheck-20261024/team-labels-reviewed.frozen.json'
    assert sha(lp)=='fa4e5913e3b48790562702e3ff8751aee159f1435ae4b6e633b3e7a27e678667'
    policy=read(POLICY);cells=collections.Counter()
    for row in read(lp)['labels']:
        grades=[i['grade'] if i['training_mask'] else None for i in row['items']]
        quality='error' if any(g is not None and g<2 for g in grades) else 'normal' if len(grades)==4 and all(g==2 for g in grades) else 'unknown'
        if not row['quality_training_candidate']:quality='unknown'
        cells[row['person'],row['exercise'],quality,policy['assignments'][row['source_id']]]+=1
    audit=read(P/'person-quality-split-audit.json')
    assert all(cells[a['person'],a['exercise'],a['quality'],a['split']]==a['count'] for a in audit['cells'])
    save('verification.frozen.json',dict(parent_models_labels_inputs_and_comparison_hashes_verified=True,
        original_pose_pixel_dimensions_and_frame_mapping_verified=True,new_Train_sources_screened=9,
        Train_source_Test_hash_overlap=0,whole5Hz_sources=1,whole5Hz_frames=154,
        proposed_partial_drafts=1,new_eligible_error_training_labels=0,new_fit_rows=0,
        person_quality_Train_Test_cells_recomputed=True,original_roles_preserved=True,empty_cells=37,
        exact80_20_achieved=False,same_Test_rows=51,same_AI_Rule_paired_items=163,
        same_comparison_preserved_not_new_accuracy_measurement=True,
        independent_final_Test_opened=False,production_changed=False))
    report='''스쿼트 잔여 오류 원본·좌표 검토 / v5

성능 상태: 기존 개선 후보의 노출 개발 AI5/6·정상오판0/13, 같은 입력 Rule1/6·0/13 유지. 이번 단계는 원본/라벨 감사이며 모델이나 문턱을 변경하지 않았다. 5/6은 반복노출한 한 사람의 개발 결과다.
놓친r09와가까운정상r05/문턱충돌정상r12의 초기 및 실제 모델사용복귀3프레임을원본과같은좌표변환으로겹쳐직접검토했다. 선택된r09에서다른사람/배경을따르는뚜렷한오연결은없었다. r05도유사몸통전경과높이를보여정상/부분1점경계의추가정답검토가필요하다. r12에는몸방향변화도보인다. 관절정답이나기존등급을모델예측에맞춰수정하지않았고6오류분모도보존했다.
기존MultiView Train 측면bad 중 앞선6개를제외한나머지9개전부를1Hz시간순추가선별했다. 이중005는원본전체154표본/5Hz를검토해상단에도무릎·골반굽힘이남는부분복귀의심구간을1점초안으로기록했다. 그러나원본이이미굽힌상태에서시작해운동전기립기준이없으므로training_grade=null,현재초기기준모델학습은보류다. 나머지8개는선별에서새실패확정없음이며정상점수정답으로자동편입하지않았다. bad제공자태그를복귀실패정답으로복사하지않았다.
새학습가능실패0/실제fit0/새심한0점0. 기존모델·라벨·입력·비교전체해시와팀원본별정책/라벨해시검증. 사람별정상·오류분할80칸을재집계해기존배정/37빈칸그대로확인. 정확80/20완료아님. 새9원본과진단Test중복0.
동일Test51행163대응항목은변경없이유지한다. 모델불변으로재학습/재선택/불필요한동일추론반복은하지않았다. 스쿼트복귀Test는정상6/오류0,AI오판0/6·Rule2/6으로독립오류검출률은아직검증불가. 최종Test미개봉/미배포.
다음필요작업은r05/r09의목표경계합의,초기기립없는완결부분복귀사례를사용할수있는별도특징계약검토와Train상태·품질라벨분리다. 현재초안을그대로고정초기길이모델에넣지않는다.
근거: visual-pose-review.frozen.json, pose-evidence.frozen.json, review-decisions.frozen.json, partial-return-draft.frozen.json, verification.frozen.json.
'''
    p=R/'RESULTS.txt';assert not p.exists();p.write_text(report)
    s=B/'project-coordination/status/model-evaluation.json';status=read(s)
    status['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    status['state']='checkpoint_saved_goal_unmet'
    status['current_work']['latest_pose_audit']=str(R.relative_to(B))
    status['current_work']['additional_multiview_Train_sources_screened']=9
    status['current_work']['new_partial_draft_missing_initial_ready']=1
    status['current_work']['new_eligible_training_error_labels_this_checkpoint']=0
    status['evidence'].extend(str((R/n).relative_to(B)) for n in ['RESULTS.txt','verification.frozen.json','visual-pose-review.frozen.json','partial-return-draft.frozen.json'])
    status['next_action']='残余r09/r05목표경계검토/초기기립없는실제부분복귀초안의별도특징계약정리. 기존분할과최종Test보호,숄더150병합결과공유.'
    s.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(read(R/'verification.frozen.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
