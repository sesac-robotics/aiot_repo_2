"""직접 재검토한 결정만 반영해 새 실험 라벨과 변경 이력을 저장한다.

원본 라벨·기존 모델·팀 저장소를 덮어쓰지 않는다. 등급은 결정 파일에서만 읽는다.
"""
import copy
import hashlib
import html
import json
from collections import Counter
from pathlib import Path
from datetime import datetime

ROOT=Path(__file__).resolve().parent


def export():
    """재검토 라벨의 점수 합계·관측 여부·참여자 분할을 검증하고 저장한다.

    Returns:
        None: 라벨 JSON, 변경 이력, 보충 계획과 HTML 검토표를 저장한다.
    Raises:
        AssertionError: 직접 검토 결정 누락, 구간 오류 또는 분할 누출 발생 시.
    """
    original=json.loads((ROOT/'manifest.json').read_text());decisions=json.loads((ROOT/'direct_decisions.json').read_text());rows=[];changes=[]
    for before in original:
        n=before['review_number'];d=decisions[str(n)];r=copy.deepcopy(before);r['criterion_levels']=d.get('levels',r['criterion_levels'])
        assert all(type(v) is int and 0<=v<=2 or v is None for v in r['criterion_levels'])
        r['previous_score']=before['score'];r['original_segment']=before['segment'];r['original_note_ko']=before['note_ko']
        if 'corrected_frames' in d:r['segment']['start_frame'],r['segment']['end_frame']=d['corrected_frames']
        a,b=r['segment']['start_frame'],r['segment']['end_frame'];assert a<b
        r['segment']['evidence_times_seconds']=[round(a/r['segment']['fps'],3),round(b/r['segment']['fps'],3)]
        r['score']=None if None in r['criterion_levels'] else sum(w*l/2 for w,l in zip(r['criterion_weights'],r['criterion_levels']))
        r['audit_status']='Codex_rechecked_experimental';r['review_version']='visual-audit-v6';r['note_ko']=d['note_ko'];r['labels_descriptive']=[d['note_ko']]
        r['review_method']=d['method'];r['expert_validated']=False;r['source_aware_review']=True
        r['total_score_hold_reason']=d.get('training_hold')
        r['eligible_for_total_score_training']=r['score'] is not None and not r['total_score_hold_reason']
        r['eligible_for_experimental_training']=r['eligible_for_total_score_training']
        r['required_input_views']=['side','synchronized_front'] if r['exercise']=='squats' else ['side']
        r['training_feature_cache_invalidated']=r['segment']['start_frame']!=before['segment']['start_frame'] or r['segment']['end_frame']!=before['segment']['end_frame']
        evidence=[r['review_sheet']]
        for stem,key in [('detail','detail_reviewed'),('dense','dense_reviewed')]:
            if d.get(key):evidence.append(str(ROOT/f'{stem}_{n:03d}.jpg'))
        r['audit_evidence']=evidence;r['decision_sha256']=hashlib.sha256(json.dumps(d,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        if r['criterion_levels']!=before['criterion_levels'] or r['training_feature_cache_invalidated'] or r['total_score_hold_reason']:
            changes.append(dict(number=n,id=r['id'],old_levels=before['criterion_levels'],new_levels=r['criterion_levels'],old_score=before['score'],new_score=r['score'],old_frames=[before['segment']['start_frame'],before['segment']['end_frame']],new_frames=[a,b],training_hold=r['total_score_hold_reason'],note_ko=d['note_ko']))
        rows.append(r)
    assert len(rows)==71 and len({r['id'] for r in rows})==71
    for key in ['person','video_sha256']:
        groups={}
        for r in rows:groups.setdefault(r[key],set()).add(r['split'])
        assert all(len(v)==1 for v in groups.values())
    for ex in ['pushups','squats']:(ROOT/f'{ex}_labels.json').write_text(json.dumps([r for r in rows if r['exercise']==ex],ensure_ascii=False,indent=2))
    summary={'reviewed':len(rows),'new_video_labels':0,'expert_validated':False,'continuous_playback_reviewed':False,'source_aware':True,'all_test_material_previously_seen':True,'production_ready':False,'model_retrained':False,'exercises':{}}
    for ex in ['pushups','squats']:
        rs=[r for r in rows if r['exercise']==ex];usable=[r for r in rs if r['eligible_for_total_score_training']]
        summary['exercises'][ex]=dict(reviewed=len(rs),numeric_drafts=sum(r['score'] is not None for r in rs),training_eligible_candidates=len(usable),eligible_by_split=dict(Counter(r['split'] for r in usable)),held=len(rs)-len(usable),people=len(set(r['person'] for r in rs)),item_grades={item:dict(Counter(str(r['criterion_levels'][j]) for r in rs)) for j,item in enumerate(rs[0]['criterion_ids'])})
    summary['changed_or_held']=len(changes);summary['boundary_changes']=sum(r['training_feature_cache_invalidated'] for r in rows);summary['grade_changes']=sum(c['old_levels']!=c['new_levels'] for c in changes)
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(ROOT/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
    plan=dict(priority=['반복 경계 수정 구간의 관절 특징 재추출','관측 가능한 입력 시점에 맞는 총점 모델 소규모 재학습','정상·부분 미달·명확한 미달과 복귀 실패 사례를 여러 사람으로 보충','새 사람의 독립 평가 자료를 별도로 확보'],local_video_shortage=False,missing_types={'pushups':['상단 복귀 부분 미달·실패','정렬·하강의 중간 수준','손바닥 바닥 지지와 손잡이 지지 도메인 분리'],'squats':['서기 복귀 부분 미달·실패','5명 밖의 새로운 참여자','정면·측면 입력 모두 있는 사례']},youtube_candidates=[dict(url='https://www.youtube.com/watch?v=lnR_kb5Wjf8',publisher='FitnessFAQs',purpose='푸시업 범위·몸통 오류의 기준 비교 및 보충 후보',status='검색 확인, 영상 직접 라벨링·학습 미실시'),dict(url='https://www.youtube.com/watch?v=T0ek1FssRBU',publisher='CrossFit',purpose='스쿼트 수행 요점과 오류의 기준 비교 및 보충 후보',status='공식 설명·검색 확인, 영상 직접 라벨링·학습 미실시')],external_media_policy='영상별 이용 조건을 확인하고 실제 구간·촬영 방향을 직접 검토한 후 학습 후보 여부를 정한다. 제목만으로 정답을 부여하지 않는다.')
    (ROOT/'supplement_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2))
    body=['<!doctype html><meta charset="utf-8"><title>자세 점수 라벨 재검토</title><style>body{font:16px sans-serif;max-width:1300px;margin:32px auto;background:#16212a;color:#e5f0f4}a{color:#63d8c6}table{border-collapse:collapse;width:100%}td,th{padding:9px;border:1px solid #62717a}details{padding:10px;border:1px solid #62717a;margin:8px}img{width:100%}video{max-width:720px;width:100%}</style><h1>스쿼트·푸시업 점수 라벨 재검토</h1><p>기존 71개 총점 초안을 직접 재검토했습니다. 시간순 표본 프레임 검토이며 연속 영상 재생·전문가 검증과 구별합니다. 정상·오류 제목을 정답으로 복사하지 않습니다.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre><p>배점은 프로젝트 기준입니다. 항목 미관측 시 총점은 비우고, 반복 경계가 바뀌면 기존 관절 특징은 재사용하지 않습니다. 스쿼트 총점에는 동기화 정면·측면 입력이 필요합니다.</p>']
    for r in rows:
        body.append('<details><summary>'+html.escape(f"{r['review_number']:03d} {r['id']} | 이전 {r['previous_score']} → 초안 {r['score']} | "+('학습 후보' if r['eligible_for_total_score_training'] else '총점 학습 보류'))+'</summary><p>'+html.escape(r['note_ko'])+'</p><p>'+html.escape(str(r['criterion_levels']))+'</p>')
        for path in r['audit_evidence']:body.append('<a href="'+html.escape(Path(path).as_uri())+'">근거 확대</a><img src="'+html.escape(Path(path).as_uri())+'">')
        clip = f"#t={r['segment']['start_frame']/r['segment']['fps']},{r['segment']['end_frame']/r['segment']['fps']}"
        body.append('<video controls preload="none" src="'+html.escape(Path(r['video_path']).as_uri()+clip)+'"> </video></details>')
    (ROOT/'index.html').write_text('\n'.join(body));print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':export()
