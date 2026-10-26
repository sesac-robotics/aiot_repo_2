"""직접 항목 라벨과 모델 검증 결과를 확인 가능한 보고서로 저장한다."""
import collections
import datetime
import html
import json
from pathlib import Path
OUT=Path(__file__).resolve().parent


def build():
    """직접 검토 수·등급별 오차·남은 작업을 HTML과 진행 JSON으로 저장한다.

    Returns:
        None: index.html과 상위 progress.json을 저장한다.
    Raises:
        FileNotFoundError: 학습 보고서 또는 라벨 파일이 없는 경우.
    """
    r=json.loads((OUT/'image_training_report.json').read_text());previous=json.loads((OUT/'training_report.json').read_text());labels=json.loads((OUT/'labels.json').read_text());features=json.loads((OUT/'feature_report.json').read_text());blocks=[]
    for ex,data in r['results'].items():
        blocks.append('<h2>'+('푸시업' if ex=='pushups' else '스쿼트')+'</h2>')
        for item,value in data['items'].items():
            old=previous['results'][ex]['items'][item];rows=''.join(f"<tr><td>{g}</td><td>{v['samples']}</td><td>{v['MAE']:.3f}</td></tr>" for g,v in value['validation_per_grade'].items())
            blocks.append(f"<h3>{html.escape(item)}</h3><p>동일 검증 자료에서 3D 중심 모델 평균 오차 {old['validation_MAE_grade']:.3f} → 2D·3D 선택 모델 {value['validation_MAE_grade']:.3f} (0~2 등급 단위).</p><table><tr><th>초안 등급</th><th>검증 개수</th><th>평균 등급 차이</th></tr>{rows}</table><p><b>채점 제공 보류:</b> {html.escape(' · '.join(value['hold_reasons']))}</p>")
    evidence=''.join(f"<tr><td>{r['exercise']}</td><td>{r['split']}</td><td><a href='{Path(r['evidence_image']).name}'>{html.escape(r['id'])}</a></td><td>{html.escape(str(r['criterion_levels']))}</td><td>{html.escape(r['note_ko'])}</td></tr>" for r in labels)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>직접 라벨 보충과 항목 모델 개선</title><style>body{font-family:system-ui,sans-serif;max-width:1100px;margin:35px auto;padding:20px;line-height:1.75;color:#183044;background:#f5f8fa}table{border-collapse:collapse}td,th{border:1px solid #bccbd6;padding:8px;text-align:left;vertical-align:top}a{color:#00639b}.note{padding:18px;background:#fff0d3;border-left:5px solid #d28b22}details{margin:25px 0}</style><h1>직접 라벨 보충과 항목 모델 개선</h1><p>2026-10-21 · 기존 소스를 수정하지 않고 새 파일로 실험을 연결했습니다.</p><div class="note"><b>최종 자세 점수 모델이 완성된 상태는 아닙니다.</b> 정상 사례에 치우친 평균 오차를 오류 등급별로 다시 점검해 채점 제공을 보류했습니다. 웹에서는 관절 영상·보류 근거와 선택한 경우의 진단용 모델 등급을 확인할 수 있습니다.</div><h2>이번에 완료한 일</h2><ol><li>QEVD 학습·검증 후보 46개를 직접 표본 장면으로 검토했습니다. 자동 스크립트가 점수 라벨을 생성하지 않았습니다.</li><li>관측 가능한 항목만 0·1·2 등급으로 기록했습니다. 정면 영상에서 몸통 정렬 등이 보이지 않는 경우 null로 남겼습니다. 따라서 이번 보충분에는 완성된 100점 총점 라벨이 없습니다.</li><li>45개 관절 추출 대상 중 40개가 기존 관측률 기준을 통과했습니다. 관측 부족 5개와 모든 항목이 가려진 1개를 억지로 학습에 넣지 않았습니다.</li><li>기존 자료와 합쳐 학습·검증 93구간의 2D 화면 특징을 RTX Heavy로 추가 추출했습니다. 시험 분할은 이번 개선에 사용하지 않았습니다.</li><li>등급별 가중 학습과 2D·3D 특징 선택을 비교했습니다. 특징 방식은 학습 참여자 교차검증으로 선택했습니다.</li><li>기존 웹에 새 모델 화면 링크를 실행 시 추가했습니다. 원래 파일과 규칙 점수 결과는 보존했습니다.</li></ol><p><a href="http://127.0.0.1:8766/model/">로컬 모델 분석 화면</a> · <a href="labels.json">직접 항목 라벨</a> · <a href="image_training_report.json">최신 모델 검증</a> · <a href="image_gate_policy.json">검증 기준</a></p><h2>수치 해석</h2><p>아래는 Codex의 프로젝트 초안 라벨과 모델 예측의 차이입니다. 전문가 정확도·임상 점수가 아닙니다. 기존 검증 자료를 반복 조회한 개발 실험이며, 독립 최종 시험이 아닙니다. 평균 오차 감소만으로 학습이 완료됐다고 판단하지 않습니다.</p>'''+''.join(blocks)+'''<h2>왜 아직 총점을 제공하지 않나</h2><p>스쿼트 깊은 하강 검증이 1명·1표본뿐이고 푸시업 몸통 오류 표본도 작습니다. 손·발 지지는 검증 오차가 줄었지만 사람을 바꾼 교차검증에서는 단순 비교 기준보다 안정적으로 낫지 않았습니다. 상단 복귀 등은 오류 등급 자체가 부족합니다. 촬영 각도·출처와 등급이 엮이는 문제도 남아 있습니다.</p><h2>다음 순서</h2><ol><li>동일 촬영 방향에서 정상·부분 수행·명확한 이탈을 여러 참여자에게 확보해 출처·각도 편향을 줄입니다.</li><li>깊은 스쿼트와 푸시업 중간 등급·복귀 실패 사례를 추가 검토합니다. 없는 실패를 영상 종료만으로 만들어 라벨링하지 않습니다.</li><li>오류 등급별 검증과 새 독립 시험을 통과한 항목만 채점용으로 전환합니다.</li><li>플랭크·Y-T-W·숄더 프레스의 직접 라벨·신규 점수 모델 학습은 아직 완료되지 않았습니다. 대상 동작·촬영 방향 확인부터 진행해야 합니다.</li><li>자세 점수 이후 오류 분류·교정 모델과 연계합니다. 현재 작업으로 해당 모델이 완성됐다고 하지 않습니다.</li></ol><details><summary>직접 검토 46개 근거와 장면</summary><table><tr><th>종목</th><th>분할</th><th>장면</th><th>항목 등급</th><th>직접 판단 근거</th></tr>'''+evidence+'''</table></details></html>'''
    (OUT/'index.html').write_text(page)
    progress=dict(stage='direct_item_labels_and_diagnostic_web_ready_scoring_held',time=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(),additional_direct_reviews=len(labels),additional_valid_features=sum(v['valid'] for v in features),additional_full_score_labels=0,image_feature_segments=93,test_split_used=False,production_ready=False,web_url='http://127.0.0.1:8766/model/',report=str(OUT/'index.html'),remaining=['balanced_same_view_and_grade_labels','new_independent_test','plank_ytw_shoulder_direct_score_models','error_classification_and_correction'])
    (OUT.parent/'progress.json').write_text(json.dumps(progress,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
