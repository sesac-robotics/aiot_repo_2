"""라벨 재검토와 총점 회귀 실험 결과를 읽기 쉬운 검토표로 묶는다."""
import json
import html
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parent


def main():
    """현재 결과와 다음 보충 과제를 저장하고 HTML 앞에 요약을 넣는다.

    Returns:
        None: summary.json, index.html, 상위 progress.json을 갱신한다.
    Raises:
        FileNotFoundError: 학습 결과가 아직 없을 경우 발생한다.
    """
    report=json.loads((ROOT/'training_report.json').read_text());summary=json.loads((ROOT/'summary.json').read_text());summary['model_retrained']=True;summary['new_model_version']=report['version'];summary['active_web_model_changed']=False
    (ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    text=['<section><h2>재검토한 점수로 총점을 직접 예측한 실험</h2><p>학습 정답은 위에서 직접 재검토한 프로젝트 점수 초안입니다. 모델은 총점을 직접 예측하며 추론 때 관절각 규칙으로 점수를 대체하지 않습니다. 기존 웹의 활성 모델은 바꾸지 않았습니다.</p><table><tr><th>운동·입력</th><th>학습 / 검증 반복 수</th><th>사람별 교차검증 평균 오차</th><th>같은 교차검증 중앙값 기준</th><th>기존 검증 오차</th></tr>']
    for ex,r in report['results'].items():
        name='푸시업·측면' if ex=='pushups' else '스쿼트·정면+측면'
        text.append(f"<tr><td>{name}</td><td>{r['train_samples']} / {r['validation_samples']}</td><td>{r['participant_CV_MAE_points']:.1f}점</td><td>{r['participant_CV_baseline_MAE_points']:.1f}점</td><td>{r['validation_MAE_points']:.1f}점</td></tr>")
    text.append('</table><p>오차는 AI가 검토한 초안과의 차이입니다. 실제 전문가 정확도·정답률이 아닙니다. 검증 사람을 반복 사용했으므로 독립 최종 시험으로 해석하지 않습니다. 소규모 학습 곡선만으로 충분한 학습량·포화점을 확정하지 않습니다.</p><h2>다음 보충 순서</h2><ol><li>기존 자료에서 일반 푸시업의 복귀 부분 미달·실패와 스쿼트 복귀 실패를 선별합니다. 무릎 푸시업은 별도 변형으로 표시합니다.</li><li>정상·부분 미달·명확한 미달을 여러 사람으로 보충합니다. 반복 수만 늘려 참여자 부족을 가리지 않습니다.</li><li>스쿼트의 정면·측면 총점과 측면 단독으로 가능한 항목 점수를 분리합니다.</li><li>새로운 사람의 독립 평가 자료를 확보한 뒤 점수 공개 여부를 판단합니다.</li></ol><p>보충 후보: <a href="https://www.youtube.com/watch?v=lnR_kb5Wjf8">FitnessFAQs 푸시업 오류</a> · <a href="https://www.youtube.com/watch?v=T0ek1FssRBU">CrossFit 스쿼트 수행 요점·오류</a>. 후보를 찾은 상태이며 아직 해당 영상을 내려받아 라벨링하거나 학습하지 않았습니다.</p></section>')
    page=(ROOT/'index.html').read_text();needle='<p>배점은 프로젝트 기준입니다.';start=page.index('<pre>');end=page.index('</pre>',start)+len('</pre>');page=page[:start]+'<pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'+page[end:];page=page.replace(needle,'\n'.join(text)+needle,1);(ROOT/'index.html').write_text(page)
    p=ROOT.parent/'progress.json';old=json.loads(p.read_text());(ROOT/'previous_progress.json').write_text(json.dumps(old,ensure_ascii=False,indent=2));old.update(stage='audited_score_drafts_and_direct_total_regression_experiment_complete_scoring_held',time=datetime.now(ZoneInfo('Asia/Seoul')).isoformat(),audit_report=str(ROOT/'index.html'),reviewed_existing_score_drafts=71,total_score_candidates=59,audit_holds=12,model_experiment=report['version'],new_model_production_ready=False,training_running=False,active_web_model_changed=False);p.write_text(json.dumps(old,ensure_ascii=False,indent=2))
    print('검토표·진행 기록 갱신 완료')

if __name__=='__main__':main()
