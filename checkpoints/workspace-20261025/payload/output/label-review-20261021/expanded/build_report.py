"""직접 라벨과 실험 모델의 결과를 확인 가능한 HTML 보고서로 저장한다."""
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent


def build():
    """두 운동의 저장된 실험 결과를 표·곡선·증거 링크로 출력한다.

    Returns:
        None: index.html과 learning_curve.png를 저장한다.
    Raises:
        FileNotFoundError: 학습 보고서가 없는 경우.
    """
    reports=[json.loads((OUT/f'{ex}_training_report.json').read_text()) for ex in ['pushups','squats']]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    blocks=[]
    for ax,r,name in zip(axes,reports,['푸시업','스쿼트']):
        curve=r['learning_curve'];ax.errorbar([v['train_samples'] for v in curve],[v['validation_MAE_mean'] for v in curve],yerr=[v['validation_MAE_std'] for v in curve],marker='o',capsize=4,label='Model / 5 seeds')
        ax.plot([v['train_samples'] for v in curve],[v['mean_baseline_MAE'] for v in curve],'--',label='Training mean baseline');ax.set(title=r['exercise'],xlabel='Training samples',ylabel='Validation MAE (points)');ax.legend();ax.grid(alpha=.2)
        group=json.loads((OUT/'group_validation.json').read_text())[r['exercise']]
        labels=json.loads((OUT/f"{r['exercise']}_labels.json").read_text());rows=[]
        for v in labels:
            evidence=v.get('evidence_images',[v.get('evidence_image')]);links=' '.join(f'<a href="{html.escape(str(Path(p).relative_to(OUT)))}">장면 {i+1}</a>' for i,p in enumerate(evidence))
            rows.append(f"<tr><td>{html.escape(v['id'])}</td><td>{v['split']}</td><td>{v['score'] if v['score'] is not None else '판단 보류'}</td><td>{links}</td><td>{html.escape(v['note_ko'])}</td></tr>")
        blocks.append(f"<h2>{name}</h2><p>관절 추출 통과: {sum(r['counts'].values())}개 / 학습·검증·시험: {r['counts']} / 선택한 학습 표본 {r['selected_training_samples']}개</p><p>시험 평균 절대오차 <b>{r['test_MAE_points']:.1f}점</b>, 학습 평균값만 출력한 비교 기준 {r['test_mean_baseline_MAE_points']:.1f}점.</p><p>학습 자료 안에서 참여자를 나눈 교차검증: 모델 {group['pooled_MAE']:.1f}점 / 평균값 비교 기준 {group['pooled_baseline_MAE']:.1f}점. <a href='group_validation.json'>참여자별 결과</a></p><p><a href='{r['exercise']}_training_report.json'>수치·분할·예측 상세</a> · <a href='{r['exercise']}_labels.json'>QEVD 확장 라벨</a> · <a href='{r['exercise']}_exclusions.json'>제외 근거</a></p><details><summary>라벨 근거와 장면 보기</summary><table><tr><th>반복 식별자</th><th>분할</th><th>초안 점수</th><th>증거</th><th>판단 근거</th></tr>{''.join(rows)}</table></details>")
    fig.savefig(OUT/'learning_curve.png',dpi=150);plt.close(fig)
    doc='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>자세 점수 모델 1차 검증</title><style>body{font-family:sans-serif;max-width:1100px;margin:35px auto;padding:20px;line-height:1.7;background:#f8fafc;color:#183044}table{border-collapse:collapse;font-size:14px}td,th{border:1px solid #ccd6df;padding:8px;vertical-align:top}img{max-width:100%}details{background:white;padding:15px}a{color:#005b9c}.notice{border-left:5px solid #d59015;padding:12px;background:#fff4da}</style><h1>자세 점수 모델 1차 검증</h1><p>2026-10-21 · 이 보고서는 로컬 실험 기록입니다.</p><div class="notice"><b>전문가 정확도 검증 전입니다.</b> 시험 오차는 Codex가 표본 장면에 부여한 프로젝트 초안 점수와 모델 예측의 차이입니다. 실제 운동 평가 정확도(%)가 아닙니다. 실사용 배포는 보류합니다. 두 모델 모두 전체 학습 표본을 쓴 검증 오차가 평균 점수만 출력하는 비교 기준보다 컸습니다. 시험 결과만으로 성공했다고 판단할 수 없습니다.</div><h2>무엇을 했나</h2><p>문제가 있는 기존 자동 라벨 119개를 신규 학습에서 차단했습니다. 푸시업 48파일과 스쿼트 40반복 후보에서 중복·충돌·판단 불가를 정리하고, 직접 장면을 확인해 점수 초안을 만들었습니다. 점수는 항목별 0·1·2단계에 제안 가중치를 적용한 라벨이며 전문가 정답이 아닙니다.</p><p>MediaPipe Heavy는 RTX에서 관절을 추출합니다. 별도 ExtraTrees 회귀 모델은 관절 특징과 점수 초안을 학습해 새 입력의 점수를 예측합니다. 추론 시 규칙 점수 합산으로 대체하지 않습니다.</p><h2>학습 곡선</h2><p>오차 막대는 같은 검증 집합에 대한 표본 선택·모델 시드 5회의 표준편차입니다. 독립 참여자 검증의 신뢰구간이 아닙니다. 표본 수가 작아 학습 포화나 필요한 최종 영상 수를 확정할 수 없습니다.</p><img src="learning_curve.png" alt="검증 MAE 학습 곡선">'''+''.join(blocks)+'''<p><a href="review_queue.csv">팀 검토용 CSV (AI 초안과 별도 검토 점수란)</a> · <a href="verification.json">산출물 검사 결과</a></p><h2>한계와 다음 진행 순서</h2><ol><li>프로젝트 배점의 독립 재검토와 대표 장면의 팀 검토를 통해 라벨 기준을 보정합니다. 같은 AI의 재검토만으로 객관적 정답을 확보한 것으로 보지 않습니다.</li><li>스쿼트는 학습 3명·검증 1명·시험 1명으로 작습니다. 다중 시점 라벨의 오류가 측면 입력에서는 안 보일 수도 있습니다. 오류 항목별 관측 가능성을 점검하고 참여자 단위 교차검증으로 일반화 오차를 확인합니다.</li><li>현재 시험 집합은 이미 확인했습니다. 이 결과로 모델을 조정하면 후속 시험은 새 독립 자료로 확인해야 합니다.</li><li>플랭크·Y-T-W·숄더 프레스는 기존 자동 점수를 재사용하지 않고 종목별 기준·대표 장면 검토부터 같은 절차로 진행합니다. 이 세 종목의 신규 점수 모델은 아직 학습하지 않았습니다.</li><li>점수 모델 검증이 끝난 범위부터 웹 실험 옵션으로 연결하고, 이후 유선혁님 담당 오류 분류·교정과 연계합니다.</li></ol><p>촬영 방향·속도·전신 관측을 포함한 범용 평가를 완성한 상태가 아닙니다. 기존 실행 프로그램과 저장 결과는 보존했습니다.</p></html>'''
    (OUT/'index.html').write_text(doc)


if __name__=='__main__':build()
