"""Render recorded training metrics and a real held-out video prediction."""
from pathlib import Path
import html,json
import numpy as np
from inference import PostureErrorClassifier
BASE=Path(__file__).resolve().parent

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads((BASE/'outputs/metrics.json').read_text());history=json.loads((BASE/'outputs/history.json').read_text());splits=json.loads((BASE/'outputs/splits.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,rows in history.items():
        axes[0].plot([r['epoch'] for r in rows],[r['loss'] for r in rows],label=name)
        axes[1].plot([r['epoch'] for r in rows],[r['validation_macro_f1'] for r in rows],label=name)
    axes[0].set_title('Training loss');axes[1].set_title('Validation macro F1')
    for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(BASE/'outputs/training_curves.png',dpi=130);plt.close(fig)
    classifier=PostureErrorClassifier();ids=splits['test']['video_ids'];selected={r['video_id']:r for r in json.loads((BASE/'data/selected.json').read_text())}
    cases=[]
    # One held-out example per available target combination; chosen by source labels, never by prediction correctness.
    for target in [(0,0),(1,0),(0,1),(1,1)]:
        vid=next((v for v in ids if tuple(selected[v]['target'])==target),None)
        if not vid:continue
        cache=np.load(BASE/'cache'/f'{vid}.npz');prediction=classifier.predict_landmarks(cache['landmarks'],float(cache['aspect']))
        cases.append(dict(video_id=vid,source_target=selected[vid]['target'],prediction=prediction,video_path=selected[vid]['video_path']))
    (BASE/'outputs/example_predictions.json').write_text(json.dumps(cases,indent=2))
    rows=''.join(f'<tr><td>{name}</td><td>{report[name]["normal_vs_selected_error"]["accuracy"]:.3f}</td><td>{report[name]["precision_macro"]:.3f}</td><td>{report[name]["recall_macro"]:.3f}</td><td>{report[name]["macro_f1"]:.3f}</td></tr>' for name in ['cnn','gru','rule_proxy','majority_train'])
    cards=''
    for case in cases:
        relative=Path(case['video_path']).relative_to(BASE.parent/'qevd-selected')
        src='/media/qevd/'+str(relative)
        p=case['prediction'];cards+=f'<article><h3>Held-out test: {case["video_id"]}</h3><video controls preload="metadata" src="{html.escape(src)}"></video><p>Source targets [shallow, back not straight]: {case["source_target"]}</p><p>Model: {html.escape(str(p["scores"]))}</p><p>Detected: {html.escape(str(p["detected_errors"]))}</p><p>{html.escape(" ".join(p["feedbacks"]))}</p></article>'
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>스쿼트 실제 모델 학습 결과</title><style>body{{font:16px system-ui;background:#f3f6fb;color:#20334e}}main{{max-width:1100px;margin:auto;padding:24px}}table{{border-collapse:collapse;width:100%;background:white}}th,td{{padding:12px;border:1px solid #ddd;text-align:left}}article{{background:white;padding:20px;border-radius:12px;margin:20px 0}}video{{width:100%;max-height:340px;background:#111}}p{{line-height:1.7;overflow-wrap:anywhere}}.notice{{padding:18px;background:#fff0ce}}img{{max-width:100%}}</style><main><h1>스쿼트 실제 모델 학습 결과</h1><p>선택 모델: {report['selected_model'].upper()} · 학습 {splits['train']['videos']} / 검증 {splits['validation']['videos']} / 원본 test {splits['test']['videos']}개 · 피험자 겹침 0</p><p class="notice">아래 점수는 학습된 모델로 각 영상 전체를 미리 계산한 실제 예측입니다. 재생 시점별 실시간 판정은 아닙니다. 스쿼트 두 오류만 평가합니다. 원본 clip 라벨 재현 성능이며 전문가 판정·실시간 ESP32-CAM 성능은 별도 검증이 필요합니다. Rule proxy의 몸통 기울기는 등 굽음과 동일하지 않습니다.</p><table><tr><th>Model</th><th>Binary accuracy</th><th>Precision macro</th><th>Recall macro</th><th>Macro F1</th></tr>{rows}</table><p>Binary accuracy: 정상 후보 대 선택 오류. 나머지: 두 오류별 다중 라벨 평균. 임곗값·모델 선택은 검증셋만 사용했습니다.</p><img src="/training_curves.png" alt="Training and validation curves"><h2>원본 test 영상의 실제 예측</h2><p>각 원본 라벨 조합에서 첫 영상을 선택했습니다. 예측 성공 여부로 예시를 고르지 않았습니다.</p>{cards}</main></html>'''
    (BASE/'outputs/report.html').write_text(page)
    # Reuse existing localhost dataset video server; retain original demo.
    web=BASE.parent/'shoulder-qevd/test_app/web'
    (web/'training_result.html').write_text(page)
    (web/'training_curves.png').write_bytes((BASE/'outputs/training_curves.png').read_bytes())
    print(BASE/'outputs/REPORT.md')

if __name__=='__main__':main()
