"""저장된 다종목 실험 지표와 지원 범위를 표·화면 자료로 정리합니다. 보고서 생성이 새로운 독립 평가나 학습을 의미하지 않습니다.

Five-exercise report, including sparse-support warnings and held-out predictions.
"""

import html
import json
from collections import Counter
from pathlib import Path

import numpy as np
from suite_config import BASE, SUITE, TASKS
from suite_inference import ExerciseClassifier


def fmt(x):
    return "평가 불가" if x is None else f"{x:.3f}"


def main():
    summaries = []
    parts = []
    samples = []
    old = BASE / "outputs/metrics.json"
    if old.exists():
        squat = json.loads(old.read_text())
        selected = squat["selected_model"]
        m = squat[selected]
        summaries.append(
            dict(
                exercise="squats",
                model=selected,
                binary_accuracy=m["normal_vs_selected_error"]["accuracy"],
                macro_f1_all_targets=m["macro_f1"],
                coverage="2/2",
                scope="기존 스쿼트 1차 모델: 두 오류, 정상 후보 대 선택 오류 정확도",
            )
        )
    for exercise, config in TASKS.items():
        out = SUITE / exercise / "outputs"
        if not (out / "metrics.json").exists():
            continue
        report = json.loads((out / "metrics.json").read_text())
        splits = json.loads((out / "splits.json").read_text())
        name = report["selected_model"]
        m = report["models"][name]
        summaries.append(
            dict(
                exercise=exercise,
                model=name,
                binary_accuracy=m["selected_error_vs_other"]["accuracy"],
                macro_f1_all_targets=m["macro_f1_all_targets"],
                macro_f1_supported=m["macro_f1_supported"],
                coverage=f'{m["evaluated_targets"]}/{m["total_targets"]}',
                test_videos=splits["test"]["videos"],
                test_selected_error_positives=splits["test"][
                    "any_selected_error_count"
                ],
                scope=config["scope"],
            )
        )
        cards = ""
        records = {
            r["video_id"]: r
            for r in json.loads(
                (SUITE / exercise / "selected.json").read_text()
            )
        }
        ids = splits["test"]["video_ids"]
        model = ExerciseClassifier(exercise)
        # First source-positive and first other-source example, not selected by model success.
        chosen = []
        for positive in [True, False]:
            vid = next(
                (v for v in ids if bool(any(records[v]["target"])) == positive),
                None,
            )
            if vid:
                chosen.append(vid)
        for vid in chosen:
            r = records[vid]
            with np.load(SUITE / "cache" / f"{vid}.npz") as cache:
                prediction = model.predict_landmarks(
                    cache["landmarks"], float(cache["aspect"])
                )
            samples.append(
                dict(
                    exercise=exercise,
                    video_id=vid,
                    source_target=r["target"],
                    prediction=prediction,
                )
            )
            root = BASE.parent / (
                "shoulder-qevd"
                if r["dataset"] == "shoulder"
                else "qevd-selected"
            )
            src = (
                "/media/"
                + r["dataset"]
                + "/"
                + str(Path(r["video_path"]).relative_to(root))
            )
            expected = [
                t for t, value in zip(config["targets"], r["target"]) if value
            ]
            cards += f'<article><h4>원본 test {html.escape(vid)}</h4><video controls preload="metadata" src="{html.escape(src)}"></video><p>원본 대상 라벨: {html.escape(str(expected))}</p><p>실제 모델 예측: {html.escape(str(prediction["detected_errors"]))}</p><p>{html.escape(" ".join(prediction["feedbacks"]))}</p><details><summary>확률 및 상태</summary><pre>{html.escape(json.dumps(prediction,indent=2,ensure_ascii=False))}</pre></details></article>'
        rows = "".join(
            f'<tr><td>{html.escape(target)}</td><td>{values["train_positive_support"]}</td><td>{values["validation_positive_support"]}</td><td>{values["positive_support"]}</td><td>{fmt(values["precision"])}</td><td>{fmt(values["recall"])}</td><td>{fmt(values["f1"])}</td></tr>'
            for target, values in m["per_target"].items()
        )
        models = "".join(
            f'<tr><td>{n}</td><td>{v["selected_error_vs_other"]["accuracy"]:.3f}</td><td>{fmt(v["macro_f1_all_targets"])}</td><td>{fmt(v["macro_f1_supported"])}</td><td>{v["evaluated_targets"]}/{v["total_targets"]}</td></tr>'
            for n, v in {
                **report["models"],
                "majority_train": report["majority_baseline"],
            }.items()
        )
        parts.append(
            f'<section id="{exercise}"><h2>{config["name"]}</h2><p>{html.escape(config["scope"])}</p><p>학습/검증/test: {report["split_sizes"]}. 피험자 겹침 0. test 중 선택 오류 양성 {splits["test"]["any_selected_error_count"]}개.</p><table><tr><th>모델</th><th>선택 오류/기타 정확도</th><th>전체 오류 Macro F1</th><th>평가 가능 오류 Macro F1</th><th>오류 평가 범위</th></tr>{models}</table><h3>선택 모델 {name.upper()}의 유형별 평가</h3><table><tr><th>유형</th><th>학습 양성</th><th>검증 양성</th><th>test 양성</th><th>Precision</th><th>Recall</th><th>F1</th></tr>{rows}</table>{cards}</section>'
        )
    SUITE.mkdir(exist_ok=True)
    (SUITE / "summary.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2)
    )
    (SUITE / "example_predictions.json").write_text(
        json.dumps(samples, ensure_ascii=False, indent=2)
    )
    md = [
        "# 5종목 실제 학습 결과",
        "",
        "| 종목 | 선택 모델 | 선택 오류/기타 정확도 | 전체 오류 Macro F1 | 오류 평가 범위 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in summaries:
        md.append(
            f'| {r["exercise"]} | {r["model"]} | {r["binary_accuracy"]:.3f} | {fmt(r["macro_f1_all_targets"])} | {r["coverage"]} |'
        )
    md += [
        "",
        "종목별 대상 오류와 평가 분포가 다르므로 숫자를 종목 간 우열로 비교하면 안 됩니다. 스쿼트 외 선택 오류 미검출은 정상 판정이 아닙니다. 어깨는 미검수 초안이며 테스트 오류 양성이 극소수입니다. YTW는 몸통 변형 미검수 상태의 탐색 실험입니다.",
        "",
        *[
            "## "
            + TASKS[k]["name"]
            + "\n\n"
            + (SUITE / k / "outputs/REPORT.md").read_text()
            for k in TASKS
            if (SUITE / k / "outputs/REPORT.md").exists()
        ],
    ]
    (SUITE / "REPORT.md").write_text("\n".join(md) + "\n")
    pending = [
        TASKS[k]["name"]
        for k in TASKS
        if not (SUITE / k / "outputs/metrics.json").exists()
    ]
    progress = (
        f"<p>학습 완료 {len(summaries)}/5종목. "
        + (
            "아직 처리 중: " + html.escape(", ".join(pending))
            if pending
            else "나머지 4종목 모델 저장 완료."
        )
        + "</p>"
    )
    names = {"squats": "스쿼트", **{k: v["name"] for k, v in TASKS.items()}}
    overview_rows = "".join(
        f'<tr><td>{names[r["exercise"]]}</td><td>{r["model"].upper()}</td><td>{fmt(r["macro_f1_all_targets"])}</td><td>{r.get("test_videos",60)}</td><td>{r["coverage"]}</td></tr>'
        for r in summaries
    )
    progress += (
        "<h2>모델 평가 요약</h2><table><tr><th>종목</th><th>모델</th><th>Macro F1</th><th>test</th><th>범위</th></tr>"
        + overview_rows
        + "</table><p>범위는 전체 대상 중 독립 테스트로 평가 가능한 오류 수입니다. 종목별 대상이 달라 점수를 단순 우열로 비교하면 안 됩니다.</p>"
    )

    page = (
        """<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>5종목 실제 모델 학습 결과</title><style>body{font:15px system-ui;background:#f3f6fb;color:#20334e;margin:0}main{max-width:1200px;margin:auto;padding:24px}p{line-height:1.8;overflow-wrap:anywhere}.notice{background:#fff0ce;padding:18px;border-radius:12px}table{border-collapse:collapse;width:100%;background:white;font-size:14px;display:block;overflow:auto}td,th{padding:10px;border:1px solid #ddd;white-space:nowrap}article{padding:20px;background:white;border:1px solid #dae2ee;margin:16px 0;border-radius:12px}video{width:100%;max-height:340px;background:#111}section{margin-top:40px}pre{overflow:auto}nav{display:flex;gap:20px;flex-wrap:wrap}a{color:#245bc1}</style><main><h1>5종목 실제 모델 학습 결과</h1>"""
        + progress
        + """<p>스쿼트 기존 모델과 종목별 CNN/GRU. 피험자를 분리하고 원본 test를 유지했습니다.</p><p class="notice">아래 예측은 학습된 모델로 영상 전체를 미리 계산한 결과입니다. 재생 시점별 실시간 판정이 아닙니다. 선택 오류가 검출되지 않아도 전체 자세가 정상이라는 뜻은 아닙니다. 어깨는 초안 100개 중 가시성 기준을 통과한 74개를 사용한 작은 실험이며, YTW는 몸통 변형 검수가 남아 있습니다. 지지가 없는 오류의 F1은 평가 불가로 표시합니다.</p><nav>"""
        + "".join(f'<a href="#{k}">{v["name"]}</a>' for k, v in TASKS.items())
        + '<a href="/training_result.html">기존 스쿼트 결과</a></nav>'
        + "".join(parts)
        + "</main></html>"
    )
    (SUITE / "report.html").write_text(page)
    (
        BASE.parent / "shoulder-qevd/test_app/web/all_training_results.html"
    ).write_text(page)
    print("Report: " + str(SUITE / "REPORT.md"))


if __name__ == "__main__":
    main()
