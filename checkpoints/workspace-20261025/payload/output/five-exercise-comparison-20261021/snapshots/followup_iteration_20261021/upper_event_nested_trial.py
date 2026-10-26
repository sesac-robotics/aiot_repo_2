"""상지 사건 확률의 판정 경계를 학습 참여자 안에서만 선택하는 실험이다.

앞선 개발 실패를 본 뒤 설계한 후향 개발 실험이다. 독립 정확도로
표현하지 않으며 새 예약 시험 자료를 읽거나 학습하지 않는다.
"""

import pathlib
import html

import joblib
import numpy as np

import audit
from upper_support_event_model import event_features, fit
from plank_partial_item_trial import load_rows
from resources import check_resources

THRESHOLDS = (0.25, 0.5, 0.75)


def choose_threshold(predictions: list[dict]) -> float:
    """학습 내부의 참여자 제외 예측으로 등급별 오류 비중을 같게 맞춘다.

    Args:
        predictions: 해당 학습 폴드 안 target 0·1과 probability 0~1 목록.

    Returns:
        float: 0.25·0.5·0.75 중 클래스 평균 오류가 작은 경계.
        동률이면 큰 경계를 택한다. 외부 검증 라벨을 받지 않는다.

    Raises:
        ValueError: 빈 예측 또는 범위 밖 확률·사건 값.
    """
    if not predictions or any(
        r["target"] not in (0, 1) or not 0 <= r["probability"] <= 1
        for r in predictions
    ):
        raise ValueError("Valid internal event predictions required")

    def loss(t):
        return float(
            np.mean(
                [
                    np.mean(
                        [
                            (r["probability"] >= t) != bool(r["target"])
                            for r in predictions
                            if r["target"] == c
                        ]
                    )
                    for c in {r["target"] for r in predictions}
                ]
            )
        )

    return min(THRESHOLDS, key=lambda t: (loss(t), -t))


def group_predictions(rows: list[dict]) -> list[dict]:
    """학습 자료 내부에서 참여자·같은 원본을 통째로 제외해 예측한다.

    Args:
        rows: x 특징·person·source·target·id가 있는 학습 자료 목록.

    Returns:
        list[dict]: 입력 참여자와 원본을 학습에서 제외한 확률 예측.

    Raises:
        ValueError: 학습 폴드가 비거나 원본·참여자 누출이 있는 경우.
    """
    result = []
    for held in sorted({r["person"] for r in rows}):
        testing = [r for r in rows if r["person"] == held]
        hashes = {r["source"] for r in testing}
        training = [
            r for r in rows if r["person"] != held and r["source"] not in hashes
        ]
        if not training:
            raise ValueError("Empty inner training fold")
        model = fit(training)
        classes = list(model.classes_)
        for row in testing:
            p = (
                float(model.predict_proba([row["x"]])[0][classes.index(1)])
                if 1 in classes
                else 0.0
            )
            result.append(
                dict(id=row["id"], target=row["target"], probability=p)
            )
    return result


def run() -> dict:
    """고정 사건 모델의 같은 학습 자료에서 중첩 참여자 제외 진단을 한다.

    Returns:
        dict: 외부 폴드별 내부 선택 경계·확률·혼동 결과와 별도 모델 경로.
        외부 제외 라벨은 경계·결측 대체값·모델 학습에 사용하지 않는다.

    Raises:
        OSError: 관절·모델·보고서 접근 실패.
        ValueError: 초안·특징·참여자 연결 오류.
        RuntimeError: 자원 보호 실패.
    """
    baseline = audit.read_json(audit.OUT / "upper_support_event_results.json")
    targets = {r["id"]: r for r in baseline["records"]}
    sequences = {
        r["id"]: r["seq"] for r in load_rows(True) if r["split"] == "train"
    }
    for entry in audit.read_json(
        audit.OUT / "qevd_upper_event_extraction.json"
    )["records"]:
        with np.load(entry["extraction"]["cache"]) as data:
            sequences[entry["id"]] = data["image_xy"].copy()
    rows = []
    for identifier, row in targets.items():
        x, _ = event_features(sequences[identifier])
        if x is None:
            raise ValueError("Event input eligibility changed")
        rows.append(dict(row, x=x))
    predictions = []
    for held in sorted({r["person"] for r in rows}):
        check_resources()
        testing = [r for r in rows if r["person"] == held]
        hashes = {r["source"] for r in testing}
        training = [
            r for r in rows if r["person"] != held and r["source"] not in hashes
        ]
        threshold = choose_threshold(group_predictions(training))
        model = fit(training)
        classes = list(model.classes_)
        for row in testing:
            p = (
                float(model.predict_proba([row["x"]])[0][classes.index(1)])
                if 1 in classes
                else 0.0
            )
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=held,
                    target=row["target"],
                    probability=p,
                    inner_selected_threshold=threshold,
                    event=p >= threshold,
                )
            )
    final_threshold = choose_threshold(group_predictions(rows))
    model = fit(rows)
    path = audit.OUT / "upper_support_event_trial/nested_threshold_model.joblib"
    joblib.dump(
        dict(
            model=model,
            threshold=final_threshold,
            train_ids=list(targets),
            expert_validated=False,
            output="visible_upper_support_change_probability_not_posture_score",
        ),
        path,
    )
    metrics = dict(
        TP=sum(r["event"] and r["target"] == 1 for r in predictions),
        FN=sum(not r["event"] and r["target"] == 1 for r in predictions),
        FP=sum(r["event"] and r["target"] == 0 for r in predictions),
        TN=sum(not r["event"] and r["target"] == 0 for r in predictions),
    )
    report = dict(
        predictions=predictions,
        metrics=metrics,
        final_training_only_threshold=final_threshold,
        threshold_candidates=THRESHOLDS,
        baseline_fixed_threshold_metrics=baseline["metrics"],
        training_records=len(rows),
        person_groups=len({r["person"] for r in rows}),
        outer_labels_used_to_select_threshold=False,
        designed_after_prior_development_diagnostic=True,
        independent_final_test=False,
        expert_validated=False,
        adopted=False,
        model_sha256=audit.digest(path),
    )
    audit.save_json(audit.OUT / "upper_event_nested_results.json", report)
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>상지 사건 중첩 진단</title><style>body{font:16px/1.6 system-ui;max-width:1000px;margin:auto;padding:24px}td,th{padding:10px;border-bottom:1px solid #aaa}</style><h1>상지 사건 판정 경계의 중첩 참여자 진단</h1><p><a href="unified_status.html">통합 현황</a> · <a href="upper_event_nested_results.json">전체 결과</a> · <a href="upper_support_event.html">첫 고정 경계 실패 포함 결과</a></p><p>첫 개발 진단을 보고 설계한 후향 실험입니다. 바깥 참여자와 원본을 통째로 제외한 뒤, 남은 학습 참여자 안에서만 0.25·0.5·0.75 경계를 선택했습니다. 바깥 정답은 경계 선택·결측 대체·학습에 쓰지 않습니다. 예약한 새 시험 출처는 이 모델에 사용하지 않았습니다.</p><table><tr><th>방식</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr>'
    ]
    for name, m in [
        ("고정 .75 AI", baseline["metrics"]["ai_event"]),
        ("중첩 경계 AI", metrics),
        ("상완 변위 규칙", baseline["metrics"]["rule_event"]),
    ]:
        parts.append(
            f'<tr><td>{name}</td><td>{m["TP"]}</td><td>{m["FN"]}</td><td>{m["FP"]}</td><td>{m["TN"]}</td></tr>'
        )
    parts.append(
        "</table><p>변경 사건 확률을 총점·정상 확률로 바꾸지 않습니다. 변경이 없다는 예측도 전신 지지 지속성 2등급을 보증하지 않습니다. 새 참여자의 오류 유형과 카메라·몸통 회전 대조 자료가 더 필요하여 공통 모델에 채택하지 않았습니다.</p></html>"
    )
    (audit.OUT / "upper_event_nested.html").write_text(
        "".join(parts), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(run()["metrics"])
