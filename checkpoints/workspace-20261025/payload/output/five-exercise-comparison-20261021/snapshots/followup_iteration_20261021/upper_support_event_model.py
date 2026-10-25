"""하체가 잘린 영상의 상지 지지 위치 변경을 별도 사건 AI로 학습한다.

출력은 사건 확률이며 자세 총점·정상 확률이 아니다. 변경 근거가 있으면
지속성 2등급을 배제할 수 있지만, 근거 없음으로 2등급을 부여하지 않는다.
"""

import collections
import pathlib

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

import audit
from plank_partial_item_trial import load_rows
from resources import check_resources


def event_features(sequence: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """보이는 쪽의 어깨·팔꿈치 상대 변화를 상완 길이로 정규화한다.

    Args:
        sequence: N×33×2 픽셀 좌표. 미관측은 NaN이다.

    Returns:
        tuple: 특징 벡터 또는 None, 관측률·선택 측면·상완 길이 비율 변위.
        얼굴·골반·다리는 입력하지 않는다. 최소 5표본과 어깨·팔꿈치의
        동시 관측률 80%를 요구한다. 결과에 점수 라벨은 포함하지 않는다.

    Raises:
        ValueError: 좌표 형상 또는 빈 입력 오류.
    """
    seq = np.asarray(sequence, dtype=float)
    if seq.ndim != 3 or seq.shape[1:] != (33, 2) or not len(seq):
        raise ValueError("Expected nonempty N x 33 x 2 pixels")
    choices = []
    for side, s, e in [("left", 11, 13), ("right", 12, 14)]:
        valid = np.isfinite(seq[:, [s, e]]).all((1, 2))
        length = np.linalg.norm(seq[:, e] - seq[:, s], axis=1)
        valid &= length > 1
        choices.append((float(valid.mean()), side, s, e, valid, length))
    coverage, side, s, e, valid, length = max(choices, key=lambda r: r[0])
    note = dict(
        coverage=coverage, selected_side=side, relative_displacement=None
    )
    window = max(2, len(seq) // 5)
    if (
        len(seq) < 5
        or coverage < 0.8
        or valid[:window].sum() < 2
        or valid[-window:].sum() < 2
    ):
        return None, note
    scale = float(np.median(length[valid]))
    relative = (seq[:, e] - seq[:, s]) / scale
    relative[~valid] = np.nan
    first = np.median(relative[:window][valid[:window]], axis=0)
    relative -= first
    last = np.median(relative[-window:][valid[-window:]], axis=0)
    delta = float(np.linalg.norm(last))
    note["relative_displacement"] = delta
    vector = []
    for column in relative.T:
        finite = column[np.isfinite(column)]
        vector.extend(
            np.percentile(finite, [10, 50, 90]).tolist()
            + [float(np.std(finite))]
        )
        vector.extend(
            column[np.linspace(0, len(column) - 1, 16).astype(int)].tolist()
        )
    vector.extend([delta, coverage])
    return np.asarray(vector), note


def fit(rows: list[dict]):
    """해당 학습 폴드만으로 결측 대체와 사건·참여자 가중치를 계산한다.

    Args:
        rows: x 특징·target 이진 사건·person 그룹이 있는 학습 목록.

    Returns:
        sklearn.Pipeline: 0·1 사건 분류기. predict_proba는 사건 확률이다.

    Raises:
        ValueError: 빈 학습 자료 또는 특징 차원 오류.
    """
    if not rows:
        raise ValueError("Empty event fitting cohort")
    pairs = collections.Counter((r["target"], r["person"]) for r in rows)
    groups = {
        c: len({p for k, p in pairs if k == c})
        for c in {r["target"] for r in rows}
    }
    weights = np.asarray(
        [
            1
            / (
                len(groups)
                * groups[r["target"]]
                * pairs[r["target"], r["person"]]
            )
            for r in rows
        ]
    )
    weights /= weights.mean()
    model = make_pipeline(
        SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ),
        ExtraTreesClassifier(
            n_estimators=128, min_samples_leaf=2, random_state=21021, n_jobs=1
        ),
    )
    model.fit(
        np.asarray([r["x"] for r in rows]),
        np.asarray([r["target"] for r in rows]),
        extratreesclassifier__sample_weight=weights,
    )
    return model


def run() -> dict:
    """기존 직접 초안과 새 3개 상지 사건 초안으로 참여자 제외 진단을 한다.

    Returns:
        dict: 실제 학습·참여자 수·직접 라벨 계보·이진 사건 혼동 결과.
        새 예약 시험 출처는 학습·모델 선택에 사용하지 않는다. 전신 점수
        모델을 교체하지 않고 알려진 개발 자료의 별도 사건 실험으로 저장한다.

    Raises:
        OSError: 직접 초안·관절·모델·결과 접근 실패.
        ValueError: 초안·특징·원본/참여자 분할 오류.
        RuntimeError: 자원 보호 실패.
    """
    check_resources()
    data = []
    unavailable = []
    for row in load_rows(True):
        if row["split"] != "train":
            continue
        target = None
        if row["grades"][3] == 2:
            target = 0
        if row["id"] in [
            "error_review_colossus_forward_core",
            "support_pannell_stop",
        ]:
            target = 1
        if target is None:
            continue
        x, note = event_features(row["seq"])
        if x is None:
            unavailable.append(dict(id=row["id"], observation=note))
            continue
        data.append(
            dict(
                id=row["id"],
                person=row["person"],
                source=row["source"],
                x=x,
                target=target,
                observation=note,
                target_origin="reused_nonexpert_direct_continuity_annotation_and_visible_arm_event",
                new_direct_review=False,
            )
        )
    direct_path = audit.OUT / "qevd_upper_event_direct.json"
    direct = audit.read_json(direct_path)
    extraction = audit.read_json(audit.OUT / "qevd_upper_event_extraction.json")
    if extraction["direct_sha256"] != audit.digest(direct_path):
        raise ValueError("New direct event labels changed")
    paths = {r["id"]: r["extraction"]["cache"] for r in extraction["records"]}
    for row in direct["reviews"]:
        with np.load(paths[row["id"]]) as cache:
            seq = cache["image_xy"].copy()
        x, note = event_features(seq)
        if x is None:
            unavailable.append(dict(id=row["id"], observation=note))
            continue
        data.append(
            dict(
                id=row["id"],
                person=row["person"],
                source=row["video_sha256"],
                x=x,
                target=row["visible_upper_support_change"],
                observation=note,
                target_origin="new_nonexpert_direct_temporal_upper_support_event",
                new_direct_review=True,
            )
        )
    predictions = []
    for person in sorted({r["person"] for r in data}):
        testing = [r for r in data if r["person"] == person]
        hashes = {r["source"] for r in testing}
        training = [
            r
            for r in data
            if r["person"] != person and r["source"] not in hashes
        ]
        model = fit(training)
        classes = list(model.classes_)
        for row in testing:
            probability = (
                float(model.predict_proba([row["x"]])[0][classes.index(1)])
                if 1 in classes
                else 0.0
            )
            predictions.append(
                dict(
                    id=row["id"],
                    held_person=person,
                    target=row["target"],
                    ai_probability=probability,
                    ai_event=probability >= 0.75,
                    rule_event=row["observation"]["relative_displacement"]
                    >= 0.20,
                    relative_displacement=row["observation"][
                        "relative_displacement"
                    ],
                )
            )
    metrics = {}
    for method in ["ai_event", "rule_event"]:
        metrics[method] = dict(
            TP=sum(r[method] and r["target"] == 1 for r in predictions),
            FN=sum(not r[method] and r["target"] == 1 for r in predictions),
            FP=sum(r[method] and r["target"] == 0 for r in predictions),
            TN=sum(not r[method] and r["target"] == 0 for r in predictions),
        )
    model = fit(data)
    folder = audit.OUT / "upper_support_event_trial"
    folder.mkdir(exist_ok=True)
    joblib.dump(
        dict(
            model=model,
            train_ids=[r["id"] for r in data],
            threshold=0.75,
            output="visible_upper_support_change_probability_not_posture_score",
            expert_validated=False,
        ),
        folder / "model.joblib",
    )
    report = dict(
        records=[{k: v for k, v in r.items() if k != "x"} for r in data],
        unavailable=unavailable,
        training_records=len(data),
        person_groups=len({r["person"] for r in data}),
        positive_person_groups=len(
            {r["person"] for r in data if r["target"] == 1}
        ),
        person_heldout_predictions=predictions,
        metrics=metrics,
        expert_validated=False,
        independent_final_test=False,
        canonical_full_score_model_modified=False,
        normal_grade_from_no_event=False,
        thresholds_frozen_before_this_diagnostic=dict(
            ai=0.75, rule_upper_arm_ratio=0.20
        ),
        new_direct_sha256=audit.digest(direct_path),
    )
    audit.save_json(audit.OUT / "upper_support_event_results.json", report)
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>상지 지지 변경 사건 AI</title><style>body{font:16px/1.6 system-ui;max-width:1100px;margin:auto;padding:24px}td,th{padding:10px;border-bottom:1px solid #aaa}</style><h1>상지 지지 변경 사건 AI·규칙</h1><p><a href="unified_status.html">통합 현황</a> · <a href="upper_support_event_results.json">전체 결과</a></p><p>얼굴·골반·다리가 없어도 어깨·팔꿈치가 충분히 보이면 상지 지지 위치 변경 사건을 평가하는 별도 분기입니다. 사건 확률을 자세 총점이나 정상 확률로 바꾸지 않습니다. 변화 근거가 없으면 지속성 2등급을 부여하지 않습니다. 새 직접 초안은 3참여자 그룹이며 기존 직접 초안 재사용과 구별합니다.</p>'
    ]
    parts.append(
        f'<p>관측 통과 {len(data)}구간 · {report["person_groups"]}참여자 그룹 · 변경 사건 {report["positive_person_groups"]}그룹. 알려진 개발 자료의 참여자 제외 진단이며 독립 최종 정확도가 아닙니다.</p><table><tr><th>방식</th><th>TP</th><th>FN</th><th>FP</th><th>TN</th></tr>'
    )
    for method, m in metrics.items():
        parts.append(
            f'<tr><td>{method}</td><td>{m["TP"]}</td><td>{m["FN"]}</td><td>{m["FP"]}</td><td>{m["TN"]}</td></tr>'
        )
    parts.append(
        "</table><p>몸통 회전·카메라 회전·관절 추적 오류를 지지 변경으로 오인할 수 있습니다. 머리만 움직이는 새 장면은 사건 음성 초안으로 포함했습니다. 제공자의 오류 문자열을 점수로 변환하지 않았습니다.</p></html>"
    )
    (audit.OUT / "upper_support_event.html").write_text(
        "".join(parts), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    print(run()["metrics"])
