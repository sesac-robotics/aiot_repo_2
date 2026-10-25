"""5종 보강 후보의 공통 검증·오류·근거·관절 영상을 로컬 웹에 연결한다."""

import html
import json
import pathlib
import statistics
import time

import cv2
import joblib
import numpy as np

import audit
import recovery_train
import render
import three_training
import recovery_w_phase
from allfive_round1 import OUT
from allfive_round1_features import new_rows
from allfive_round1_train import predict
from multiview_squat_batch2 import OUT as SQUAT_BATCH2
from multiview_squat_batch2_train import mask_reviewed_frames
from recovery_finalize import LocalLinks
from recovery_prepare import ROOT
from recovery_w_shape_trial import classify as w_classify

NAMES = dict(
    pushups="푸시업",
    squats="스쿼트",
    shoulder="숄더 프레스",
    plank="플랭크",
    ytw="YTW",
)


def row_html(values: list) -> str:
    """이스케이프한 표 한 행을 반환한다.

    Args:
        values: 병렬 열 값. None은 판정 불가로 표시한다.

    Returns:
        str: 정적 HTML 행. 수치 단위·점수를 변환하지 않는다.
    """
    return (
        "<tr>"
        + "".join(
            "<td>" + html.escape("판정 불가" if x is None else str(x)) + "</td>"
            for x in values
        )
        + "</tr>"
    )


def paired_summary(predictions: list[dict]) -> dict:
    """네 출력이 모두 있는 동일 항목·총점의 전후 오차만 집계한다.

    Args:
        predictions: 후보와 baseline 예측, 직접 초안이 연결된 고정 검증 행.

    Returns:
        dict: 4방식 공통 총점·항목 교집합 수와 두 모델 전후 오차. 판정 불가
        사례는 정확도 분모로 섞지 않고 원래 제출 수·가용 수를 따로 보존한다.
    """
    common = [
        p
        for p in predictions
        if p["target_score"] is not None
        and all(
            p[k] is not None and p["baseline"][k] is not None
            for k in ("ai_score", "rule_score")
        )
    ]
    items = {}
    for p in predictions:
        for i, item in enumerate(p["criterion_ids"]):
            g = p["levels"][i]
            if g is None or any(
                p[k][i] is None or p["baseline"][k][i] is None
                for k in ("ai_levels", "rule_levels")
            ):
                continue
            values = items.setdefault(
                item,
                dict(
                    records=0,
                    baseline_ai=[],
                    baseline_rule=[],
                    candidate_ai=[],
                    candidate_rule=[],
                ),
            )
            values["records"] += 1
            for stage, source in [
                ("baseline", p["baseline"]),
                ("candidate", p),
            ]:
                values[stage + "_ai"].append(abs(source["ai_levels"][i] - g))
                values[stage + "_rule"].append(
                    abs(source["rule_levels"][i] - g)
                )
    return dict(
        submitted=len(predictions),
        common_total_records=len(common),
        total={
            stage: {
                method: (
                    float(
                        np.mean(
                            [
                                abs(
                                    (
                                        p["baseline"]
                                        if stage == "baseline"
                                        else p
                                    )[method]
                                    - p["target_score"]
                                )
                                for p in common
                            ]
                        )
                    )
                    if common
                    else None
                )
                for method in ("ai_score", "rule_score")
            }
            for stage in ("baseline", "candidate")
        },
        common_items={
            item: {
                k: float(np.mean(v)) if isinstance(v, list) else v
                for k, v in values.items()
            }
            for item, values in items.items()
        },
    )


def profile(model: dict, row: dict) -> dict:
    """같은 관측 특징에서 두 방식 계산 시간과 재계산 일치를 측정한다.

    Args:
        model: 이번 별도 후보 모델.
        row: 이미 추출된 관측 특징. 정답 등급은 계산에 사용하지 않는다.

    Returns:
        dict: 공통 항목 수·3회 중앙값 ms·출력 일치. Heavy·특징·영상 비용은
        제외하며 CPU 25% 보호 환경의 수치이지 서비스 전체 지연은 아니다.

    Raises:
        ValueError: 측정 경로가 정식 후보 예측과 다른 경우.
    """
    expected = predict(model, row)
    indices = [
        i
        for i, (a, b) in enumerate(
            zip(expected["ai_levels"], expected["rule_levels"])
        )
        if a is not None and b is not None
    ]
    ai_times = []
    rule_times = []
    for repeat in range(3):
        started = time.perf_counter()
        ai = []
        for i in indices:
            n = row["notes"][i]
            estimator = model[recovery_train.head_key(row, i)]["model"]
            value = (
                estimator["model"].predict([n["measurement"]])[0]
                if isinstance(estimator, dict)
                else estimator.predict([n["feature"]])[0]
            )
            ai.append(float(np.clip(value, 0, 2)))
        ai_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        rules = []
        for i in indices:
            n = row["notes"][i]
            head = model[recovery_train.head_key(row, i)]
            kind = head.get("explicit_rule_kind")
            if kind == "W_shape_v1":
                g = int(
                    w_classify(
                        np.asarray([n["feature"]]),
                        np.asarray([head["W_policy"]]),
                    )[0, 0]
                )
            elif kind in ("plank_scalar_v1", "knee_heel_pitch_v1"):
                g = three_training.rule_grade(n, head["boundary"])
                if (
                    kind == "knee_heel_pitch_v1"
                    and head["pitch_boundary"] is not None
                ):
                    a, b = head["pitch_boundary"]
                    g = min(
                        g,
                        (
                            2
                            if n["feature"][9] <= a
                            else 1 if n["feature"][9] <= b else 0
                        ),
                    )
            else:
                g = (
                    n["rule_grade"]
                    if "rule_grade" in n
                    else three_training.rule_grade(n, head["boundary"])
                )
            rules.append(g)
        rule_times.append((time.perf_counter() - started) * 1000)
        if not np.allclose(
            ai, [expected["ai_levels"][i] for i in indices]
        ) or rules != [expected["rule_levels"][i] for i in indices]:
            raise ValueError("Profiling differs from canonical prediction")
    return dict(
        id=row["id"],
        common_items=len(indices),
        ai_median_ms=statistics.median(ai_times) if indices else None,
        rule_median_ms=statistics.median(rule_times) if indices else None,
        same_feature_repeats=3,
        consistent=True,
        Heavy_features_decode_excluded=True,
    )


def run() -> dict:
    """모든 저장 지문을 확인하고 근거·그룹 제외 예시·비교 웹을 만든다.

    Returns:
        dict: 파일 연결·지문·전체 영상 디코딩 확인. 브라우저 재생·전문가
        검증·독립 최종 시험 또는 서비스 모델 변경은 수행하지 않는다.

    Raises:
        ValueError: 지문·공통 검증·입력·영상·표시 연결 불일치.
        OSError: 저장 자료·영상·HTML 접근 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    report = audit.read_json(OUT / "comparison.json")
    if not report["completed"] or len(report["exercises"]) != 5:
        raise ValueError("Five exercise comparison is incomplete")
    checks = {
        name: sha == audit.digest(OUT / name)
        for name, sha in report["input_sha256"].items()
    }
    checks.update(
        {
            name: sha == audit.digest(pathlib.Path(__file__).with_name(name))
            for name, sha in report["code_sha256"].items()
        }
    )
    checks.update(
        {
            name: sha == audit.digest(pathlib.Path(name))
            for name, sha in report["preserved"].items()
        }
    )
    for r in report["exercises"]:
        checks[r["exercise"] + "_model"] = r["model_sha256"] == audit.digest(
            OUT / "models" / (r["exercise"] + ".joblib")
        )
    if not all(checks.values()):
        raise ValueError("Comparison fingerprints changed")
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    receipts = {
        r["id"]: r["extraction"]
        for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    tracking = {
        r["id"]: r
        for r in audit.read_json(OUT / "subject_tracking_review.json")[
            "records"
        ]
    }
    features = {
        r["id"]: r
        for r in audit.read_json(OUT / "candidate_input_manifest.json")
    }
    comparisons = {
        r["exercise"]: paired_summary(r["fixed_predictions"])
        for r in report["exercises"]
    }
    audit.save_json(OUT / "paired_development_comparison.json", comparisons)
    videos = []
    timings = []
    for experiment in report["exercises"]:
        exercise = experiment["exercise"]
        oof = {r["id"]: r for r in experiment["group_cv_predictions"]}
        choices = [
            r
            for r in annotations.values()
            if r["exercise"] == exercise and r["id"] in oof
        ]
        chosen = next(
            (
                r
                for r in choices
                if any(
                    g is not None and g < 2
                    for g in r["assessment"]["criterion_levels"]
                )
            ),
            choices[0] if choices else None,
        )
        if chosen is None:
            continue
        identifier = chosen["id"]
        prediction = oof[identifier]
        rec = receipts[identifier]
        folder = OUT / "previews" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(rec["cache"]) as cache:
            data = {k: cache[k].copy() for k in cache.files}
        data["image_xy"] = mask_reviewed_frames(
            data["image_xy"],
            data["frame_numbers"],
            tracking[identifier].get("unreliable_frame_intervals", []),
        )
        np.savez_compressed(
            folder
            / (
                "side.npz"
                if exercise in ("pushups", "squats", "plank")
                else "primary.npz"
            ),
            **data,
        )
        details = {}
        if exercise == "ytw":
            details = dict(
                evidence=dict(
                    phase_detection=recovery_w_phase.detect(data["image_xy"]),
                    w_shape=dict(
                        morphology=dict(grade=prediction["rule_levels"][2])
                    ),
                )
            )
        path = render.overlay(
            chosen,
            dict(
                prediction,
                cache_folder=str(folder),
                display_unit="/100",
                rule_details=details,
            ),
        )
        cap = cv2.VideoCapture(str(path))
        expected = int(cap.get(7))
        decoded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if decoded == expected // 2:
                    cv2.imwrite(str(folder / "preview.jpg"), frame)
                decoded += 1
        finally:
            cap.release()
        if decoded != expected or not decoded:
            raise ValueError("Video decoding failed")
        videos.append(
            dict(
                exercise=exercise,
                id=identifier,
                path=str(path),
                sha256=audit.digest(path),
                decoded_frames=decoded,
                display_prediction="connected_group_out_of_fold_not_training_fit",
                ai_score=prediction["ai_score"],
                rule_score=prediction["rule_score"],
            )
        )
        source = features[identifier]
        row = dict(
            source,
            weights=chosen["assessment"]["criterion_weights"],
            criterion_ids=chosen["assessment"]["criterion_ids"],
            notes=[
                dict(n, measurement=None, higher_is_better=True)
                for n in source["items"]
            ],
        )
        # 감사 manifest는 측정값을 생략하므로 해당 입력을 원본 캐시로 다시 계산한다.
    # 속도는 학습 입력 재구성 없이도 검증되는 별도 계산 manifest를 이용한다.
    reconstructed, _ = new_rows()
    for exercise in NAMES:
        choices = [r for r in reconstructed if r["exercise"] == exercise]
        if not choices:
            continue
        selected = choices[0]
        saved = features[selected["id"]]
        for note, observed in zip(selected["notes"], saved["items"]):
            note.update(
                feature=observed["feature"],
                bank=observed["bank"],
                eligible=observed["eligible"],
            )
        model = joblib.load(OUT / "models" / (exercise + ".joblib"))
        timings.append(profile(model, selected))
    page = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><title>5종 보유 자료 보강·후보 비교</title><style>body{font:16px/1.6 sans-serif;max-width:1250px;margin:24px auto;padding:16px;background:#f5f7fa;color:#213047}table{width:100%;border-collapse:collapse;background:white}th,td{border:1px solid #cbd4df;padding:9px;text-align:left;overflow-wrap:anywhere}details{background:white;padding:12px;margin:14px 0}img{max-width:100%}video{max-width:100%;max-height:550px}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style>',
        '<h1>5종 보유 자료 우선 보강 — 별도 후보</h1><p><a href="../index.html">보존된 5종 기준 결과</a> · <a href="../multiview_squat/batch2/index.html">앞선 스쿼트 후보</a></p>',
        "<p>QEVD 20개 원본(푸시업 4·스쿼트 4·플랭크 8·YTW 4)과 보유 MultiView 숄더 프레스 4개의 시간순 표본을 직접 검토했습니다. 앉아서 수행한 숄더 1개는 목표 변형에서 제외했습니다. 플랭크는 보유 해군 측면 원본의 겹치지 않는 2개 유지 구간을 추가 검토했습니다. 같은 원본·참여자의 반복을 새 참여자로 세지 않습니다.</p>",
        "<p>점수 초안 18구간과 별도 상지 변경 사건 주석 8개입니다. 직접 검토는 시간순 표본 검토이며 연속 전체 영상·전문가 정답이 아닙니다. 제공자 문자열·Rule 결과를 점수 정답으로 변환하지 않았습니다. 관절 표시 25구간의 필수 부위만 직접 대조했고, 불안정 프레임·불일치 단계를 제외했습니다. 초안 수가 곧 학습 입력 수는 아닙니다.</p>",
        "<p>새 구간은 기존 train 역할을 유지했습니다. 고정 개발 검증은 재학습에 넣지 않았고, 새 학습 입력은 연결된 참여자·원본·녹화 그룹을 통째로 제외하는 3폴드에서 비교했습니다. 아래 고정 검증은 이미 확인한 개발 자료이며 독립 최종 시험이 아닙니다.</p>",
        "<p>MediaPipe Heavy는 관절 추출기이며 별도 ExtraTrees AI 점수와 기하 Rule Base를 비교합니다. 측면 발 기울기는 화면 투영 특징으로 실제 지면 접촉을 보증하지 않습니다. W 단계·형태 후보는 이전 개발 실패를 계기로 설계했으며 숫자 선택은 각 학습 폴드에서만 수행했습니다. 기본 후보에서는 정면 Y·전경의 Rule 총점을 채우지 않았습니다. 별도 투영 규칙 실험은 아래에서 직접 확인된 관측 교집합으로만 비교합니다.</p>",
        "<h2>실제 사용 수와 고정 개발 검증</h2><table><tr><th>운동</th><th>학습 구간 / 메타 그룹 / 총점 조건 충족</th><th>새 초안 / 사용 항목 라벨</th><th>공통 총점 수</th><th>AI 오차 전→후</th><th>Rule 오차 전→후</th></tr>",
    ]
    for r in report["exercises"]:
        p = comparisons[r["exercise"]]
        page.append(
            row_html(
                [
                    NAMES[r["exercise"]],
                    f"{r['training_segments']} / {r['training_metadata_groups']} / {r['training_full_observed']}",
                    f"{r['new_score_drafts']} / {r['new_eligible_item_labels']}",
                    p["common_total_records"],
                    str(p["total"]["baseline"]["ai_score"])
                    + " → "
                    + str(p["total"]["candidate"]["ai_score"]),
                    str(p["total"]["baseline"]["rule_score"])
                    + " → "
                    + str(p["total"]["candidate"]["rule_score"]),
                ]
            )
        )
    page.append(
        "</table><p>총점 오차는 100점 단위이며 기준·후보의 AI·규칙 네 출력이 모두 있는 같은 구간만 비교했습니다. 전체 제출·판정 가능 수와 항목별 비교는 아래에 별도 표시합니다. 학습 구간 수에는 부분 항목만 사용하는 구간도 포함됩니다.</p>"
    )
    for r in report["exercises"]:
        e = r["exercise"]
        p = comparisons[e]
        page.append(
            "<h2>"
            + NAMES[e]
            + "</h2><h3>같은 항목의 전후 오차 (등급 0~2)</h3><table><tr><th>항목</th><th>공통 수</th><th>AI 전→후</th><th>Rule 전→후</th></tr>"
        )
        for item, v in p["common_items"].items():
            page.append(
                row_html(
                    [
                        item,
                        v["records"],
                        f"{v['baseline_ai']:.4f} → {v['candidate_ai']:.4f}",
                        f"{v['baseline_rule']:.4f} → {v['candidate_rule']:.4f}",
                    ]
                )
            )
        page.append(
            "</table><details><summary>학습 등급 분포·단일 등급·관측률·교차검증</summary><pre>"
            + html.escape(
                json.dumps(
                    dict(
                        heads=r["heads"],
                        fixed_baseline=r["fixed_baseline"],
                        fixed_candidate=r["fixed_candidate"],
                        group_cv=r["group_cv"],
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            + "</pre></details>"
        )
        page.append(
            "<p>정상 또는 오류 한 등급만 학습한 항목은 다른 등급을 구분하는 능력이 확인되지 않았습니다. 교차검증은 학습 내부 평가이며 고정 검증과 섞지 않습니다.</p>"
        )
        for v in [v for v in videos if v["exercise"] == e]:
            page.append(
                "<p>"
                + html.escape(v["id"])
                + ' — 학습 시 해당 연결 그룹 전체를 제외한 예측입니다. 총점이 없으면 N/A로 표시합니다.</p><video controls preload="metadata" src="'
                + pathlib.Path(v["path"]).relative_to(OUT).as_posix()
                + '"></video>'
            )
        for ann in [a for a in annotations.values() if a["exercise"] == e]:
            f = features[ann["id"]]
            page.append(
                "<details><summary>"
                + html.escape(ann["id"])
                + "</summary><p>"
                + html.escape(ann["assessment"]["note_ko"])
                + "</p><pre>"
                + html.escape(
                    json.dumps(
                        dict(levels=f["levels"], items=f["items"]),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre>"
            )
            for image in [
                *ann["evidence_images"],
                tracking[ann["id"]]["evidence_image"],
            ]:
                page.append(
                    '<img loading="lazy" src="'
                    + pathlib.Path(image).relative_to(OUT).as_posix()
                    + '">'
                )
            page.append("</details>")
    event_path = OUT / "upper_event_comparison.json"
    if event_path.exists():
        page.append(
            "<h2>플랭크 정면의 상지 변경 사건 — 점수와 별도</h2><p>보이는 상지 지지 위치 변경·중단의 확률·사건 비교입니다. 사건 미관측을 정상 자세 등급이나 총점으로 바꾸지 않습니다.</p><pre>"
            + html.escape(
                json.dumps(
                    audit.read_json(event_path), ensure_ascii=False, indent=2
                )
            )
            + "</pre>"
        )
    for relative, title, script in (
        (
            "ordinal_trial/comparison.json",
            "학습 내부 AI 방식·특징 선택 추가 실험",
            "allfive_round1_ordinal.py",
        ),
        (
            "ytw_projected_rule_trial/comparison.json",
            "YTW 네 항목 투영 규칙 후보",
            "ytw_projected_rule_trial.py",
        ),
    ):
        trial = audit.read_json(OUT / relative)
        if not trial["completed"]:
            raise ValueError("Supplemental trial is incomplete")
        checks[relative + ":code"] = trial["code_sha256"] == audit.digest(
            pathlib.Path(__file__).with_name(script)
        )
        checks[relative + ":input"] = trial["input_sha256"] == audit.digest(
            OUT / "candidate_input_manifest.json"
        )
        if not checks[relative + ":code"] or not checks[relative + ":input"]:
            raise ValueError("Supplemental trial fingerprint mismatch")
        if "exercises" in trial:
            shown = [
                dict(
                    exercise=r["exercise"],
                    selection=r["selection"],
                    paired_comparison=r["paired_comparison"],
                )
                for r in trial["exercises"]
            ]
            note = "선택용 내부 교차검증 오차는 독립 정확도가 아닙니다. 같은 고정 개발 검증에서 악화된 종목도 표시하며 자동 승격하지 않습니다."
        else:
            shown = {
                k: trial[k]
                for k in (
                    "training_heads",
                    "connected_group_cv",
                    "fixed_development",
                    "manual_observability_intersection",
                    "automatic_deployment_view_gate_verified",
                    "anatomical_hinge_angle_measured",
                )
            }
            note = "Y·T 도달과 팔 펴기, W의 팔 굽힘·손·팔꿈치 위치, 전경의 몸통 투영 비율·좌우 흔들림을 명시 조건으로 계산했습니다. 보이는 특징과 직접 라벨이 모두 있는 오프라인 교집합의 비교입니다. 실제 고관절 각도·서비스 자동 구도 판정·독립 최종 시험은 검증하지 않았습니다."
        page.append(
            "<h2>"
            + title
            + "</h2><p>"
            + note
            + "</p><pre>"
            + html.escape(json.dumps(shown, ensure_ascii=False, indent=2))
            + '</pre><p><a href="'
            + relative
            + '">전체 실험 JSON</a></p>'
        )
    page.append(
        "<h2>같은 입력의 계산 시간·일관성</h2><p>CPU 25%·메모리 3GiB 보호 환경의 동일 관측 특징 3회 중앙값 ms입니다. 공통 Heavy·특징 추출·디코딩 비용을 제외하므로 서비스 전체 지연이나 실제 재촬영 일관성은 아닙니다.</p><pre>"
        + html.escape(json.dumps(timings, ensure_ascii=False, indent=2))
        + "</pre>"
    )
    for name in (
        "comparison.json",
        "paired_development_comparison.json",
        "score_annotations.json",
        "fine_grained_labels.json",
        "fine_grained_labels_with_worker_ids.json",
        "manual_decisions.json",
        "subject_tracking_review.json",
        "candidate_input_manifest.json",
        "feature_changes.json",
    ):
        page.append('<p><a href="' + name + '">' + name + "</a></p>")
    (OUT / "index.html").write_text("\n".join(page), encoding="utf-8")
    parser = LocalLinks()
    parser.feed((OUT / "index.html").read_text())
    missing = [p for p in parser.paths if not (OUT / p).exists()]
    if missing:
        raise ValueError("Missing local report links")
    result = dict(
        comparison_sha256=audit.digest(OUT / "comparison.json"),
        code_sha256=audit.digest(pathlib.Path(__file__)),
        fingerprints=checks,
        videos=videos,
        all_frames_decoded=True,
        score_profiles=timings,
        local_links_checked=len(parser.paths),
        missing_local_links=missing,
        browser_rendering_verified=False,
        independent_final_test=False,
        promoted=False,
    )
    audit.save_json(OUT / "report_verification.json", result)
    return dict(
        videos=len(videos),
        local_links_checked=len(parser.paths),
        fingerprints_valid=True,
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
