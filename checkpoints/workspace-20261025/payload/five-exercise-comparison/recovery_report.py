"""재학습 후보의 수량·동일 검증 비교·한계를 기존 로컬 웹에 연결한다."""

import html
import json
import pathlib

import audit
from recovery_prepare import ROOT


def run() -> dict:
    """저장된 비교 결과에서 읽기 전용 HTML과 검증 요약을 만든다.

    Returns:
        dict: HTML 경로와 운동별 동일 교집합 전후 오차. 웹 브라우저에서
        렌더링·재생 검증을 수행했다는 뜻은 아니다. 점수 단위는 100점이다.

    Raises:
        OSError: 입력·HTML 저장·로컬 연결 실패.
        ValueError: 같은 이름의 연결이 다른 목적지를 가리키는 경우.
    """
    fitted = audit.read_json(ROOT / "comparison.json")
    sample_path = ROOT / "sample_count_review.json"
    sample = audit.read_json(sample_path) if sample_path.exists() else {}
    grade_groups = {}
    if sample.get("comparison_sha256") == audit.digest(
        ROOT / "comparison.json"
    ):
        grade_groups = {
            (row["exercise"], head["item"], head["view"]): head.get(
                "grade_metadata_groups", {}
            )
            for row in sample["exercises"]
            for head in row["item_view_counts"]
        }
    names = dict(
        pushups="푸시업",
        squats="스쿼트",
        shoulder="숄더 프레스",
        plank="플랭크",
        ytw="YTW",
    )
    content = [
        '<!doctype html><meta charset="utf-8"><title>보유 자료 라벨 보강</title>',
        "<style>body{font:16px sans-serif;max-width:1250px;margin:32px auto;padding:16px;color:#213047;background:#f5f7fa}table{border-collapse:collapse;background:white;width:100%;margin:16px 0}td,th{padding:10px;border:1px solid #d5dce5;text-align:left}details{padding:12px;background:white;margin:12px 0}a{color:#165bb3}p{line-height:1.6}</style>",
        "<h1>보유 자료 라벨 보강·재학습 후보</h1><p>전문가 정답이 아닌 직접 시간순 표본 검토 초안으로 진행한 개발 실험입니다. 기존 공통 모델을 교체하지 않았고, 독립 최종 시험은 완료하지 않았습니다. 70개는 최소 개발 목표이며 정확도 보장이 아닙니다.</p>",
        "<p>MediaPipe Heavy는 관절 추출기입니다. AI 점수 회귀와 Rule Base는 같은 관절·관측 정책에서 별도로 계산합니다. 제공자 정상·오류 문자열과 규칙 점수는 AI의 정답·입력으로 변환하지 않습니다.</p>",
        "<table><tr><th>운동</th><th>실제 학습 입력 / 70</th><th>person·worker 메타 그룹 / 연결 그룹</th><th>네 항목 라벨·관측 모두 충족</th><th>부분 입력</th><th>동일 개발 검증 AI / 규칙 MAE</th></tr>",
    ]
    changes = []
    allfive_trial = ROOT / "allfive_round1/comparison.json"
    if (
        allfive_trial.exists()
        and (allfive_trial.parent / "index.html").exists()
    ):
        current = audit.read_json(allfive_trial)
        if current.get("completed") and current.get("preserved", {}).get(
            str(ROOT / "comparison.json")
        ) == audit.digest(ROOT / "comparison.json"):
            content.insert(
                4,
                '<p><a href="allfive_round1/index.html">최신 5종 보유 자료 보강: 직접 초안·추적 검토·AI/규칙 재학습·추가 실험</a> — 아래 보존 기준 모델과 구분한 별도 후보입니다. 악화된 결과와 미완료 조건도 함께 표시합니다.</p>',
            )
    batch2_trial = ROOT / "multiview_squat/batch2/comparison.json"
    if batch2_trial.exists() and (batch2_trial.parent / "index.html").exists():
        batch2 = audit.read_json(batch2_trial)
        if batch2.get("preserved", {}).get(
            str(ROOT / "comparison.json")
        ) == audit.digest(ROOT / "comparison.json"):
            content.insert(
                4,
                '<p><a href="multiview_squat/batch2/index.html">최신 보유 스쿼트 12묶음: 20구간 라벨·재학습·두 점수 비교</a> — 추적 검토 후 공통 총점 개발 검증 5구간입니다. 별도 후보이며 기존 5종 모델에 반영하지 않았습니다.</p>',
            )
    multiview_trial = ROOT / "multiview_squat/comparison.json"
    if (
        multiview_trial.exists()
        and (ROOT / "multiview_squat/index.html").exists()
    ):
        candidate = audit.read_json(multiview_trial)
        if candidate.get("source_comparison_sha256") == audit.digest(
            ROOT / "comparison.json"
        ):
            content.insert(
                4,
                '<p><a href="multiview_squat/index.html">보유 MultiView 스쿼트: 시점 대조·라벨 5개·별도 재학습 비교</a> — 아래 5종 기준 결과와 구분한 추가 개발 실험입니다.</p>',
            )
    for row in fitted["exercises"]:
        e = row["exercise"]
        m = row["fixed_validation"]
        error = m["paired_total_MAE_points"]
        content.append(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(x)) + "</td>"
                for x in [
                    names[e],
                    str(row["training_effective_segments"]) + " / 70",
                    str(row["training_people"])
                    + " / "
                    + str(row.get("training_connected_groups", "집계 전")),
                    row.get("training_four_item_observed_segments", "집계 전"),
                    row.get(
                        "training_partial_or_unobserved_segments", "집계 전"
                    ),
                    f"{error['ai_score']} / {error['rule_score']} (공통 {m['paired_total_records']}개)",
                ]
            )
            + "</tr>"
        )
        paired = [
            p
            for p in row["fixed_predictions"]
            if p["target_score"] is not None
            and all(
                p[k] is not None and p["before"][k] is not None
                for k in ["ai_score", "rule_score"]
            )
        ]
        change = dict(exercise=e, common_records=len(paired))
        for stage in ["before", "after"]:
            change[stage] = {
                k: (
                    sum(
                        abs(
                            (p["before"] if stage == "before" else p)[k]
                            - p["target_score"]
                        )
                        for p in paired
                    )
                    / len(paired)
                    if paired
                    else None
                )
                for k in ["ai_score", "rule_score"]
            }
        changes.append(change)
    content.append(
        "</table><p>부분 입력은 일부 항목만 학습한 구간입니다. 같은 사람의 다른 반복과 여러 촬영 구도는 새 참여자로 세지 않습니다. QEVD의 worker 기반 그룹은 같은 데이터셋 안에서도 확인된 실제 참여자 수가 아닙니다. 실제 인원 수로 해석하지 않습니다. 촬영 배경 연결은 보수적인 분할 보조 정보이며 동일인 전체를 확인한 결과가 아닙니다. YTW의 전경 대리 특징과 총점은 검증된 완성 모델로 해석할 수 없습니다.</p>"
    )
    content.append(
        "<p>네 항목 라벨·관측이 모두 있는 구간 70개까지의 최소 부족 수: "
        + " · ".join(
            names[r["exercise"]]
            + " "
            + str(max(0, 70 - r["training_four_item_observed_segments"]))
            for r in fitted["exercises"]
        )
        + "개. 이 수를 채워도 오류 등급·참여자 다양성과 별도 시험 검증이 필요합니다. YTW는 이 조건을 만족해도 Y·전경 규칙 총점 검증이 별도로 남습니다.</p>"
    )
    content.append(
        "<p>플랭크의 별도 정지 몸통 항목 실험은 원본 이미지 56장·직접 등급 초안 54장·공통 평가 37장을 보존합니다. 영상의 네 항목 학습 수에 합산하지 않습니다. 정지 이미지에 지지 지속성 점수나 가정한 FPS를 붙이지 않았습니다.</p>"
    )
    content.append(
        "<p>단일 시도 경계와 일부 촬영 관측 제한은 직접 검토한 메타데이터를 사용합니다. 이 결과는 원본 영상 전체에서 완전히 자동으로 운동·시도·관측 가능성을 찾아내는 성능 검증이 아닙니다.</p>"
    )
    variant_path = ROOT / "variant_policy_exclusions.json"
    if variant_path.exists():
        content.append(
            "<p>반복 감사에서 앉은 숄더 프레스 후속 구간이 기존 서기 제외 조건을 우회한 오류를 수정했습니다. "
            + str(len(audit.read_json(variant_path)["records"]))
            + "개 초안은 보존하고 이번 학습에서 제외했습니다. 표의 수량과 성능은 수정 후 결과입니다.</p>"
        )
    content.append(
        '<p><a href="visual_recording_links.json">촬영 자료 연결</a> · <a href="contact_scene_audit.json">자동 배경 대응 감사</a> · <a href="recording_split_quarantine.json">검증 분할 충돌 제외 기록</a> · <a href="fit_input_manifest.json">실제 학습·검증 입력 기록</a></p>'
    )
    for name, label in [
        ("W_failure_followup.json", "W 복귀 오선택·상완 모양 오판 직접 재점검"),
        (
            "held_multiview_squat_inventory.json",
            "보유 MultiView 스쿼트 미통합 목록·기존 역할",
        ),
        (
            "variant_policy_exclusions.json",
            "서기 기준과 다른 변형의 학습 제외 기록",
        ),
        (
            "targeted_variant_direct_exclusions.json",
            "이번 직접 검토에서 제외한 앉은 변형",
        ),
        (
            "plank_single_screen_direct_review.json",
            "플랭크 추가 600개 선별·12개 직접 검토 기록",
        ),
        (
            "ytw_extension_direct_review.json",
            "YTW 팔 도달 후보 선별·직접 검토 기록",
        ),
        ("ytw_extension_direct_decisions.json", "YTW 추가 항목별 초안 근거"),
    ]:
        if (ROOT / name).exists():
            content.append('<p><a href="' + name + '">' + label + "</a></p>")
    trial_path = ROOT / "trunk_direction_trial/comparison.json"
    if trial_path.exists():
        trial = audit.read_json(trial_path)
        if trial["comparison_sha256"] == audit.digest(ROOT / "comparison.json"):
            content.append(
                "<h2>푸시업 몸통 방향 특징 별도 실험</h2><p>학습 내부 그룹 검증으로만 특징·회귀기를 선택했습니다. 공통 모델·규칙을 교체하지 않았으며, 선택 검증 오차를 독립 성능으로 해석하지 않습니다.</p><pre>"
                + html.escape(
                    json.dumps(
                        {
                            k: v
                            for k, v in trial.items()
                            if k not in ("validation_predictions",)
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + '</pre><p><a href="trunk_direction_trial/comparison.json">실험 원자료</a></p>'
            )
    w_path = ROOT / "W_calibration_trial.json"
    if w_path.exists():
        trial = audit.read_json(w_path)
        if trial["comparison_sha256"] == audit.digest(ROOT / "comparison.json"):
            content.append(
                "<h2>W 명시 규칙 경계 보정 실험</h2><p>직접 초안만 보정 목표로 쓰고 학습 그룹 3폴드에서 경계를 다시 선택했습니다. 고정 개발 4개는 보정에 쓰지 않았습니다. 공통 규칙은 보존합니다.</p><pre>"
                + html.escape(json.dumps(trial, ensure_ascii=False, indent=2))
                + '</pre><p><a href="W_calibration_trial.json">W 보정·실패 사례 원자료</a></p>'
            )
    for name, title in [
        ("W_phase_trial.json", "W 단계 선택 보완·AI 항목 재학습"),
        ("W_shape_trial.json", "팔꿈치가 몸통 아래에 붙는 모양의 규칙 보완"),
    ]:
        path = ROOT / name
        if path.exists():
            trial = audit.read_json(path)
            if trial["comparison_sha256"] == audit.digest(
                ROOT / "comparison.json"
            ):
                content.append(
                    "<details><summary>"
                    + title
                    + "</summary><p>개발 오판 사례를 보고 설계한 별도 실험입니다. 독립 시험이 아니며 공통 모델·규칙을 교체하지 않았습니다. 오차 단위는 W 0~2등급입니다.</p><pre>"
                    + html.escape(
                        json.dumps(
                            {k: v for k, v in trial.items() if k != "records"},
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    + '</pre><a href="'
                    + name
                    + '">전체 근거·실패 사례</a></details>'
                )
    w_preview_path = ROOT / "W_preview_manifest.json"
    if w_preview_path.exists():
        preview = audit.read_json(w_preview_path)
        if preview["comparison_sha256"] == audit.digest(
            ROOT / "comparison.json"
        ):
            content.append(
                "<h2>W 오판 사례 전후 관절 영상 — 항목 0~2등급</h2><p>영상의 숫자는 W 항목 등급입니다. YTW 전체 100점 총점이 아닙니다. 개발에 참고한 같은 사례로 이전·실험 후보를 비교합니다.</p>"
            )
            for record in preview["records"]:
                relative = (
                    pathlib.Path(record["path"]).relative_to(ROOT).as_posix()
                )
                content.append(
                    "<p>"
                    + html.escape(
                        str(
                            {
                                k: v
                                for k, v in record.items()
                                if k not in ("path", "sha256")
                            }
                        )
                    )
                    + '</p><video controls preload="metadata" style="max-width:700px" src="'
                    + html.escape(relative, quote=True)
                    + '"></video>'
                )
    for row, change in zip(fitted["exercises"], changes):
        e = row["exercise"]
        content.append(
            "<details><summary>"
            + names[e]
            + " — 항목별 분포·개발 검증</summary><table><tr><th>항목</th><th>구도</th><th>학습 수 / 그룹</th><th>등급 0·1·2 분포</th><th>등급별 메타 그룹 수 — 실제 인원 미확인</th><th>단일 등급</th></tr>"
        )
        for head in row["heads"]:
            content.append(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(x)) + "</td>"
                    for x in [
                        head["key"][1],
                        head["key"][2],
                        f"{head['records']} / {head['people']}",
                        head["grade_counts"],
                        grade_groups.get(tuple(head["key"]), "집계 전"),
                        head["single_grade_model"],
                    ]
                )
                + "</tr>"
            )
        content.append(
            "</table><p>기존 입력만 학습한 후보와 보강 입력을 포함한 후보의 동일 유효 검증 교집합 비교:</p><pre>"
            + html.escape(json.dumps(change, ensure_ascii=False, indent=2))
            + "</pre><p>참여자·원본 연결 그룹 교차검증. 반복 구간은 같은 그룹으로 묶음:</p><pre>"
            + html.escape(
                json.dumps(row["group_cv"], ensure_ascii=False, indent=2)
            )
            + "</pre></details>"
        )
    content.append(
        '<p><a href="comparison.json">모델 비교 원자료</a> · <a href="sample_count_review.json">70개 목표·항목별 집계</a></p><h2>추가 직접 라벨의 근거</h2>'
    )
    preview_path = ROOT / "preview_manifest.json"
    if preview_path.exists():
        preview = audit.read_json(preview_path)
        if preview["comparison_sha256"] == audit.digest(
            ROOT / "comparison.json"
        ):
            content.append(
                "<h2>점수 계산 시간·반복 재현성</h2><p>고정 개발 검증의 같은 입력 특징에서 반복한 결과입니다. 관절 추출·특징 생성 시간은 제외되어 전체 영상 처리 속도가 아니며, 재촬영 일관성도 아닙니다.</p><pre>"
                + html.escape(
                    json.dumps(
                        preview.get(
                            "fixed_development_validation_grading_profile"
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                + "</pre>"
            )
            content.append("<h2>관절 표시 학습 예제 — 독립 성능 시험 아님</h2>")
            for example in preview["records"]:
                content.append(
                    "<h3>"
                    + html.escape(example["id"])
                    + "</h3><pre>"
                    + html.escape(
                        json.dumps(
                            dict(
                                reference=example["reference"],
                                prediction=example["prediction"],
                                observation=example["notes"],
                            ),
                            ensure_ascii=False,
                            indent=2,
                        )
                    )
                    + "</pre>"
                )
                for video in example["videos"]:
                    relative = (
                        pathlib.Path(video["path"]).relative_to(ROOT).as_posix()
                    )
                    content.append(
                        "<p>"
                        + html.escape(video["view"])
                        + '</p><video controls preload="metadata" style="max-width:600px;max-height:500px" src="'
                        + html.escape(relative, quote=True)
                        + '"></video>'
                    )
    for path in sorted(ROOT.glob("full_repetition_score_annotations_*.json")):
        for row in audit.read_json(path):
            links = []
            for image in row["evidence_images"]:
                relative = pathlib.Path(image).relative_to(ROOT).as_posix()
                links.append(
                    '<a href="'
                    + html.escape(relative, quote=True)
                    + '">시간순 검토판</a>'
                )
            content.append(
                "<details><summary>"
                + html.escape(row["id"])
                + "</summary><p>"
                + html.escape(str(row["assessment"]["criterion_levels"]))
                + " — "
                + html.escape(row["assessment"]["note_ko"])
                + "</p>"
                + " · ".join(links)
                + "</details>"
            )
    content.append("<h2>보유 QEVD YTW 직접 검토 초안</h2>")
    for row in audit.read_json(ROOT / "additional_score_annotations.json"):
        if row["exercise"] != "ytw":
            continue
        links = []
        for image_path in row["evidence_images"]:
            relative = pathlib.Path(image_path).relative_to(ROOT).as_posix()
            links.append(
                '<a href="'
                + html.escape(relative, quote=True)
                + '">직접 검토 근거</a>'
            )
        content.append(
            "<details><summary>"
            + html.escape(row["id"])
            + "</summary><p>"
            + html.escape(str(row["assessment"]["criterion_levels"]))
            + " · 총점 "
            + html.escape(str(row["assessment"]["score"]))
            + "</p><p>"
            + html.escape(row["assessment"]["note_ko"])
            + "</p>"
            + " · ".join(links)
            + "</details>"
        )
    (ROOT / "index.html").write_text("\n".join(content), encoding="utf-8")
    target = audit.BASE / "output/three-exercise-comparison-20261022"
    link = target / "five-exercise-recovery"
    if link.is_symlink() and link.resolve() != ROOT:
        raise ValueError("Unexpected existing recovery link")
    if not link.exists():
        link.symlink_to(ROOT, target_is_directory=True)
    page = target / "index.html"
    source = page.read_text(encoding="utf-8")
    marker = 'href="five-exercise-recovery/index.html"'
    if marker not in source:
        source = source.replace(
            "</body>",
            "<p><a "
            + marker
            + ">보유 자료 라벨 보강·70개 목표·별도 재학습 후보</a></p></body>",
        )
        page.write_text(source, encoding="utf-8")
    result = dict(
        html=str(ROOT / "index.html"),
        paired_before_after=changes,
        browser_render_verified=False,
        comparison_sha256=audit.digest(ROOT / "comparison.json"),
    )
    audit.save_json(ROOT / "report_verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
