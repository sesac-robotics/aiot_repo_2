"""미사용 새 출처에서 동결 플랭크 총점·항목 AI와 규칙을 한 번 비교한다."""

import html
import pathlib
import shutil

import cv2
import joblib
import numpy as np

import audit
import benchmark
from extract import extract_view
from plank_item_observation import assess, features, ITEMS
from plank_partial_item_trial import load_rows
from plank_support_motion import detect
import render
import rule_score


def run() -> dict:
    """사전 예약·모델 동결·초안 해시를 확인하고 새 입력만 추출한다.

    Returns:
        dict: 같은 관측 항목의 동결 AI·규칙 결과와 새 원본 그룹 수.
        비전문가 전향적 소규모 확인이며 5종 최종 정확도 검증이 아니다.
        결과를 본 뒤 모델·경계를 변경하거나 새 자료로 학습하지 않는다.

    Raises:
        OSError: 모델·라벨·영상·결과 접근 실패.
        ValueError: 해시·입력 특징 불일치·학습 원본/참여자 중복.
        RuntimeError: 완료 결과 재평가 또는 자원 보호 실패.
    """
    target = audit.OUT / "autonomous_prospective_results.json"
    if target.exists() and audit.read_json(target).get("completed"):
        raise RuntimeError("One-pass prospective results already frozen")
    policy_path = audit.OUT / "autonomous_prospective_freeze.json"
    policy = audit.read_json(policy_path)
    for record in policy["files"].values():
        if audit.digest(pathlib.Path(record["path"])) != record["sha256"]:
            raise ValueError("Frozen file changed")
    if (
        audit.digest(pathlib.Path(rule_score.__file__))
        != policy["files"]["common_rule"]["sha256"]
    ):
        raise ValueError("Common rule changed")
    if (
        audit.digest(audit.OUT / "labels.json")
        != policy["training_labels_sha256"]
    ):
        raise ValueError("Fitting labels changed")
    direct_path = audit.OUT / "autonomous_prospective_direct.json"
    direct_hash = audit.digest(direct_path)
    direct = audit.read_json(direct_path)
    if direct["model_freeze_sha256"] != audit.digest(policy_path):
        raise ValueError("Direct/freeze lineage mismatch")
    drafts = {r["id"]: r for r in direct["reviews"]}
    train = [r for r in load_rows(True) if r["split"] == "train"]
    people = {r["person"] for r in train}
    sources = {r["source"] for r in train}
    full = joblib.load(policy["files"]["full_ai"]["path"])
    models = {
        item: joblib.load(policy["files"][item]["path"]) for item in ITEMS
    }
    report = (
        audit.read_json(target)
        if target.exists()
        else dict(
            records=[],
            completed=False,
            direct_sha256=direct_hash,
            freeze_sha256=audit.digest(policy_path),
            sources_used_for_fitting=False,
            expert_validated=False,
            all_five_final_test=False,
        )
    )
    if report["direct_sha256"] != direct_hash:
        raise ValueError("Direct draft changed after first prediction")
    for export in audit.read_json(
        audit.OUT / "autonomous_prospective_plan_exports.json"
    )["records"]:
        draft = drafts[export["id"]]
        if (
            draft["source_sha256"] in sources
            or "youtube_channel_" + draft["person_group"] in people
        ):
            raise ValueError("New source/person overlaps fitting data")
        if any(r["id"] == export["id"] for r in report["records"]):
            continue
        row = dict(
            id=export["id"],
            exercise="plank",
            video_path=export["video_path"],
            video_sha256=export["video_sha256"],
            required_views=[
                (
                    "side"
                    if draft["total_score"] is not None
                    else "item_only_review"
                )
            ],
            segment=dict(
                start_frame=0, end_frame=export["frames"] - 1, fps=export["fps"]
            ),
        )
        extraction = extract_view(row, "side", row["video_path"])
        seq, _, holds = benchmark.load_input(row)
        observation = assess(
            seq,
            export["frames"] / export["fps"],
            [g is not None for g in draft["draft_levels"]],
        )
        x, names = benchmark.input_features(seq, None)
        if names != full["names"]:
            raise ValueError("Full feature order mismatch")
        raw = np.clip(full["model"].predict([x])[0], 0, 2)
        approved = (
            draft["total_score"] is not None
            and not holds
            and all(r["eligible"] for r in observation["items"])
        )
        rule = rule_score.evaluate("plank", seq, None, ITEMS, [40, 25, 20, 15])
        item_results = []
        for index, item in enumerate(ITEMS):
            obs = observation["items"][index]
            pred = None
            if obs["eligible"]:
                x, names = features(seq, index, obs["side"])
                if names != models[item]["names"]:
                    raise ValueError("Item feature order mismatch")
                pred = float(
                    np.clip(models[item]["model"].predict([x])[0], 0, 2)
                )
            item_results.append(
                dict(
                    criterion_id=item,
                    draft_grade=draft["draft_levels"][index],
                    ai_grade=pred,
                    rule_grade=obs["rule_grade"],
                    observation=obs,
                )
            )
        report["records"].append(
            dict(
                id=row["id"],
                source_id=draft["source_id"],
                person_group=draft["person_group"],
                source_group=draft["source_group"],
                role=draft["scope"],
                extraction=extraction,
                common_joint_holds=holds,
                full_total_approved=approved,
                direct_total=draft["total_score"],
                full_ai_score=(
                    float(np.dot(raw, [40, 25, 20, 15]) / 2)
                    if approved
                    else None
                ),
                full_rule_score=rule["score"] if approved else None,
                items=item_results,
                motion_evidence=detect(seq),
                video_path=export["video_path"],
            )
        )
        audit.save_json(target, report)
    paired = [
        r
        for r in report["records"]
        if r["full_total_approved"] and r["full_rule_score"] is not None
    ]
    report.update(
        completed=True,
        new_source_groups=len({r["source_group"] for r in report["records"]}),
        prospective_full_source_groups=len({r["source_group"] for r in paired}),
        paired_full_records=len(paired),
        full_ai_MAE_points=(
            float(
                np.mean(
                    [
                        abs(r["full_ai_score"] - r["direct_total"])
                        for r in paired
                    ]
                )
            )
            if paired
            else None
        ),
        full_rule_MAE_points=(
            float(
                np.mean(
                    [
                        abs(r["full_rule_score"] - r["direct_total"])
                        for r in paired
                    ]
                )
            )
            if paired
            else None
        ),
    )
    if audit.digest(direct_path) != direct_hash:
        raise ValueError("Direct drafts changed")
    audit.save_json(target, report)
    return report


def render_result() -> pathlib.Path:
    """동결 결과와 구간별 관절 영상·초안 근거를 HTML에 표시한다.

    Returns:
        pathlib.Path: 결과 HTML 절대 경로. 총점과 부분 사건을 구별한다.

    Raises:
        OSError: 결과·캐시·HTML 접근 실패.
        ValueError: 관절 영상 디코딩 실패.
        RuntimeError: 자원 보호 실패.
    """
    report = audit.read_json(audit.OUT / "autonomous_prospective_results.json")
    direct = {
        r["id"]: r
        for r in audit.read_json(
            audit.OUT / "autonomous_prospective_direct.json"
        )["reviews"]
    }
    esc = html.escape
    parts = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>새 출처 동결 모델 확인</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:auto;padding:24px;background:#14212c;color:#eef5ff}section{padding:18px;margin:18px 0;background:#243a4b;border-radius:12px}a{color:#8cdfef}td,th{padding:9px;border-bottom:1px solid #678}video{max-width:100%;width:850px}</style><h1>사전 예약 새 출처의 동결 모델 확인</h1><p><a href="unified_status.html">통합 현황</a> · <a href="autonomous_prospective_results.json">모든 결과</a> · <a href="autonomous_prospective_freeze.json">모델·규칙 동결</a></p><p>새 3출처를 학습에 사용하지 않고 모델·규칙을 먼저 고정했습니다. 기본 플랭크 총점 구간 두 개는 같은 사람·원본 하나에 속하므로 두 독립 참여자로 세지 않습니다. 나머지는 의도적 지지 변경 변형의 부분 사건입니다. 비전문가 직접 초안과의 소규모 전향 확인이며 전문가 검증·5종 최종 시험 완료가 아닙니다.</p>'
    ]
    decoded = {}
    parts.append(
        "<p>위 승인 총점은 동결한 총점 모델·규칙의 결과입니다. 아래 항목 AI·규칙은 "
        "필수 관절과 측면을 항목별로 선택하는 별도 실험판이며, 승인 총점의 "
        "분해값이 아닙니다. 합산하여 위 총점을 재구성하지 마세요. "
        '<a href="autonomous_failure_analysis.json">사후 오류·측면 선택 차이 분석</a>'
        "</p>"
    )
    for row in report["records"]:
        draft = direct[row["id"]]
        parts.append(
            f'<section><h2>{esc(row["id"])}</h2><p>{esc(draft["note_ko"])}</p><p>직접 총점 {row["direct_total"]} · 승인 AI {row["full_ai_score"]} · 승인 규칙 {row["full_rule_score"]}</p><table><tr><th>항목</th><th>직접 초안</th><th>항목 AI</th><th>항목 규칙</th></tr>'
        )
        for item in row["items"]:
            parts.append(
                f'<tr><td>{esc(item["criterion_id"])}</td><td>{item["draft_grade"]}</td><td>{item["ai_grade"]}</td><td>{item["rule_grade"]}</td></tr>'
            )
        folder = audit.OUT / "autonomous_prospective_overlays" / row["id"]
        folder.mkdir(exist_ok=True, parents=True)
        shutil.copyfile(row["extraction"]["cache"], folder / "side.npz")
        video = render.overlay(
            dict(id=row["id"], exercise="plank", video_path=row["video_path"]),
            dict(
                cache_folder=str(folder),
                ai_score=row["full_ai_score"],
                rule_score=row["full_rule_score"],
            ),
        )
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        count = 0
        while True:
            ok, pixels = cap.read()
            if not ok:
                break
            if count == 15:
                cv2.imwrite(str(folder / "preview.jpg"), pixels)
            count += 1
        cap.release()
        if count < 5:
            raise ValueError("Prospective overlay cannot be decoded")
        decoded[row["id"]] = count
        parts.append(
            f'</table><video controls preload="metadata" poster="{(folder/"preview.jpg").relative_to(audit.OUT)}" src="{video.relative_to(audit.OUT)}"></video><p><a href="{esc(draft["source_url"])}">{esc(draft["creator"])}</a> · CC BY 표기 확인, 구간 분리·크롭·무음·관절 표시 적용.</p></section>'
        )
    parts.append("</html>")
    page = audit.OUT / "autonomous_prospective.html"
    page.write_text("".join(parts), encoding="utf-8")
    audit.save_json(
        audit.OUT / "autonomous_prospective_render_checks.json",
        dict(decoded_overlay_frames=decoded, local_browser_visual_check=False),
    )
    return page


if __name__ == "__main__":
    run()
    print(render_result())
