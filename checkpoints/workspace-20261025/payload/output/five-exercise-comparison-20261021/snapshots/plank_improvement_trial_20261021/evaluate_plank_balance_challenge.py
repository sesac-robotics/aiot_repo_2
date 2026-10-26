"""학습에 넣지 않은 신규 출처에서 고정 AI·규칙 실험안을 비교한다."""

import joblib
import numpy as np

import audit
import benchmark
from extract import extract_view
import plank_rule_trial
import rule_score


def run() -> dict:
    """정상 측면·사선 전환 구간을 추출하고 관측 가능한 항목만 비교한다.

    Returns:
        dict: 새 2개 구간의 관절 관측·세 AI·두 규칙 결과. 직접 초안이
        없는 항목과 사선 총점은 None으로 유지한다. 개발 후보 확인이며
        독립 최종 시험·정확도 보증이 아니다. 새 자료는 학습에 넣지 않는다.

    Raises:
        OSError: 원본·캐시·모델 접근 실패.
        ValueError: 원본/참여자 누출·입력 특징·동결 초안 불일치.
        RuntimeError: 자원 보호 또는 관절 추출 실패.
    """
    direct_path = audit.OUT / "plank_balance_challenge_direct.json"
    frozen = audit.digest(direct_path)
    direct = audit.read_json(direct_path)
    drafts = {r["id"]: r for r in direct["reviews"]}
    train_ids = {
        r["id"]
        for r in audit.read_json(audit.OUT / "comparison.json")["results"][
            "plank"
        ]["rule_only_previews"]
    }
    train = [
        r
        for r in audit.read_json(audit.OUT / "labels.json")
        if r["id"] in train_ids
    ]
    hashes = {r.get("source_original_sha256", r["video_sha256"]) for r in train}
    people = {r["person"] for r in train}
    results = []
    for export in audit.read_json(
        audit.OUT / "plank_balance_challenge_plan_exports.json"
    )["records"]:
        if (
            export["source_sha256"] in hashes
            or "youtube_channel_" + export["person_group"] in people
        ):
            raise ValueError("New source/person overlaps fitting groups")
        draft = drafts[export["id"]]
        row = dict(
            id=export["id"],
            exercise="plank",
            video_sha256=export["video_sha256"],
            required_views=[
                (
                    "side"
                    if export["id"] == "amanda_challenge"
                    else "oblique_item_only"
                )
            ],
            segment=dict(
                start_frame=0, end_frame=export["frames"] - 1, fps=export["fps"]
            ),
        )
        extraction = extract_view(row, "side", export["video_path"])
        seq, front, holds = benchmark.load_input(row)
        x, names = benchmark.input_features(seq, front)
        methods = {}
        approved_total = draft["total_score"] is not None and not holds
        for name in ["unweighted_leaf2", "balanced_leaf2", "balanced_leaf1"]:
            bundle = joblib.load(
                audit.OUT / "plank_balance_trial" / f"{name}.joblib"
            )
            if (
                names != bundle["names"]
                or set(bundle["train_ids"]) != train_ids
            ):
                raise ValueError("Model feature or fitting cohort mismatch")
            levels = np.clip(bundle["model"].predict([x])[0], 0, 2)
            masked = [
                float(p) if d is not None else None
                for p, d in zip(levels, draft["draft_levels"], strict=True)
            ]
            methods[name] = dict(
                levels=masked,
                score=(
                    float(np.dot(levels, [40, 25, 20, 15]) / 2)
                    if approved_total
                    else None
                ),
            )
        for name, module in [
            ("original_rule", rule_score),
            ("trial_rule", plank_rule_trial),
        ]:
            rule = (
                module.evaluate(
                    "plank",
                    seq,
                    None,
                    direct["criterion_ids"],
                    direct["weights"],
                )
                if name == "original_rule"
                else module.evaluate(seq)
            )
            methods[name] = dict(
                levels=[
                    p if d is not None else None
                    for p, d in zip(
                        rule["levels"], draft["draft_levels"], strict=True
                    )
                ],
                score=rule["score"] if approved_total else None,
                evidence=rule["evidence"],
            )
        results.append(
            dict(
                id=row["id"],
                extraction=extraction,
                common_joint_holds=holds,
                full_score_approved=approved_total,
                view_contract="side" if approved_total else "oblique_item_only",
                direct_levels=draft["draft_levels"],
                direct_total=draft["total_score"],
                methods=methods,
            )
        )
    if frozen != audit.digest(direct_path):
        raise ValueError("Direct drafts changed")
    report = dict(
        results=results,
        direct_labels_sha256=frozen,
        train_source_overlap=False,
        independent_final_test=False,
        expert_validated=False,
        adopted=False,
        new_sources_used_for_fitting=False,
        common_rule_modified=False,
        rule_trial_sha256=audit.digest(
            audit.BASE / "five-exercise-comparison/plank_rule_trial.py"
        ),
    )
    report["input_adjustment"] = (
        audit.read_json(audit.OUT / "plank_challenge_input_adjustment.json")
        if (audit.OUT / "plank_challenge_input_adjustment.json").exists()
        else None
    )
    audit.save_json(audit.OUT / "plank_balance_challenge_results.json", report)
    return report


if __name__ == "__main__":
    print(
        [
            (r["id"], r["common_joint_holds"], r["methods"])
            for r in run()["results"]
        ]
    )
