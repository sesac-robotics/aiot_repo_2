"""직접 구간과 일치한 W 미수행 후보 하나와 전역 오판 위험을 웹에 기록한다."""

import html
import json
import pathlib
import re

import cv2
import numpy as np

import audit
import render
from allfive_round1 import OUT
from recovery_finalize import LocalLinks
from w_observed_absence_trial import DEST


def run() -> dict:
    """추가 관절 영상과 자동 진단을 기존 5종 후보 웹에 연결한다.

    Returns:
        dict: 새 승인 특징 수·기존 정상 초안 충돌 수·영상·파일 검사.
        기존 미검토 진단을 직접 검토나 새 정답 라벨로 바꾸지 않는다.

    Raises:
        ValueError: 입력·코드·모델·영상·직접 시점 연결 지문 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
        OSError: 원본·표시 결과·감사 자료 저장 실패.
    """
    trial = audit.read_json(DEST / "comparison.json")
    verification = audit.read_json(OUT / "report_verification.json")
    if not trial["completed"] or trial[
        "candidate_input_sha256"
    ] != audit.digest(OUT / "candidate_input_manifest.json"):
        raise ValueError("Trial input changed")
    if trial["model_sha256"] != audit.digest(DEST / "model.joblib") or trial[
        "code_sha256"
    ] != audit.digest(
        pathlib.Path(__file__).with_name("w_observed_absence_trial.py")
    ):
        raise ValueError("Trial code/model changed")
    accepted = [
        r
        for r in trial["observed_absence_candidates"]
        if r["accepted_as_feature_not_label"]
    ]
    conflicts = [
        r
        for r in trial["observed_absence_candidates"]
        if r["reference_W"] == 2 and r["measured_shape_grade"] < 2
    ]
    videos = []
    annotations = {
        r["id"]: r for r in audit.read_json(OUT / "score_annotations.json")
    }
    receipts = {
        r["id"]: r["extraction"]
        for r in audit.read_json(OUT / "extraction.json")["records"]
    }
    predictions = {r["id"]: r for r in trial["group_cv_predictions"]}
    for item in accepted:
        identifier = item["id"]
        ann = annotations[identifier]
        rec = receipts[identifier]
        folder = DEST / "previews" / identifier
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(rec["cache"]) as data:
            np.savez_compressed(
                folder / "primary.npz",
                **{k: data[k].copy() for k in data.files}
            )
        first, last = item["T_window"]
        details = dict(
            evidence=dict(
                phase_detection=dict(
                    phases=dict(
                        T_attempt=dict(start_index=first, end_index=last),
                        observed_return_without_W=item["window"],
                    )
                ),
                w_shape=dict(
                    morphology=dict(grade=item["measured_shape_grade"])
                ),
            )
        )
        video = render.overlay(
            ann,
            dict(
                predictions[identifier],
                cache_folder=str(folder),
                display_unit="/100",
                rule_details=details,
            ),
        )
        cap = cv2.VideoCapture(str(video))
        expected = int(cap.get(7))
        decoded = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if decoded == item["window"]["anchor_index"]:
                    cv2.imwrite(str(folder / "preview.jpg"), frame)
                decoded += 1
        finally:
            cap.release()
        if not decoded or decoded != expected:
            raise ValueError("W preview decoding failed")
        videos.append(
            dict(
                exercise="ytw",
                id=identifier + "_observed_absence_trial",
                path=str(video),
                sha256=audit.digest(video),
                decoded_frames=decoded,
                display_prediction="connected_group_out_of_fold_not_training_fit",
            )
        )
    qualification = dict(
        automatic_candidates=len(trial["observed_absence_candidates"]),
        new_direct_window_and_tracking_approved_features=len(accepted),
        existing_normal_draft_conflicts=len(conflicts),
        existing_conflicts_automatically_adopted=False,
        score_labels_created=0,
        full_posture_approval=False,
        common_rule_promoted=False,
    )
    audit.save_json(DEST / "qualification.json", qualification)
    content = (
        '<section id="observed-W-return"><h2>W 미수행 관측 후보 — 구간별 제한 검증</h2><p>T 뒤 충분히 관측된 팔 내림과 중간 W 손·굽힘 근거의 부재를 확인하는 별도 후보입니다. 가림이나 단순 인식 실패를 0점으로 바꾸지 않습니다. 새 직접 시점·관절 검토와 일치한 1구간의 기존 W 0등급 라벨에 관측 특징을 연결했습니다. 규칙 결과를 새 라벨로 복사하지 않았습니다.</p><p>자동 진단 11개 중 기존 정상 W 초안과 충돌하는 2개도 발견했습니다. 이 진단들은 이번에 직접 재검토·학습 승인하지 않았고 공통 규칙에 반영하지 않았습니다. 규칙의 전역 적용은 아직 부적절합니다.</p><pre>'
        + html.escape(
            json.dumps(
                dict(
                    qualification=qualification,
                    fixed_development=trial["fixed_development"],
                    group_cv=trial["group_cv"],
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        + "</pre>"
    )
    for video in videos:
        content += (
            '<video controls preload="metadata" src="'
            + pathlib.Path(video["path"]).relative_to(OUT).as_posix()
            + '"></video>'
        )
    content += '<p>영상은 해당 연결 그룹을 학습에서 제외한 예측입니다. Y 등 미관측 항목이 남아 총점은 N/A이며, W 표시만 추가 근거로 확인합니다.</p><p><a href="w_observed_absence_trial/comparison.json">W 추가 실험 전체</a> · <a href="failure_analysis.json">5종 실패 분석</a> · <a href="phase_alignment_verification.json">새 YTW 단계·라벨 구간 대조</a></p></section>'
    index = (OUT / "index.html").read_text()
    start = "<!-- OBSERVED_W_START -->"
    end = "<!-- OBSERVED_W_END -->"
    index = re.sub(
        re.escape(start) + ".*?" + re.escape(end), "", index, flags=re.S
    )
    index += "\n" + start + content + end
    (OUT / "index.html").write_text(index, encoding="utf-8")
    parser = LocalLinks()
    parser.feed(index)
    missing = [p for p in parser.paths if not (OUT / p).exists()]
    if missing:
        raise ValueError("Postprocessed links missing")
    verification["videos"] = [
        v
        for v in verification["videos"]
        if not v["id"].endswith("_observed_absence_trial")
    ] + videos
    verification.update(
        local_links_checked=len(parser.paths),
        missing_local_links=missing,
        postprocessing=dict(
            script=pathlib.Path(__file__).name,
            script_sha256=audit.digest(pathlib.Path(__file__)),
            trial_comparison_sha256=audit.digest(DEST / "comparison.json"),
            index_sha256=audit.digest(OUT / "index.html"),
        ),
    )
    audit.save_json(OUT / "report_verification.json", verification)
    return dict(
        qualification=qualification,
        videos=len(verification["videos"]),
        links=len(parser.paths),
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
