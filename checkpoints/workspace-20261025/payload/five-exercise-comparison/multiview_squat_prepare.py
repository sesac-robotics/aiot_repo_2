"""보유 MultiView 스쿼트의 학습·개발 검증 시도와 시점별 검토판을 준비한다."""

import json
import pathlib
import time

import audit
import external_review
from recovery_prepare import ROOT

OUT = ROOT / "multiview_squat"


def run() -> dict:
    """기존 역할이 확인된 4개 메타 참여자의 원본과 첫 8초를 준비한다.

    Returns:
        dict: 원본 8묶음·각 2시점 검토판의 자동 준비 수. 동일 시도·동기화·
        점수 라벨은 생성하지 않는다. test·미확정 역할은 사용하지 않는다.

    Raises:
        OSError: 보유 목록·원본·검토판 저장 실패.
        ValueError: 기존 역할·원본 연결 불일치.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    OUT.mkdir(exist_ok=True)
    (OUT / "review").mkdir(exist_ok=True)
    inventory = audit.read_json(ROOT / "held_multiview_squat_inventory.json")[
        "records"
    ]
    canonical = {}
    for row in audit.read_json(
        audit.BASE
        / "output/three-exercise-comparison-20261022/score_annotations.json"
    ):
        if row["id"].startswith("subject_"):
            subject = row["id"].split("_")[1]
            canonical.setdefault(subject, row)
            if (
                canonical[subject]["person"] != row["person"]
                or canonical[subject]["split"] != row["split"]
            ):
                raise ValueError("Canonical MultiView role differs")
    index = {r["id"]: r for r in inventory}
    path = OUT / "preparation.json"
    existing = audit.read_json(path) if path.exists() else dict(records=[])
    ready = {r["id"]: r for r in existing["records"]}
    records = []
    for subject in ("001", "005", "002", "004"):
        for quality in ("good", "bad"):
            identifier = f"subject_{subject}_squat_{quality}"
            parent = canonical[subject]
            if parent["split"] not in ("train", "validation"):
                raise ValueError(
                    "Only existing development roles are permitted"
                )
            row = dict(
                id=identifier,
                exercise="squats",
                person=parent["person"],
                worker_id=parent["worker_id"],
                split=parent["split"],
                provider_quality=quality,
                provider_strings_not_scores=True,
                views={},
                directly_reviewed=False,
                synchronization_verified=False,
            )
            old = ready.get(identifier, {"views": {}})
            for view in ("side", "front"):
                source = index[identifier + "_" + view]
                if source["existing_role"] != parent["split"]:
                    raise ValueError("Source role changed")
                receipt = old["views"].get(view)
                if receipt is None:
                    receipt = external_review.contact_sheet(
                        pathlib.Path(source["path"]),
                        OUT / "review" / (identifier + "_" + view + ".jpg"),
                        0,
                        8,
                        24,
                    )
                    time.sleep(0.5)
                elif receipt["video_sha256"] != audit.digest(
                    pathlib.Path(source["path"])
                ):
                    raise ValueError("Reviewed source changed")
                row["views"][view] = receipt
            records.append(row)
            audit.save_json(
                path,
                dict(
                    records=records,
                    completed=False,
                    automatic_preparation_not_direct_review=True,
                ),
            )
    audit.save_json(
        path,
        dict(
            records=records,
            completed=True,
            automatic_preparation_not_direct_review=True,
        ),
    )
    return dict(
        recording_candidates=len(records),
        view_sheets=sum(len(r["views"]) for r in records),
        score_labels_created=0,
    )


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False))
