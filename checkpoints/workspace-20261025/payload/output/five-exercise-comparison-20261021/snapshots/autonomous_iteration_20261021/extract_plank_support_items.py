"""직접 표시한 팔꿈치 이탈·운동 종료 구간을 Heavy로 추출한다."""

import numpy as np

import audit
from extract import extract_view
from plank_item_observation import assess


def run() -> dict:
    """초안을 동결한 세 구간의 항목별 관측·캐시 계보를 저장한다.

    Returns:
        dict: 관절 추출·항목별 관측·원본 연결 결과. 종료 사건을
        잘못된 자세나 총점 오류 라벨로 바꾸지 않는다.

    Raises:
        OSError: 직접 초안·원본·캐시 접근 실패.
        ValueError: 원본·직접 초안 해시 또는 ID 연결 불일치.
        RuntimeError: 추출 또는 자원 보호 실패.
    """
    path = audit.OUT / "plank_support_enrichment_direct.json"
    frozen = audit.digest(path)
    direct = {r["id"]: r for r in audit.read_json(path)["reviews"]}
    records = []
    for export in audit.read_json(
        audit.OUT / "plank_support_enrichment_plan_exports.json"
    )["records"]:
        draft = direct[export["id"]]
        row = dict(
            id=export["id"],
            exercise="plank",
            video_sha256=export["video_sha256"],
            required_views=["item_only_review"],
            segment=dict(
                start_frame=0, end_frame=export["frames"] - 1, fps=export["fps"]
            ),
        )
        extraction = extract_view(
            row,
            "primary",
            export["video_path"],
            audit.OUT / "plank_support_cache",
        )
        with np.load(extraction["cache"]) as cache:
            seq = cache["image_xy"].copy()
        records.append(
            dict(
                id=row["id"],
                extraction=extraction,
                observation=assess(
                    seq,
                    export["frames"] / export["fps"],
                    [v is not None for v in draft["draft_levels"]],
                ),
                total_score=None,
                source_group=draft["source_group"],
                person_group=draft["person_group"],
                kind=draft["kind"],
            )
        )
    if frozen != audit.digest(path):
        raise ValueError("Direct drafts changed during extraction")
    report = dict(
        records=records,
        direct_sha256=frozen,
        independent_final_test=False,
        expert_validated=False,
        full_score_labels_created=0,
    )
    audit.save_json(audit.OUT / "plank_support_extraction.json", report)
    return report


if __name__ == "__main__":
    print([(r["id"], r["observation"]) for r in run()["records"]])
