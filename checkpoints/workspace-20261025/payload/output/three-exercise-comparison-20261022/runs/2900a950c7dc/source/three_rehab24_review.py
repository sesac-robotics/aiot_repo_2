"""새 동기화 스쿼트 자료에서 원본 참여자별 검토 구간을 사전 선정한다.

제공자 정상·오류는 선정에만 사용한다. 9명 전체는 개발 자료이며 새 최종
시험으로 부르지 않는다. REHAB26과 실제 인물 연결이 미확인이라 혼합 학습하지 않는다.
"""

import csv
import pathlib

import audit
import external_review
import three_protocol
from resources import check_resources


def run() -> dict:
    """참여자마다 최대 세 구간의 동기 검토판을 만든다. 오류가 적으면 정상 후보로 보충한다.

    Returns:
        dict: 원본·양 시점·공유 프레임·검토판·선정 이유 목록.
        점수 라벨과 직접 검토 판정은 생성하지 않는다.

    Raises:
        OSError: 원본·CSV·검토판 접근 실패.
        ValueError: 대응 영상 또는 충분한 전면 구간이 없는 경우.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    target = root / "rehab24_review_preparation.json"
    if target.exists():
        return audit.read_json(target)
    with (root / "Segmentation.csv").open() as file:
        rows = [
            r
            for r in csv.DictReader(file, delimiter=";")
            if r["exercise_id"] == "6" and r["cam17_orientation"] == "front"
        ]
    selected = []
    for person in sorted({r["person_id"] for r in rows}, key=int):
        candidates = [r for r in rows if r["person_id"] == person]
        good = [r for r in candidates if r["correctness"] == "1"]
        bad = [r for r in candidates if r["correctness"] == "0"]
        if not good:
            raise ValueError("No verified frontal reference candidates")
        chosen = [good[len(good) // 2]]
        for row in [*bad[:1], *bad[-1:], *good]:
            if row not in chosen and len(chosen) < 3:
                chosen.append(row)
        selected.extend(chosen)
    folder = root / "review/rehab24"
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for item in selected:
        check_resources()
        id = f"rehab24_{item['video_id']}_rep{item['repetition_number']}"
        video_root = audit.BASE / "data/rehab24-squat/Ex6"
        front = video_root / (item["video_id"] + "-Camera17-30fps.mp4")
        side = video_root / (
            item["video_id"] + "-Camera18-30fps-transposed.mp4"
        )
        sheets = {}
        for view, path in [("side", side), ("front", front)]:
            sheets[view] = external_review.contact_sheet(
                path,
                folder / f"{id}_{view}.jpg",
                int(item["first_frame"]) / 30,
                int(item["last_frame"]) / 30,
                24,
            )
        results.append(
            dict(
                id=id,
                exercise="squats",
                person="rehab24_" + item["person_id"],
                split="train",
                evaluation_role="new_development_not_independent_final",
                dataset="REHAB24",
                video_path=str(side),
                front_path=str(front),
                video_sha256=sheets["side"]["video_sha256"],
                front_sha256=sheets["front"]["video_sha256"],
                segment=dict(
                    start_frame=int(item["first_frame"]),
                    end_frame=int(item["last_frame"]),
                    fps=30.0,
                ),
                required_input_views=["side", "synchronized_front"],
                source_labels=["correctness=" + item["correctness"]],
                provider_annotation=item,
                sheets=sheets,
                direct_review=False,
                expert_validated=False,
            )
        )
        audit.save_json(
            root / "rehab24_review_checkpoint.json",
            dict(records=results, completed=False),
        )
    report = dict(
        records=results,
        clips_prepared=len(results),
        temporal_sheets=len(results) * 2,
        directly_reviewed=0,
        labels_created=0,
        provider_error_candidates=sum(
            r["provider_annotation"]["correctness"] == "0" for r in results
        ),
        requested_error_quota_not_available_for_all_people=True,
        new_development_person_groups=9,
        cross_dataset_identity_linkage_verified=False,
        REHAB24_and_REHAB26_fit_together=False,
        independent_final_exam=False,
    )
    audit.save_json(target, report)
    return report


if __name__ == "__main__":
    print(run()["clips_prepared"])
