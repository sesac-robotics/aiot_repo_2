"""3종의 기존 직접 라벨을 QEVD JSON으로 내보내고 새 검토 후보를 준비한다.

기존 시험 자료는 노출된 과거 자료로 기록한다. 새 후보 선택은 사람·원본을
보존하며 제공자 문자열은 후보 균형에만 사용하고 점수는 생성하지 않는다.
"""

import collections
import hashlib
import pathlib

import cv2

import audit
import external_review
import three_protocol
from resources import check_resources


def export_existing() -> dict:
    """3종 기존 라벨을 기본 네 필드 JSON과 점수 확장 JSON으로 저장한다.

    Returns:
        dict: 운동별 초안 수와 QEVD 원본·참여자 레지스트리 해시.
        원래 labels_descriptive를 제공자 원문으로 복원하며 검토 메모는
        assessment.note_ko로 보존한다. 변환 자체는 직접 재검토가 아니다.

    Raises:
        OSError: 원본 JSON 읽기·결과 쓰기 실패.
        ValueError: 레코드 계약·참여자 연결 불일치.
    """
    root = three_protocol.OUT
    qevd = audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
    originals = {
        pathlib.Path(r["video_path"]).stem: r for r in audit.read_json(qevd)
    }
    rows = [
        r
        for r in audit.read_json(audit.OUT / "labels.json")
        if r["exercise"] in three_protocol.EXERCISES
    ]
    persons = sorted(
        {r["person"] for r in rows if not r["person"].startswith("qevd_")}
    )
    worker_map = {p: 1000000 + index for index, p in enumerate(persons)}
    exported, basic, workers = [], [], []
    for row in rows:
        dataset = (
            "QEVD"
            if row["person"].startswith("qevd_")
            else (
                "REHAB26" if row["person"].startswith("rehab_") else "MultiView"
            )
        )
        record = three_protocol.convert(row, dataset=dataset)
        record["assessment"]["note_ko"] = row.get("note_ko")
        if dataset == "QEVD":
            original = originals[pathlib.Path(row["video_path"]).stem]
            record["labels"] = original["labels"]
            record["labels_descriptive"] = original["labels_descriptive"]
        else:
            record["worker_id"] = worker_map[row["person"]]
            record["provenance"][
                "worker_id_assignment"
            ] = "local_external_namespace_not_provider_QEVD_worker"
            record["labels_descriptive"] = []
        record["evaluation_role"] = (
            "historically_exposed_test_not_new_final"
            if row["split"] == "test"
            else "development_" + row["split"]
        )
        three_protocol.validate_record(record)
        exported.append(record)
        basic.append(
            {
                key: record[key]
                for key in (
                    "video_path",
                    "labels",
                    "labels_descriptive",
                    "split",
                )
            }
        )
        workers.append(
            {
                "video_path": record["video_path"],
                "worker_id": record["worker_id"],
            }
        )
    audit.save_json(root / "fine_grained_labels.json", basic)
    audit.save_json(root / "fine_grained_labels_with_worker_ids.json", workers)
    audit.save_json(root / "score_annotations.json", exported)
    audit.save_json(root / "external_worker_registry.json", worker_map)
    report = dict(
        records=len(rows),
        by_exercise=dict(collections.Counter(r["exercise"] for r in rows)),
        original_QEVD_json_sha256=audit.digest(qevd),
        all_converted_records_valid=True,
        conversion_is_new_direct_review=False,
        native_base_keys=[
            "video_path",
            "labels",
            "labels_descriptive",
            "split",
        ],
        score_extension="assessment",
        no_directory_moves=True,
    )
    audit.save_json(root / "json_conversion_verification.json", report)
    return report


def prepare() -> dict:
    """미라벨 푸시업 후보와 숄더 프레스 기존 시험 후보의 검토판을 만든다.

    Returns:
        dict: 후보·검토판·기존 노출 여부. 자동 생성은 직접 검토가 아니다.

    Raises:
        OSError: 영상·JSON 접근 실패.
        ValueError: 영상 표본 준비 또는 메타데이터 실패.
        RuntimeError: 자원 보호 조건 위반.
    """
    root = three_protocol.OUT
    target = root / "candidate_preparation.json"
    if target.exists():
        return audit.read_json(target)
    labels = audit.read_json(audit.OUT / "labels.json")
    known = {r["id"] for r in labels}
    known_people = {r["person"] for r in labels}
    pool = audit.read_json(
        audit.BASE / "output/overnight-20261020/score_candidates.json"
    )
    selected = [
        r
        for r in pool
        if r["id"] not in known
        and r["exercise"] == "pushups"
        and r["id"].startswith("subject_")
        and r["id"].endswith("_side")
    ]
    selected += [
        r
        for r in pool
        if r["exercise"] == "shoulder"
        and r["split"] == "test"
        and r["id"].endswith("_front")
        and r["id"] not in known
    ]
    # QEVD는 미노출 사람만 12명 선정한다. 원래 시험·검증 참여자는 이동하지 않는다.
    for split, count in [("train", 8), ("validation", 4)]:
        eligible = [
            r
            for r in pool
            if r["exercise"] == "pushups"
            and r["id"].startswith("qevd_")
            and r["split"] == split
            and r["person"] not in known_people
        ]
        eligible.sort(
            key=lambda r: hashlib.sha256(
                ("three-20261022" + r["id"]).encode()
            ).hexdigest()
        )
        picked = set()
        for term in [
            "no obvious issue",
            "butt",
            "knees",
            "range",
            "elbows",
            "no obvious issue",
            "butt",
            "knees",
        ]:
            if len(picked) >= count:
                break
            row = next(
                (
                    r
                    for r in eligible
                    if r["person"] not in picked
                    and any(term in s for s in r["source_labels"])
                ),
                None,
            )
            if row:
                selected.append(row)
                picked.add(row["person"])
    folder = root / "review"
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for row in selected:
        check_resources()
        image_path = folder / (row["id"] + ".jpg")
        metadata_path = image_path.with_suffix(".json")
        if image_path.exists() and metadata_path.exists():
            sheet = audit.read_json(metadata_path)
            if sheet["video_sha256"] != audit.digest(pathlib.Path(row["path"])):
                raise ValueError("Previously prepared source changed")
        else:
            capture = cv2.VideoCapture(
                row["path"], cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
            )
            try:
                duration = (capture.get(7) - 1) / capture.get(5)
            finally:
                capture.release()
            sheet = external_review.contact_sheet(
                pathlib.Path(row["path"]), image_path, 0, min(8.0, duration), 36
            )
        records.append(
            dict(
                row,
                video_path=row["path"],
                video_sha256=sheet["video_sha256"],
                sheet=sheet,
                participant_previously_labelled=row["person"] in known_people,
                independent_final_test=False,
                direct_review=False,
            )
        )
        audit.save_json(
            root / "candidate_preparation_checkpoint.json",
            dict(records=records, completed=False),
        )
    report = dict(
        records=records,
        automatically_prepared=len(records),
        direct_reviews=0,
        labels_created=0,
        no_test_to_train_moves=True,
    )
    audit.save_json(target, report)
    return report


if __name__ == "__main__":
    three_protocol.initialize()
    print(export_existing())
    print(prepare()["automatically_prepared"])
