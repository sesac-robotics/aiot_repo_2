"""AI-Hub Training 맨몸 중첩 ZIP의 종목과 Validation 중복을 조사한다.

제공자 조건은 점수로 변환하지 않는다. 압축 묶음 번호는 실제 원시 TAR와
대조하기 전까지 다운로드 후보로만 표시하고 직접 검토·학습 완료로 세지 않는다.
"""

import collections
import io
import json
import pathlib
import re
import zipfile
import audit
from resources import check_resources

TARGETS = {"플랭크", "푸시업", "Y - Exercise"}


def run() -> dict:
    """17개 내부 ZIP을 읽어 경로·참여자 토큰·조건·CRC 검사를 저장한다.

    Returns:
        dict: 묶음별 종목 수와 잠정 참여자, Validation 중복 경로·토큰.
        원시 이미지·시간 정보·실제 촬영 방향·점수 정확도는 미확인이다.

    Raises:
        OSError: 압축 또는 결과 파일 접근 실패.
        zipfile.BadZipFile: ZIP 구조·멤버 CRC 오류.
        json.JSONDecodeError: 라벨 JSON 오류.
        ValueError: 이미지 경로가 예상 형식과 다른 경우.
        RuntimeError: 자원 보호 조건 초과.
    """
    check_resources()
    folder = audit.OUT / "aihub_documents"
    source = folder / "training_body_new_220128.zip"
    validation = audit.read_json(folder / "validation_body_inventory.json")
    validation_people = {
        p for r in validation["records"] for p in r["person_tokens"]
    }
    validation_images = {
        im["path"] for r in validation["records"] for im in r["images"]
    }
    validation_labels = {r["label_member"] for r in validation["records"]}
    batches, records = [], []
    all_people, duplicate_images, duplicate_labels = set(), set(), set()
    for_duplicate_check = set()
    repeated_training_images = set()
    dimensions = collections.Counter()
    with zipfile.ZipFile(source) as outer:
        for info in outer.infolist():
            check_resources()
            if info.is_dir():
                continue
            if not info.filename.endswith(".zip"):
                raise ValueError("Unexpected Training member")
            nested = outer.read(
                info
            )  # 전체 내부 ZIP을 읽어 외부 멤버 CRC도 검사한다.
            batch_records = []
            with zipfile.ZipFile(io.BytesIO(nested)) as archive:
                for index, member in enumerate(archive.infolist()):
                    if index % 150 == 0:
                        check_resources()
                    if member.is_dir():
                        continue
                    raw = archive.read(
                        member
                    )  # 2D·3D 모든 파일의 CRC를 검사한다.
                    if not member.filename.endswith(".json"):
                        continue
                    if member.filename.endswith("-3d.json"):
                        dimensions["3D"] += 1
                        continue
                    dimensions["2D"] += 1
                    data = json.loads(raw)
                    metadata = data["type_info"]
                    people, cameras, paths = set(), set(), set()
                    sample = None
                    for frame in data["frames"]:
                        for view, content in frame.items():
                            if not isinstance(content, dict) or not content.get(
                                "img_key"
                            ):
                                continue
                            key = content["img_key"]
                            match = re.fullmatch(
                                r"(.+)_([A-Za-z])-([0-9]+)\.jpg",
                                pathlib.PurePosixPath(key).name,
                            )
                            if match is None:
                                raise ValueError("Unexpected image filename")
                            people.add(match[1].split("-")[-1])
                            cameras.add(match[2])
                            paths.add(key)
                            sample = sample or key
                    duplicate_images.update(paths & validation_images)
                    duplicate_labels.update(
                        {member.filename} & validation_labels
                    )
                    repeated_training_images.update(paths & for_duplicate_check)
                    for_duplicate_check.update(paths)
                    all_people.update(people)
                    row = dict(
                        container=info.filename,
                        label_member=member.filename,
                        exercise=metadata["exercise"],
                        posture=metadata.get("pose"),
                        provider_type=data["type"],
                        conditions=metadata.get("conditions", []),
                        person_tokens=sorted(people),
                        camera_tokens=sorted(cameras),
                        frame_records=len(data["frames"]),
                        unique_image_paths=len(paths),
                        example_image_path=sample,
                        original_provider_split="Training",
                        direct_visual_review=False,
                        training_usable=False,
                    )
                    records.append(row)
                    batch_records.append(row)
            exercises = []
            for name in sorted({r["exercise"] for r in batch_records}):
                rows = [r for r in batch_records if r["exercise"] == name]
                exercises.append(
                    dict(
                        exercise=name,
                        label_records=len(rows),
                        person_tokens=sorted(
                            {p for r in rows for p in r["person_tokens"]}
                        ),
                        provider_all_conditions_true=sum(
                            all(c["value"] is True for c in r["conditions"])
                            for r in rows
                        ),
                        provider_any_condition_false=sum(
                            any(c["value"] is False for c in r["conditions"])
                            for r in rows
                        ),
                    )
                )
            batches.append(
                dict(
                    container=info.filename,
                    exercises=exercises,
                    CRC_checked=True,
                    raw_archive_pairing_confirmed=False,
                )
            )
            print(info.filename, len(batch_records), flush=True)
    targets = []
    for name in sorted(TARGETS):
        rows = [r for r in records if r["exercise"] == name]
        targets.append(
            dict(
                exercise=name,
                label_records=len(rows),
                containers=sorted({r["container"] for r in rows}),
                provisional_person_tokens=sorted(
                    {p for r in rows for p in r["person_tokens"]}
                ),
                provider_all_conditions_true=sum(
                    all(c["value"] is True for c in r["conditions"])
                    for r in rows
                ),
                provider_any_condition_false=sum(
                    any(c["value"] is False for c in r["conditions"])
                    for r in rows
                ),
            )
        )
    report = dict(
        source=str(source),
        source_sha256=audit.digest(source),
        nested_archives=len(batches),
        label_counts=dict(dimensions),
        all_archive_member_CRC_checked=True,
        batches=batches,
        targets=targets,
        provisional_person_tokens=sorted(all_people),
        distinct_people_verified=False,
        validation_overlap=dict(
            person_tokens=sorted(all_people & validation_people),
            exact_image_path_count=len(duplicate_images),
            exact_image_path_examples=sorted(duplicate_images)[:20],
            exact_label_path_count=len(duplicate_labels),
            exact_label_path_examples=sorted(duplicate_labels)[:20],
            identity_and_visual_duplicates_still_to_check=True,
        ),
        repeated_training_image_path_count=len(repeated_training_images),
        records=records,
        images_seen=0,
        Heavy_extractions=0,
        score_labels_created=0,
        raw_training_archive_numbers_confirmed=False,
        split_policy_ko="참여자 토큰 중복은 같은 내부 분할로 묶음. 토큰 비중복도 실제 인물·재사용 이미지 비중복을 보장하지 않으므로 원시 대조 전 최종 시험으로 확정하지 않음.",
    )
    audit.save_json(folder / "training_body_inventory.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(
        dict(
            label_counts=result["label_counts"],
            targets=result["targets"],
            validation_overlap=result["validation_overlap"],
        ),
        flush=True,
    )
