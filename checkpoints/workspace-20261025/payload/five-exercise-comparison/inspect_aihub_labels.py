"""AI-Hub Validation 맨몸 라벨의 종목·상태·영상 연결 후보를 조사한다.

제공자 상태와 관절은 메타데이터로만 보존한다. 직접 자세 검토·Heavy 추출·
점수 라벨 생성은 하지 않으며 원시 이미지의 존재도 아직 보장하지 않는다.
"""

import collections
import pathlib
import re
import zipfile
import json
import audit


def run() -> dict:
    """완료된 라벨 ZIP을 읽고 원본 이미지 선택 목록을 저장한다.

    Returns:
        dict: 운동별 2D 라벨 수, 잠정 참여자 토큰·카메라·조건 패턴.
        프레임 번호는 파일명 숫자이며 FPS·실제 연속 시간은 미확인이다.

    Raises:
        OSError: 입력·결과 접근 실패.
        zipfile.BadZipFile: ZIP 또는 멤버 손상.
        json.JSONDecodeError: JSON 구문 오류.
        ValueError: 이미지 경로·라벨 구조가 예상과 다른 경우.
    """
    folder = audit.OUT / "aihub_documents"
    source = folder / "validation_body_01.zip"
    records = []
    dimensions = collections.Counter()
    with zipfile.ZipFile(source) as archive:
        for member in archive.namelist():
            if not member.lower().endswith(".json"):
                continue
            if member.endswith("-3d.json"):
                dimensions["3D"] += 1
                continue
            dimensions["2D"] += 1
            data = json.loads(archive.read(member))
            info = data["type_info"]
            images = []
            persons, cameras, frame_numbers = set(), set(), set()
            frame_view_counts = []
            for frame in data["frames"]:
                count = 0
                for view, content in frame.items():
                    if not isinstance(content, dict) or not content.get(
                        "img_key"
                    ):
                        continue
                    key = content["img_key"]
                    parts = pathlib.PurePosixPath(key).parts
                    match = re.fullmatch(
                        r"(.+)_([A-Za-z])-([0-9]+)\.jpg", parts[-1]
                    )
                    if match is None or len(parts) < 5:
                        raise ValueError("Unexpected image key: " + key)
                    persons.add(match[1].split("-")[-1])
                    cameras.add(match[2])
                    frame_numbers.add(int(match[3]))
                    images.append(
                        dict(path=key, view=view, active=content.get("active"))
                    )
                    count += 1
                frame_view_counts.append(count)
            records.append(
                dict(
                    label_member=member,
                    exercise=info["exercise"],
                    posture=info.get("pose"),
                    provider_type=data["type"],
                    conditions=info.get("conditions", []),
                    description=info.get("description"),
                    person_tokens=sorted(persons),
                    camera_tokens=sorted(cameras),
                    frame_records=len(data["frames"]),
                    frame_view_counts=frame_view_counts,
                    image_references=len(images),
                    unique_frame_numbers=sorted(frame_numbers),
                    images=images,
                    raw_image_presence_confirmed=False,
                    direct_visual_review=False,
                    training_usable=False,
                    original_provider_split="Validation",
                    final_independent_test_assignment=None,
                )
            )
    summaries = []
    for name in sorted({r["exercise"] for r in records}):
        rows = [r for r in records if r["exercise"] == name]
        patterns = collections.Counter(
            tuple((c["condition"], c["value"]) for c in r["conditions"])
            for r in rows
        )
        summaries.append(
            dict(
                exercise=name,
                label_records=len(rows),
                provisional_person_tokens=sorted(
                    {p for r in rows for p in r["person_tokens"]}
                ),
                distinct_people_verified=False,
                camera_tokens=sorted(
                    {p for r in rows for p in r["camera_tokens"]}
                ),
                image_references=sum(r["image_references"] for r in rows),
                frame_record_counts=sorted({r["frame_records"] for r in rows}),
                provider_condition_patterns=[
                    dict(
                        conditions=[
                            dict(condition=k, value=v) for k, v in pattern
                        ],
                        label_records=count,
                    )
                    for pattern, count in patterns.items()
                ],
            )
        )
    report = dict(
        source=str(source),
        source_sha256=audit.digest(source),
        label_counts=dict(dimensions),
        exercises=summaries,
        records=records,
        score_labels_created=0,
        Heavy_caches_created=0,
        directly_reviewed_videos=0,
        metadata_only=True,
        raw_archive_member_mapping_confirmed=False,
        candidate_raw_archive=dict(
            path="013.피트니스자세/2.Validation/원시데이터/valid_body_data.tar",
            file_key=69365,
            bytes=28184729600,
            basis="공식 목록의 같은 Validation 맨몸 자료. 내부 멤버는 미확인.",
        ),
        fps_confirmed=False,
        camera_directions_confirmed=False,
        split_policy_ko="제공자 Validation 명칭을 독립 최종 시험으로 그대로 쓰지 않음. 참여자·원본·반복·다중 구도를 함께 묶고 기존 Training 라벨과 중복 확인 후 내부 분할 결정.",
    )
    audit.save_json(folder / "validation_body_inventory.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(report["label_counts"])
    for row in report["exercises"]:
        print(
            row["exercise"],
            row["label_records"],
            row["provisional_person_tokens"],
            row["camera_tokens"],
            row["frame_record_counts"],
        )
