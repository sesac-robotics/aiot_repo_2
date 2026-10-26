"""받을 원시 자료에서 구도·오류 상태를 검토할 이미지 경로를 선택한다.

선택에는 제공자 조건을 사용하지만 점수는 만들지 않는다. 각 플랭크 조건과
Y 정상 조건의 시간 표본을 고르며 이미지 직접 검토·학습 편입과 구분한다.
"""

import collections
import pathlib
import audit


def run() -> dict:
    """카메라별 처음·중간·끝 이미지 경로와 검토 목록을 저장한다.

    Returns:
        dict: 검토용 이미지 경로·라벨·조건·잠정 사람 토큰. 시간 단위는
        파일명 프레임 번호이며 실제 FPS와 초 단위 간격은 미확인이다.

    Raises:
        OSError: 라벨 인벤토리 읽기·결과 저장 실패.
        KeyError: 필수 경로·조건이 없는 경우.
    """
    folder = audit.OUT / "aihub_documents"
    source = audit.read_json(folder / "validation_body_inventory.json")
    selections = []
    for row in source["records"]:
        normal = all(c["value"] is True for c in row["conditions"])
        if row["exercise"] != "플랭크" and not (
            row["exercise"] == "Y - Exercise" and normal
        ):
            continue
        cameras = collections.defaultdict(list)
        for image in row["images"]:
            camera = pathlib.PurePosixPath(image["path"]).parent.name.rsplit(
                "_", 1
            )[-1]
            cameras[camera].append(image["path"])
        chosen = []
        for camera, paths in sorted(cameras.items()):
            paths = sorted(set(paths))
            chosen.extend(
                paths[i] for i in sorted({0, len(paths) // 2, len(paths) - 1})
            )
        selections.append(
            dict(
                label=row["label_member"],
                exercise=row["exercise"],
                person_tokens=row["person_tokens"],
                conditions=row["conditions"],
                provider_all_conditions_true=normal,
                images=chosen,
                direct_reviewed=False,
                training_usable=False,
            )
        )
    report = dict(
        selections=selections,
        selected_image_count=sum(len(r["images"]) for r in selections),
        score_labels_created=0,
        images_seen=0,
        raw_image_presence_confirmed=False,
        extraction_budget_bytes=1024**3,
        purpose_ko="플랭크 모든 제공자 조건의 구도·시간 표본과 Y 정상 조건의 구도 확인 후보. 이미지 미확인.",
    )
    audit.save_json(folder / "raw_review_selection.json", report)
    return report


if __name__ == "__main__":
    report = run()
    print(
        dict(
            records=len(report["selections"]),
            images=report["selected_image_count"],
        )
    )
