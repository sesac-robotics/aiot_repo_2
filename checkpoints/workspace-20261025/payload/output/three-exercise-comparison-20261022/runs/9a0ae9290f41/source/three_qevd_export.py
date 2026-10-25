"""원본 QEVD와 같은 기본 JSON 및 연결 가능한 독립 점수 주석을 내보낸다.

영상 파일은 이동하지 않는다. QEVD 원본 경로·문자열·분할·참여자를 보존하고
외부 영상에는 충돌 없는 논리 경로를 부여한다. 실제 경로와 개발 검증 분할은
별도 manifest로 연결한다. 정상·오류 문자열에서 점수를 생성하지 않는다.
"""

import copy
import pathlib

import audit
import three_protocol


def build_bundle(
    rows: list[dict], originals: list[dict], workers: list[dict]
) -> dict:
    """기본 필드를 보존하며 여러 구간 주석을 영상 단위 JSON에 연결한다.

    Args:
        rows: 독립 직접 점수 초안 목록. 점수 확장 계약을 만족해야 한다.
        originals: 원본 QEVD 네 필드 레코드 목록.
        workers: 원본 QEVD 영상 경로·정수 참여자 목록.

    Returns:
        dict: 기본 라벨·참여자·구간 점수·파일 연결·개발 분할 목록.
        외부 자료의 validation은 기본 train으로 표현하고 실제 분할은
        development_splits에 보존한다. 학습기가 이 필드를 따라야 한다.

    Raises:
        KeyError: QEVD 원본 또는 참여자 연결을 찾을 수 없는 경우.
        ValueError: 중복 주석·원본 충돌·분할 누출·참여자 불일치가 있는 경우.
    """
    source = {pathlib.Path(r["video_path"]).stem: r for r in originals}
    worker_map = {r["video_path"]: r["worker_id"] for r in workers}
    base, participants, media, splits = {}, {}, {}, {}
    annotations, ids, group_splits = [], set(), {}
    for row in rows:
        three_protocol.validate_record(row)
        if row["id"] in ids:
            raise ValueError("Duplicate assessment ID")
        ids.add(row["id"])
        dataset = row["provenance"]["dataset"]
        if dataset == "QEVD":
            original = source[pathlib.Path(row["video_path"]).stem]
            native = copy.deepcopy(original)
            worker = worker_map[native["video_path"]]
            if row["worker_id"] != worker:
                raise ValueError("QEVD worker differs from original annotation")
            if row["split"] == "test" and native["split"] != "test":
                raise ValueError(
                    "Native training video cannot become final test"
                )
            if native["split"] == "test" and row["split"] != "test":
                raise ValueError("Native test video cannot enter development")
        else:
            native = dict(
                video_path=f"./external/{dataset}/{row['video_sha256']}.mp4",
                labels=copy.deepcopy(row["labels"]),
                labels_descriptive=copy.deepcopy(row["labels_descriptive"]),
                split="test" if row["split"] == "test" else "train",
            )
            worker = row["worker_id"]
            if type(worker) is not int:
                raise ValueError("External participant registry ID required")
        key = native["video_path"]
        for group in [
            ("person", row["person"]),
            ("source", row["video_sha256"]),
        ]:
            if group in group_splits and group_splits[group] != row["split"]:
                raise ValueError("Participant or raw video spans local splits")
            group_splits[group] = row["split"]
        if key in base and base[key] != native:
            if dataset == "QEVD" or base[key]["split"] != native["split"]:
                raise ValueError("Conflicting source labels for one video")
            # 한 긴 원본의 여러 구간 문자열은 원문을 유지한 합집합으로 연결한다.
            for field in ("labels", "labels_descriptive"):
                native[field] = list(
                    dict.fromkeys(base[key][field] + native[field])
                )
        if key in participants and participants[key]["worker_id"] != worker:
            raise ValueError("Conflicting participant for one video")
        base[key] = native
        participants[key] = dict(video_path=key, worker_id=worker)
        item = media.setdefault(
            key,
            dict(
                video_path=key,
                local_paths=[],
                sha256=row["video_sha256"],
                dataset=dataset,
            ),
        )
        if item["sha256"] != row["video_sha256"]:
            raise ValueError("One logical video refers to different content")
        if row["video_path"] not in item["local_paths"]:
            item["local_paths"].append(row["video_path"])
        splits[key] = dict(
            video_path=key,
            split=row["split"],
            evaluation_role=row.get(
                "evaluation_role", "development_" + row["split"]
            ),
            native_split=native["split"],
        )
        extended = copy.deepcopy(row)
        extended["video_path"] = key
        extended["local_video_path"] = row["video_path"]
        extended["provenance"]["native_split"] = native["split"]
        extended["provenance"]["worker_id_namespace"] = (
            "QEVD_original" if dataset == "QEVD" else "local_external_registry"
        )
        annotations.append(extended)
    keys = sorted(base)
    return dict(
        fine_grained_labels=[base[k] for k in keys],
        fine_grained_labels_with_worker_ids=[participants[k] for k in keys],
        score_annotations=annotations,
        media_manifest=[media[k] for k in keys],
        development_splits=[splits[k] for k in keys],
    )


def export() -> dict:
    """현재 3종 초안을 읽어 QEVD 기본 형식의 별도 배포 묶음을 저장한다.

    Returns:
        dict: 영상·구간 수, 연결 검증 결과와 원본 파일 해시.
        변환은 라벨 재검토·학습·최종 정확도 검증을 의미하지 않는다.

    Raises:
        OSError: 입력 또는 출력 파일 접근 실패.
        ValueError: 원본·분할·참여자 연결 불일치.
    """
    root = three_protocol.OUT
    paths = [
        root / "score_annotations.json",
        root / "additional_score_annotations.json",
    ]
    rows = [
        row for path in paths if path.exists() for row in audit.read_json(path)
    ]
    original = (
        audit.BASE / "data/qevd-selected/raw/coach/fine_grained_labels.json"
    )
    workers = (
        audit.BASE
        / "data/qevd-selected/raw/annotations/fine_grained_labels_with_worker_ids.json"
    )
    bundle = build_bundle(
        rows, audit.read_json(original), audit.read_json(workers)
    )
    destination = root / "qevd-json"
    destination.mkdir(exist_ok=True)
    for name, content in bundle.items():
        audit.save_json(destination / (name + ".json"), content)
    report = dict(
        videos=len(bundle["fine_grained_labels"]),
        scored_segments=len(bundle["score_annotations"]),
        exact_base_keys=["video_path", "labels", "labels_descriptive", "split"],
        base_split_values=["train", "test"],
        local_validation_must_follow="development_splits.json",
        original_QEVD_values_preserved=True,
        references_resolve=True,
        expert_validated=False,
        source_sha256={
            str(p): audit.digest(p)
            for p in [original, workers, *paths]
            if p.exists()
        },
        warning_ko="기본 train에는 로컬 validation이 포함되므로 학습·검증은 반드시 development_splits.json으로 분리한다. 점수 초안은 전문가 정답이 아니다.",
    )
    audit.save_json(destination / "verification.json", report)
    return report


if __name__ == "__main__":
    print(export())
