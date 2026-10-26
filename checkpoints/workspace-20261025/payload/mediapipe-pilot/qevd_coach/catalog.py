"""QEVD 클립 주석을 운동별 비교 과제로 변환한다.

오류 라벨 부재를 정상으로 간주하지 않는다. 명시한 기준 속성만 있는
클립을 비교군으로 사용하며, 그 비교군도 의학적 정상 판정은 아니다.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# 선택한 4종 및 YTW 순서 오류의 내장 SSD 원본 경로.
DATA = ROOT.parent.parent / "data" / "qevd-selected"
PROFILES = {
    "pushups": {
        "name": "푸시업",
        "source": "pushups",
        "reference": ["90 degrees", "deep", "shoulder-width"],
        "errors": {
            "elevated hips": [
                "엉덩이가 높음",
                "몸통과 골반 정렬을 확인하세요.",
            ],
            "shallow": [
                "내려가는 깊이 부족",
                "팔꿈치 굽힘과 내려가는 범위를 확인하세요.",
            ],
            "butt sinking": [
                "골반이 처짐",
                "골반이 아래로 처지는지 확인하세요.",
            ],
            "legs on the floor": [
                "다리가 바닥에 닿음",
                "선택한 일반 푸시업 자세인지 확인하세요.",
            ],
            "head down": ["고개를 숙임", "목과 몸통 정렬을 확인하세요."],
        },
    },
    "squats": {
        "name": "스쿼트",
        "source": "squats",
        "reference": [
            "no obvious issue",
            "shoulder-width",
            "90 degrees",
            "over 90 degrees",
        ],
        "require_reference": "no obvious issue",
        "errors": {
            "back not straight": [
                "등 정렬 흐트러짐",
                "등과 몸통 정렬을 확인하세요.",
            ],
            "shallow": ["앉는 깊이 부족", "앉는 깊이가 충분한지 확인하세요."],
        },
    },
    "plank": {
        "name": "팔꿈치 플랭크",
        "source": "elbow plank",
        "reference": ["average"],
        "errors": {
            "high hips": ["엉덩이가 높음", "어깨·골반·발목 정렬을 확인하세요."],
            "butt sinking": [
                "골반이 처짐",
                "골반이 아래로 처지는지 확인하세요.",
            ],
            "legs and hips on the floor": [
                "다리와 골반이 바닥에 닿음",
                "버티는 자세가 유지되는지 확인하세요.",
            ],
            "on knees": [
                "무릎을 대고 수행",
                "일반형과 무릎 지지형을 구분하세요.",
            ],
            "head up": ["고개를 듦", "목이 과하게 젖혀지는지 확인하세요."],
            "head down": ["고개를 숙임", "목과 몸통 정렬을 확인하세요."],
            "arms too wide": [
                "팔 간격이 넓음",
                "팔꿈치 위치와 간격을 확인하세요.",
            ],
        },
    },
    "bird_dog": {
        "name": "버드독 대응 동작",
        "source": "opposite arm and leg lifts (on knees)",
        "reference": ["average"],
        "errors": {
            "back not straight, arching up": [
                "등을 위로 둥글게 말음",
                "등이 위로 굽는지 확인하세요.",
            ],
            "back not straight, dropping down": [
                "등이 아래로 처짐",
                "몸통과 골반 정렬을 확인하세요.",
            ],
            "only arms": [
                "팔만 듦",
                "반대쪽 팔과 다리를 함께 움직이는지 확인하세요.",
            ],
            "only legs": [
                "다리만 듦",
                "반대쪽 팔과 다리를 함께 움직이는지 확인하세요.",
            ],
            "low arm and leg": [
                "팔·다리 높이가 낮음",
                "팔과 다리의 뻗는 범위를 확인하세요.",
            ],
            "not alternating": [
                "좌우 교대하지 않음",
                "동작 중 좌우 교대를 확인하세요.",
            ],
            "hands too far forward, not under shoulders": [
                "손이 어깨보다 앞에 있음",
                "손과 어깨 위치를 확인하세요.",
            ],
            "knees not under hips": [
                "무릎과 골반 위치 불일치",
                "무릎이 골반 아래인지 확인하세요.",
            ],
            "losing balance": ["균형 흔들림", "몸통이 흔들리는지 확인하세요."],
        },
    },
}


def build_catalog() -> dict:
    """내장 SSD의 선택 원본을 조사해 기준군 또는 지원 오류 클립을 반환한다.

    Returns:
        영상·작업자 식별자·원본 라벨·오류·분할 및 전체 라벨 집계.
        공식 시험 사람과 겹치는 학습 사람의 영상은 별도 제외한다.
    Raises:
        FileNotFoundError: 원본 명세나 영상 경로가 없는 경우.
        ValueError: JSON 주석이 잘못된 경우.
    """
    with (DATA / "manifests/short_clips.csv").open() as source:
        rows = list(csv.DictReader(source))
    sources = {v["source"]: k for k, v in PROFILES.items()}
    selected, counts = [], {k: {} for k in PROFILES}
    test_workers = set()
    for row in rows:
        if row["split"] == "test":
            test_workers.add(row["worker_id"])
    # 선택 자료 밖의 공식 시험 참여자도 학습 중복 검사에 포함한다.
    global_workers = DATA / "manifests/official_test_workers.json"
    if global_workers.is_file():
        test_workers.update(json.loads(global_workers.read_text()))
    excluded_overlap = 0
    for row in rows:
        labels = json.loads(row["labels"])
        names = {label.partition(" - ")[0] for label in labels}
        matches = names & sources.keys()
        if len(matches) != 1 or len(names) != 1:
            continue
        key = sources[next(iter(matches))]
        profile = PROFILES[key]
        attrs = [label.partition(" - ")[2] for label in labels]
        for attr in attrs:
            counts[key][attr] = counts[key].get(attr, 0) + 1
        errors = [attr for attr in attrs if attr in profile["errors"]]
        reference = bool(attrs) and set(attrs) <= set(profile["reference"])
        required = profile.get("require_reference")
        reference = reference and (not required or required in attrs)
        if not errors and not reference:
            continue
        worker = row["worker_id"]
        if not worker:
            continue
        if row["split"] == "test":
            split = "test"
        elif worker in test_workers:
            excluded_overlap += 1
            continue
        else:
            bucket = int(hashlib.sha256(worker.encode()).hexdigest()[:8], 16)
            split = "validation" if bucket % 5 == 0 else "train"
        path = DATA / row["video_path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        selected.append(
            dict(
                id=row["video_id"],
                path=str(path),
                worker=worker,
                exercise=key,
                split=split,
                labels=labels,
                errors=errors,
                reference=reference,
            )
        )
    return dict(
        items=selected,
        label_counts=counts,
        excluded_train_worker_overlap=excluded_overlap,
        profiles=PROFILES,
        semantics="클립별 오류 대 기준 속성 비교. 오류 부재는 정상 아님.",
    )
