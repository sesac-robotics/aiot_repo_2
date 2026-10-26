"""얼굴·신원 추정 없이 영상 바깥 배경의 유사 촬영 자료를 분할에서 묶는다."""

import json
import pathlib

import cv2
import numpy as np

import audit
from recovery_prepare import ROOT
from resources import check_resources


def run() -> dict:
    """보유 개발 QEVD의 배경 유사도를 보수적 촬영 그룹으로 저장한다.

    Returns:
        dict: 자동 근접 배경 쌍·연결 그룹·분할 충돌. 얼굴·실제 참여자
        동일성 판정이 아니며 다른 촬영 환경의 동일인을 발견하지 못한다.
        고정 밝기 차이 기준은 모델 점수·라벨·검증 오차로 보정하지 않는다.

    Raises:
        OSError: 원본·선택 목록·출력 접근 실패.
        ValueError: 원본 영상 표본 디코딩 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    records = audit.read_json(ROOT / "input_plan.json")["records"]
    for path in ROOT.glob("additional_score_annotations.json"):
        records.extend(audit.read_json(path))
    selected = {}
    for row in records:
        if not row["id"].startswith("qevd_") or row["split"] == "test":
            continue
        root = "_".join(row["id"].split("_")[:2])
        selected[root] = row
    ids = []
    features = []
    previews = {}
    roles = {}
    for identifier, row in sorted(selected.items()):
        check_resources()
        cap = cv2.VideoCapture(row["video_path"])
        count = int(cap.get(7))
        samples = []
        for fraction in [0.1, 0.5, 0.9]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, round((count - 1) * fraction))
            ok, frame = cap.read()
            if not ok:
                cap.release()
                raise ValueError("Background sample unavailable")
            tiny = (
                cv2.resize(
                    frame, (64, 36), interpolation=cv2.INTER_AREA
                ).astype(float)
                / 255
            )
            # 중앙 수행자 영역은 사용하지 않고 바깥 좌우 배경만 비교한다.
            samples.append(
                np.concatenate([tiny[:, :12].ravel(), tiny[:, 52:].ravel()])
            )
            if fraction == 0.5:
                previews[identifier] = cv2.resize(frame, (256, 144))
        cap.release()
        ids.append(identifier)
        features.append(np.median(samples, axis=0))
        roles[identifier] = row["split"]
    parent = list(range(len(ids)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    pairs = []
    for i in range(len(ids)):
        for j in range(i):
            distance = float(np.mean(np.abs(features[i] - features[j])))
            if distance <= 0.035:
                parent[find(i)] = find(j)
                pairs.append(
                    dict(
                        left=ids[j],
                        right=ids[i],
                        border_color_MAE_0_1=distance,
                        splits=[roles[ids[j]], roles[ids[i]]],
                    )
                )
    components = {}
    for i, identifier in enumerate(ids):
        components.setdefault(find(i), []).append(identifier)
    groups = [
        dict(
            id="automatic_background_" + str(k),
            records=v,
            basis_ko="원본 3시점 중앙값의 좌우 바깥 배경 색상 MAE ≤ 0.035. 자동 보수적 촬영 연결이며 직접 신원 확인 아님.",
            person_identity_verified=False,
            automatically_grouped=True,
        )
        for k, v in components.items()
        if len(v) > 1
    ]
    target = ROOT / "visual_recording_links.json"
    manual = audit.read_json(target) if target.exists() else dict(groups=[])
    manual["groups"] = [
        g for g in manual["groups"] if not g.get("automatically_grouped")
    ] + groups
    manual["scope"] = (
        "manual_observed_links_plus_conservative_border_similarity_not_exhaustive_identity_audit"
    )
    audit.save_json(target, manual)
    sheet = np.full((max(1, min(16, len(pairs))) * 174, 540, 3), 25, np.uint8)
    for n, pair in enumerate(
        sorted(pairs, key=lambda p: p["border_color_MAE_0_1"])[:16]
    ):
        for column, key in enumerate(["left", "right"]):
            x = column * 270
            y = n * 174
            sheet[y + 25 : y + 169, x : x + 256] = previews[pair[key]]
            cv2.putText(
                sheet,
                pair[key] + " " + roles[pair[key]],
                (x + 2, y + 17),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
    cv2.imwrite(str(ROOT / "review/recording_similarity_pairs.jpg"), sheet)
    report = dict(
        development_videos=len(ids),
        automatic_pairs=pairs,
        automatic_groups=groups,
        cross_role_pairs=[p for p in pairs if len(set(p["splits"])) > 1],
        direct_visual_review=False,
        face_or_identity_recognition=False,
        not_exhaustive=True,
    )
    audit.save_json(ROOT / "recording_similarity_audit.json", report)
    return dict(
        development_videos=len(ids),
        near_background_pairs=len(pairs),
        cross_role_pairs=len(report["cross_role_pairs"]),
    )


if __name__ == "__main__":
    print(json.dumps(run()))
