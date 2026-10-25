"""기존 검토판의 바깥 배경 특징만으로 보수적인 촬영 연결을 자동 검사한다."""

import json
import pathlib

import cv2
import numpy as np

import audit
from recovery_prepare import ROOT
from resources import check_resources


def run() -> dict:
    """기존 접촉판 첫 장면의 배경 ORB 대응을 분할 연결 후보로 저장한다.

    Returns:
        dict: 자동 배경 대응 쌍·그룹·분할 충돌과 수동 연결 재탐지 수.
        얼굴 또는 실제 참여자 신원 확인이 아니며 완전 중복 감사도 아니다.
        영상·점수 추론 없이 이미 저장된 검토판만 읽는다.

    Raises:
        OSError: 검토판·목록·출력 접근 실패.
        ValueError: 지원하지 않는 검토판 형상 또는 이미지 읽기 실패.
        RuntimeError: 온도·메모리 보호 기준 위반.
    """
    cv2.setNumThreads(1)
    rows = audit.read_json(ROOT / "input_plan.json")["records"]
    rows.extend(audit.read_json(ROOT / "additional_score_annotations.json"))
    selected = {}
    for row in rows:
        if row["id"].startswith("qevd_") and row["split"] != "test":
            selected["_".join(row["id"].split("_")[:2])] = row
    orb = cv2.ORB_create(
        nfeatures=1200, edgeThreshold=4, patchSize=15, fastThreshold=8
    )
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    records = []
    for identifier, row in sorted(selected.items()):
        check_resources()
        path = row.get("evidence_image") or row["evidence_images"][0]
        im = cv2.imread(path)
        if im is None:
            raise ValueError("Contact image unavailable")
        if im.shape[1] == 1536:
            frame = im[32:176, :256]
        elif im.shape[1] == 2100:
            frame = cv2.resize(im[:180, :320], (256, 144))
        else:
            raise ValueError("Unverified contact sheet layout")
        gray = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        mask = np.ones(gray.shape, np.uint8) * 255
        mask[:, 76:180] = 0
        points, desc = orb.detectAndCompute(gray, mask)
        records.append(
            dict(
                id=identifier,
                split=row["split"],
                points=points,
                desc=desc,
                frame=frame,
            )
        )
    parents = list(range(len(records)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    pairs = []
    for i, left in enumerate(records):
        if left["desc"] is None:
            continue
        for j, right in enumerate(records[:i]):
            if right["desc"] is None or len(right["desc"]) < 2:
                continue
            matches = [
                m
                for pair in matcher.knnMatch(left["desc"], right["desc"], k=2)
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < 40 and m.distance < 0.7 * n.distance
            ]
            if len(matches) < 12:
                continue
            a = np.float32([left["points"][m.queryIdx].pt for m in matches])
            b = np.float32([right["points"][m.trainIdx].pt for m in matches])
            matrix, inliers = cv2.findHomography(
                a, b, cv2.RANSAC, 3, maxIters=500
            )
            if matrix is None or inliers is None:
                continue
            valid = inliers.ravel().astype(bool)
            accepted = a[valid]
            if (
                valid.sum() < 12
                or valid.mean() < 0.65
                or (accepted[:, 0] < 76).sum() < 4
                or (accepted[:, 0] >= 180).sum() < 4
            ):
                continue
            if (
                not np.isfinite(matrix).all()
                or not 0.5 < abs(np.linalg.det(matrix[:2, :2])) < 2
            ):
                continue
            parents[find(i)] = find(j)
            pairs.append(
                dict(
                    left=left["id"],
                    right=right["id"],
                    splits=[left["split"], right["split"]],
                    background_inliers=int(valid.sum()),
                    automatic_only=True,
                )
            )
    components = {}
    for i, row in enumerate(records):
        components.setdefault(find(i), []).append(row["id"])
    target = ROOT / "visual_recording_links.json"
    links = audit.read_json(target)
    manual = [g for g in links["groups"] if not g.get("automatically_grouped")]
    links["groups"] = manual + [
        dict(
            id="contact_background_" + str(k),
            records=v,
            basis_ko="기존 검토판 좌우 바깥 배경의 ORB 대응·양쪽 분포·호모그래피 검증. 자동 촬영 연결이며 얼굴 신원 확인 아님.",
            automatically_grouped=True,
            person_identity_verified=False,
        )
        for k, v in components.items()
        if len(v) > 1
    ]
    audit.save_json(target, links)
    report = dict(
        videos=len(records),
        pairs=pairs,
        automatic_groups=len(links["groups"]) - len(manual),
        cross_role_pairs=[p for p in pairs if len(set(p["splits"])) > 1],
        directly_reviewed=False,
        identity_verified=False,
        not_exhaustive=True,
        manual_groups_redetected=[
            dict(
                id=g["id"],
                connected=len(
                    {
                        find(
                            next(
                                i for i, r in enumerate(records) if r["id"] == s
                            )
                        )
                        for s in g["records"]
                        if s in selected
                    }
                )
                == 1,
            )
            for g in manual
        ],
    )
    audit.save_json(ROOT / "contact_scene_audit.json", report)
    return dict(
        videos=len(records),
        pairs=len(pairs),
        cross_role_pairs=len(report["cross_role_pairs"]),
        manual_groups_redetected=report["manual_groups_redetected"],
    )


if __name__ == "__main__":
    print(json.dumps(run()))
