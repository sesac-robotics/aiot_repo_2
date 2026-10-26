"""탐색 디코딩에서 순차 디코딩으로 변경한 관절 캐시의 차이를 기록한다.

같은 원본의 추출 회귀 검사이며 재촬영·새 참여자에 대한 일관성이 아니다.
"""

import numpy as np

import audit
from resources import check_resources


def run() -> dict:
    """보존한 v1 캐시와 현재 v2 캐시의 프레임·관측·좌표 차이를 저장한다.

    Returns:
        dict: 시점별 프레임 일치 여부, 관측 마스크 변경 수, 픽셀 오차.
        GPU 추론과 디코딩 변경이 함께 포함되므로 원인을 분리하지 않는다.

    Raises:
        OSError: 보존 캐시 또는 현재 캐시를 읽을 수 없는 경우.
        RuntimeError: 자원 보호 조건을 위반한 경우.
    """
    previous = audit.OUT / "snapshots/pre_sequential_v2/heavy"
    rows = []
    for old_path in sorted(previous.glob("*/*.npz")):
        check_resources()
        new_path = audit.OUT / "heavy" / old_path.relative_to(previous)
        if not new_path.exists():
            continue
        with np.load(old_path) as old, np.load(new_path) as new:
            equal_frames = np.array_equal(
                old["frame_numbers"], new["frame_numbers"]
            )
            row = dict(
                id=old_path.parent.name,
                view=old_path.stem,
                frame_indices_equal=equal_frames,
            )
            if equal_frames:
                before, after = old["image_xy"], new["image_xy"]
                before_mask = np.isfinite(before).all(2)
                after_mask = np.isfinite(after).all(2)
                observed = before_mask & after_mask
                displacement = np.linalg.norm(
                    after[observed] - before[observed], axis=1
                )
                row.update(
                    observation_mask_changed_joints=int(
                        np.sum(before_mask != after_mask)
                    ),
                    jointly_observed_joints=int(observed.sum()),
                    mean_pixel_displacement=(
                        float(displacement.mean())
                        if len(displacement)
                        else None
                    ),
                    max_pixel_displacement=(
                        float(displacement.max()) if len(displacement) else None
                    ),
                )
            rows.append(row)
    report = dict(
        comparisons=rows,
        view_pairs=len(rows),
        all_frame_indices_equal=all(r["frame_indices_equal"] for r in rows),
        same_original_videos=True,
        independent_final_test=False,
        real_recapture_consistency=False,
        purpose_ko="디코딩 구현 변경 전후의 같은 원본 관절 캐시 회귀 검사. 추론·디코딩 영향은 분리하지 않음.",
    )
    audit.save_json(audit.OUT / "cache_parity.json", report)
    return report


if __name__ == "__main__":
    print(run())
