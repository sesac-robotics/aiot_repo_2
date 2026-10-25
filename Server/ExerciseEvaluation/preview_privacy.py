"""점수 입력은 변경하지 않고 표시용 복사본의 얼굴에 모자이크를 적용합니다."""

import base64

import cv2
import numpy as np


def mosaic_faces(frame, xy):
    """보존본의 얼굴 영역·타일 계산을 유지하고 적용 여부만 반환합니다."""
    h, w = frame.shape[:2]
    head = xy[:11][np.isfinite(xy[:11]).all(axis=1)]
    if len(head) < 5:
        return False
    lo, hi = head.min(axis=0), head.max(axis=0)
    span = hi[0] - lo[0]
    if span < 12 or span > w * 0.45:
        return False
    cx = (lo[0] + hi[0]) / 2
    cy = (lo[1] + hi[1]) / 2 + span * 0.05
    bw, bh = span * 1.15, max(hi[1] - lo[1], span * 1.12)
    x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
    x1, y1 = min(w, int(cx + bw / 2)), min(h, int(cy + bh / 2))
    if x1 <= x0 or y1 <= y0:
        return False
    roi = frame[y0:y1, x0:x1]
    tile = cv2.resize(
        roi,
        (max(8, (x1 - x0) // 6), max(8, (y1 - y0) // 6)),
        interpolation=cv2.INTER_AREA,
    )
    frame[y0:y1, x0:x1] = cv2.resize(
        tile, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST
    )
    return True


def encode_masked_preview(frame, xy):
    """동일 프레임의 관절만 사용하며 미검출 때 원본 영상을 반환하지 않습니다."""
    preview = frame.copy()
    mode = "face"
    if not mosaic_faces(preview, xy):
        h, w = preview.shape[:2]
        tile = cv2.resize(
            preview, (min(20, w), min(15, h)), interpolation=cv2.INTER_AREA
        )
        preview = cv2.resize(tile, (w, h), interpolation=cv2.INTER_NEAREST)
        mode = "full_frame"
    ok, jpeg = cv2.imencode(".jpg", preview)
    if not ok:
        raise RuntimeError("모자이크 미리보기를 만들 수 없습니다.")
    return base64.b64encode(jpeg).decode("ascii"), mode
