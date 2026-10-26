"""규칙 점수 초안과 사람의 검토 라벨을 별도로 보존한다.

점수는 0~100점이며 승인 여부만으로 전문가 정답이라고 간주하지 않는다.
동일 인물의 학습ㆍ평가 혼입을 막기 위해 검토 시 익명 참여자 코드를 받는다.
"""

import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading

REVIEW_LOCK = threading.Lock()


def load_review(folder):
    """검토 파일을 읽거나 빈 검토 상태를 반환한다.

    Args:
        folder: 결과가 저장된 작업 폴더.
    Returns:
        버전 번호와 반복별 검토 사전. 저장된 검토가 없으면 버전은 0이다.
    Raises:
        ValueError: 저장된 JSON이 손상된 경우.
    """
    path = Path(folder) / "review.json"
    return (
        json.loads(path.read_text())
        if path.exists()
        else {"revision": 0, "items": {}, "history": []}
    )


def save_review(folder, payload):
    """검토 결과를 검증하고 원본 점수와 분리해 원자적으로 저장한다.

    Args:
        folder: result.json이 있는 작업 폴더.
        payload: 반복 번호, 검토 상태, 익명 인물 코드, 측면 확인, 점수,
            메모, 기대 revision을 담은 사전. 승인ㆍ수정 점수 단위는 점이다.
    Returns:
        갱신된 검토 사전. 원래 규칙 점수는 덮어쓰지 않는다.
    Raises:
        ValueError: 형식ㆍ점수ㆍ구도 확인ㆍ참여자 코드가 잘못된 경우.
        RuntimeError: 다른 검토가 먼저 저장되어 revision이 달라진 경우.
    """
    folder = Path(folder)
    with REVIEW_LOCK:
        original = (folder / "result.json").read_bytes()
        result = json.loads(original)
        rep = payload.get("number")
        status = payload.get("status")
        if type(rep) is not int or not any(
            x["number"] == rep for x in result["repetitions"]
        ):
            raise ValueError("반복 번호가 잘못되었습니다.")
        if status not in ["accepted", "corrected", "rejected"]:
            raise ValueError("검토 상태가 잘못되었습니다.")
        subject = payload.get("subject_id", "")
        if not isinstance(subject, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,40}", subject
        ):
            raise ValueError(
                "이름 대신 영문ㆍ숫자의 익명 참여자 코드를 입력하세요."
            )
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > 1000:
            raise ValueError("메모는 1000자 이하입니다.")
        score = None
        if status != "rejected":
            if payload.get("side_confirmed") is not True:
                raise ValueError("측면 촬영 여부를 먼저 확인하세요.")
            if status == "accepted":
                score = next(
                    x["score"]
                    for x in result["repetitions"]
                    if x["number"] == rep
                )
            else:
                score = payload.get("score")
                if (
                    type(score) not in (float, int)
                    or not math.isfinite(score)
                    or not 0 <= score <= 100
                ):
                    raise ValueError("수정 점수는 0~100의 숫자여야 합니다.")
        review = load_review(folder)
        if payload.get("revision") != review["revision"]:
            raise RuntimeError(
                "다른 검토가 저장됐습니다. 화면을 다시 열어 주세요."
            )
        record = {
            "number": rep,
            "status": status,
            "subject_id": subject,
            "side_confirmed": payload.get("side_confirmed") is True,
            "reviewed_score": score,
            "note": note,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "expert_validated": False,
        }
        review["revision"] += 1
        review["items"][str(rep)] = record
        review["history"].append(dict(record, revision=review["revision"]))
        temp = folder / "review.pending.json"
        temp.write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temp.replace(folder / "review.json")
        return review


def export_csv(folder, reviewed_only=False):
    """규칙 초안과 검토 점수를 구분한 UTF-8 CSV 바이트를 반환한다.

    Args:
        folder: 결과ㆍ검토 파일이 있는 작업 폴더.
        reviewed_only: 참이면 승인ㆍ수정된 행만 출력한다.
    Returns:
        CSV 바이트. 정답 부재 행의 reviewed_score는 빈칸이다.
    Raises:
        FileNotFoundError: 분석 결과가 없는 경우.
    """
    folder = Path(folder)
    raw = (folder / "result.json").read_bytes()
    result = json.loads(raw)
    review = load_review(folder)
    digest = hashlib.sha256(raw).hexdigest()
    stream = io.StringIO()
    fields = [
        "job_id",
        "exercise",
        "number",
        "start_seconds",
        "end_seconds",
        "draft_score",
        "reviewed_score",
        "label_status",
        "subject_id",
        "rules_version",
        "result_sha256",
        "expert_validated",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for rep in result["repetitions"]:
        record = review["items"].get(str(rep["number"]), {})
        if record.get("source_sha256") != digest:
            record = {}
        status = record.get("status", "unreviewed")
        if reviewed_only and status not in ["accepted", "corrected"]:
            continue
        writer.writerow(
            dict(
                job_id=folder.name,
                exercise=result["exercise"],
                number=rep["number"],
                start_seconds=rep["start"],
                end_seconds=rep["end"],
                draft_score=rep["score"],
                reviewed_score=record.get("reviewed_score", ""),
                label_status=status,
                subject_id=record.get("subject_id", ""),
                rules_version=result["rules"]["version"],
                result_sha256=digest,
                expert_validated=False,
            )
        )
    return ("\ufeff" + stream.getvalue()).encode("utf-8")
