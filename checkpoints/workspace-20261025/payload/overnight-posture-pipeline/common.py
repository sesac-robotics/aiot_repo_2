"""야간 실험의 경로·원자적 저장·진행 기록을 제공한다. 데이터는 내장 SSD만 읽는다."""
from pathlib import Path
import datetime
import hashlib
import json
import os

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'output/overnight-20261020'
OUT.mkdir(parents=True, exist_ok=True)


def save(path, value):
    """JSON 객체를 UTF-8로 원자 저장한다.

    Args:
        path: 대상 파일 경로.
        value: JSON 직렬화 가능한 값. 물리 단위는 해당 필드에 명시한다.
    Raises:
        ValueError: NaN 또는 직렬화할 수 없는 값이 있는 경우.
        OSError: 저장 실패.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    temp.replace(path)


def digest(path):
    """파일의 SHA256 문자열을 반환한다. 읽기 실패는 호출자에게 전달한다."""
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def progress(stage, **details):
    """현재 단계와 UTC 오프셋이 포함된 시각을 저장하고 출력한다."""
    value = dict(stage=stage, time=datetime.datetime.now().astimezone().isoformat(), **details)
    save(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
