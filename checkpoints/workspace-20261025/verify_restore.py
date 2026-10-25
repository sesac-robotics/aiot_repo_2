"""감사된 작업 스냅샷을 검사하고 충돌 없는 대상 경로에 복원합니다.

원본 데이터셋·공유 문서·실행 환경은 복원하지 않습니다. 복원은 학습이나
서비스 실행을 시작하지 않으며, 기존 내용이 다른 파일은 덮어쓰지 않습니다.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def digest(path):
    """큰 모델도 일정 메모리로 읽어 SHA-256을 계산합니다."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_path(root, name):
    """경로 탈출·심볼릭 링크를 거쳐 기존 파일을 덮어쓰는 것을 막습니다."""
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Unsafe snapshot path: " + name)
    path = root.joinpath(*relative.parts)
    for parent in (path, *path.parents):
        if parent == root.parent:
            break
        if parent.is_symlink():
            raise ValueError("Symlink path refused: " + str(parent))
    return path


def verify(root, records):
    """전체 파일의 존재·길이·해시가 명세와 일치하는지 검사합니다."""
    failures = []
    for row in records:
        path = safe_path(root, row["path"])
        if not path.is_file():
            failures.append((row["path"], "missing"))
        elif path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            failures.append((row["path"], "hash_or_size_mismatch"))
    if failures:
        raise ValueError(json.dumps(failures, ensure_ascii=False))
    return len(records)


def restore(destination, records):
    """모든 충돌을 먼저 검사하고 바이트가 같은 기존 파일만 재사용합니다."""
    verify(HERE / "payload", records)
    for row in records:
        path = safe_path(destination, row["path"])
        if path.exists() and (not path.is_file() or digest(path) != row["sha256"]):
            raise FileExistsError("Refusing to overwrite: " + str(path))
    for row in records:
        path = safe_path(destination, row["path"])
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against a file appearing after preflight.
        with path.open("xb") as target, (HERE / "payload" / row["path"]).open("rb") as source:
            shutil.copyfileobj(source, target)
    return verify(destination, records)


def main():
    """기본 검사는 읽기 전용이며 --restore가 있을 때만 파일을 씁니다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-workspace", type=Path)
    parser.add_argument("--restore", type=Path)
    args = parser.parse_args()
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    rows = manifest["files"]
    if args.restore:
        count = restore(args.restore.absolute(), rows)
    else:
        count = verify(args.verify_workspace.absolute() if args.verify_workspace else HERE / "payload", rows)
    print(json.dumps({"verified_files": count, "model_fit_or_service_started": False}))


if __name__ == "__main__":
    main()
