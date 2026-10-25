"""검증된 중복 데이터의 물리 사본과 해제 완료 압축본을 정리한다.

팀 저장소·진행 중 다운로드·점수 라벨·실험 결과는 수정하지 않는다.
데이터셋 해제본의 논리 경로는 유지하고 SHA-256이 같은 파일만 하드링크로
통합한다. ZIP은 모든 멤버가 해제본과 동일한지 검증한 뒤에만 제거한다.
"""

import collections
import hashlib
import os
import pathlib
import shutil
import uuid
import zipfile
import audit
from resources import check_resources


def stream_digest(stream) -> str:
    """파일 스트림의 현재 위치부터 끝까지 SHA-256을 반환한다.

    Args:
        stream: 바이너리 읽기 스트림. 바이트 단위 입력이다.

    Returns:
        str: 64자리 SHA-256. 스트림 소유권은 호출자에게 있다.

    Raises:
        OSError: 읽기 실패.
    """
    digest = hashlib.sha256()
    while block := stream.read(1024 * 1024):
        digest.update(block)
    return digest.hexdigest()


def run() -> dict:
    """명시한 로컬 데이터셋만 검증하고 중복을 정리해 기록한다.

    Returns:
        dict: 압축 삭제·하드링크 교체의 파일·해시·바이트와 전후 여유 공간.
        다운로드 동시 진행 때문에 실제 여유 공간 차이는 순수 회수량과 다르다.

    Raises:
        ValueError: 예상 경로·해시·용량·변경 상태가 일치하지 않는 경우.
        OSError: 읽기·링크·교체·삭제·기록 실패.
        zipfile.BadZipFile: ZIP 손상 또는 CRC 오류.
        RuntimeError: 온도·전원·메모리 보호 중단.
    """
    root = (
        audit.BASE / "data/A Multi-View Raw Video Dataset of Seven Fitness Ex"
    )
    names = (
        "Dataset Exercise-wise",
        "Dataset Exercise Quality-wise",
        "Dataset Subject-wise",
    )
    before_free = shutil.disk_usage(root).free
    files, archive_checks = {}, []
    for name in names:
        check_resources()
        archive_path = root / (name + ".zip")
        if not archive_path.is_file():
            raise ValueError(
                "Expected archive missing; inspect previous cleanup report"
            )
        original_archive_stat = archive_path.stat()
        count = 0
        with zipfile.ZipFile(archive_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                if count % 40 == 0:
                    check_resources()
                member = pathlib.PurePosixPath(info.filename)
                if (
                    member.is_absolute()
                    or ".." in member.parts
                    or member.parts[0] != name
                ):
                    raise ValueError("Unexpected ZIP member path")
                target = root.joinpath(*member.parts)
                if target.is_symlink() or not target.is_file():
                    raise ValueError("Extracted member missing or symlink")
                stat = target.stat()
                if stat.st_size != info.file_size:
                    raise ValueError("Extracted member size differs")
                with target.open("rb") as stream:
                    disk_sha = stream_digest(stream)
                with archive.open(info) as stream:
                    zip_sha = stream_digest(stream)
                if disk_sha != zip_sha:
                    raise ValueError("ZIP and extracted member content differs")
                files[str(target)] = dict(
                    sha256=disk_sha,
                    bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns,
                    device=stat.st_dev,
                    inode=stat.st_ino,
                    mode=stat.st_mode,
                    uid=stat.st_uid,
                    gid=stat.st_gid,
                )
                count += 1
        archive_checks.append(
            dict(
                path=str(archive_path),
                bytes=original_archive_stat.st_size,
                mtime_ns=original_archive_stat.st_mtime_ns,
                members_verified=count,
                every_member_SHA256_matches=True,
                deleted=False,
            )
        )
        print("verified", name, count, flush=True)
    # 삭제·교체 전에 모든 대조 결과를 저장한다.
    report = dict(
        archive_checks=archive_checks,
        verified_files=files,
        hardlinks=[],
        completed=False,
        free_bytes_before=before_free,
        download_and_team_repository_untouched=True,
    )
    destination = audit.OUT / "disk_cleanup_verified_duplicates.json"
    audit.save_json(destination, report)
    groups = collections.defaultdict(list)
    for path, metadata in files.items():
        groups[
            (
                metadata["device"],
                metadata["bytes"],
                metadata["sha256"],
                metadata["mode"],
                metadata["uid"],
                metadata["gid"],
            )
        ].append(path)
    recovered_links = 0
    for index, paths in enumerate(groups.values()):
        if index % 40 == 0:
            check_resources()
        canonical = pathlib.Path(paths[0])
        for duplicate_name in paths[1:]:
            duplicate = pathlib.Path(duplicate_name)
            original = files[duplicate_name]
            current = duplicate.stat()
            if (
                current.st_size != original["bytes"]
                or current.st_mtime_ns != original["mtime_ns"]
            ):
                raise ValueError("Data changed after verification")
            if current.st_ino == canonical.stat().st_ino:
                continue
            temp = duplicate.with_name(
                duplicate.name + ".dedup-" + uuid.uuid4().hex
            )
            try:
                os.link(canonical, temp)
                os.replace(temp, duplicate)
            finally:
                if temp.exists():
                    temp.unlink()
            if not os.path.samefile(canonical, duplicate):
                raise ValueError("Hardlink verification failed")
            recovered_links += current.st_size
            report["hardlinks"].append(
                dict(
                    path=str(duplicate),
                    canonical=str(canonical),
                    sha256=original["sha256"],
                    logical_bytes_preserved=current.st_size,
                )
            )
    audit.save_json(destination, report)
    for row in archive_checks:
        path = pathlib.Path(row["path"])
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (
            row["bytes"],
            row["mtime_ns"],
        ):
            raise ValueError("Archive changed after verification")
        # 모든 보존 경로가 검증한 실제 파일을 가리키는지 삭제 직전에 재확인한다.
        for name, metadata in files.items():
            item = pathlib.Path(name)
            if not item.is_file() or item.stat().st_size != metadata["bytes"]:
                raise ValueError("Preserved dataset file missing")
        path.unlink()
        row["deleted"] = True
        audit.save_json(destination, report)
    report.update(
        completed=True,
        hardlink_logical_duplicate_bytes_removed=recovered_links,
        deleted_archive_bytes=sum(
            r["bytes"] for r in archive_checks if r["deleted"]
        ),
        free_bytes_after=shutil.disk_usage(root).free,
    )
    audit.save_json(destination, report)
    return report


if __name__ == "__main__":
    result = run()
    print(
        {
            key: result[key]
            for key in (
                "completed",
                "deleted_archive_bytes",
                "hardlink_logical_duplicate_bytes_removed",
                "free_bytes_before",
                "free_bytes_after",
            )
        }
    )
