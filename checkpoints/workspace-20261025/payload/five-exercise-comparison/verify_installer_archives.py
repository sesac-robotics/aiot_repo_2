"""삭제 후보 설치 압축파일과 설치·해제본을 SHA-256으로 대조한다.

압축 내용의 프로그램은 실행하지 않는다. 일치 여부만 JSON으로 기록하며
삭제는 수행하지 않는다. 원본 AI-Hub 모델·학습 데이터는 대상이 아니다.
"""

import pathlib
import subprocess
import tarfile
import zipfile
import hashlib
import audit
from resources import check_resources


def digest_stream(stream) -> str:
    """바이너리 스트림을 끝까지 읽어 SHA-256을 반환한다.

    Args:
        stream: 읽기 가능한 바이트 스트림.

    Returns:
        str: 64자리 SHA-256. 스트림은 호출자가 닫는다.

    Raises:
        OSError: 읽기 실패.
    """
    result = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        result.update(chunk)
    return result.hexdigest()


def run() -> dict:
    """Ollama 설치 압축과 소스 ZIP 두 개의 보존본 일치를 조사한다.

    Returns:
        dict: 각 압축의 대조 수와 불일치·누락 목록. 완전 일치만 삭제 후보다.

    Raises:
        OSError: 파일 또는 zstd 실행 실패.
        tarfile.TarError: TAR 구조 오류.
        zipfile.BadZipFile: ZIP 구조·CRC 오류.
        RuntimeError: 자원 보호 또는 zstd 압축 해제 실패.
    """
    check_resources()
    runtime = audit.BASE / "local-ai-runtime"
    archive_path = runtime / "ollama-linux-amd64.tar.zst"
    rows = []
    if archive_path.exists():
        issues, count = [], 0
        process = subprocess.Popen(
            ["zstd", "-dc", "--", str(archive_path)], stdout=subprocess.PIPE
        )
        try:
            with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
                for member in archive:
                    if count % 20 == 0:
                        check_resources()
                    name = pathlib.PurePosixPath(member.name)
                    if name.is_absolute() or ".." in name.parts:
                        raise ValueError("Unsafe installer member")
                    target = runtime.joinpath(*name.parts)
                    if member.isfile():
                        count += 1
                        if (
                            not target.is_file()
                            or target.stat().st_size != member.size
                        ):
                            issues.append(
                                dict(
                                    member=member.name,
                                    reason="missing_or_size_differs",
                                )
                            )
                            continue
                        with archive.extractfile(member) as stream:
                            inside = digest_stream(stream)
                        if inside != audit.digest(target):
                            issues.append(
                                dict(
                                    member=member.name, reason="SHA256_differs"
                                )
                            )
                    elif member.issym():
                        if (
                            not target.is_symlink()
                            or str(target.readlink()) != member.linkname
                        ):
                            issues.append(
                                dict(
                                    member=member.name, reason="symlink_differs"
                                )
                            )
            # zstd는 TAR 끝의 남은 패딩도 출력할 수 있으므로 끝까지 소비한다.
            while process.stdout.read(1024 * 1024):
                pass
            if process.wait(timeout=30) != 0:
                raise RuntimeError("Installer zstd verification failed")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.kill()
                process.wait()
        rows.append(
            dict(
                archive=str(archive_path),
                bytes=archive_path.stat().st_size,
                installed_root=str(runtime),
                regular_files_checked=count,
                issues=issues,
                safe_duplicate_to_delete=not issues and count > 0,
            )
        )
    for name in ("Transformer_Rehabilitation-main.zip", "PosePro-main.zip"):
        check_resources()
        path = pathlib.Path.home() / "Downloads" / name
        if not path.exists():
            continue
        issues, count = [], 0
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                name_path = pathlib.PurePosixPath(member.filename)
                if name_path.is_absolute() or ".." in name_path.parts:
                    raise ValueError("Unsafe source ZIP member")
                target = (
                    audit.BASE / "pretrained-model-evaluation" / str(name_path)
                )
                count += 1
                if (
                    not target.is_file()
                    or target.stat().st_size != member.file_size
                ):
                    issues.append(
                        dict(
                            member=member.filename,
                            reason="missing_or_size_differs",
                        )
                    )
                    continue
                with archive.open(member) as stream:
                    inside = digest_stream(stream)
                if inside != audit.digest(target):
                    issues.append(
                        dict(member=member.filename, reason="SHA256_differs")
                    )
        rows.append(
            dict(
                archive=str(path),
                bytes=path.stat().st_size,
                regular_files_checked=count,
                issues=issues,
                safe_duplicate_to_delete=not issues and count > 0,
            )
        )
    report = dict(records=rows, deleted=False)
    audit.save_json(
        audit.BASE
        / "output/storage-audit-20261021/installer-verification.json",
        report,
    )
    return report


if __name__ == "__main__":
    result = run()
    print(
        [
            dict(
                archive=r["archive"],
                checked=r["regular_files_checked"],
                issues=len(r["issues"]),
                safe=r["safe_duplicate_to_delete"],
            )
            for r in result["records"]
        ]
    )
