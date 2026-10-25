"""승인된 사용자 캐시·검증된 설치 압축·중복 다운로드를 정리한다.

현재 Codex 런타임·프로젝트 데이터·개인 보안 파일·고유 미확인 파일은
보존한다. 열려 있는 캐시 파일은 건너뛰고 삭제·이동 내역을 JSON으로 남긴다.
"""

import datetime
import hashlib
import os
import pathlib
import shutil
import subprocess
import audit


def run() -> dict:
    """한정된 캐시와 중복 파일만 정리하고 실제 여유 공간 변화를 반환한다.

    Returns:
        dict: 경로별 처리·보존·이동 기록과 실제 여유 바이트. 캐시는 실행 중
        앱에 의해 재생성될 수 있으므로 정리 후 용량이 다시 증가할 수 있다.

    Raises:
        OSError: 조회·삭제·이동·기록 실패.
        ValueError: 허용 경로·설치본·활성 파일 검증이 실패한 경우.
        RuntimeError: 현재 런타임 또는 현재 Python 환경이 없는 경우.
    """
    home = pathlib.Path.home()
    folder = audit.BASE / "output/storage-audit-20261021"
    output = folder / "cache-and-installers-removal.json"
    runtime = home / ".cache/codex-runtimes/codex-primary-runtime"
    if not (runtime / "runtime.json").is_file():
        raise RuntimeError("Current Codex runtime missing")
    runtime_identity = (
        runtime.stat().st_dev,
        runtime.stat().st_ino,
        audit.digest(runtime / "runtime.json"),
    )
    before = shutil.disk_usage(home).free
    opened = set()
    for proc in pathlib.Path("/proc").glob("[0-9]*"):
        try:
            for descriptor in (proc / "fd").iterdir():
                try:
                    stat = descriptor.stat()
                    opened.add((stat.st_dev, stat.st_ino))
                except OSError:
                    pass
        except OSError:
            pass
    report = dict(
        started_at=datetime.datetime.now().astimezone().isoformat(),
        free_bytes_before=before,
        records=[],
        skipped_open_files=0,
        preserved_current_runtime=str(runtime),
        completed=False,
    )
    audit.save_json(output, report)
    targets = [
        home / ".cache/pip",
        home / ".arduino15/staging",
        home / ".cache/arduino",
        home / ".cache/thumbnails",
        home / ".cache/google-chrome/Default/Cache",
        home / ".cache/google-chrome/Default/Code Cache",
        home / ".cache/google-chrome/Default/GPUCache",
        home / ".config/Slack/Cache",
        home / ".config/Slack/Code Cache",
        home / ".config/Slack/Service Worker/CacheStorage",
        home / ".config/Code/CachedExtensionVSIXs",
        home / ".cache/codex-runtimes/codex-runtime-install-pHeVzy",
        home / ".cache/codex-runtimes/codex-runtime-install-60tQkU",
    ]
    for relative in ("Default", "codex-browser-app"):
        for kind in ("Cache", "Code Cache", "GPUCache"):
            targets.append(home / ".cache/Codex" / relative / kind)
    for partition in (home / ".cache/Codex/Default/Partitions").glob("*"):
        for kind in ("Cache", "Code Cache", "GPUCache"):
            targets.append(partition / kind)
    for target in targets:
        if not target.exists():
            continue
        if target.is_symlink() or target.resolve() != target:
            raise ValueError("Unexpected cache path alias")
        if (
            runtime == target
            or runtime in target.parents
            or target in runtime.parents
        ):
            raise ValueError("Current runtime overlaps cleanup")
        row = dict(
            path=str(target),
            deleted_files=0,
            deleted_logical_bytes=0,
            skipped_open_files=0,
            complete=False,
            kind="regenerable_cache_or_old_install_staging",
        )
        report["records"].append(row)
        audit.save_json(output, report)
        for base, directories, names in os.walk(
            target, topdown=False, followlinks=False
        ):
            for name in names:
                path = pathlib.Path(base) / name
                try:
                    stat = path.lstat()
                    if (stat.st_dev, stat.st_ino) in opened:
                        row["skipped_open_files"] += 1
                        continue
                    path.unlink()
                    row["deleted_files"] += 1
                    row["deleted_logical_bytes"] += stat.st_size
                except FileNotFoundError:
                    pass
            for name in directories:
                path = pathlib.Path(base) / name
                try:
                    if path.is_symlink():
                        path.unlink()
                    else:
                        path.rmdir()
                except OSError:
                    pass
        row["complete"] = True
        report["skipped_open_files"] += row["skipped_open_files"]
        audit.save_json(output, report)
    verification = audit.read_json(folder / "installer-verification.json")
    for row in verification["records"]:
        path = pathlib.Path(row["archive"])
        if row["safe_duplicate_to_delete"] and path.is_file():
            if path.stat().st_size != row["bytes"]:
                raise ValueError("Installer archive changed")
            path.unlink()
            report["records"].append(
                dict(
                    path=str(path),
                    kind="verified_installer_or_source_archive",
                    deleted=True,
                    deleted_logical_bytes=row["bytes"],
                )
            )
        elif path.exists():
            report["records"].append(
                dict(
                    path=str(path),
                    kind="archive_not_deleted",
                    reason="Retained copy does not match all archive members",
                )
            )
    # 알 수 없는 고유 다운로드는 보존하고, 동일한 내용의 사본만 제거한다.
    downloads = home / "Downloads"
    digest_groups = {}
    for path in sorted(downloads.glob(".com.google.Chrome.*")):
        if path.is_symlink() or not path.is_file():
            continue
        stat = path.stat()
        if (stat.st_dev, stat.st_ino) in opened:
            continue
        digest = audit.digest(path)
        if digest in digest_groups:
            keep = digest_groups[digest]
            if (
                not keep.is_file()
                or path.stat().st_mtime_ns != stat.st_mtime_ns
            ):
                raise ValueError("Download changed while checking duplicate")
            path.unlink()
            report["records"].append(
                dict(
                    path=str(path),
                    kind="byte_identical_download_duplicate",
                    retained=str(keep),
                    sha256=digest,
                    deleted=True,
                    deleted_logical_bytes=stat.st_size,
                )
            )
        else:
            digest_groups[digest] = path
    diagram = downloads / "SR-005-workout-session-sequences.drawio"
    if diagram.is_file():
        destination = audit.BASE / "references/downloads" / diagram.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            raise ValueError("Diagram destination already exists")
        before_stat = diagram.stat()
        digest = audit.digest(diagram)
        os.rename(diagram, destination)
        after_stat = destination.stat()
        if (before_stat.st_dev, before_stat.st_ino) != (
            after_stat.st_dev,
            after_stat.st_ino,
        ):
            raise ValueError("Diagram move was not atomic")
        report["records"].append(
            dict(
                path=str(diagram),
                moved_to=str(destination),
                kind="project_diagram_moved",
                sha256=digest,
                copied=False,
            )
        )
    current_identity = (
        runtime.stat().st_dev,
        runtime.stat().st_ino,
        audit.digest(runtime / "runtime.json"),
    )
    if current_identity != runtime_identity:
        raise RuntimeError("Current runtime changed")
    subprocess.run(
        [str(runtime / "dependencies/node/bin/node"), "--version"], check=True
    )
    subprocess.run(
        [
            str(audit.BASE / "aiot_repo_2/R2D2/bin/python"),
            "-c",
            'import numpy, cv2, sys; assert sys.prefix == "/home/jayden/SeSACPAI-Mid/aiot_repo_2/R2D2"; print("R2D2 imports OK")',
        ],
        check=True,
    )
    report.update(
        completed=True,
        finished_at=datetime.datetime.now().astimezone().isoformat(),
        free_bytes_after=shutil.disk_usage(home).free,
        current_runtime_verified=True,
        project_environment_verified=True,
        separate_old_project_preserved=True,
    )
    report["observed_free_bytes_increase"] = report["free_bytes_after"] - before
    audit.save_json(output, report)
    return report


if __name__ == "__main__":
    result = run()
    print(
        {
            k: result[k]
            for k in (
                "completed",
                "observed_free_bytes_increase",
                "free_bytes_after",
                "skipped_open_files",
            )
        }
    )
