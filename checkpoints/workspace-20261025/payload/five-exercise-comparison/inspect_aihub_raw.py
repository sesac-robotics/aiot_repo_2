"""완료된 AI-Hub 원시 TAR의 라벨 대응과 소량 검토 이미지 추출을 수행한다.

미완료 다운로드를 읽지 않는다. 선택 경로만 고정 폴더에 저장하며 원본을
변경·삭제하지 않는다. 제공자 상태를 점수 또는 직접 검토 결과로 바꾸지 않는다.
"""

import argparse
import contextlib
import hashlib
import pathlib
import re
import shutil
import tarfile
import audit
from resources import check_resources
from segmented_reader import SegmentedReader

EXPECTED_PAYLOAD_BYTES = 28184729600


def canonical_image_path(name: str) -> str | None:
    """TAR 경로의 Day 폴더부터 시작하는 라벨 대응 키를 반환한다.

    Args:
        name: TAR 내부 멤버 경로. 단위 없음.

    Returns:
        str | None: 안전한 Day 경로. JPG·Day 폴더가 없으면 None이다.

    Raises:
        ValueError: 부모 경로 이동이나 절대 경로를 포함한 경우.
    """
    path = pathlib.PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe archive member path")
    if path.suffix.lower() != ".jpg":
        return None
    for i, part in enumerate(path.parts):
        if part.startswith("Day"):
            return str(pathlib.PurePosixPath(*path.parts[i:]))
    return None


def run(source: pathlib.Path) -> dict:
    """전체 목표 경로를 대조하고 1GiB 한도 안에서 검토 표본만 추출한다.

    Args:
        source: 다운로드 완료된 valid_body_data TAR 또는 이를 감싼 TAR 경로.

    Returns:
        dict: 일치·누락 경로 수, 선택 이미지 해시·용량·원본 파일 상태.
        JPG 디코딩·시각 검토·FPS·점수 평가·학습은 수행하지 않는다.

    Raises:
        ValueError: 미완료·예상하지 않은 페이로드·경로 충돌·추출 한도 초과.
        OSError: 원본·이미지·보고서 접근 실패.
        tarfile.TarError: TAR 형식 또는 데이터 오류.
        RuntimeError: 온도·전원·메모리 보호 중단.
    """
    source = pathlib.Path(source).resolve()
    if source.suffix.lower() != ".tar":
        raise ValueError("Only completed TAR files may be inspected")
    check_resources()
    before = source.stat()
    folder = audit.OUT / "aihub_documents"
    plan = audit.read_json(folder / "raw_review_selection.json")
    if (
        shutil.disk_usage(audit.OUT).free
        < plan["extraction_budget_bytes"] + 10 * 1024**3
    ):
        raise RuntimeError(
            "Keep at least 10GiB free after selective extraction"
        )
    requested = set(
        (folder / "target_image_paths.txt").read_text().splitlines()
    )
    selected = {p for r in plan["selections"] for p in r["images"]}
    matched, extracted = set(), []
    total = 0
    with contextlib.ExitStack() as stack:
        outer = stack.enter_context(tarfile.open(source, "r:"))
        if before.st_size == EXPECTED_PAYLOAD_BYTES:
            payload = outer
            wrapper_member = None
        else:
            candidates = []
            for member in outer.getmembers():
                name = pathlib.PurePosixPath(member.name).name
                match = re.fullmatch(
                    r"valid_body_data\.tar\.part([0-9]+)", name
                )
                if member.isfile() and match:
                    candidates.append(
                        (int(match[1]), member.offset_data, member.size)
                    )
            if sum(row[2] for row in candidates) != EXPECTED_PAYLOAD_BYTES:
                raise ValueError("Incomplete or unexpected raw payload")
            raw_stream = stack.enter_context(source.open("rb"))
            joined = stack.enter_context(
                SegmentedReader(raw_stream, candidates)
            )
            wrapper_member = dict(
                segment_count=len(candidates), no_merged_copy_created=True
            )
            payload = stack.enter_context(
                tarfile.open(fileobj=joined, mode="r:")
            )
        seen = set()
        for index, member in enumerate(payload):
            if index % 500 == 0:
                check_resources()
            if not member.isfile():
                continue
            key = canonical_image_path(member.name)
            if key in requested:
                matched.add(key)
            if key not in selected:
                continue
            if key in seen:
                raise ValueError("Duplicate selected image path")
            seen.add(key)
            total += member.size
            if (
                total > plan["extraction_budget_bytes"]
                or member.size > 20 * 1024**2
            ):
                raise ValueError("Selected image extraction budget exceeded")
            content = payload.extractfile(member).read()
            target = audit.OUT / "aihub_raw_review" / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            extracted.append(
                dict(
                    path=key,
                    bytes=len(content),
                    sha256=hashlib.sha256(content).hexdigest(),
                )
            )
    after = source.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Source changed during inspection")
    report = dict(
        source=str(source),
        source_bytes=before.st_size,
        source_mtime_ns=before.st_mtime_ns,
        source_whole_file_sha256=None,
        wrapper_member=wrapper_member,
        target_paths=len(requested),
        target_paths_matched=len(matched),
        target_paths_missing=sorted(requested - matched),
        selected_images_requested=len(selected),
        selected_images_extracted=len(extracted),
        selected_images_missing=sorted(selected - seen),
        extracted_bytes=total,
        image_files=extracted,
        complete=True,
        image_decode_checked=False,
        directly_reviewed=False,
        score_labels_created=0,
        Heavy_extractions=0,
        independent_final_test_assignment=None,
    )
    audit.save_json(folder / "raw_archive_inspection.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    result = run(parser.parse_args().source)
    print(
        {
            key: result[key]
            for key in (
                "target_paths_matched",
                "target_paths",
                "selected_images_extracted",
                "extracted_bytes",
            )
        }
    )
