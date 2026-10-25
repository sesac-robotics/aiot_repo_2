"""AI-Hub 다운로드 완료 여부와 압축파일 구성만 읽는 로컬 인수 검사다.

원본을 풀거나 코드를 실행하지 않는다. 비상업적 교육 목적과 학습 후 원본
제거 계획은 사용자 의사이며 개별 파일의 이용 허락·모델 호환성을 뜻하지 않는다.
"""

import datetime
import pathlib
import zipfile
import audit

POLICY_URL = (
    "https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105"
)
DATASET_URL = "https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=231"


def inspect_archive(path: pathlib.Path) -> dict:
    """완료된 ZIP의 중앙 디렉터리와 작은 설명 파일만 읽는다.

    Args:
        path: 완료된 ZIP 경로. .crdownload·.part 파일은 거부한다.

    Returns:
        dict: 멤버 수·압축 전후 바이트·경로 이상·가중치 후보·작은 설명문.
        실제 모델 입력·학습 가능 여부는 이 목록만으로 확정하지 않는다.

    Raises:
        ValueError: 미완료 다운로드 또는 ZIP이 아닌 경로.
        OSError: 파일 접근 실패.
        zipfile.BadZipFile: ZIP 디렉터리 손상 또는 형식 오류.
    """
    path = pathlib.Path(path)
    if path.suffix.lower() != ".zip":
        raise ValueError("Only completed ZIP files may be inspected")
    before = path.stat()
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [i.filename for i in infos]
        invalid = [
            name
            for name in names
            if pathlib.PurePosixPath(name.replace("\\", "/")).is_absolute()
            or ".." in pathlib.PurePosixPath(name.replace("\\", "/")).parts
        ]
        descriptions = []
        for info in infos:
            base = pathlib.PurePosixPath(info.filename).name.lower()
            if (
                base.startswith(("readme", "license", "requirements"))
                and info.file_size <= 65536
                and not info.is_dir()
                and not info.flag_bits & 1
            ):
                descriptions.append(
                    dict(
                        name=info.filename,
                        text=archive.read(info).decode(
                            "utf-8", errors="replace"
                        ),
                    )
                )
                if len(descriptions) >= 12:
                    break
        result = dict(
            path=str(path),
            bytes=before.st_size,
            members=len(infos),
            uncompressed_bytes=sum(i.file_size for i in infos),
            unsafe_paths=invalid,
            files_not_extracted=True,
            code_not_executed=True,
            model_candidates=[
                n
                for n in names
                if pathlib.PurePosixPath(n).suffix.lower()
                in (".pth", ".pt", ".onnx", ".h5", ".ckpt", ".pkl")
            ][:100],
            source_code_count=sum(n.endswith(".py") for n in names),
            json_count=sum(n.endswith(".json") for n in names),
            member_sample=names[:100],
            descriptions=descriptions,
            compatibility_confirmed=False,
        )
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Archive changed while inspected")
    return result


def run() -> dict:
    """다운로드 폴더의 진행 상태와 AI-Hub 활용 계획을 별도 JSON에 저장한다.

    Returns:
        dict: 진행 파일 크기와 오늘 완료된 ZIP 인수 결과. 원본 삭제는 실행하지
        않으며 학습·검증 완료와 보관 범위를 확인한 뒤 사용자 계획을 이행한다.

    Raises:
        OSError: 다운로드 폴더 또는 결과 접근 실패.
    """
    folder = pathlib.Path.home() / "Downloads"
    today = datetime.datetime.now().date()
    recent = [
        p
        for p in folder.iterdir()
        if p.is_file()
        and datetime.datetime.fromtimestamp(p.stat().st_mtime).date() == today
    ]
    pending = [
        dict(path=str(p), bytes=p.stat().st_size, content_identified=False)
        for p in recent
        if p.suffix.lower() in (".crdownload", ".part")
    ]
    completed = []
    for path in recent:
        if path.suffix.lower() == ".zip":
            try:
                completed.append(inspect_archive(path))
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                completed.append(dict(path=str(path), error=str(error)))
    previous_path = audit.OUT / "aihub_intake.json"
    previous = audit.read_json(previous_path) if previous_path.exists() else {}
    managed = [
        dict(path=str(path), bytes=path.stat().st_size)
        for path in (audit.BASE / "data/aihub231").glob("*/*.tar")
        if path.is_file()
    ]
    report = dict(
        previous,
        managed_files=managed,
        checked_at=datetime.datetime.now().astimezone().isoformat(),
        user_intent_ko="SeSAC Physical 로보틱스 과정의 비상업적 학습; 학습 후 원본 제거",
        user_reported_download_in_progress=bool(pending),
        pending=pending,
        completed_archives=completed,
        dataset_url=DATASET_URL,
        policy_url=POLICY_URL,
        dataset_and_model_terms_still_to_inspect=True,
        official_model_task="RGB ResNet+TSM exercise type/state estimation; not confirmed 0-100 score regression",
        next_checks_ko=[
            "모델 코드·가중치·실행 환경·입출력 확인",
            "플랭크 및 전경 YTW 운동 정의·구도·참여자·연속 프레임 여부 확인",
            "제공자 상태 문자열을 점수로 변환하지 않고 항목별 직접 초안과 구분",
            "같은 참여자·원본·다중 구도·반복을 같은 분할로 묶음",
            "학습과 검증 완료 후 원본 제거 범위와 재현용 모델·비식별 지표 보관 확인",
        ],
        original_deleted=False,
        source_media_shared=False,
        final_test_or_training_records_added=0,
    )
    audit.save_json(audit.OUT / "aihub_intake.json", report)
    return report


if __name__ == "__main__":
    result = run()
    print(
        dict(
            pending_files=len(result["pending"]),
            completed_archives=len(result["completed_archives"]),
        )
    )
