"""개발 중인 팀 모듈을 읽기 전용으로 고정하여 한 세션의 버전을 보장한다."""
import hashlib
from pathlib import Path

SOURCE = Path('/home/jayden/SeSACPAI-Mid/team-posture-scoring/Server/AIService')


def read_source():
    """공개 점수 모듈과 배포 번들을 읽는다. 원본 파일은 수정하지 않는다."""
    paths = sorted((SOURCE/'app/scoring').glob('*.py'))
    paths += sorted(p for p in (SOURCE/'artifacts/posture_score').iterdir() if p.is_file())
    return {str(p.relative_to(SOURCE)):p.read_bytes() for p in paths}


def digest(files):
    """상대 경로·파일 길이·내용을 포함한 SHA-256 지문을 반환한다."""
    h = hashlib.sha256()
    for name, content in sorted(files.items()):
        h.update(name.encode()+b'\0'+str(len(content)).encode()+b'\0'+content)
    return h.hexdigest()


def prepare_snapshot():
    """두 번 읽은 지문이 같은 경우만 로컬 실행본을 생성한다.

    Returns:
        tuple[Path,str]: 고정 모듈 루트와 코드·모델 통합 지문.

    Raises:
        RuntimeError: 파일 쓰기 중이라 일관된 버전을 확보하지 못한 경우.
    """
    for _ in range(3):
        files = read_source()
        version = digest(files)
        if version != digest(read_source()):
            continue
        root = Path(__file__).parent/'runtime'/version
        for name, content in files.items():
            path = root/name
            path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                path.write_bytes(content)
            elif path.read_bytes()!=content:
                raise RuntimeError('고정 실행본 지문 불일치')
        (root/'app/__init__.py').touch(exist_ok=True)
        return root, version
    raise RuntimeError('팀 소스 갱신 중입니다. 잠시 후 다시 실행하세요.')
