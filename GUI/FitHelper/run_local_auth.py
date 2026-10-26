"""팀 회원 서버를 수정하지 않고 로컬 DB·인증 키를 공급하는 실행기입니다.

테스트 계정이나 기존 개인 DB를 배포하지 않습니다. DB 스키마 생성과 회원
처리는 저장소의 원본 AIFitService 수명 및 라우터에 위임합니다.
"""

import argparse
import os
import secrets

from Server.ExerciseEvaluation.paths import DATA_ROOT


def configure_local_auth():
    """환경 설정이 없을 때만 Git 제외 폴더에 개인 설정을 준비합니다."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.environ.setdefault("AIFIT_DB_PATH", str(DATA_ROOT / "AIFit.db"))
    if not os.environ.get("AIFIT_JWT_SECRET") and not os.environ.get(
        "JWT_SECRET"
    ):
        key_file = DATA_ROOT / "local-jwt.key"
        try:
            with os.fdopen(
                os.open(key_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600),
                "w",
            ) as stream:
                stream.write(secrets.token_urlsafe(48))
        except FileExistsError:
            pass
        os.environ["AIFIT_JWT_SECRET"] = key_file.read_text().strip()


def main():
    """로컬 주소에서 원본 회원 서버를 실행합니다. 카메라는 열지 않습니다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    configure_local_auth()
    import uvicorn

    uvicorn.run(
        "Server.AIFitService.main:app", host="127.0.0.1", port=args.port
    )


if __name__ == "__main__":
    main()
