"""이 파일을 열고 VS Code 오른쪽 위 ▶ 버튼을 누르세요.

브라우저에서 운동 영상과 교정 문구 예시를 자동으로 엽니다.
추가 패키지 설치 없이 Python 3으로 실행합니다.
"""
from pathlib import Path
import argparse
import errno
import sys
import threading
import webbrowser

sys.path.insert(0, str(Path(__file__).resolve().parent / "test_app"))
from server import make_server


def main(default_page="/?dataset=shoulder&evidence=1&autoplay=1&demo=1", model_results=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    try:
        server = make_server(args.port)
    except OSError as error:
        if error.errno != errno.EADDRINUSE:
            raise
        server = make_server(0)
    url = f"http://127.0.0.1:{server.server_port}{default_page}"
    print("\n5종목 학습 모델 결과 실행" if model_results else "\n운동 영상 + 교정 문구 데모 실행", flush=True)
    print(f"화면 주소: {url}", flush=True)
    print("요약표와 종목별 테스트 영상, 실제 모델 예측을 확인하세요." if model_results else "영상 왼쪽 / 교정 문구 오른쪽. 다른 사례는 목록에서 선택하세요.", flush=True)
    print("영상 전체로 미리 계산한 모델 예측입니다. 재생 시점별 실시간 판별 화면은 아닙니다." if model_results else "저장된 라벨 예시이며 자동 자세 판별 결과는 아닙니다.", flush=True)
    print("종료: 이 터미널에서 Ctrl+C\n", flush=True)
    if not args.no_browser:
        threading.Thread(target=webbrowser.open, args=(url,), kwargs={"new": 2}, daemon=True).start()
    with server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
