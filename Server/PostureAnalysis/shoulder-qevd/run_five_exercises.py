"""VS Code에서 이 파일을 열고 오른쪽 위 ▶ 버튼을 누르세요.
5개 운동의 라벨 유형 목록에서 영상과 영어 교정 문구를 확인합니다.
"""

from build_five_exercises import main as build
from run_demo import main as run

if __name__ == "__main__":
    build()
    run(default_page="/five_exercises.html")
