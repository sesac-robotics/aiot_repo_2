"""교정과 점수를 함께 실행하는 전용 진입점입니다."""

from .server import main

if __name__ == "__main__":
    main(fixed_mode="combined")
