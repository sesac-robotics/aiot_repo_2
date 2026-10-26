# 브랜치 검증 기록

- 검증일: 2026-10-21 (Asia/Seoul)
- 대상: `Server/PostureAnalysis`의 GitHub 업로드용 복사본
- 실행: `python -m unittest test_pipeline test_suite test_live test_api test_suite_api`
- 결과: 총 23개 중 22개 통과, 1개 skip.
- skip 사유: 전체 추출 특징 배열 `data/squat.npz`는 로컬 자료로 유지하므로, 전체 train 배열에서 정규화를 재계산하는 검증은 전처리 후 실행해야 합니다.
- 확인 범위: 학습/검증/시험 worker 분리, 종목별 추론과 캐시 확률 일치, 관절 누락 거부, API 입력 검증, 실시간 세션 초기화, 교정 문구 안정화, 중지 후 마지막 결과 보존.
- Python 소스 문법, JSON 파싱, 커밋 대상 확장자 및 크기를 확인했습니다.
- 원본 영상·사진·실시간 로그는 커밋하지 않았습니다. 수치 관절 fixture 5개와 학습된 모델은 포함했습니다.
- 모델 성능은 저장 데이터 시험 결과입니다. 실제 ESP32-CAM 사람 테스트의 오류별 정답 대비 정확도를 검증한 결과가 아닙니다.
