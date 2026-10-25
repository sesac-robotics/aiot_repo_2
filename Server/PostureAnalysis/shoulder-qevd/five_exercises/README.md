# 5종목 공통 라벨링 데이터

`../build_five_exercises.py`를 실행해 재생성합니다. `../run_five_exercises.py`는 영상 확인 화면을 엽니다.

- `manifests/label_definitions.json`: 공통 코드, 운동, 원본 라벨, 영어 설명, 교정 문구, 유형 및 출처.
- `manifests/annotations.json`: 영상별 다중 라벨 코드, 실제 경로, split, worker, 검토 상태.
- `raw/coach/`: QEVD와 같은 JSON 필드 구성. 원본 라벨 문구와 split을 유지합니다.
- `manifests/short_clips.csv`: QEVD와 같은 열 구성. 영상 복제 없이 절대 경로로 기존 영상을 참조합니다.
- `checks/validation_report.json`: 경로/코드/ID/피험자 분할 검사.

전체 18,752개 중 QEVD 18,652개는 기존 원본 라벨을 재사용하고, 어깨 100개는 기존 초안을 사용합니다. 라벨 70종 모두가 오류는 아닙니다. 정상 후보, 자세 특성, 동작 속도·순서, 가시성, 모호하거나 충돌하는 라벨도 별도 category로 보존합니다.

교정 문구는 초안이며 새 전문가 검수는 없습니다. training_ready=false는 이번 통합본의 검수 기준이며 QEVD 원본 자체가 학습 불가능하다는 의미는 아닙니다. YTW는 서 있는 몸통/숙인 몸통 변형을 아직 영상별 판정하지 않았습니다. segments=null, landmarks=null이며 오류 발생 프레임을 뜻하지 않습니다. 실제 학습 전 목표 변형 확인, 수동 검수, 포즈 추출이 필요합니다. 어깨의 미확정 40개와 보류 10개는 제외했습니다.
