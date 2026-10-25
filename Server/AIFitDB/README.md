

<!-- integration-standards-20261025 -->

# AIFit DB 구성요소

`AIFit.sql`은 SQLite 기본 스키마입니다. UserInfo → UserBodyInfo, ExerciseSession → Exercise → ExerciseRecord/ExerciseReport의 관계를 외래 키로 표현합니다. Routine은 분석 결과와 세션에서 공유할 수 있으므로 참조가 남아 있으면 삭제하지 않습니다. ExerciseRecord의 `(exercise_id, set_no, rep_no)` UNIQUE 제약이 같은 반복의 저장 중복을 제한합니다.

백엔드 `database/connection.py`가 연결·외래 키 활성화·커밋/롤백·종료를 소유합니다. `init_db`는 기본 스키마, 사진 분석과 workout 초기화는 기존 추가 컬럼/테이블을 준비합니다. 이 방식은 빈 DB 시작을 복구하지만 장기적으로 명시적 버전 마이그레이션 도구가 필요한 구조입니다. 데이터베이스 제약과 Python 입력 검증은 서로 보완하며 어느 하나가 다른 것을 대체하지 않습니다.

`db_test.py`는 import 시 DB를 수정하지 않도록 main으로 감쌌습니다. 두 데모는 명시적으로 실행하면 데모 데이터를 저장하거나 삭제할 수 있습니다. 자동 검증은 운영 DB 대신 `tests/integration/test_backend.py`의 임시 DB를 사용합니다. `.db` 파일과 로컬 실행 산출물은 Git에서 제외합니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [db_test.py](db_test.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `main` |
| [db_test_v2.py](db_test_v2.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `check`, `main` |
