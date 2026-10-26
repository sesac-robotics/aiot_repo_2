

<!-- integration-standards-20261025 -->

## 통합 구조 설명

실행은 저장소 루트의 `python -m GUI.FitHelper.main`입니다. `AIFIT_SERVER_URL`을 백엔드 주소로 지정합니다. 생략하면 `http://127.0.0.1:8000`입니다. 파일을 직접 실행하기 위해 `sys.path`를 바꾸는 방식을 제거했습니다.

`main → QApplication.aifit_context → LoginWindow → WelcomeWindow` 순서로 객체가 생성됩니다. ApplicationContext는 하나의 AuthSession을 소유하며 토큰과 프로필을 함께 설정·해제합니다. 다른 API 클라이언트로 토큰을 복제하지 않습니다. AuthSession은 Qt·requests를 모르는 순수 상태 객체이고, ApplicationContext는 Qt 실행과 상태 객체를 연결하는 조립 경계입니다.

로그인 어댑터는 `/api/auth/login` 뒤 `/api/me` 검증까지 성공해야 상태를 게시합니다. 둘 중 하나가 실패하면 이전 토큰도 제거합니다. RequestTask는 QRunnable을 스레드 풀에 제출하고 QObject의 슬롯에서 결과를 받습니다. 화면 콜백이 실행되는 곳은 GUI 스레드입니다. 작업자는 위젯을 직접 수정하지 않습니다. 로그인·회원가입·프로필 요청 중 중복 제출을 막습니다. 사진 분석·운동·리포트 화면은 기존 QThread 작업자를 유지합니다.

운동 화면의 수치는 백엔드/AI 세션 상태의 표시입니다. 프레임 수신 스레드가 별도 횟수를 계산하지 않습니다. VideoReceiver는 바이트 파서 → OpenCV JPEG 디코딩 → RGB 변환 → QImage 복사 → frame_received 신호 순서로 영상을 전달합니다. QImage 복사는 원본 NumPy 버퍼가 해제된 뒤에도 표시 메모리를 보존하기 위해 필요합니다.

`login_user.py`, `video_gui.py`, `jwt요청용 코드 (절대지우지마셈)`은 이전 독립 시험 경로입니다. 현재 인증 흐름과 혼동하지 않도록 보존 목적을 표시합니다. 관리자 GUI 디렉터리는 구현된 서비스가 아닙니다. 캐시는 분석 ID의 편의 저장이며 서버 정답이나 인증 저장소가 아닙니다.

구조 검증은 `tests/integration/test_gui.py`에 있습니다. Qt offscreen 환경에서 요청이 다른 스레드에서 실행되는지, GUI 이벤트 처리가 계속 가능한지, 결과 콜백이 GUI 스레드인지 검사합니다. 이 검사는 전체 화면 배치·실기 동작 완료를 보장하지 않습니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [GUI/AIFitAdmin/main.py](../AIFitAdmin/main.py) | 관리자 GUI 자리표시자입니다. 현재 구현된 관리자 기능이 없으므로 포트폴리오에서 완성 기능으로 설명하지 않습니다. | 패키지 경계/선언 |
| [GUI/FitHelper/application_context.py](application_context.py) | QApplication이 소유하는 의존성을 조립합니다. 네트워크 연결은 만들지 않습니다. | `ApplicationContext`, `get_context` |
| [GUI/FitHelper/bodyAnalysisLoading.py](bodyAnalysisLoading.py) | 사진 분석 대기 화면과 상태 표시 타이머를 관리합니다. HTTP 분석 작업의 완료 상태와 화면 표시 상태를 구분합니다. | `BodyAnalysisLoadingWindow` |
| [GUI/FitHelper/body_analysis_cache.py](body_analysis_cache.py) | 사용자별 최근 체형 분석 ID를 저장·조회하고 부위 상태를 화면 문구로 변환합니다. 인증 토큰이나 분석 정답을 저장하는 모듈이 아닙니다. | `account_key`, `load_body_id`, `save_body_id`, `clear_body_id`, `area_groups`, `format_area_report`, `format_area_summary` |
| [GUI/FitHelper/body_analysis_client.py](body_analysis_client.py) | 체형 사진 업로드와 분석 조회를 HTTP 요청으로 변환합니다. 인증 상태는 login_client의 단일 세션에서 읽고, 업로드 파일은 성공·실패와 관계없이 닫습니다. | `BodyAnalysisAPIError`, `get_status`, `upload_photos`, `get_analysis` |
| [GUI/FitHelper/body_analysis_report.py](body_analysis_report.py) | 서버의 체형 분석 결과를 표시하고 루틴 화면으로 이동합니다. 화면은 모델 추론이나 DB 쿼리를 수행하지 않습니다. | `BodyAnalysisReportWindow` |
| [GUI/FitHelper/body_upload_window.py](body_upload_window.py) | 정면·좌측·우측·후면 사진 선택, 비동기 업로드, 분석 상태 폴링을 연결합니다. 작업 완료 신호를 받은 GUI 스레드에서 화면을 갱신합니다. | `BodyAnalysisLoadingWindow`, `BodyUploadWindow` |
| [GUI/FitHelper/domain/auth_session.py](domain/auth_session.py) | 로그인 상태의 단일 소유자입니다. Qt와 HTTP 라이브러리에 의존하지 않습니다. | `AuthSession` |
| [GUI/FitHelper/history_window.py](history_window.py) | 날짜별 운동 리포트를 조회해 달력과 상세 결과에 표시합니다. 네트워크 작업과 Qt 위젯 갱신의 스레드를 구분합니다. | `HistoryDetailWindow` |
| [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/__init__.py](jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [GUI/FitHelper/jwt요청용 코드 (절대지우지마셈)/login.py](jwt%EC%9A%94%EC%B2%AD%EC%9A%A9%20%EC%BD%94%EB%93%9C%20%28%EC%A0%88%EB%8C%80%EC%A7%80%EC%9A%B0%EC%A7%80%EB%A7%88%EC%85%88%29/login.py) | 이 파일의 구체적인 계약은 공개 함수와 상위 README에 명시합니다. | `request_json`, `response_message`, `LoginWindow` |
| [GUI/FitHelper/login_client.py](login_client.py) | 인증 HTTP 어댑터입니다. 인증 상태는 QApplication 컨텍스트 한 곳에 저장합니다. | `LoginClient`, `login`, `get_access_token`, `get_user`, `update_user`, `logout` |
| [GUI/FitHelper/login_user.py](login_user.py) | 이전 로그인 API를 시험하던 호환용 클라이언트입니다. 현재 GUI의 인증 진입점은 login_client와 login_window이며 이 파일은 과거 호출 계약 비교용으로 보존합니다. | `LoginError`, `login` |
| [GUI/FitHelper/login_window.py](login_window.py) | 로그인·회원가입 입력을 검증하고 요청 작업을 제출합니다. 인증 성공 후 대시보드로 이동하며 요청 중에는 중복 제출을 막습니다. | `RegistrationCompleteWindow`, `RegistrationWindow`, `LoginWindow` |
| [GUI/FitHelper/main.py](main.py) | QApplication, 인증 세션 컨텍스트와 로그인 화면을 조립하는 GUI 진입점입니다. 컴포넌트는 실행 위치와 무관한 패키지 경로로 가져옵니다. | `main` |
| [GUI/FitHelper/profile_window.py](profile_window.py) | 프로필 조회·변경, 비밀번호 변경, 탈퇴 확인 화면을 연결합니다. HTTP 요청은 작업 스레드에서 수행하고 완료 콜백에서 위젯과 인증 상태를 갱신합니다. | `ProfileAPIError`, `ProfileSettingsWindow` |
| [GUI/FitHelper/request_worker.py](request_worker.py) | 시간이 걸리는 요청을 Qt 스레드 풀에서 실행하고 결과만 화면으로 전달합니다. | `RequestTask` |
| [GUI/FitHelper/routine_window.py](routine_window.py) | 분석 결과에 포함된 주간 루틴을 조회해 운동 이름·세트·횟수·유지 시간으로 표시합니다. 이전 응답 형식은 화면 경계에서 읽기 형식으로 변환합니다. | `FullRoutineWindow` |
| [GUI/FitHelper/video_gui.py](video_gui.py) | 독립 영상 수신 GUI 시험 화면입니다. 본 서비스의 운동 화면은 workout_window이며 이 파일의 카운터를 업무 세션의 권위로 사용하지 않습니다. | `VideoReceiver`, `VideoWindow` |
| [GUI/FitHelper/video_receiver.py](video_receiver.py) | JWT를 첨부해 MJPEG 영상을 수신하고 JPEG를 디코딩해 QImage 신호를 보냅니다. RGB 메모리를 복사해 수신 버퍼 수명과 화면 메모리를 분리합니다. | `VideoReceiver` |
| [GUI/FitHelper/welcome_window.py](welcome_window.py) | 로그인 후 대시보드, 체형 분석 요약과 하위 화면 이동을 담당합니다. 분석 ID 캐시와 서버의 분석 완료 상태를 구분합니다. | `WelcomeWindow` |
| [GUI/FitHelper/workout_window.py](workout_window.py) | 운동 시작·중지·재개·종료 명령과 라이브 상태 조회를 연결합니다. 화면 카운터는 서버 응답의 투영이며 별도 운동 횟수 판정자가 아닙니다. | `WorkoutWindow` |
