# 임시 JWT 로그인 시험

`python main.py`는 기존 디자인(`mainLoginWindow.ui`)을 유지합니다. 로그인 버튼은 이 폴더의 JWT API 코드를 사용해 `/api/auth/login`과 `/api/me`를 시험합니다. 기존 회원가입 화면과 나머지 GUI는 유지됩니다.

- 시험 화면: `jwt_auth_test/login.py`
- 기존 디자인과 JWT 시험을 연결하는 실행 파일: `main.py`와 `login_window.py`
- 이전 실행 파일 백업: `archive/pre_jwt_auth_test/main.py`
- 시험 GUI의 API 기본 주소: `http://192.168.15.118:8000` (기존 GUI 서버 주소)
- 다른 주소를 시험할 때는 실행 전 `AIFIT_SERVER_URL` 환경 변수로 지정합니다.

이 코드는 `POST /api/auth/login`, `POST /api/auth/register`, 로그인 후 `GET /api/me`를 호출합니다. 서버가 이 경로와 응답 필드(`success`, 로그인 시 `access_token` 및 `user`)를 제공해야 JWT 검증까지 통과합니다. 기존 서버의 `/api/login` 또는 `/api/register`만 제공된다면 경로가 맞지 않아 실패할 수 있습니다.

JWT 시험을 중단하고 기존 방식으로 돌아가려면 `archive/pre_jwt_auth_test/main.py`와 `archive/pre_jwt_auth_test/login_window.py`를 각각 현재 위치의 파일로 복사하면 됩니다.
