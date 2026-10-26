

<!-- integration-standards-20261025 -->

# 개발·시험 도구

wired_camera_link는 유선 시험망 설정을 검증하고 외부 명령 어댑터로 적용·원복합니다. 실제 네트워크를 변경하는 apply 동작과 입력 검증을 구분합니다. test_wired_camera_link는 가짜 명령 실행기로 대역 충돌·잘못된 주소·실패 시 이전 연결 복구를 확인합니다.

t9_share는 허용된 공유 루트의 파일 목록·전송을 HTTP/SSH 경계에서 처리합니다. filesystem은 경로 제한, server는 전송과 인증, __main__은 CLI 조립을 담당합니다. 공유 도구 테스트는 로컬 임시 디렉터리와 루프백 서비스를 사용하며 실제 팀 계정으로 파일을 전송하지 않습니다. 선택 의존성인 paramiko를 requirements에서 설치해야 합니다.

운영 앱을 시작하거나 네트워크를 바꾸는 도구를 단순 문서 확인을 위해 실행하지 않습니다. 테스트의 mock 통과와 실제 두 PC의 네트워크 연결 성공은 별도 결과입니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [tools/check_source_contracts.py](check_source_contracts.py) | 통합 브랜치의 파싱·설명·의존 방향·import 경로 계약을 검사합니다. | `check_sources`, `main` |
| [tools/t9_share/__init__.py](t9_share/__init__.py) | 이 디렉터리의 공개 패키지 경계를 정의합니다. 구체적인 역할과 의존 방향은 상위 README의 구성요소 설명에 정리합니다. | 패키지 경계/선언 |
| [tools/t9_share/__main__.py](t9_share/__main__.py) | 패키지를 명시적으로 실행할 때 명령행 진입점에 제어를 전달합니다. import만으로 작업을 시작하지 않습니다. | `run`, `check_wired_route`, `create_host_key`, `start_share`, `mount_share`, `main` |
| [tools/t9_share/filesystem.py](t9_share/filesystem.py) | 공유 루트 내부의 일반 파일만 읽는 SFTP 어댑터. | `RootReader`, `ReadHandle`, `ReadOnlySftp` |
| [tools/t9_share/server.py](t9_share/server.py) | 공개키와 상대 IP로 제한한 읽기 전용 SSH/SFTP 서버. | `load_public_key`, `fingerprint`, `KeyAuth`, `serve_connection`, `serve` |
| [tools/t9_share/test_share.py](t9_share/test_share.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `RootReaderTest`, `SftpIntegrationTest`, `KeyTest` |
| [tools/test_wired_camera_link.py](test_wired_camera_link.py) | 이 구성요소의 입력·출력 및 실패 처리 계약을 검증합니다. 합성 입력이나 저장된 시험 자료의 검사 범위는 각 테스트 이름과 단언문에 명시합니다. | `LinkTests` |
| [tools/wired_camera_link.py](wired_camera_link.py) | 우분투에서 카메라 수신 노트북 간 유선 시험 연결을 설정한다. | `run_command`, `LinkSettings`, `validate_settings`, `NetworkLink`, `LinkWindow`, `main` |
