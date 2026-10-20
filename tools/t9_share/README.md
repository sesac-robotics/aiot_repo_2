# 유선 T9 데이터 공유 도구

이영준 노트북에 연결된 T9 데이터를 유선혁 노트북에서 폴더처럼 읽는다.
학습은 **유선혁 노트북의 CPU/GPU**에서 실행한다. 이 도구는 학습 코드나
모델을 실행하지 않는다. 기존 카메라 프로그램·AI 서버는 수정하지 않는다.

| 구분 | 이영준: T9 제공 | 유선혁: 학습 |
| --- | --- | --- |
| 기존 카메라 연결 도구 역할 | B | A |
| 유선 IPv4 예시 | 192.168.50.2/24 | 192.168.50.1/24 |
| 공유 입력 폴더 | /run/media/jayden/T9/Data | ~/t9-data |
| 실행 명령 | serve | mount |
| 모델·특징 캐시·체크포인트 저장 | 공유 폴더에는 쓰지 않음 | 본인 로컬 디스크 |

공유 폴더는 읽기 전용이다. 공개키가 일치하는 상대 IP 한 개만 접속하며,
셸·명령 실행·포트 전달을 제공하지 않는다. SSH로 암호화하고 서버 키도
검증한다. 심볼릭 링크와 상위 경로 이동은 지원하지 않는다.
OS 계정·SSH 데몬·방화벽·네트워크 설정은 변경하지 않는다.
키 파일은 저장소 밖에 보관하며 개인키는 서로 보내지 않는다.

## 1. 양쪽 준비

LAN 케이블만 꽂으면 IPv4가 자동 설정되는 것은 아니다. 양쪽에서 기존
카메라 연결 도구를 실행해 위 표의 IP를 적용한 후 연결 확인을 한다.
이미 설정되어 있으면 다시 설정하지 않는다. 인터넷은 Wi-Fi를 유지한다.

```bash
cd ~/SeSACPAI-Mid/aiot_repo_2
/usr/bin/python3 tools/wired_camera_link.py
```

유선혁 노트북에서 아직 이 브랜치를 받지 않았다면 미완료 작업을 보존한 뒤
`git fetch origin`, `git switch --track origin/feat/wired-camera-link`로
연결한다. 이미 로컬 브랜치가 있으면 `git switch feat/wired-camera-link`,
`git pull --ff-only`를 사용한다. 변경이 있어 전환이 거절되면 강제 전환하지
않는다. 아래 명령들은 모두 저장소 루트에서 실행한다.

**각자 기존 학습용 가상환경을 활성화한 뒤** 별도 의존성을 설치한다.
학습 프레임워크 요구 버전과 충돌하면 이 도구 전용 가상환경을 사용한다.

```bash
python -m pip install -r tools/t9_share/requirements.txt
python -m tools.t9_share --help
```

유선혁 노트북에 SSHFS가 없으면 Ubuntu 패키지를 한 번 설치한다.

```bash
sudo apt install sshfs
```

## 2. 유선혁: 접속용 키 생성 및 공개키 전달 (최초 한 번)

기존 `~/.ssh/t9_training` 파일이 있다면 덮어쓰지 않고 기존 키를 사용한다.
새로 만드는 경우 아래 명령을 실행하고 암호는 본인 터미널에서 입력한다.

```bash
mkdir -p ~/.ssh
chmod 700 ~/.ssh
ssh-keygen -t ed25519 -f ~/.ssh/t9_training -C t9-training
ssh-keygen -lf ~/.ssh/t9_training.pub
```

**`t9_training.pub`만** 이영준에게 전달한다. 암호나 확장자 없는 개인키
`t9_training`은 보내지 않는다. 공개키 파일 내용과 지문을 서로 확인한다.

## 3. 이영준: 공유 실행

받은 공개키를 예를 들어 `~/Downloads/sunhyuk_t9.pub`로 저장한다.
T9가 실제 마운트되어 있고 데이터 폴더가 보이는지 먼저 확인한다.

```bash
findmnt -T /run/media/jayden/T9/Data
ls /run/media/jayden/T9/Data
python -m tools.t9_share serve \
  --bind 192.168.50.2 --peer 192.168.50.1 \
  --root /run/media/jayden/T9/Data \
  --client-key ~/Downloads/sunhyuk_t9.pub
```

`--root`는 공유 허용된 특정 데이터셋 폴더로 좁혀도 된다. 예를 들어
`--root /run/media/jayden/T9/Data/REHAB26`을 쓰면 상대의 마운트 최상위가
REHAB26 내용이다. 공유가 데이터셋 사용 승인을 대신하지 않으므로,
공유 대상은 팀원이 이용할 수 있는 데이터로 정한다.

서버가 출력한 키 지문을 확인하고 다음 **공개 신뢰 파일**을 유선혁에게
전달한다. 개인 서버 키 `host_key`는 보내지 않는다.

```text
~/.local/share/aiot-t9-share/t9_known_hosts
```

이 파일은 개인키가 아니라 서버 공개키다. 이후 재실행 시 같은 서버 키를
사용한다. IP나 `--port`를 바꾸면 새 주소가 적힌 신뢰 파일을 다시 전달한다.
서버 터미널은 켜 둔다. 자동 시작·백그라운드 등록은 하지 않는다.

## 4. 유선혁: 폴더 연결

받은 신뢰 파일을 `~/Downloads/t9_known_hosts`로 저장하고 지문을 비교한다.

```bash
ssh-keygen -lf ~/Downloads/t9_known_hosts
python -m tools.t9_share mount \
  --host 192.168.50.2 \
  --identity ~/.ssh/t9_training \
  --known-hosts ~/Downloads/t9_known_hosts \
  --mountpoint ~/t9-data
ls ~/t9-data
```

공유 포트는 양쪽 모두 기본 8022다. 변경하면 `serve`와 `mount`의
`--port`를 같은 값으로 지정한다. `t9`는 이 도구 안의 가상 접속 이름이며
이영준의 Ubuntu 계정 비밀번호는 필요 없다.

## 5. 학습 코드에서 사용

전체 Data를 공유했다면 다음 경로가 학습 입력이다.

```text
~/t9-data/QEVD
~/t9-data/REHAB26
~/t9-data/Pose-Estimation-Model-Comparison
```

기존 학습 코드의 데이터 루트 설정을 이 경로로 바꾼다. 아래는 경로를
읽는 예시이며 모든 학습 프로그램이 이 환경변수를 자동 지원하는 것은 아니다.

```bash
export T9_DATA_ROOT="$HOME/t9-data"
mkdir -p ~/training-output
```

```python
import os
import pathlib

data_root = pathlib.Path(os.environ["T9_DATA_ROOT"])
qevd_root = data_root / "QEVD"
output_root = pathlib.Path.home() / "training-output"
```

- **출력·전처리 캐시·체크포인트·로그는 `~/training-output` 등 로컬에 저장한다.**
  원본 옆에 캐시를 쓰는 데이터 로더는 저장 위치 설정을 변경해야 한다.
- 기존 CSV/JSON에 `/run/media/jayden/T9/Data/...` 절대 경로가 있으면,
  데이터 로더에서 그 접두사만 `~/t9-data`의 실제 절대 경로로 바꿔 읽는다.
  T9 원본 메타데이터를 직접 수정하지 않는다. 전처리된 데이터에 다른
  절대 경로가 들어 있는지도 확인한다.
- 처음에는 DataLoader `num_workers=0` 등 작은 읽기 병렬도로 확인하고
  측정 후 늘린다. 프레임 단위 무작위 원격 읽기가 느리면 필요한 데이터나
  추출 특징을 유선혁 노트북에 복사·캐시해 학습한다. 복사 가능 범위는
  해당 데이터 사용 조건을 따른다.
- 유선 연결은 T9를 물리적으로 꽂은 것과 속도가 같지 않다. 실제 속도는
  양쪽 LAN·어댑터·SSH 처리·파일 크기에 따라 다르며 GPU 사용률로 확인한다.
- 학습 중 두 노트북을 잠자기 상태로 만들거나 T9·LAN 케이블을 분리하지 않는다.
  연결이 끊기면 학습은 읽기 오류로 중단되거나 대기할 수 있다.

## 6. 종료 및 문제 해결

학습을 먼저 멈춘 뒤 **유선혁 노트북에서** 연결을 해제한다.

```bash
python -m tools.t9_share unmount --mountpoint ~/t9-data
```

이후 이영준 서버 터미널에서 Ctrl+C를 누른다. 마지막으로 양쪽 카메라 연결
도구의 ‘원복’을 누른다. T9는 사용이 끝난 뒤 Ubuntu에서 안전 제거한다.

| 증상 | 확인 |
| --- | --- |
| 유선 경로가 없다는 오류 | 카메라 연결 도구에서 양쪽 IPv4 적용. Wi-Fi 경로를 대신 사용하지 않음 |
| 서버 IP 불일치 | `ip -br addr`로 이영준의 실제 유선 IPv4 확인 |
| 연결 거절·시간 초과 | 서버 실행·양쪽 IP·8022 포트·방화벽 확인. 방화벽 전체를 끄지 않음 |
| Permission denied (publickey) | 상대 공개키와 mount의 개인키가 한 쌍인지 확인 |
| 서버 키 불일치 | 서버 지문을 직접 비교. 검증 옵션을 끄거나 기존 신뢰를 무조건 삭제하지 않음 |
| 폴더가 비어 있지 않음 | 다른 빈 마운트 폴더 사용. 기존 파일 삭제 불필요 |
| 쓰기 실패 | 정상 동작. 출력·캐시 경로를 로컬 디스크로 변경 |
| 해제 시 busy | 학습·파일 탐색기를 닫고 터미널을 마운트 폴더 밖으로 이동 |

## 구성·검증

- `filesystem.py`: 공유 루트 기준 파일 조회 및 임의 바이트 위치 읽기.
- `server.py`: 공개키 인증, 상대 IP 제한, SSH 연결 정리.
- `__main__.py`: 명령 입력 검증·유선 경로 확인·SSHFS 연결.
- `test_share.py`: 합성 파일의 실제 SSH 읽기·쓰기 거부·경로 탈출·인증 시험.

```bash
python -m unittest tools.t9_share.test_share -v
```

팀 기준 확인: 공통 안내 v6(3932327), Python v3(3604597), 설계·모듈화
v3(3866722), Git v3(3899580), 한글 주석 v2(6979834), Git 입문 v2(6947031).
URL은 `https://r2d2sesac.atlassian.net/wiki/spaces/YLI4jLZJ4ELw/pages/` 뒤에
각 ID를 붙여 확인한다. 기존 공통 DB/API 계약을 변경하지 않는 독립 도구다.

SFTP 구현 API: https://docs.paramiko.org/en/stable/api/sftp.html
