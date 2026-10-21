# AIFit GUI 영상 클라이언트 (Ubuntu)

`main.py`를 실행하면 PyQt6 GUI가 열리고 AIFit 서버의 MJPEG 스트림을 표시합니다.

## 1. 가상환경 생성 및 활성화

```bash
sudo apt update
sudo apt install python3-venv python3-pip
mkdir -p ~/venv
python3 -m venv ~/venv/aifit_gui
source ~/venv/GUI_Video/bin/activate
```

## 2. 패키지 설치

압축을 푼 폴더로 이동한 후:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 서버 주소 설정

`video_gui.py`의 기본 주소는 `http://192.168.15.118:8000`입니다.
실제 AIFit 서버 IP와 다르면 코드의 기본값을 바꾸거나 다음처럼 환경변수를 지정하세요.

```bash
export AIFIT_SERVER_URL="http://실제서버IP:8000"
```

URL에 `<` 또는 `>` 기호를 넣지 마세요.

## 4. 실행

```bash
python main.py
```

GUI에서 **영상 시작** 버튼을 누르면 `/api/video/stream`에 연결합니다.

## 5. 서버 측 조건

AIFit 서버가 실행 중이어야 합니다. 서버는 `POST /api/video/frame`으로 AI 서버가 보낸 JPEG 프레임을 받은 뒤 `GET /api/video/stream`으로 GUI에 전달합니다. GUI는 카메라를 직접 촬영하지 않습니다.

연결 상태 확인:

```bash
curl -i http://실제서버IP:8000/api/video/status
```

`has_frame`이 `true`여야 영상 프레임이 수신된 상태입니다.
