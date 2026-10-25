

<!-- integration-standards-20261025 -->

## 통합된 장치 코드

Arduino 스케치 진입점은 폴더와 이름이 같은 `ImageStreamer.ino`입니다. 기존 CameraWebServer 예제 파일의 이름을 Arduino 빌드 규칙에 맞췄습니다. 카메라 설정 구조체는 `{}`로 초기화하고 센서 포인터 실패를 검사합니다. Wi-Fi 접속 정보는 저장소에 직접 넣지 않습니다.

```bash
cp Device/ImageStreamer/wifi_credentials.example.h Device/ImageStreamer/wifi_credentials.h
# 위 로컬 파일에 네트워크 값을 입력합니다. Git에서 제외됩니다.
arduino-cli compile --fqbn esp32:esp32:esp32cam Device/ImageStreamer
```

`connection_state.h`는 Idle/Connecting/Ready/TimedOut을 enum class로 표현합니다. setup은 카메라를 구성하고 연결을 요청합니다. loop는 접속 상태와 unsigned 밀리초 경과를 검사해 연결 성공 시 서버를 한 번만 시작합니다. 30초가 지나면 연결을 종료하고 설정 확인 후 재시작하도록 안내합니다. Ready 이후 반복 loop가 웹 서버를 다시 시작하지 않습니다. unsigned 차이는 millis 오버플로 시에도 짧은 경과 시간을 올바르게 계산합니다.

상태 전이는 하드웨어 없는 순수 함수이므로 `tests/firmware/test_connection_state.cpp`에서 성공·시간 초과·중복 호출·타이머 오버플로를 C++17로 검증합니다. ESP32 코어 3.3.12와 arduino-cli 1.5.1에서 전체 스케치 컴파일도 확인했습니다. 이는 실제 Wi-Fi·영상 스트림·장치 장시간 동작 검증과 구분합니다. 업로드는 별도 명시적 작업입니다.

`app_httpd.cpp`는 Espressif CameraWebServer 예제 기반 서버·JPEG 처리 코드, `camera_pins.h`는 보드별 핀 매핑, `camera_index.h`는 압축 웹 UI 자산입니다. 해당 파일을 프로젝트 도메인 계층으로 재작성하거나 압축 배열을 서식 변경하지 않습니다. SDK 내부의 동적 메모리 사용까지 제거했다고 주장하지 않습니다. 코어 버전 업그레이드 시 호환성 검증이 필요합니다. ESP32Cam.py는 Qt로 영상을 직접 확인하는 독립 PC 시험 도구입니다.

### 소스별 책임과 공개 진입점

| 소스 | 책임과 해석 | 주요 공개 진입점 |
|---|---|---|
| [Device/ImageStreamer/ESP32Cam.py](ESP32Cam.py) | ESP32 영상을 직접 확인하는 독립 Qt 시험 도구입니다. 자동 서비스 진입점이 아니며 명시적 실행 시에만 카메라를 엽니다. | `VideoThread`, `MainWindow` |
