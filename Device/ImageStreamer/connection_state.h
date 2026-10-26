#ifndef AIFIT_CONNECTION_STATE_H
#define AIFIT_CONNECTION_STATE_H

#include <cstdint>

namespace connection {
/** @brief 네트워크 연결 수명입니다. Ready 이후 서버를 중복 시작하지 않습니다.
 */
enum class State : std::uint8_t { kIdle, kConnecting, kReady, kTimedOut };

/**
 * @brief 연결 결과와 경과 시간만으로 다음 상태를 결정하는 순수 함수입니다.
 * @param state 현재 연결 상태입니다.
 * @param connected Wi-Fi 드라이버가 연결 완료를 보고했는지 나타냅니다.
 * @param elapsed_ms 연결 시작부터 경과한 밀리초입니다. unsigned 차이로
 * 계산합니다.
 * @param timeout_ms 연결 제한 시간(밀리초)입니다.
 * @return 새 상태입니다. 완료·실패 상태는 추가 입력에 반응하지 않습니다.
 */
constexpr State advance(State state, bool connected, std::uint32_t elapsed_ms,
                        std::uint32_t timeout_ms) noexcept {
  if (state != State::kConnecting) {
    return state;
  }
  if (connected) {
    return State::kReady;
  }
  return elapsed_ms >= timeout_ms ? State::kTimedOut : State::kConnecting;
}
}  // namespace connection

#endif  // AIFIT_CONNECTION_STATE_H
