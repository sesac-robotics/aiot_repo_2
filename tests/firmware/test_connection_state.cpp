#include <cassert>
#include <cstdint>

#include "Device/ImageStreamer/connection_state.h"

int main() {
  using connection::advance;
  using connection::State;
  constexpr std::uint32_t kTimeout = 30000U;
  static_assert(advance(State::kIdle, true, 0U, kTimeout) == State::kIdle);
  assert(advance(State::kConnecting, false, 29999U, kTimeout) ==
         State::kConnecting);
  assert(advance(State::kConnecting, false, 30000U, kTimeout) ==
         State::kTimedOut);
  assert(advance(State::kConnecting, true, 30000U, kTimeout) == State::kReady);
  assert(advance(State::kReady, true, 50000U, kTimeout) == State::kReady);
  assert(advance(State::kTimedOut, true, 50000U, kTimeout) == State::kTimedOut);
  constexpr std::uint32_t kStart = 0xFFFFFFF0U;
  constexpr std::uint32_t kNow = 0x00000010U;
  static_assert(kNow - kStart == 32U);
  assert(advance(State::kConnecting, false, kNow - kStart, kTimeout) ==
         State::kConnecting);
  return 0;
}
