#!/usr/bin/env bash
# 역할: 앱과 독립된 사용자 서비스로 야간 라벨링·학습을 시작하거나 재개한다.
# 입력: 없음. 출력: 서비스 활성 상태. 물리 단위 없음.
# 실패: 사용자 systemd 또는 서비스 시작 실패 시 해당 종료 코드를 전달한다.
set -euo pipefail
systemctl --user daemon-reload
systemctl --user start posture-overnight-20261020.service
systemctl --user is-active posture-overnight-20261020.service
