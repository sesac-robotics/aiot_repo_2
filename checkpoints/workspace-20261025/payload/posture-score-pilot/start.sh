#!/usr/bin/env bash
# 별도 로컬 서버를 실행한다. 원본 영상과 결과는 jobs에 저장한다.
set -euo pipefail
cd "$(dirname "$0")"
exec /home/jayden/SeSACPAI-Mid/aiot_repo_2/R2D2/bin/python server.py
