"""부족한 지지·복귀·좌우 기울기 오류를 목표로 별도 외부 검색을 실행한다."""

import three_external_discovery

if __name__ == "__main__":
    report = three_external_discovery.run(
        queries=[
            ("pushups", "push up mistakes failed pushup"),
            ("pushups", "Roller Derby Athletics push ups mistakes"),
            ("shoulder", "standing dumbbell shoulder press wrong form"),
            ("squats", "squat hip shift uneven squat mistakes"),
        ],
        report_name="external_discovery_errors",
    )
    print(len(report["records"]))
