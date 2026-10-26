"""현재 지지·복귀 오류에 맞는 원제작자 후보를 추가로 검색한다."""

import three_external_discovery

if __name__ == "__main__":
    print(
        len(
            three_external_discovery.run(
                queries=[
                    ("pushups", "push up worm common faults"),
                    (
                        "shoulder",
                        "dumbbell shoulder press common faults side lean",
                    ),
                ],
                report_name="external_discovery_faults",
            )["records"]
        )
    )
