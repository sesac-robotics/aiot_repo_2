"""새 오류 후보의 개요 검토판을 이전 결과와 분리해 준비한다."""

import three_discovery_review

if __name__ == "__main__":
    print(
        len(
            three_discovery_review.run(
                "external_discovery_errors.json", "external_errors"
            )["records"]
        )
    )
