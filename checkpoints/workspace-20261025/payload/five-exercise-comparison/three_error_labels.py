"""새 오류 원본의 직접 등급 결정을 기존 QEVD 점수 주석에 연결한다."""

import three_external_labels

if __name__ == "__main__":
    print(three_external_labels.run("error_label_decisions.json"))
