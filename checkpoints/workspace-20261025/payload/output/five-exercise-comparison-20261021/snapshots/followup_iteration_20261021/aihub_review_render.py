"""AI-Hub 자료 위치·정리·검토 범위를 기존 로컬 웹에 표시한다."""

import html
import pathlib
import audit


def section() -> str:
    """현재 인수·정리·직접 검토 기록으로 HTML 구역을 반환한다.

    Returns:
        str: 한국어 상태와 검토판 링크. 점수·학습 완료를 생성하지 않는다.

    Raises:
        OSError: 필수 결과 JSON을 읽을 수 없는 경우.
        KeyError: 필수 진단 구조가 없는 경우.
    """
    folder = audit.OUT / "aihub_documents"
    raw = audit.read_json(folder / "raw_archive_inspection.json")
    views = audit.read_json(folder / "raw_decode_and_sheets.json")
    cleanup = audit.read_json(
        audit.OUT / "disk_cleanup_verified_duplicates.json"
    )
    wrappers = audit.read_json(
        audit.OUT / "disk_cleanup_download_wrappers.json"
    )
    recovered = (
        cleanup["deleted_archive_bytes"]
        + cleanup["hardlink_logical_duplicate_bytes_removed"]
        + wrappers["deleted_bytes"]
    )
    parts = [
        '<section id="aihub"><h2>AI-Hub 자료 이동·선택 추출</h2>',
        '<p><a href="aihub_documents/file_relocation.json">자료 이동</a> · '
        '<a href="aihub_documents/raw_archive_inspection.json">원시 목록 대조</a> · '
        '<a href="aihub_documents/training_body_summary.json">Training 라벨 조사</a> · '
        '<a href="disk_cleanup_verified_duplicates.json">중복 정리 검증</a></p>',
        "<p>AI-Hub 원시 자료와 제공 모델은 프로젝트 data/aihub231/raw 및 model 폴더로 이동했습니다. "
        "같은 파일 시스템의 동일 inode 이동을 확인했으며 복사본은 만들지 않았습니다. "
        "라벨·문서와 분석 결과는 이 프로젝트의 output 폴더에 보존했습니다.</p>",
        f'<p>원시 27개 분할 조각은 합친 복사본 없이 읽었습니다. 목표 경로 {raw["target_paths_matched"]:,}/{raw["target_paths"]:,}개 일치, '
        f'선택 표본 {raw["selected_images_extracted"]:,}장만 추출해 {raw["extracted_bytes"]/1e6:.1f}MB를 사용했습니다. '
        f'{views["decoded_images"]:,}장 해시·JPEG 디코딩을 검사했습니다. '
        f"별도 중복 정리로 약 {recovered/1e9:.1f}GB를 회수했습니다. "
        "정리한 기존 데이터의 논리 경로와 내용은 유지했으며 동일 파일은 하드링크를 공유합니다. 변환 결과는 별도 파일에 저장합니다.</p>",
        f'<p>직접 검토: 제공자 정상 조건의 검토판 14장, 카메라 5개 × 시간 표본 3개씩 총 {views["images_directly_reviewed"]}개. '
        "전체 945장의 직접 검토가 아닙니다. 이 최초 구도 검토에서는 점수 라벨을 만들지 않았습니다.</p>",
        "<p>Y-Exercise는 바닥에 엎드린 Y 들기로 확인했습니다. 전경 YTW 전체 자료로 편입하지 않습니다. "
        "플랭크 C 구도는 측면에 가깝지만 기구 기둥에 가림이 있고, B는 측면 사선 후보입니다. "
        "준비·종료 장면이 섞여 제공자 정상 조건을 모든 프레임 점수로 복사하지 않습니다. FPS와 실제 연속 시간도 미확인입니다.</p>",
        "<p>Training: 플랭크 207·푸시업 832·Y 204개 라벨, 세 종목 참여자 토큰 26종. "
        "세 종목별 Validation 토큰 중복은 없지만 전체 맨몸 종목에서는 공통 토큰 5종이 있습니다. "
        "원시 인물·중복 대조 전 독립 시험으로 확정하지 않습니다.</p>",
        "<p>비상업적 SeSAC 교육·학습 목적과 학습 후 원본 제거 계획을 기록했습니다. "
        '<a href="https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=231">출처: AI-Hub 피트니스 자세 이미지, 한국지능정보사회진흥원 사업결과</a>. '
        "로컬 검토 화면이며 원본 재배포를 하지 않습니다.</p>",
    ]
    static_path = audit.OUT / "plank_static_review/comparison.json"
    if static_path.exists():
        static = audit.read_json(static_path)
        parts.append(
            '<p><a href="plank_static.html">후속 플랭크 정지 정렬 항목 AI/규칙 개발 비교</a>: '
            f'B f17 원본 {static["source_images"]}장을 7개 검토판에서 직접 확인하고 '
            f'정렬 항목 초안 {static["directly_drafted_trunk_labels"]}개를 Heavy 결과 전에 저장했습니다. '
            f'Heavy IMAGE 관측 {static["observable_images"]}장 중 라벨과 공통 비교 가능한 것은 {static["common_evaluated_images"]}장입니다. '
            "골반 밴드·기구 지지를 배제할 수 없고 시간 정보도 없어 기본 플랭크 총점 학습 자료 추가는 0개입니다. "
            "AI는 단순 평균 기준선을 넘지 못했으며 공통 총점 모델에 채택하지 않았습니다.</p>"
        )
    for row in views["sheets"]:
        path = pathlib.Path(row["image"]).relative_to(audit.OUT)
        title = row["exercise"] + " / " + ", ".join(row["person_tokens"])
        parts.append(
            f"<details><summary>{html.escape(title)} — 직접 구도 검토 표본</summary>"
            f'<p>{html.escape(row.get("finding_ko", "직접 검토 전"))}</p>'
            f'<img src="{html.escape(str(path))}" alt="{html.escape(title)} 카메라·시간 표본"></details>'
        )
    parts.append("</section>")
    return "".join(parts)


def run() -> None:
    """구도 검토 HTML을 로컬 출력 폴더에 저장한다.

    Returns:
        None: aihub_review.html을 저장하고 종료한다.

    Raises:
        OSError: 결과 읽기 또는 HTML 저장 실패.
        KeyError: 필수 진단 항목 누락.
    """
    head = '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AI-Hub 자료 확인</title><style>body{font:16px/1.7 system-ui;max-width:1100px;margin:auto;padding:24px;background:#14212c;color:#edf4fa}a{color:#80dcff}img{max-width:100%}details{padding:12px;border-bottom:1px solid #567}</style><p><a href="index.html">점수 비교</a> · <a href="w_analysis.html">W 원인 분석</a></p>'
    (audit.OUT / "aihub_review.html").write_text(
        head + section() + "</html>", encoding="utf-8"
    )


if __name__ == "__main__":
    run()
