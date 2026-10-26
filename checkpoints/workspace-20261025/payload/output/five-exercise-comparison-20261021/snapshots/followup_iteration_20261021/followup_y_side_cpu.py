"""새 측면 Y 후보 한 원본의 짧은 표본만 CPU 저부하로 검토·추출한다.

GPU 추출기의 AC 필수 보호를 우회하지 않는다. CPU delegate와 엄격한
배터리·온도 보호를 별도로 쓰며 이전 GPU 처리 시간 측정에 합산하지 않는다.
"""

import argparse
import os
import pathlib
import subprocess
import sys

import cv2
import numpy as np

import audit
from cpu_analysis_guard import check_resources
from external_review import contact_sheet
from extract import extract_view
from ytw_side_components import evaluate
from render import overlay

SOURCE_ID = "Yp-C_Si1gHE"


def check_cpu() -> dict:
    """짧은 CPU 작업의 자원 조건을 확인하고 측정값을 반환한다.

    Returns:
        dict: CPU70℃·GPU60℃·배터리80% 중단 조건의 현재 자원값.

    Raises:
        RuntimeError: 저부하 CPU 보호 조건 위반.
    """
    return check_resources("short_heavy_cpu")


def prepare() -> dict:
    """CC를 확인한 원본을 360p로 확보하고 두 6초 구간만 표본화한다.

    Returns:
        dict: 라이선스·원본 해시·두 검토판·원본 PNG 계보. 아직 점수 라벨은 없다.

    Raises:
        OSError: 원본·메타데이터·검토 이미지 접근 실패.
        ValueError: CC 라이선스 미확인 또는 프레임 읽기 실패.
        subprocess.SubprocessError: 90초 제한 내 다운로드 실패.
        RuntimeError: CPU 보호 중단.
    """
    check_cpu()
    metadata = audit.OUT / f"sources/youtube_{SOURCE_ID}.metadata.json"
    info = audit.read_json(metadata)
    if (
        info.get("license")
        != "Creative Commons Attribution license (reuse allowed)"
    ):
        raise ValueError("Verified reusable CC license required")
    source = audit.OUT / f"sources/youtube_{SOURCE_ID}.mp4"
    if not source.exists():
        env = dict(os.environ, PYTHONPATH=str(audit.BASE / "data/tools/yt-dlp"))
        subprocess.run(
            [
                sys.executable,
                "-m",
                "yt_dlp",
                "--load-info-json",
                str(metadata),
                "--match-filters",
                'license~="Creative Commons"',
                "-f",
                "bestvideo[height<=360][filesize<=80M][ext=mp4][vcodec^=avc1]",
                "--max-filesize",
                "80M",
                "--socket-timeout",
                "15",
                "--retries",
                "1",
                "--no-progress",
                "--no-continue",
                "-o",
                str(source),
            ],
            env=env,
            timeout=90,
            check=True,
        )
    records = []
    for name, start, end in [
        ("followup_josh_side_A", 80, 86),
        ("followup_josh_side_B", 105, 111),
    ]:
        sheet = contact_sheet(
            source,
            audit.OUT / f"review/{name}.jpg",
            start,
            end,
            36,
            resource_checker=check_cpu,
        )
        cap = cv2.VideoCapture(
            str(source), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        images = []
        try:
            for fraction in [0.2, 0.5, 0.8]:
                index = round((start + (end - start) * fraction) * sheet["fps"])
                cap.set(1, index)
                ok, pixels = cap.read()
                if not ok:
                    raise ValueError("Y candidate exact frame unavailable")
                target = audit.OUT / f"review/{name}_{index}.png"
                if not cv2.imwrite(str(target), pixels):
                    raise OSError("Y candidate PNG write failed")
                images.append(dict(frame=index, path=str(target)))
        finally:
            cap.release()
        records.append(
            dict(id=name, start=start, end=end, sheet=sheet, raw_images=images)
        )
    report = dict(
        source_id=SOURCE_ID,
        source_url=f"https://www.youtube.com/watch?v={SOURCE_ID}",
        creator=info["uploader"],
        person_group=info["channel_id"],
        source_group=f"youtube_{SOURCE_ID}_all_segments",
        license=info["license"],
        metadata_sha256=audit.digest(metadata),
        video_path=str(source),
        video_sha256=audit.digest(source),
        bytes=source.stat().st_size,
        records=records,
        labels_created=0,
        delegation="CPU_only_short_review",
    )
    audit.save_json(audit.OUT / "followup_y_side_preparation.json", report)
    return report


def extract() -> dict:
    """직접 표시한 최대 두 4초 이하 구간만 CPU Heavy로 추출·대조한다.

    Returns:
        dict: CPU delegate 캐시와 전경·Y 측면 성분 결과. 양팔 Y 전체나
        YTW 총점을 생성하지 않는다. 기존 GPU 런타임 표에 합산하지 않는다.

    Raises:
        OSError: 직접 초안·원본·캐시 접근 실패.
        ValueError: 구간 길이·개수·계보 또는 규칙 동결 불일치.
        RuntimeError: CPU Heavy 추출·자원 보호 실패.
    """
    path = audit.OUT / "followup_y_side_direct.json"
    frozen = audit.digest(path)
    direct = audit.read_json(path)
    if not 0 < len(direct["reviews"]) <= 2:
        raise ValueError("At most two short side intervals permitted")
    results = []
    for draft in direct["reviews"]:
        segment = draft["segment"]
        duration = (
            segment["end_frame"] - segment["start_frame"] + 1
        ) / segment["fps"]
        if not 0 < duration <= 4.1:
            raise ValueError(
                "CPU candidate interval must be at most four seconds"
            )
        row = dict(
            draft, required_views=["directly_verified_side_Y_component_only"]
        )
        result = extract_view(
            row,
            "primary",
            draft["video_path"],
            audit.OUT / "followup_y_side_CPU_cache",
            delegate="CPU",
            resource_checker=check_cpu,
        )
        with np.load(result["cache"]) as cache:
            seq = cache["image_xy"].copy()
        rule = evaluate(seq, draft["verified_side_view"])
        if draft["Y_sagittal_grade"] is None:
            rule["unscoped_Y_geometry_diagnostic_only"] = rule[
                "y_sagittal_component_grade"
            ]
            rule["y_sagittal_component_grade"] = None
        results.append(
            dict(
                id=draft["id"],
                extraction=result,
                direct_hinge_grade=draft["hinge_grade"],
                direct_Y_sagittal_grade=draft["Y_sagittal_grade"],
                rule=rule,
            )
        )
    if audit.digest(path) != frozen:
        raise ValueError("Y direct drafts changed during extraction")
    report = dict(
        records=results,
        direct_sha256=frozen,
        delegate="CPU",
        full_Y_or_YTW_total_created=False,
        expert_validated=False,
        independent_final_test=False,
        learning_status="new_observed_components_only_not_enough_people_for_Y_AI",
    )
    audit.save_json(audit.OUT / "followup_y_side_results.json", report)
    return report


def render_short_overlays() -> dict:
    """이미 추출한 최대 9개 짧은 구간을 CPU로 표시하고 전체 디코딩을 검사한다.

    Returns:
        dict: 원본·관절 계보와 MP4·중간 정지화면·디코딩 수. W는 직접
        표시 구간의 0~2등급이며 플랭크 총점·Y 성분과 합산하지 않는다.

    Raises:
        OSError: 캐시·원본·표시 결과 접근 실패.
        ValueError: 4초 초과 구간, 프레임·영상 디코딩 오류.
        RuntimeError: CPU 보호 조건 위반.
    """
    check_cpu()
    extraction = audit.read_json(audit.OUT / "followup_extraction.json")
    caches = {
        r["id"]: r
        for key in ["plank_records", "W_records"]
        for r in extraction[key]
    }
    plank = audit.read_json(audit.OUT / "followup_plank_learning_results.json")
    predictions = {
        r["id"]: r
        for key in ["LOPO_predictions", "new_validation"]
        for r in plank[key]
    }
    plan = []
    for draft in audit.read_json(audit.OUT / "followup_plank_direct.json")[
        "reviews"
    ]:
        r = predictions[draft["id"]]
        plan.append(
            dict(
                id=draft["id"],
                source=draft["video_path"],
                cache=caches[draft["id"]]["cache"],
                ai=r["after_ai"],
                rule=r["rule_score"],
                unit="total_points_0_100",
                core=None,
            )
        )
    drafts = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "followup_W_direct.json")[
            "reviews"
        ]
    }
    for r in audit.read_json(audit.OUT / "followup_W_learning_results.json")[
        "new_development_validation"
    ]:
        if r["ai"] is None:
            continue
        draft = drafts[r["id"]]
        plan.append(
            dict(
                id=r["id"],
                source=draft["video_path"],
                cache=caches[r["id"]]["cache"],
                ai=r["ai"],
                rule=r["component_rule"],
                unit="W_grade_0_2_human_window",
                core=draft["direct_W_core_frames"],
            )
        )
    drafts = {
        r["id"]: r
        for r in audit.read_json(audit.OUT / "followup_y_side_direct.json")[
            "reviews"
        ]
    }
    for r in audit.read_json(audit.OUT / "followup_y_side_results.json")[
        "records"
    ]:
        plan.append(
            dict(
                id=r["id"],
                source=drafts[r["id"]]["video_path"],
                cache=r["extraction"]["cache"],
                ai=None,
                rule=None,
                unit="Y_hinge_components_only_no_total_score",
                core=None,
            )
        )
    if len(plan) > 9:
        raise ValueError("Only nine fixed short overlays permitted")
    results = []
    for row in plan:
        check_cpu()
        folder = audit.OUT / "followup_overlays" / row["id"]
        folder.mkdir(parents=True, exist_ok=True)
        with np.load(row["cache"]) as cache:
            seq = cache["image_xy"].copy()
            frames = cache["frame_numbers"].copy()
        if row["core"]:
            a, b = row["core"]
            mask = (frames >= a) & (frames <= b)
            seq, frames = seq[mask], frames[mask]
        if not 0 < len(seq) <= 41:
            raise ValueError("Only at most four-second pose overlays permitted")
        np.savez_compressed(
            folder / "primary.npz", image_xy=seq, frame_numbers=frames
        )
        video = overlay(
            dict(exercise="component", video_path=row["source"]),
            dict(
                cache_folder=str(folder),
                ai_score=row["ai"],
                rule_score=row["rule"],
                display_unit="/2" if row["unit"].startswith("W_grade") else "",
            ),
            resource_checker=check_cpu,
        )
        cap = cv2.VideoCapture(
            str(video), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1]
        )
        count = 0
        preview = None
        try:
            while True:
                ok, pixels = cap.read()
                if not ok:
                    break
                if count == len(seq) // 2:
                    preview = pixels
                count += 1
        finally:
            cap.release()
        if count != len(seq) or preview is None:
            raise ValueError("Overlay frame decode mismatch")
        png = folder / "preview.jpg"
        if not cv2.imwrite(str(png), preview):
            raise OSError("Overlay preview write failed")
        results.append(
            dict(
                id=row["id"],
                video=str(video),
                preview=str(png),
                decoded_frames=count,
                unit=row["unit"],
                parent_cache=row["cache"],
                parent_cache_sha256=audit.digest(pathlib.Path(row["cache"])),
                selected_original_frames=frames.tolist(),
                new_extraction=False,
                ai_display_value=row["ai"],
                rule_display_value=row["rule"],
            )
        )
    report = dict(
        records=results, CPU_only=True, completed=True, new_Heavy_extractions=0
    )
    audit.save_json(audit.OUT / "followup_overlay_verification.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "extract", "render"])
    args = parser.parse_args()
    print(
        prepare()
        if args.mode == "prepare"
        else extract() if args.mode == "extract" else render_short_overlays()
    )
