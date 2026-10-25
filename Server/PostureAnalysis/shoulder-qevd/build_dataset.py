"""제공자 숄더 프레스 원본과 메타데이터를 정리합니다. 영상 경로·라벨·분할의 연결을 유지하는 데이터 어댑터입니다.

Package existing visual-review drafts in the local QEVD schemas.

No model-generated or inferred labels are silently treated as verified truth.
Source videos and qevd-selected are read only. Re-running verifies existing copies.
"""

import csv
import hashlib
import html
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT.parent / "shoulder-labeling"
QEVD = ROOT.parent / "qevd-selected"
EXERCISE = "shoulder press"
TEST_SUBJECTS = {
    "subject_001",
    "subject_006",
    "subject_011",
    "subject_018",
    "subject_024",
}

DEFINITIONS = {
    "NO_OBVIOUS_ISSUE": (
        "no obvious issue",
        "No obvious form issue is indicated by the source label",
        "정상 자세 후보",
        "기준 동작의 자세와 궤적을 유지해주세요.",
        "Keep following the reference movement and arm path.",
    ),
    "TOP_WIDE": (
        "arms too wide at the top",
        "User's arms spread too far outward at the top compared with the reference press",
        "상단에서 양팔이 바깥으로 벌어짐",
        "올릴 때 양팔이 바깥으로 퍼지지 않도록 기준 영상의 궤적에 맞춰주세요.",
        "Keep your arms from spreading outward at the top; follow the reference arm path.",
    ),
    "TOP_INCOMPLETE": (
        "incomplete upward range",
        "User lowers the weights before reaching the upward range shown in the reference press",
        "상단 도달 범위 부족",
        "기준 영상처럼 팔을 충분히 올린 뒤 내려주세요.",
        "Reach the upward position shown in the reference before lowering your arms.",
    ),
    "HANDS_BEHIND_HEAD": (
        "hands move behind the head",
        "User's hands move behind the head while the elbows remain high during lowering",
        "하강 시 손이 머리 뒤로 이동",
        "기준 프레스 동작처럼 어깨 옆에서 위로 밀어주세요.",
        "Follow the reference press path from beside your shoulders rather than folding the weights behind your head.",
    ),
    "FOREARM_INWARD": (
        "forearms tilt inward",
        "User's wrists move inward relative to the elbows during lowering",
        "하강 시 전완이 안쪽으로 기울어짐",
        "내릴 때 손만 안쪽으로 모으지 말고 손목과 팔꿈치의 정렬을 기준 영상에 맞춰주세요.",
        "Avoid moving only your hands inward as you lower; match wrist and elbow alignment to the reference.",
    ),
    "ELBOW_PATH_NARROW": (
        "elbow path narrower than reference",
        "User's elbows move closer to the torso at the bottom than in the reference press variation",
        "하단 팔꿈치 경로가 기준보다 좁음",
        "기준 프레스 동작을 따르는 경우 팔꿈치 간격을 기준 영상에 맞춰 조금 넓혀주세요.",
        "If following this reference press variation, adjust your elbow spacing slightly wider to match the reference.",
    ),
    "ARM_ASYMMETRY": (
        "uneven arm height and timing",
        "User's arms reach different heights and move at different times during the simultaneous press",
        "좌우 팔 높이·타이밍 불균형",
        "양팔을 같은 높이와 속도로 함께 올려주세요.",
        "Raise both arms together at matching heights and speeds for this simultaneous press variation.",
    ),
}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    source_rows = sorted(
        read(REVIEW / "video_inventory.json"),
        key=lambda r: (r["subject_id"], r["source_quality"], r["view"]),
    )
    examples = {
        (r["subject_id"], r["view"]): r
        for r in read(REVIEW / "error_examples.json")
    }
    # Keep review images and short example clips portable with the dataset.
    for directory in ("review", "examples"):
        output = ROOT / "docs/evidence" / directory
        output.mkdir(parents=True, exist_ok=True)
        for source in (REVIEW / directory).iterdir():
            if source.suffix.lower() in {".jpg", ".mp4"}:
                shutil.copy2(source, output / source.name)
    qe_labels = read(QEVD / "raw/coach/fine_grained_labels.json")
    qe_workers = read(
        QEVD / "raw/annotations/fine_grained_labels_with_worker_ids.json"
    )
    qe_ids = {Path(r["video_path"]).stem for r in qe_labels}
    qe_worker_ids = {r["worker_id"] for r in qe_workers}
    worker_map = {
        s: 900000 + int(s.split("_")[1])
        for s in sorted({r["subject_id"] for r in source_rows})
    }
    assert not (set(worker_map.values()) & qe_worker_ids)

    annotations, feedbacks, workers, manifest = [], [], [], []
    reviews, pending, quarantine, mapping, hashes = [], [], [], [], []
    for i, row in enumerate(source_rows):
        video_id = f"{90000000 + i:08d}"
        assert video_id not in qe_ids
        original = Path(row["source_path"])
        state = row["screening_status"]
        if state in {"duplicate_label_conflict", "exercise_identity_review"}:
            disposition = "quarantine"
            target_rel = f"quarantine/videos/{video_id}.mp4"
        elif state == "screened_unresolved":
            disposition = "pending"
            target_rel = f"pending/videos/{video_id}.mp4"
        else:
            disposition = "labeled_draft"
            target_rel = f"raw/fit/part1/{video_id}.mp4"
        target = ROOT / target_rel
        target.parent.mkdir(parents=True, exist_ok=True)
        original_sha = sha(original)
        if target.exists():
            assert (
                sha(target) == original_sha
            ), f"Unexpected existing output: {target}"
        else:
            shutil.copy2(original, target)
        assert sha(target) == original_sha
        hashes.append(
            {
                "video_id": video_id,
                "video_path": target_rel,
                "sha256": original_sha,
            }
        )

        split = "test" if row["subject_id"] in TEST_SUBJECTS else "train"
        common = {
            "video_id": video_id,
            "video_path": target_rel,
            "worker_id": worker_map[row["subject_id"]],
            "subject_id": row["subject_id"],
            "view": row["view"],
            "source_quality": row["source_quality"],
            "source_path": str(original),
            "duration_sec": row["duration_sec"],
            "disposition": disposition,
            "source_screening_status": state,
            "reserved_split": split,
            "sha256": original_sha,
        }
        mapping.append(common)
        if disposition != "labeled_draft":
            item = {
                **common,
                "reason_ko": row["notes_ko"],
                "labels": [],
                "training_ready": False,
            }
            (pending if disposition == "pending" else quarantine).append(item)
            continue

        codes = (
            ["NO_OBVIOUS_ISSUE"]
            if row["source_quality"] == "good"
            else row["candidate_codes"]
        )
        assert codes
        labels = [f"{EXERCISE} - {DEFINITIONS[c][0]}" for c in codes]
        descriptive = [f"{EXERCISE} - {DEFINITIONS[c][1]}" for c in codes]
        qe_path = f"./{video_id}.mp4"
        # Exact key sets and value types of the local QEVD files.
        annotations.append(
            {
                "video_path": qe_path,
                "labels": labels,
                "labels_descriptive": descriptive,
                "split": split,
            }
        )
        feedbacks.append(
            {
                "video_path": qe_path,
                "feedbacks": [DEFINITIONS[c][4] for c in codes],
                "split": split,
            }
        )
        workers.append(
            {"video_path": qe_path, "worker_id": worker_map[row["subject_id"]]}
        )
        manifest.append(
            {
                "video_id": video_id,
                "video_path": target_rel,
                "split": split,
                "worker_id": worker_map[row["subject_id"]],
                "labels": json.dumps(labels, ensure_ascii=False),
                "labels_descriptive": json.dumps(
                    descriptive, ensure_ascii=False
                ),
            }
        )
        original_example = (
            examples.get((row["subject_id"], row["view"]))
            if row["source_quality"] == "bad"
            else None
        )
        ex = (
            json.loads(json.dumps(original_example))
            if original_example
            else None
        )
        if ex:
            ex["comparison_image"] = (
                f"docs/evidence/examples/{Path(ex['comparison_image']).name}"
            )
            for quality in ("good", "bad"):
                ex[quality][
                    "clip_path"
                ] = f"docs/evidence/examples/{Path(ex[quality]['clip_path']).name}"
        review = {
            **common,
            "split": split,
            "error_codes": codes,
            "annotation_status": "draft",
            "expert_verified": False,
            "training_ready": False,
            "label_origin": (
                "source_binary_good_weak_label"
                if row["source_quality"] == "good"
                else (
                    "visual_comparison_example"
                    if ex
                    else "sampled_visual_screening_candidate"
                )
            ),
            "feedback_origin": "assistant_authored_template_not_source_ground_truth",
            "labels_ko": [DEFINITIONS[c][2] for c in codes],
            "feedbacks_ko": [DEFINITIONS[c][3] for c in codes],
            "labels_en": [DEFINITIONS[c][0] for c in codes],
            "labels_descriptive_en": descriptive,
            "feedbacks_en": [DEFINITIONS[c][4] for c in codes],
            "notes_ko": row["notes_ko"],
            "temporal_annotation": "clip_level_candidate_only",
            "segments": None,
            "segments_status": "not_annotated",
            "landmarks": None,
            "landmarks_status": "not_extracted",
            "sampled_frame_times_approx_sec": row["approx_sample_times_sec"],
            "review_image": f"docs/evidence/review/{Path(row['comparison_sheet']).name}",
            "example_evidence": ex,
        }
        reviews.append(review)

    write(ROOT / "raw/coach/fine_grained_labels.json", annotations)
    write(ROOT / "raw/coach/feedbacks_short_clips.json", feedbacks)
    write(
        ROOT / "raw/annotations/fine_grained_labels_with_worker_ids.json",
        workers,
    )
    fields = [
        "video_id",
        "video_path",
        "split",
        "worker_id",
        "labels",
        "labels_descriptive",
    ]
    (ROOT / "manifests").mkdir(exist_ok=True)
    with (ROOT / "manifests/short_clips.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(manifest)
    write(
        ROOT / "manifests/official_test_workers.json",
        sorted({str(r["worker_id"]) for r in manifest if r["split"] == "test"}),
    )
    write(
        ROOT / "manifests/split_policy.json",
        {
            "short_clips": "Custom subject-disjoint train/test split for the shoulder extension. This is not an official QEVD split.",
            "workers": "worker_id = 900000 + source subject number; same subject stays in one split across all views and qualities. No claim that people are disjoint across source datasets.",
            "paths": "Manifest video_path is relative to shoulder-qevd root; JSON ./ID.mp4 resolves by basename via the manifest, exactly as in qevd-selected.",
            "test_subject_ids": sorted(TEST_SUBJECTS),
            "annotation_status": "All 100 labeled rows are draft. Source-good weak labels and visual-error candidates are distinguished in annotation_review.json. No rows are expert-verified.",
            "pending": "40 unresolved clips are outside raw/coach and short_clips.csv.",
            "quarantine": "10 duplicate-conflict/exercise-identity clips are outside raw/coach and short_clips.csv. No source files were deleted.",
            "rare_classes": "ARM_ASYMMETRY has only one draft example. Do not infer class-level evaluation validity from this pilot split.",
        },
    )
    write(ROOT / "manifests/annotation_review.json", reviews)
    write(ROOT / "manifests/source_video_mapping.json", mapping)
    write(ROOT / "pending/annotations_pending.json", pending)
    write(ROOT / "quarantine/annotations_quarantine.json", quarantine)
    write(
        ROOT / "manifests/label_definitions.json",
        [
            {
                "code": code,
                "exercise": EXERCISE,
                "label": f"{EXERCISE} - {v[0]}",
                "labels_descriptive": f"{EXERCISE} - {v[1]}",
                "name_en": v[0],
                "name_ko": v[2],
                "annotation_type": (
                    "normal_candidate"
                    if code == "NO_OBVIOUS_ISSUE"
                    else "error_candidate"
                ),
                "feedback_en": v[4],
                "feedback_ko": v[3],
                "reference_relative": code != "NO_OBVIOUS_ISSUE",
                "thresholds": None,
                "validated": False,
            }
            for code, v in DEFINITIONS.items()
        ],
    )
    write(
        ROOT / "manifests/exercise_catalog.json",
        [
            {
                "exercise_id": "squats",
                "display_ko": "스쿼트",
                "label_prefixes": ["squats"],
                "source_dataset": "qevd-selected",
            },
            {
                "exercise_id": "pushups",
                "display_ko": "푸시업",
                "label_prefixes": ["pushups"],
                "source_dataset": "qevd-selected",
            },
            {
                "exercise_id": "elbow_plank",
                "display_ko": "엘보 플랭크",
                "label_prefixes": ["elbow plank"],
                "source_dataset": "qevd-selected",
            },
            {
                "exercise_id": "ytw",
                "display_ko": "YTW",
                "label_prefixes": [
                    "standing ytw",
                    "standing twy",
                    "standing ywt",
                    "standing wyt",
                    "standing wty",
                    "standing tyw",
                ],
                "source_dataset": "qevd-selected",
                "note": "Order variants remain distinct in the labels; grouped only for the five-exercise catalog.",
            },
            {
                "exercise_id": "shoulder_press",
                "display_ko": "덤벨 숄더 프레스",
                "label_prefixes": [EXERCISE],
                "source_dataset": "shoulder-qevd",
                "annotation_status": "draft",
            },
        ],
    )
    (ROOT / "checks").mkdir(exist_ok=True)
    with (ROOT / "checks/sha256.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, ["video_id", "video_path", "sha256"])
        writer.writeheader()
        writer.writerows(hashes)
    counts = {
        "source_videos": len(source_rows),
        "labeled_draft_videos": len(annotations),
        "source_good_weak_labels": sum(
            r["source_quality"] == "good" for r in reviews
        ),
        "visual_error_candidates": sum(
            r["source_quality"] == "bad" for r in reviews
        ),
        "error_examples_with_detailed_evidence": sum(
            r["example_evidence"] is not None for r in reviews
        ),
        "pending_videos": len(pending),
        "quarantine_videos": len(quarantine),
        "label_counts": dict(
            Counter(l for r in annotations for l in r["labels"])
        ),
        "split_counts": dict(Counter(r["split"] for r in annotations)),
        "source_subjects": len(worker_map),
        "expert_verified": 0,
        "training_ready": 0,
    }
    write(
        ROOT / "checks/build_report.json",
        {
            "dataset": "shoulder-qevd",
            "version": "0.1-draft",
            "created_date": "2026-10-21",
            "schema_compatible_with": "qevd-selected local short-clip JSON and CSV key sets",
            "original_videos_copied_without_transcoding": True,
            "original_files_sha256_matched": True,
            "qevd_source_modified": False,
            "counts": counts,
        },
    )
    esc = html.escape
    page = [
        '<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>어깨 QEVD 형식 데이터셋</title><style>body{font:15px system-ui,sans-serif;margin:0;background:#f4f6fa;color:#18293f}main{max-width:1180px;margin:auto;padding:28px}p{line-height:1.7}.note{padding:18px;background:#fff0ce;border-left:4px solid #d2a12e}.card{background:white;padding:22px;margin:22px 0;border-radius:14px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #ddd;vertical-align:top}a{color:#245ab5}img{width:100%;max-width:960px;height:auto}video{width:48%;max-height:380px;background:#101820}input{padding:12px;width:min(90%,500px);font:inherit}code{font-size:12px}details{padding:10px}@media(max-width:650px){main{padding:12px}td,th{padding:7px}video{width:100%}}</style><main><h1>어깨 운동 · QEVD 호환 데이터셋</h1><p>100개 라벨 초안 · 정상 후보 70 / 오류 후보 30 · 미확정 40 / 보류 10개 별도 보존</p><div class="note"><b>0.1-draft</b> — 모든 라벨은 검수 전 초안입니다. 정상 후보는 원본 good에서 파생했고, 오류 후보는 기존 시각 검토를 옮겼습니다. MediaPipe 좌표와 오류 구간 경계는 포함하지 않았습니다.</div><p><a href="README.md">설명</a> · <a href="raw/coach/fine_grained_labels.json">QEVD 라벨 JSON</a> · <a href="raw/coach/feedbacks_short_clips.json">피드백 JSON</a> · <a href="manifests/short_clips.csv">CSV 목록</a> · <a href="manifests/annotation_review.json">English labels · review notes</a></p><section class="card"><h2>향후 5종목 구성</h2><p>스쿼트 · 푸시업 · 엘보 플랭크 · YTW — 기존 QEVD<br>덤벨 숄더 프레스 — 이 패키지</p><p>기존 QEVD는 수정하지 않았습니다. 어깨 영상·참여자 ID는 현재 QEVD와 충돌하지 않는 별도 범위를 사용합니다.</p></section><section class="card"><h2>라벨 초안 100개</h2><input id="query" placeholder="참여자, 오류 코드, English label or feedback" aria-label="라벨 검색"><table><thead><tr><th>영상 / 참여자</th><th>라벨 / 피드백 초안</th><th>출처 / 자료</th></tr></thead><tbody>'
    ]
    for row in reviews:
        page.append(
            f'<tr class="record"><td><a href="{esc(row["video_path"])}">{row["video_id"]}.mp4</a><br>{esc(row["subject_id"])} / {esc(row["view"])}<br>{row["split"]}</td><td>{esc(", ".join(row["labels_en"]))}<br><code>{esc(", ".join(row["error_codes"]))}</code><p>{esc(" ".join(row["feedbacks_en"]))}</p></td><td><code>{esc(row["label_origin"])}</code><br><a href="{esc(row["review_image"])}">비교 시트</a></td></tr>'
        )
    page.append(
        '</tbody></table></section><section class="card"><h2>상세 비교 사례 10개</h2><p>good와 bad의 같은 동작 단계를 비교합니다. 6초 구간 전체가 오류라는 뜻이 아닙니다.</p>'
    )
    for row in reviews:
        ex = row["example_evidence"]
        if ex:
            page.append(
                f'<details><summary>{esc(row["subject_id"])} / {esc(row["view"])} — {esc(ex["error_name_ko"])}</summary><p>{esc(ex["observation_ko"])}</p><img loading="lazy" src="{esc(ex["comparison_image"])}" alt="good와 bad 비교"><p>왼쪽 good / 오른쪽 bad</p><video controls preload="none" src="{esc(ex["good"]["clip_path"])}"></video> <video controls preload="none" src="{esc(ex["bad"]["clip_path"])}"></video></details>'
            )
    page.append(
        '</section><section class="card"><h2>미확정·보류 파일</h2><p><a href="pending/annotations_pending.json">미확정 40개</a> · <a href="quarantine/annotations_quarantine.json">중복 충돌·운동 종류 확인 10개</a></p><p>이 파일들은 위 100개 라벨 목록에 넣지 않았습니다.</p></section></main><script>document.getElementById("query").addEventListener("input",function(){const q=this.value.toLowerCase();document.querySelectorAll(".record").forEach(r=>{r.hidden=!r.textContent.toLowerCase().includes(q)});});</script></html>'
    )
    (ROOT / "preview.html").write_text("".join(page))
    print(json.dumps(counts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
