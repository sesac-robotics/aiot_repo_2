"""원본 파일·라벨·구간·메타데이터의 연결과 기본 형식을 검증합니다. 형식 검증이 자세 라벨의 전문가 정답 검수를 대신하지 않습니다.

Validate schema compatibility, original video copies, IDs and split integrity.
"""

import csv
import hashlib
import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QEVD = ROOT.parent / "qevd-selected"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class LocalLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        self.links.extend(v for k, v in attrs if k in {"href", "src"})


def main():
    paths = [
        "raw/coach/fine_grained_labels.json",
        "raw/coach/feedbacks_short_clips.json",
        "raw/annotations/fine_grained_labels_with_worker_ids.json",
    ]
    payloads = {}
    for path in paths:
        actual, reference = read(ROOT / path), read(QEVD / path)
        assert isinstance(actual, list) and len(actual) == 100
        expected = set(reference[0])
        for record in actual:
            assert set(record) == expected, (path, record)
            for key in expected:
                assert type(record[key]) is type(reference[0][key]), (path, key)
            if "split" in record:
                assert record["split"] in {"train", "test"}
        payloads[path] = actual
    labels, feedbacks, workers = [payloads[p] for p in paths]
    ids = {r["video_path"] for r in labels}
    assert len(ids) == 100
    assert (
        ids
        == {r["video_path"] for r in feedbacks}
        == {r["video_path"] for r in workers}
    )
    with (ROOT / "manifests/short_clips.csv").open() as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        header = reader.fieldnames
    with (QEVD / "manifests/short_clips.csv").open() as stream:
        assert header == csv.DictReader(stream).fieldnames
    assert len(rows) == 100
    by_id = {r["video_id"]: r for r in rows}
    assert len(by_id) == 100
    worker_by_id = {Path(r["video_path"]).stem: r["worker_id"] for r in workers}
    feedback_by_id = {Path(r["video_path"]).stem: r for r in feedbacks}
    for record in labels:
        video_id = Path(record["video_path"]).stem
        row = by_id[video_id]
        assert (ROOT / row["video_path"]).is_file()
        assert record["labels"] == json.loads(row["labels"])
        assert record["labels_descriptive"] == json.loads(
            row["labels_descriptive"]
        )
        assert len(record["labels"]) == len(record["labels_descriptive"]) > 0
        assert all(
            label.startswith("shoulder press - ") for label in record["labels"]
        )
        assert (
            record["split"] == row["split"] == feedback_by_id[video_id]["split"]
        )
        assert int(row["worker_id"]) == worker_by_id[video_id]
    mapping = read(ROOT / "manifests/source_video_mapping.json")
    pending = read(ROOT / "pending/annotations_pending.json")
    quarantine = read(ROOT / "quarantine/annotations_quarantine.json")
    reviews = read(ROOT / "manifests/annotation_review.json")
    assert (len(mapping), len(pending), len(quarantine), len(reviews)) == (
        150,
        40,
        10,
        100,
    )
    assert len({r["video_id"] for r in mapping}) == 150
    all_draft_ids = {r["video_id"] for r in reviews}
    assert all_draft_ids == set(by_id)
    assert not (all_draft_ids & {r["video_id"] for r in pending + quarantine})
    assert Counter(r["source_quality"] for r in reviews) == {
        "good": 70,
        "bad": 30,
    }
    assert Counter(r["source_screening_status"] for r in quarantine) == {
        "duplicate_label_conflict": 6,
        "exercise_identity_review": 4,
    }
    assert all(
        r["annotation_status"] == "draft"
        and not r["training_ready"]
        and not r["expert_verified"]
        for r in reviews
    )
    assert all(
        r["segments"] is None and r["landmarks"] is None for r in reviews
    )
    for row in reviews:
        assert (ROOT / row["review_image"]).is_file()
        ex = row["example_evidence"]
        if ex:
            assert (ROOT / ex["comparison_image"]).is_file()
            for quality in ("good", "bad"):
                assert (ROOT / ex[quality]["clip_path"]).is_file()
    subjects = {}
    for record in mapping:
        if record["subject_id"] in subjects:
            assert subjects[record["subject_id"]] == record["reserved_split"]
        subjects[record["subject_id"]] = record["reserved_split"]
    train_workers = {r["worker_id"] for r in rows if r["split"] == "train"}
    test_workers = {r["worker_id"] for r in rows if r["split"] == "test"}
    assert not (train_workers & test_workers)
    assert len(train_workers) == 20 and len(test_workers) == 5
    assert (
        set(read(ROOT / "manifests/official_test_workers.json")) == test_workers
    )
    qe_ids = {Path(r["video_path"]).stem for r in read(QEVD / paths[0])}
    qe_workers = {r["worker_id"] for r in read(QEVD / paths[2])}
    assert not ({r["video_id"] for r in mapping} & qe_ids)
    assert not ({r["worker_id"] for r in mapping} & qe_workers)
    for record in mapping:
        assert (
            sha(ROOT / record["video_path"])
            == sha(record["source_path"])
            == record["sha256"]
        )
    parser = LocalLinks()
    parser.feed((ROOT / "preview.html").read_text())
    assert all((ROOT / link).is_file() for link in parser.links)
    from dataset_io import load_records

    assert len(load_records(ROOT)) == 0
    assert len(load_records(ROOT, include_draft=True)) == 100
    assert len(load_records(QEVD)) == 18652
    report = {
        "status": "passed",
        "meaning": "File/schema/provenance validation only; not pose-label correctness validation",
        "qe_json_key_sets_and_types_match": True,
        "qe_csv_columns_match": True,
        "all_150_source_copy_sha256_match": True,
        "all_100_labeled_records_and_paths_resolve": True,
        "existing_qevd_video_id_collisions": 0,
        "existing_qevd_worker_id_collisions": 0,
        "train_test_subject_overlap": 0,
        "split_counts": dict(Counter(r["split"] for r in rows)),
        "quarantine_excluded": 10,
        "pending_excluded": 40,
        "default_loader_unverified_drafts_loaded": 0,
        "explicit_draft_loader_rows": 100,
        "existing_qevd_loader_rows": 18652,
        "expert_verified_labels": 0,
        "preview_local_links_checked": len(parser.links),
    }
    (ROOT / "checks/validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
