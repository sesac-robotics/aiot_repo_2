"""Read QEVD-style files through their manifest, preserving label provenance."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json


def load_records(root, include_draft=False):
    root = Path(root).resolve()
    with (root / "manifests/short_clips.csv").open() as stream:
        manifest = {r["video_id"]: r for r in csv.DictReader(stream)}
    labels = json.loads((root / "raw/coach/fine_grained_labels.json").read_text())
    feedback_path = root / "derived/coach/feedbacks_short_clips.json"
    mapped_feedback = feedback_path.exists()
    if not mapped_feedback:
        feedback_path = root / "raw/coach/feedbacks_short_clips.json"
    feedbacks = {Path(r["video_path"]).stem: r["feedbacks"] for r in json.loads(feedback_path.read_text())}
    review_file = root / "manifests/annotation_review.json"
    reviews = {r["video_id"]: r for r in json.loads(review_file.read_text())} if review_file.exists() else None
    records = []
    for entry in labels:
        video_id = Path(entry["video_path"]).stem
        row = manifest[video_id]
        review = reviews.get(video_id) if reviews is not None else None
        if reviews is not None and not include_draft:
            # Missing status, pending and draft rows must never pass by default.
            if not review or review["annotation_status"] != "verified" or not review["training_ready"]:
                continue
        records.append({
            "video_id": video_id, "video_path": str(root / row["video_path"]),
            "worker_id": int(row["worker_id"]), "split": entry["split"],
            "labels": entry["labels"], "labels_descriptive": entry["labels_descriptive"],
            "feedbacks": feedbacks.get(video_id, []),
            "feedback_origin": "assistant_authored_label_mapping" if mapped_feedback else "assistant_authored_template" if review else "source_dataset",
            "feedback_status": "draft" if mapped_feedback or review else "source_dataset",
            "annotation_status": review["annotation_status"] if review else "source_dataset_annotation_unassessed",
            "label_origin": review["label_origin"] if review else "source_dataset",
            "training_ready": review["training_ready"] if review else None,
            "feedbacks_ko": review["feedbacks_ko"] if review else [],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--include-draft", action="store_true", help="Include unverified drafts for inspection/experiments; does not validate them")
    args = parser.parse_args()
    rows = load_records(args.root, include_draft=args.include_draft)
    print(json.dumps({"loaded": len(rows), "split": dict(Counter(r["split"] for r in rows)),
                      "labels": dict(Counter(l for r in rows for l in r["labels"])),
                      "note": "Explicitly includes unverified drafts" if args.include_draft else "Draft rows are excluded if a review sidecar exists"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
