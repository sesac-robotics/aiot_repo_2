"""기존 단일 종목 연구 모델의 사람 분할·정규화·후보 학습을 수행합니다. 보존된 스쿼트 산출물에 자동 재실행하지 않습니다.

Train CNN and GRU; tune only on group-disjoint validation, evaluate source test.
"""

import argparse
import copy
import csv
import hashlib
import importlib.metadata
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from features import FEATURE_NAMES, SAMPLE_FPS, SEQUENCE_LENGTH, TARGETS
from models import MODELS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"


def save(name, obj):
    (OUT / name).write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    )


def metrics(y, p):
    result = dict(
        exact_match_accuracy=accuracy_score(y, p),
        label_accuracy=float((y == p).mean()),
        precision_macro=precision_score(y, p, average="macro", zero_division=0),
        recall_macro=recall_score(y, p, average="macro", zero_division=0),
        macro_f1=f1_score(y, p, average="macro", zero_division=0),
        per_error=classification_report(
            y, p, target_names=TARGETS, output_dict=True, zero_division=0
        ),
    )
    a, b = y.any(axis=1).astype(int), p.any(axis=1).astype(int)
    result["normal_vs_selected_error"] = dict(
        accuracy=accuracy_score(a, b),
        precision=precision_score(a, b, zero_division=0),
        recall=recall_score(a, b, zero_division=0),
        macro_f1=f1_score(a, b, average="macro", zero_division=0),
        confusion_matrix=confusion_matrix(a, b, labels=[0, 1]).tolist(),
    )
    return result


def tune(y, probs):
    ts = []
    for j in range(y.shape[1]):
        options = np.arange(0.1, 0.91, 0.05)
        ts.append(
            float(
                max(
                    options,
                    key=lambda t: f1_score(
                        y[:, j], probs[:, j] >= t, zero_division=0
                    ),
                )
            )
        )
    return np.array(ts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()
    from audit_duplicates import main as audit_duplicates

    audit_duplicates()
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    data = np.load(BASE / "data/squat.npz")
    X, y, g, ids = data["X"], data["y"].astype(int), data["groups"], data["ids"]
    assert np.isfinite(X).all()
    source_train = np.flatnonzero(data["source_split"] == "train")
    test = np.flatnonzero(data["source_split"] == "test")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    a, b = next(
        splitter.split(X[source_train], y[source_train], g[source_train])
    )
    train, val = source_train[a], source_train[b]
    splits = dict(train=train, validation=val, test=test)
    for name, idx in splits.items():
        assert (
            len(idx) > 0
            and np.all(y[idx].sum(axis=0) > 0)
            and np.all(y[idx].sum(axis=0) < len(idx))
        ), name
    assert not (
        set(g[train]) & set(g[val])
        or set(g[train]) & set(g[test])
        or set(g[val]) & set(g[test])
    )
    save(
        "splits.json",
        {
            name: dict(
                videos=len(idx),
                workers=sorted(map(int, set(g[idx]))),
                video_ids=ids[idx].tolist(),
                positive_counts=dict(
                    zip(TARGETS, map(int, y[idx].sum(axis=0)))
                ),
                normal_count=int((~y[idx].any(axis=1)).sum()),
            )
            for name, idx in splits.items()
        },
    )
    mean = X[train].mean(axis=(0, 1))
    std = np.maximum(X[train].std(axis=(0, 1)), 1e-5)
    Z = (X - mean) / std
    np.savez(OUT / "normalization.npz", mean=mean, std=std)
    tensors = {
        name: torch.tensor(Z[idx], dtype=torch.float32)
        for name, idx in splits.items()
    }
    loader = DataLoader(
        TensorDataset(
            tensors["train"], torch.tensor(y[train], dtype=torch.float32)
        ),
        batch_size=48,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    pos = y[train].sum(axis=0)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            (len(train) - pos) / np.maximum(pos, 1), dtype=torch.float32
        )
    )
    report = {}
    checkpoints = {}
    histories = {}
    start = time.time()
    for name, constructor in MODELS.items():
        torch.manual_seed(42)
        model = constructor(X.shape[2], len(TARGETS))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=0.001, weight_decay=0.001
        )
        best_score = -1
        best = None
        stale = 0
        history = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
                total += loss.item() * len(xb)
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(model(tensors["validation"])).numpy()
            thresholds = tune(y[val], probs)
            score = f1_score(
                y[val], probs >= thresholds, average="macro", zero_division=0
            )
            history.append(
                dict(
                    epoch=epoch,
                    loss=total / len(train),
                    validation_macro_f1=float(score),
                )
            )
            if score > best_score + 1e-5:
                best_score = score
                best = copy.deepcopy(model.state_dict())
                best_thresholds = thresholds
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if epoch % 5 == 0:
                print(
                    f"{name} epoch={epoch} train_loss={total/len(train):.4f} val_macro_f1={score:.4f} best={best_score:.4f}",
                    flush=True,
                )
            if stale >= 12:
                break
        model.load_state_dict(best)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(tensors["test"])).numpy()
        report[name] = metrics(y[test], probs >= best_thresholds)
        report[name].update(
            validation_macro_f1=float(best_score),
            best_epoch=best_epoch,
            thresholds=best_thresholds.tolist(),
        )
        checkpoint = dict(
            state_dict=best,
            architecture=name,
            feature_count=X.shape[2],
            targets=TARGETS,
            feature_names=FEATURE_NAMES,
            sequence_length=SEQUENCE_LENGTH,
            sample_fps=SAMPLE_FPS,
            mean=mean.tolist(),
            std=std.tolist(),
            thresholds=best_thresholds.tolist(),
            label_scope="QEVD squat source clip labels; two selected errors only",
            split_seed=42,
        )
        torch.save(checkpoint, OUT / f"squat_{name}.pt")
        checkpoints[name] = checkpoint
        histories[name] = history
        with (OUT / f"{name}_test_predictions.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["video_id", "worker_id"]
                + [f"true_{t}" for t in TARGETS]
                + [f"prob_{t}" for t in TARGETS]
            )
            writer.writerows(
                [
                    [ids[i], int(g[i]), *y[i].tolist(), *p.tolist()]
                    for i, p in zip(test, probs)
                ]
            )
    # Geometry proxy baseline: depth and torso lean, tuned on validation only.
    knee = np.minimum(X[:, :, -5], X[:, :, -4])
    depth = np.percentile(knee, 10, axis=1)
    lean = np.percentile(X[:, :, -1], 90, axis=1)
    rule_thresholds = []
    for j, (feature, options) in enumerate(
        [(depth, np.arange(65, 166, 5)), (lean, np.arange(10, 86, 5))]
    ):
        rule_thresholds.append(
            float(
                max(
                    options,
                    key=lambda t: f1_score(
                        y[val, j], feature[val] >= t, zero_division=0
                    ),
                )
            )
        )
    rule = np.stack(
        [depth[test] >= rule_thresholds[0], lean[test] >= rule_thresholds[1]],
        axis=1,
    )
    report["rule_proxy"] = metrics(y[test], rule)
    report["rule_proxy"]["thresholds"] = rule_thresholds
    report["rule_proxy"][
        "limitation"
    ] = "Knee-depth and torso-lean proxies. Torso lean does not measure spinal curvature and is not equivalent to a validated back-alignment rule."
    report["majority_train"] = metrics(
        y[test], np.tile(y[train].mean(axis=0) >= 0.5, (len(test), 1))
    )
    winner = max(MODELS, key=lambda k: report[k]["validation_macro_f1"])
    torch.save(checkpoints[winner], OUT / "best.pt")
    report["selected_model"] = winner
    report["selection_basis"] = (
        "validation macro F1 only; test is not used for model selection"
    )
    report["elapsed_training_seconds"] = time.time() - start
    report["environment"] = {
        name: importlib.metadata.version(name)
        for name in [
            "torch",
            "mediapipe",
            "numpy",
            "opencv-python",
            "scikit-learn",
        ]
    }
    report["environment"].update(
        python=sys.version, device="cpu", torch_threads=torch.get_num_threads()
    )
    report["configuration"] = dict(
        seed=42,
        max_epochs=args.epochs,
        early_stopping_patience=12,
        optimizer="AdamW",
        learning_rate=0.001,
        weight_decay=0.001,
        batch_size=48,
        loss="BCEWithLogitsLoss with training pos_weight",
        sample_fps=SAMPLE_FPS,
        sequence_length=SEQUENCE_LENGTH,
        features=FEATURE_NAMES,
    )
    report["code_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in BASE.glob("*.py")
    }
    report["dataset_sha256"] = hashlib.sha256(
        (BASE / "data/squat.npz").read_bytes()
    ).hexdigest()
    report["limitations"] = [
        "Source clip-level labels, not expert reannotation or frame-level truth.",
        "Unlisted target labels are treated as negative under a closed-world annotation assumption.",
        "2D image landmarks and monocular z are view-dependent; back curvature is incompletely observed.",
        "Original source test workers preserved; validation workers held out from source train.",
        "Each video resampled to 32 steps; live windows may differ from full video duration.",
        "No accuracy guarantee on ESP32-CAM or on the other four exercises.",
    ]
    save("metrics.json", report)
    save("history.json", histories)
    lines = [
        "# 스쿼트 자세 오류 학습 결과",
        "",
        f"선택 모델: {winner} (검증 Macro F1로 선택)",
        "",
        "| 모델 | 정확도(정상/선택 오류) | Precision macro(오류별) | Recall macro(오류별) | Macro F1(오류별) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ["cnn", "gru", "rule_proxy", "majority_train"]:
        r = report[name]
        lines.append(
            f'| {name} | {r["normal_vs_selected_error"]["accuracy"]:.4f} | {r["precision_macro"]:.4f} | {r["recall_macro"]:.4f} | {r["macro_f1"]:.4f} |'
        )
    lines += [
        "",
        "정확도는 정상 후보 대 선택 오류의 이진 정확도입니다. 오류별 수치는 shallow/back not straight 다중 라벨 평균입니다. 모든 오류 조합을 맞추는 exact_match_accuracy는 metrics.json에 별도로 기록합니다.",
        "",
        "Rule proxy는 무릎 굽힘 깊이와 몸통 기울기를 사용하는 비교용 휴리스틱입니다. 몸통 기울기는 등 굽음과 동일하지 않아 기존 Rule-based 시스템과의 완전한 비교를 뜻하지 않습니다.",
        "",
        *["- " + s for s in report["limitations"]],
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                k: {
                    m: v
                    for m, v in r.items()
                    if m
                    in ["macro_f1", "normal_vs_selected_error", "best_epoch"]
                }
                for k, r in report.items()
                if isinstance(r, dict)
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
