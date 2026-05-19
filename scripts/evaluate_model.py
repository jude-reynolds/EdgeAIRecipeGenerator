#!/usr/bin/env python3
"""
Evaluate the trained model against the Fruits360 test set.

Produces:
  - Overall accuracy
  - Per-class accuracy table (sorted worst → best so problem areas are visible)
  - Best / worst 10 classes
  - Optional CSV export with --csv

If class_groups.json exists in the project root, grouped labels are applied
automatically to match whatever the model was trained on.

Run from the project root (WSL or Windows, no camera needed):
    python scripts/evaluate_model.py
    python scripts/evaluate_model.py --csv
    python scripts/evaluate_model.py --sort best
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

# Suppress TF info/warning logs for cleaner output
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"

PROJECT_ROOT  = Path(__file__).resolve().parent.parent
MODEL_PATH    = PROJECT_ROOT / "checkpoints" / "fruits360_classifier.keras"
REPORT_PATH   = PROJECT_ROOT / "models" / "retrain_report.json"
GROUPS_FILE   = PROJECT_ROOT / "class_groups.json"
TEST_DIR      = PROJECT_ROOT / "data" / "fruits-360_100x100" / "fruits-360" / "Test"

IMAGE_SIZE  = (100, 100)
BATCH_SIZE  = 64


# ── helpers ───────────────────────────────────────────────────────────────────

def load_groups():
    if not GROUPS_FILE.exists():
        return None, None
    raw = json.loads(GROUPS_FILE.read_text())
    folder_to_group = {m: g for g, members in raw.items() for m in members}
    group_names     = sorted(raw.keys())
    return folder_to_group, group_names


def build_remap_tensor(folder_class_names, folder_to_group, group_names):
    group_to_idx = {g: i for i, g in enumerate(group_names)}
    mapping      = [group_to_idx.get(folder_to_group.get(cn, cn), 0)
                    for cn in folder_class_names]
    return tf.constant(mapping, dtype=tf.int32)


def load_report():
    if not REPORT_PATH.exists():
        return None
    return json.loads(REPORT_PATH.read_text())


# ── evaluation ────────────────────────────────────────────────────────────────

def run_evaluation(model, remap_tensor=None):
    """Run model over the full test set. Returns (true_labels, pred_labels)."""
    AUTOTUNE = tf.data.AUTOTUNE

    ds = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    folder_class_names = ds.class_names

    def to_float(img, lbl):
        return tf.cast(img, tf.float32), lbl

    ds = ds.map(to_float, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    all_true = []
    all_pred = []

    total_batches = len(ds)
    for i, (images, labels) in enumerate(ds):
        print(f"\r  Evaluating … {i+1}/{total_batches} batches", end="", flush=True)
        probs = model(images, training=False)
        preds = tf.argmax(probs, axis=1, output_type=tf.int32)

        if remap_tensor is not None:
            labels = tf.gather(remap_tensor, labels)
            # preds are already in model's output space (group indices)

        all_true.extend(labels.numpy())
        all_pred.extend(preds.numpy())

    print()
    return np.array(all_true), np.array(all_pred), folder_class_names


def per_class_accuracy(true_labels, pred_labels, class_names):
    rows = []
    for idx, name in enumerate(class_names):
        mask   = true_labels == idx
        total  = int(mask.sum())
        if total == 0:
            continue
        correct = int((pred_labels[mask] == idx).sum())
        rows.append({
            "class":    name,
            "correct":  correct,
            "total":    total,
            "accuracy": correct / total,
        })
    return rows


# ── display ───────────────────────────────────────────────────────────────────

def print_summary(report, overall_acc, num_classes):
    print("\n" + "=" * 55)
    print("  Model Accuracy Summary")
    print("=" * 55)

    if report:
        p1 = report.get("phase1", {})
        p2 = report.get("phase2", {})
        print(f"  Phase 1 best val accuracy : {p1.get('best_val_accuracy', 0):.2%}"
              f"  ({p1.get('epochs_ran', '?')} epochs)")
        print(f"  Phase 2 best val accuracy : {p2.get('best_val_accuracy', 0):.2%}"
              f"  ({p2.get('epochs_ran', '?')} epochs)")
        print(f"  Classes trained on        : {report.get('num_classes', '?')}")
        print()

    print(f"  Test-set overall accuracy : {overall_acc:.2%}")
    print(f"  Classes evaluated         : {num_classes}")
    print("=" * 55)


def print_per_class_table(rows, sort_order):
    if sort_order == "best":
        sorted_rows = sorted(rows, key=lambda r: r["accuracy"], reverse=True)
    elif sort_order == "name":
        sorted_rows = sorted(rows, key=lambda r: r["class"])
    else:  # worst first (default — shows problems immediately)
        sorted_rows = sorted(rows, key=lambda r: r["accuracy"])

    col = max(len(r["class"]) for r in rows)
    header = f"{'Class':<{col}}  {'Correct':>7}  {'Total':>7}  {'Accuracy':>9}  Bar"
    sep    = "-" * (len(header) + 2)

    print(f"\n{sep}")
    print(f"  {header}")
    print(sep)
    for r in sorted_rows:
        bar_len = round(r["accuracy"] * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        print(f"  {r['class']:<{col}}  {r['correct']:>7}  {r['total']:>7}  "
              f"{r['accuracy']:>8.1%}  {bar}")
    print(sep)


def print_extremes(rows):
    sorted_rows = sorted(rows, key=lambda r: r["accuracy"])
    worst = sorted_rows[:10]
    best  = sorted_rows[-10:][::-1]

    col = max(len(r["class"]) for r in rows)

    print("\n  10 WORST classes:")
    print(f"  {'Class':<{col}}  Accuracy")
    print(f"  {'-'*(col+12)}")
    for r in worst:
        print(f"  {r['class']:<{col}}  {r['accuracy']:.1%}  ({r['correct']}/{r['total']})")

    print(f"\n  10 BEST classes:")
    print(f"  {'Class':<{col}}  Accuracy")
    print(f"  {'-'*(col+12)}")
    for r in best:
        print(f"  {r['class']:<{col}}  {r['accuracy']:.1%}  ({r['correct']}/{r['total']})")
    print()


def save_csv(rows, path: Path):
    sorted_rows = sorted(rows, key=lambda r: r["class"])
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "correct", "total", "accuracy"])
        writer.writeheader()
        for r in sorted_rows:
            writer.writerow({**r, "accuracy": f"{r['accuracy']:.4f}"})
    print(f"\n  CSV saved: {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate Fruits360 model on test set")
    parser.add_argument("--sort",   choices=["worst", "best", "name"], default="worst",
                        help="Sort order for the per-class table (default: worst first)")
    parser.add_argument("--csv",    action="store_true",
                        help="Save per-class results to scripts/evaluation_results.csv")
    parser.add_argument("--no-table", action="store_true",
                        help="Skip the full per-class table (only show summary + extremes)")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        sys.exit(1)

    # ── load ──────────────────────────────────────────────────────────────────
    print(f"\nLoading model: {MODEL_PATH.name}")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

    folder_to_group, group_names = load_groups()
    grouped = folder_to_group is not None
    if grouped:
        print(f"class_groups.json detected — evaluating {len(group_names)} grouped classes")

    report = load_report()

    # ── evaluate ──────────────────────────────────────────────────────────────
    print(f"\nRunning on test set ({TEST_DIR.name}) …")
    true_labels, pred_labels, folder_class_names = run_evaluation(
        model,
        remap_tensor=build_remap_tensor(folder_class_names, folder_to_group, group_names)
                     if grouped else None,
    )

    class_names  = group_names if grouped else folder_class_names
    overall_acc  = float((true_labels == pred_labels).mean())
    rows         = per_class_accuracy(true_labels, pred_labels, class_names)

    # ── display ───────────────────────────────────────────────────────────────
    print_summary(report, overall_acc, len(class_names))
    print_extremes(rows)
    if not args.no_table:
        print_per_class_table(rows, args.sort)

    if args.csv:
        csv_path = PROJECT_ROOT / "scripts" / "evaluation_results.csv"
        save_csv(rows, csv_path)


if __name__ == "__main__":
    main()
