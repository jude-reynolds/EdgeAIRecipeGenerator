#!/usr/bin/env python3
"""
Print a class distribution table for the Fruits360 dataset.

Shows training and test image counts per class, sorted by training count.
Optionally saves to CSV with --csv flag.

Run from the project root:
    python scripts/class_distribution.py
    python scripts/class_distribution.py --csv
    python scripts/class_distribution.py --sort name
"""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR    = PROJECT_ROOT / "data" / "fruits-360_100x100" / "fruits-360" / "Training"
TEST_DIR     = PROJECT_ROOT / "data" / "fruits-360_100x100" / "fruits-360" / "Test"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}


def count_images(directory: Path) -> int:
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTS)


def build_table():
    if not TRAIN_DIR.exists():
        print(f"ERROR: Training directory not found:\n  {TRAIN_DIR}")
        sys.exit(1)

    classes = sorted(d.name for d in TRAIN_DIR.iterdir() if d.is_dir())

    rows = []
    for cls in classes:
        train_count = count_images(TRAIN_DIR / cls)
        test_dir    = TEST_DIR / cls
        test_count  = count_images(test_dir) if test_dir.exists() else 0
        rows.append({
            "class":  cls,
            "train":  train_count,
            "test":   test_count,
            "total":  train_count + test_count,
        })

    return rows


def sort_rows(rows, key):
    if key == "name":
        return sorted(rows, key=lambda r: r["class"])
    if key == "test":
        return sorted(rows, key=lambda r: r["test"], reverse=True)
    if key == "total":
        return sorted(rows, key=lambda r: r["total"], reverse=True)
    # default: train count descending
    return sorted(rows, key=lambda r: r["train"], reverse=True)


def print_table(rows):
    col_class = max(len(r["class"]) for r in rows)
    col_class = max(col_class, 5)  # min width for header

    header = (
        f"{'Class':<{col_class}}  {'Train':>6}  {'Test':>5}  {'Total':>6}  Bar (train)"
    )
    sep = "-" * len(header)

    train_counts = [r["train"] for r in rows]
    max_train    = max(train_counts)
    bar_width    = 30

    print(sep)
    print(header)
    print(sep)
    for r in rows:
        bar_len = round(r["train"] / max_train * bar_width)
        bar     = "█" * bar_len
        print(
            f"{r['class']:<{col_class}}  {r['train']:>6}  {r['test']:>5}  {r['total']:>6}  {bar}"
        )
    print(sep)

    total_train = sum(r["train"] for r in rows)
    total_test  = sum(r["test"]  for r in rows)
    print(
        f"{'TOTAL':<{col_class}}  {total_train:>6}  {total_test:>5}  {total_train + total_test:>6}"
    )
    print(sep)


def print_summary(rows):
    train_counts = [r["train"] for r in rows]
    n = len(rows)

    mn  = min(train_counts)
    mx  = max(train_counts)
    avg = sum(train_counts) / n
    imbalance = mx / mn

    bottom5 = sorted(rows, key=lambda r: r["train"])[:5]
    top5    = sorted(rows, key=lambda r: r["train"], reverse=True)[:5]

    print("\nSummary (training images)")
    print(f"  Classes        : {n}")
    print(f"  Total images   : {sum(train_counts):,}")
    print(f"  Mean per class : {avg:.0f}")
    print(f"  Min            : {mn}  ({bottom5[0]['class']})")
    print(f"  Max            : {mx}  ({top5[0]['class']})")
    print(f"  Imbalance ratio: {imbalance:.1f}x  (max / min)")

    print("\n  5 smallest classes:")
    for r in bottom5:
        print(f"    {r['train']:>4}  {r['class']}")

    print("\n  5 largest classes:")
    for r in top5:
        print(f"    {r['train']:>4}  {r['class']}")


def save_csv(rows, path: Path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class", "train", "test", "total"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV saved to: {path}")


def main():
    parser = argparse.ArgumentParser(description="Fruits360 class distribution table")
    parser.add_argument(
        "--sort",
        choices=["train", "test", "total", "name"],
        default="train",
        help="Column to sort by (default: train, descending)",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Save results to scripts/class_distribution.csv",
    )
    args = parser.parse_args()

    rows = build_table()
    rows = sort_rows(rows, args.sort)

    print_table(rows)
    print_summary(rows)

    if args.csv:
        csv_path = PROJECT_ROOT / "scripts" / "class_distribution.csv"
        save_csv(sort_rows(rows, "name"), csv_path)  # CSV always sorted by name


if __name__ == "__main__":
    main()
