#!/usr/bin/env python3
"""
Auto-generate a class grouping file (class_groups.json) for the Fruits360 dataset.

Groups all variant classes under a common ingredient name using the first word
of the class name as the group (e.g. "Cucumber 6", "Cucumber 10" → "Cucumber").
A small set of manual overrides handles two-word names and edge cases.

Review and edit class_groups.json before retraining — the file is yours to adjust.

Run from the project root:
    python scripts/build_class_groups.py
"""

import json
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR    = PROJECT_ROOT / "data" / "fruits-360_100x100" / "fruits-360" / "Training"
OUTPUT_FILE  = PROJECT_ROOT / "class_groups.json"

# Classes whose group name should be the first TWO words rather than one.
# Add any others you want to keep as two-word ingredient names.
TWO_WORD_GROUPS = {
    "Passion Fruit",
    "Caju seed",        # → "Caju Seed" (cashew)
    "Ginger Root",
}

# Explicit overrides: class folder name → ingredient group name.
# These take priority over the auto-grouping rules.
OVERRIDES = {
    "Avocado ripe":      "Avocado",
    "Physalis with Husk":"Physalis",
    "Salak":             "Salak",
    "Kohlrabi White":    "Kohlrabi",
    "Nut Forest":        "Nut Forest",   # keep distinct from Pecan
    "Nut Pecan":         "Pecan",
    "Caju seed":         "Cashew",
    "Ginger Root":       "Ginger",
    "Pomelo Sweetie":    "Pomelo",
    "Corn Husk":         "Corn",
    "Pepper Red Wine":   "Pepper",
    "Pineapple Mini":    "Pineapple",
    "Nectarine Flat":    "Nectarine",
    "Peach Flat":        "Peach",
    "Lemon Meyer":       "Lemon",
    "Mango Red":         "Mango",
    "Grape Blue 1":      "Grape",
    "Grape Pink":        "Grape",
    "Grape White":       "Grape",
    "Grape White 2":     "Grape",
    "Passion Fruit":     "Passion Fruit",
    "Potato Red":        "Potato",
    "Potato Sweet":      "Potato",
    "Potato White":      "Potato",
    "Strawberry Wedge":  "Strawberry",
    "Tomato Heart":      "Tomato",
    "Tomato Maroon":     "Tomato",
    "Tomato Yellow":     "Tomato",
    "Tomato Cherry Red":    "Cherry Tomato",
    "Tomato Cherry Yellow": "Cherry Tomato",
    "Tomato not Ripened":   "Tomato",
    "Cherry Wax Red":    "Cherry",
    "Cherry Wax Yellow": "Cherry",
    "Cherry Wax Black":  "Cherry",
    "Melon Piel de Sapo":"Melon",
    "Grapefruit Pink":   "Grapefruit",
    "Grapefruit White":  "Grapefruit",
    "Apple Crimson Snow":"Apple",
    "Apple Pink Lady":   "Apple",
    "Apple Red Delicious":"Apple",
    "Apple Granny Smith":"Apple",
    "Apple Red Yellow 1":"Apple",
    "Apple Red Yellow 2":"Apple",
    "Pear Monster":      "Pear",
    "Pear Williams":     "Pear",
    "Pear Abate":        "Pear",
    "Pear Forelle":      "Pear",
    "Pear Kaiser":       "Pear",
    "Pear Red":          "Pear",
    "Pear Stone":        "Pear",
    "Pear Mini":         "Pear",
    "Plum 2":            "Plum",
    "Plum 3":            "Plum",
    "Banana Lady Finger":"Banana",
    "Banana Red":        "Banana",
    "Avocado Black":     "Avocado",
    "Cauliflower 1":     "Cauliflower",
    "Mandarine":         "Mandarin",
    "Cantaloupe 1":      "Cantaloupe",
    "Cantaloupe 2":      "Cantaloupe",
}


def auto_group(class_name: str) -> str:
    """Derive an ingredient group name from a class folder name."""
    if class_name in OVERRIDES:
        return OVERRIDES[class_name]
    # Check two-word prefixes
    for prefix in TWO_WORD_GROUPS:
        if class_name.startswith(prefix):
            return prefix
    # Default: first word, capitalised
    return class_name.split()[0]


def build_groups(train_dir: Path) -> dict[str, list[str]]:
    class_folders = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    groups: dict[str, list[str]] = defaultdict(list)
    for cls in class_folders:
        group = auto_group(cls)
        groups[group].append(cls)
    return dict(sorted(groups.items()))


def count_images(train_dir: Path, class_name: str) -> int:
    return sum(
        1 for f in (train_dir / class_name).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )


def print_summary(groups: dict, train_dir: Path):
    print(f"\n{'Group':<22} {'Classes':>7}  {'Images':>7}  Members")
    print("-" * 90)
    total_groups  = 0
    total_classes = 0
    total_images  = 0
    for group, members in groups.items():
        imgs = sum(count_images(train_dir, m) for m in members)
        total_groups  += 1
        total_classes += len(members)
        total_images  += imgs
        member_str = ", ".join(members[:4])
        if len(members) > 4:
            member_str += f" … (+{len(members)-4} more)"
        print(f"{group:<22} {len(members):>7}  {imgs:>7}  {member_str}")
    print("-" * 90)
    print(f"{'TOTAL':<22} {total_classes:>7}  {total_images:>7}  ({total_groups} groups)")
    print()


def main():
    print("=" * 60)
    print("Fruits360 Class Grouper")
    print("=" * 60)

    if not TRAIN_DIR.exists():
        print(f"ERROR: Training directory not found:\n  {TRAIN_DIR}")
        return

    groups = build_groups(TRAIN_DIR)

    print_summary(groups, TRAIN_DIR)

    OUTPUT_FILE.write_text(json.dumps(groups, indent=2))
    print(f"Saved: {OUTPUT_FILE}")
    print()
    print("Next steps:")
    print("  1. Open class_groups.json and review the groupings")
    print("     - Rename groups, move classes between groups, or split/merge as needed")
    print("     - Each key is the ingredient name; the list is the Fruits360 folders that map to it")
    print("  2. Run: python scripts/retrain_fixed.py")
    print("     (it will detect class_groups.json and train on the grouped labels)")
    print()


if __name__ == "__main__":
    main()
