#!/usr/bin/env python3
"""
Collect real-world training images using the OAK-D camera.

Images are saved directly into the Fruits360 Training folder under the correct
class name so they are picked up automatically by retrain_fixed.py.

Run on Windows with OAK-D plugged in:
    python collect_training_data.py

Controls during capture:
    SPACE  - toggle auto-capture on/off
    s      - capture a single frame immediately
    n      - finish current class, pick a new one
    q      - quit
"""

import os
import sys
import cv2
import depthai as dai
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LABELS_FILE  = PROJECT_ROOT / "models" / "labels.txt"
TRAIN_DIR    = PROJECT_ROOT / "data" / "fruits-360_100x100" / "fruits-360" / "Training"

# Auto-capture rate: save one image every this many frames (30 fps → every 10 = 3/sec)
AUTO_CAPTURE_INTERVAL = 10


def load_labels():
    if not LABELS_FILE.exists():
        print(f"ERROR: labels file not found at {LABELS_FILE}")
        sys.exit(1)
    return [l.strip() for l in LABELS_FILE.read_text().splitlines() if l.strip()]


def pick_class(labels: list[str]) -> str:
    """Let the user search for and confirm a class name."""
    print("\nType part of the fruit/vegetable name to search (e.g. 'pepper', 'onion'):")
    while True:
        query = input("Search: ").strip().lower()
        if not query:
            continue
        matches = [l for l in labels if query in l.lower()]
        if not matches:
            print("  No matches found. Try again.")
            continue
        print("\n  Matches:")
        for i, m in enumerate(matches):
            existing = len(list((TRAIN_DIR / m).glob("*.jpg"))) if (TRAIN_DIR / m).exists() else 0
            print(f"  [{i}] {m}  ({existing} images already)")
        choice = input("\n  Enter number to select, or press Enter to search again: ").strip()
        if choice.isdigit() and 0 <= int(choice) < len(matches):
            return matches[int(choice)]
        print("  Searching again …\n")


def setup_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cap = dai.ImgFrameCapability()
    cap.size.fixed((400, 400))
    cap.fps.fixed(30.0)
    cap.resizeMode = dai.ImgResizeMode.CROP
    queue = cam.requestOutput(cap, True).createOutputQueue()
    return pipeline, queue


def next_index(class_dir: Path) -> int:
    """Return the next available image index for this class folder."""
    existing = list(class_dir.glob("*.jpg"))
    if not existing:
        return 0
    nums = []
    for f in existing:
        stem = f.stem  # e.g. "real_0042"
        try:
            nums.append(int(stem.split("_")[-1]))
        except ValueError:
            pass
    return max(nums) + 1 if nums else len(existing)


def draw_overlay(frame, class_name, count, auto_on, interval):
    h, w = frame.shape[:2]
    # semi-transparent dark bar at bottom
    bar = frame.copy()
    cv2.rectangle(bar, (0, h - 80), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(bar, 0.5, frame, 0.5, 0, frame)

    auto_label = "AUTO ON" if auto_on else "AUTO OFF"
    auto_color = (0, 255, 0) if auto_on else (0, 120, 255)
    fps_note   = f"  (~{30 // interval}/sec)" if auto_on else ""

    cv2.putText(frame, f"Class : {class_name}", (10, h - 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, f"Saved : {count}  |  {auto_label}{fps_note}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, auto_color, 1)
    cv2.putText(frame, "SPACE=toggle auto  S=single  N=next class  Q=quit",
                (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


def collect_for_class(class_name: str, queue) -> int:
    """Run the capture loop for one class. Returns number of images saved."""
    class_dir = TRAIN_DIR / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    idx       = next_index(class_dir)
    saved     = 0
    auto_on   = False
    frame_num = 0

    print(f"\n  Capturing for: {class_name}")
    print(f"  Saving to    : {class_dir}")
    print(f"  Tip: rotate the object, vary distance, try different backgrounds.\n")

    while True:
        rgb   = queue.get()
        frame = rgb.getCvFrame()
        frame_num += 1

        do_save = False
        if auto_on and (frame_num % AUTO_CAPTURE_INTERVAL == 0):
            do_save = True

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return saved, "quit"
        elif key == ord('n'):
            return saved, "next"
        elif key == ord(' '):
            auto_on = not auto_on
            print(f"  Auto-capture {'ON' if auto_on else 'OFF'}")
        elif key == ord('s'):
            do_save = True

        if do_save:
            filename = class_dir / f"real_{idx:04d}.jpg"
            # Resize to 100x100 to match the Fruits360 dataset resolution
            small = cv2.resize(frame, (100, 100))
            cv2.imwrite(str(filename), small)
            idx  += 1
            saved += 1
            if saved % 10 == 0:
                print(f"  Saved {saved} images …")

        draw_overlay(frame, class_name, saved, auto_on, AUTO_CAPTURE_INTERVAL)
        cv2.imshow("OAK-D Data Collection", frame)

    return saved, "quit"


def main():
    print("=" * 60)
    print("OAK-D Real-World Data Collection")
    print("=" * 60)
    print(f"Dataset : {TRAIN_DIR}")
    print(f"Labels  : {LABELS_FILE}")

    labels = load_labels()
    print(f"Classes : {len(labels)}\n")

    print("Starting OAK-D camera …")
    pipeline, queue = setup_pipeline()
    pipeline.start()

    total_saved = {}

    try:
        while True:
            class_name = pick_class(labels)
            saved, action = collect_for_class(class_name, queue)
            total_saved[class_name] = total_saved.get(class_name, 0) + saved
            print(f"\n  ✓ Saved {saved} images for '{class_name}'")

            if action == "quit":
                break

            again = input("\n  Pick another class? [Y/n]: ").strip().lower()
            if again == "n":
                break

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cv2.destroyAllWindows()
        pipeline.stop()

    print("\n" + "=" * 60)
    print("Collection complete. Summary:")
    for cls, count in total_saved.items():
        print(f"  {cls}: {count} new images")
    print("\nNext steps:")
    print("  1. Switch to WSL")
    print("  2. Run: python scripts/retrain_fixed.py")
    print("     (the new images are already in the Training folder)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
