#!/usr/bin/env python3
"""
Fruits360 ingredient recognition app for Raspberry Pi 5 + OAK-D Lite.

Shows a live camera feed. Press C to capture and classify the ingredient
in frame. Recipes are looked up automatically and printed to the terminal.

Controls:
    C  - capture current frame, classify, and show recipes
    A  - manually add an ingredient by name and show its recipes
    I  - print current ingredient inventory to terminal
    S  - save session log to JSON
    Q  - quit
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import depthai as dai
from recipe_generator import RecipeGenerator, IngredientSearch

try:
    from ai_edge_litert.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
    except ImportError:
        import tensorflow as tf
        TFLiteInterpreter = tf.lite.Interpreter

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH   = PROJECT_ROOT / "models" / "fruits360_classifier.tflite"
LABELS_FILE  = PROJECT_ROOT / "models" / "labels.txt"

CONFIDENCE_THRESHOLD = 0.70
RESULT_DISPLAY_SECS  = 3.0      # how long to show the result overlay after capture


def normalize_for_recipe(label):
    """Strip trailing Fruits360 variant number: 'Apple 10' → 'apple'."""
    return re.sub(r'\s*\d+\s*$', '', label).strip().lower()


def load_labels():
    if not LABELS_FILE.exists():
        print(f"ERROR: labels not found: {LABELS_FILE}")
        sys.exit(1)
    return [l.strip() for l in LABELS_FILE.read_text().splitlines() if l.strip()]


def load_model():
    if not MODEL_PATH.exists():
        print(f"ERROR: model not found: {MODEL_PATH}")
        sys.exit(1)
    interp = TFLiteInterpreter(model_path=str(MODEL_PATH))
    interp.allocate_tensors()
    return interp


def classify(interp, frame, labels):
    """Run TFLite inference on a single frame. Returns (label, confidence)."""
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]
    h, w = inp["shape"][1], inp["shape"][2]

    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)  # [0, 255]
    interp.set_tensor(inp["index"], np.expand_dims(img, axis=0))
    interp.invoke()

    probs     = interp.get_tensor(out["index"])[0]
    idx       = int(np.argmax(probs))
    return labels[idx], float(probs[idx])


def setup_oak_d():
    pipeline = dai.Pipeline()
    cam      = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    cap      = dai.ImgFrameCapability()
    cap.size.fixed((640, 640))
    cap.fps.fixed(30.0)
    cap.resizeMode = dai.ImgResizeMode.CROP
    queue = cam.requestOutput(cap, True).createOutputQueue()
    return pipeline, queue


def draw_live_overlay(frame):
    """Minimal HUD shown on the live feed."""
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, "C=Capture  A=Add ingredient  I=Inventory  S=Save  Q=Quit",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1)


def draw_result_overlay(frame, label, confidence, timestamp, top_recipe=None):
    """Large result banner shown after a successful capture."""
    h, w = frame.shape[:2]

    banner_h = 140 if top_recipe else 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    color = (0, 220, 0) if confidence >= CONFIDENCE_THRESHOLD else (0, 140, 255)
    cv2.putText(frame, label,
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
    cv2.putText(frame, f"{confidence:.1%}  |  {timestamp}",
                (16, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    if top_recipe:
        cv2.putText(frame, f"Top recipe: {top_recipe}",
                    (16, 128), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 255), 1)


def draw_session_log(frame, log):
    """Show last 5 detections in the bottom-right corner."""
    if not log:
        return
    h, w  = frame.shape[:2]
    recent = log[-5:][::-1]
    y      = h - 46
    for entry in recent:
        text  = f"{entry['label']}  {entry['confidence']:.0%}"
        color = (180, 180, 180)
        cv2.putText(frame, text, (w - 280, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        y -= 20


def save_session(log):
    if not log:
        print("Nothing to save.")
        return
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PROJECT_ROOT / f"session_{ts}.json"
    path.write_text(json.dumps(log, indent=2))
    print(f"Session saved → {path}  ({len(log)} detections)")


def main():
    print("Loading model …")
    labels = load_labels()
    interp = load_model()
    print(f"  {len(labels)} classes loaded")

    recipe_gen = RecipeGenerator(use_local=True)
    inventory  = IngredientSearch()

    print("Starting OAK-D camera …")
    pipeline, queue = setup_oak_d()
    pipeline.start()
    print("  Ready.  Point camera at an ingredient and press C.\n")

    log              = []
    last_label       = None
    last_confidence  = 0.0
    last_capture_ts  = ""
    last_top_recipe  = None
    result_until     = 0.0      # monotonic time until result overlay hides

    while pipeline.isRunning():
        frame     = queue.get().getCvFrame()
        now       = time.monotonic()
        show_result = now < result_until

        if show_result:
            draw_result_overlay(frame, last_label, last_confidence, last_capture_ts, last_top_recipe)
        else:
            draw_live_overlay(frame)

        draw_session_log(frame, log)
        cv2.imshow("OAK-D Ingredient Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        elif key == ord("c"):
            label, confidence = classify(interp, frame, labels)
            last_label       = label
            last_confidence  = confidence
            last_capture_ts  = datetime.now().strftime("%H:%M:%S")
            result_until     = now + RESULT_DISPLAY_SECS

            entry = {
                "timestamp":  datetime.now().isoformat(),
                "label":      label,
                "confidence": round(confidence, 4),
            }
            log.append(entry)

            status = "✓" if confidence >= CONFIDENCE_THRESHOLD else "?"
            print(f"  {status} {label:<30} {confidence:.1%}  @ {last_capture_ts}")

            ingredient = normalize_for_recipe(label)
            inventory.add_ingredient(ingredient)
            recipes = recipe_gen.get_recipes(ingredient)
            last_top_recipe = recipes[0]["name"] if recipes else None
            print(recipe_gen.format_recipes(ingredient, recipes))

        elif key == ord("a"):
            print("\nIngredient name (Enter to cancel): ", end="", flush=True)
            name = input().strip().lower()
            if name:
                inventory.add_ingredient(name)
                recipes = recipe_gen.get_recipes(name)
                print(recipe_gen.format_recipes(name, recipes))

        elif key == ord("i"):
            print(inventory.suggest_meals())

        elif key == ord("s"):
            save_session(log)

    cv2.destroyAllWindows()
    pipeline.stop()
    save_session(log)
    print("Done.")


if __name__ == "__main__":
    main()
