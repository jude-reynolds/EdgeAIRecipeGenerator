#!/usr/bin/env python3
"""
Fixed retraining script for Fruits360 classifier.

Addresses the 'Cucumber 6' overtraining bias by:
  - Starting fresh from ImageNet weights (never loading the biased checkpoint)
  - Computing balanced class weights to counteract the 6.8x class imbalance
  - Augmentation layers placed INSIDE the model (not in tf.data.map) so that
    Keras manages training vs. inference mode and XLA JIT is not triggered
  - Two-phase training: frozen EfficientNetB0 backbone, then partial unfreeze
  - Exporting the corrected model to TFLite for OAK-D deployment

Run from WSL inside the project root:
    python scripts/retrain_fixed.py

Requirements: tensorflow>=2.14, numpy
GPU will be used automatically if visible to TensorFlow.
"""

# Must be set before TensorFlow is imported to suppress XLA JIT compilation
# errors that occur on WSL2 GPU environments.
import os
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import csv
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# ── paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "fruits-360_100x100" / "fruits-360"
TRAIN_DIR    = DATA_DIR / "Training"
VAL_DIR      = DATA_DIR / "Test"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
MODEL_DIR    = PROJECT_ROOT / "models"

# ── hyper-parameters ───────────────────────────────────────────────────────────
IMAGE_SIZE     = (100, 100)
INPUT_SHAPE    = (*IMAGE_SIZE, 3)
BATCH_SIZE     = 32           # same as original; increase to 64 if GPU has >8 GB VRAM
PHASE1_EPOCHS  = 20           # frozen backbone
PHASE2_EPOCHS  = 15           # partial unfreeze – fine-tuning
UNFREEZE_LAYERS = 30          # number of EfficientNetB0 tail layers to unfreeze


def configure_gpu():
    """Allow memory growth and disable XLA JIT (unstable on WSL2)."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Belt-and-suspenders: also disable via the TF API after import
    tf.config.optimizer.set_jit(False)
    if gpus:
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("No GPU detected – training on CPU (will be slow).")


def compute_class_weights(train_dir: Path):
    """
    Count images per class and return a balanced weight dict.
    Alphabetical sort matches image_dataset_from_directory label assignment.
    """
    class_names = sorted(d.name for d in train_dir.iterdir() if d.is_dir())
    counts = []
    for cls in class_names:
        n = sum(
            1 for f in (train_dir / cls).iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        counts.append(max(n, 1))

    total     = sum(counts)
    n_classes = len(class_names)
    weights   = {i: total / (n_classes * c) for i, c in enumerate(counts)}

    w_vals = list(weights.values())
    print(f"  Classes     : {n_classes}")
    print(f"  Min images  : {min(counts):,}  →  weight {max(w_vals):.3f}")
    print(f"  Max images  : {max(counts):,}  →  weight {min(w_vals):.3f}")
    print(f"  Mean images : {total / n_classes:.0f}")
    if "Cucumber 6" in class_names:
        idx = class_names.index("Cucumber 6")
        print(f"  Cucumber 6  : {counts[idx]:,} images  →  weight {weights[idx]:.3f}")
    return weights, class_names


def load_class_groups():
    """
    Load class_groups.json from the project root if it exists.
    Returns (folder_to_group dict, sorted group_names list) or (None, None).
    Format: {"GroupName": ["FolderName1", "FolderName2", ...], ...}
    """
    groups_file = PROJECT_ROOT / "class_groups.json"
    if not groups_file.exists():
        return None, None

    raw = json.loads(groups_file.read_text())
    folder_to_group = {}
    for group_name, members in raw.items():
        for member in members:
            folder_to_group[member] = group_name

    group_names = sorted(raw.keys())
    print(f"  class_groups.json loaded: {len(raw)} groups from {len(folder_to_group)} classes")
    return folder_to_group, group_names


def compute_class_weights_grouped(train_dir: Path, folder_class_names: list,
                                   folder_to_group: dict, group_names: list):
    """Compute balanced class weights summed across all folders in each group."""
    group_to_idx = {g: i for i, g in enumerate(group_names)}
    group_counts = [0] * len(group_names)

    for cls in folder_class_names:
        group = folder_to_group.get(cls, cls)
        if group not in group_to_idx:
            continue
        n = sum(
            1 for f in (train_dir / cls).iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        group_counts[group_to_idx[group]] += n

    total     = sum(group_counts)
    n_classes = len(group_names)
    weights   = {i: total / (n_classes * max(c, 1)) for i, c in enumerate(group_counts)}

    w_vals = list(weights.values())
    print(f"  Groups      : {n_classes}")
    print(f"  Min images  : {min(group_counts):,}  →  weight {max(w_vals):.3f}")
    print(f"  Max images  : {max(group_counts):,}  →  weight {min(w_vals):.3f}")
    print(f"  Mean images : {total / n_classes:.0f}")
    return weights


def apply_group_remapping(train_ds, val_ds, folder_class_names: list,
                           folder_to_group: dict, group_names: list):
    """Remap integer labels from per-folder indices to per-group indices."""
    group_to_idx   = {g: i for i, g in enumerate(group_names)}
    mapping        = [group_to_idx.get(folder_to_group.get(cn, cn), 0)
                      for cn in folder_class_names]
    mapping_tensor = tf.constant(mapping, dtype=tf.int32)

    def remap(image, label):
        return image, mapping_tensor[tf.cast(label, tf.int32)]

    AUTOTUNE  = tf.data.AUTOTUNE
    train_ds  = train_ds.map(remap, num_parallel_calls=AUTOTUNE)
    val_ds    = val_ds.map(remap,   num_parallel_calls=AUTOTUNE)
    return train_ds, val_ds


def build_datasets(class_names_ref: list):
    """
    Build tf.data pipelines.
    Normalisation uses a plain tf.cast/divide (no Keras layer inside map)
    to avoid XLA graph compilation issues.
    Augmentation lives inside the model, not here.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    # Cast to float32 but keep [0, 255] pixel range.
    # EfficientNetB0's frozen batch-norm statistics were calibrated on pixel-scale
    # inputs during ImageNet training. Dividing to [0, 1] collapses the activations
    # to near-zero after GlobalAveragePooling, making the head impossible to train.
    # The original scripts/train.py (97% accuracy) also passed [0, 255] directly.
    def to_float(image, label):
        return tf.cast(image, tf.float32), label

    train_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=True,
        seed=42,
    )
    val_raw = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        labels="inferred",
        label_mode="int",
        batch_size=BATCH_SIZE,
        image_size=IMAGE_SIZE,
        shuffle=False,
    )

    assert train_raw.class_names == class_names_ref, (
        "Class name mismatch between weight computation and dataset! "
        "Both must sort the class folders the same way."
    )

    train_ds = train_raw.map(to_float, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    val_ds   = val_raw.map(to_float,  num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
    return train_ds, val_ds


def build_model(num_classes: int):
    """
    EfficientNetB0 with augmentation layers inside the model graph.
    Keras automatically applies augmentation only during model.fit() and
    skips it during model.evaluate() / model.predict().
    Backbone starts frozen.

    Augmentation layers are added individually (not wrapped in a nested
    tf.keras.Sequential) to avoid a Keras 3 deepcopy/pickling bug that
    triggers when ModelCheckpoint saves the full model in HDF5 format.
    """
    base = EfficientNetB0(
        include_top=False,
        input_shape=INPUT_SHAPE,
        weights="imagenet",
    )
    base.trainable = False

    inputs = layers.Input(shape=INPUT_SHAPE)
    # Each augmentation layer is a direct node in the functional graph —
    # no nested Sequential wrapper, so Keras 3 can serialise safely.
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)
    x = layers.RandomZoom(0.08)(x)
    x = layers.RandomTranslation(0.08, 0.08)(x)
    x = layers.RandomBrightness(factor=0.10)(x)
    x = base(x, training=False)           # backbone always in inference mode (frozen)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs), base


def make_callbacks(checkpoint_path: Path, patience_stop=6, patience_lr=3):
    return [
        ModelCheckpoint(
            str(checkpoint_path),
            save_best_only=True,
            save_weights_only=True,   # avoids deepcopy/pickling of the full model graph
            monitor="val_accuracy",
            verbose=1,
        ),
        EarlyStopping(
            patience=patience_stop,
            monitor="val_accuracy",
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-8,
            verbose=1,
        ),
    ]


def save_history_csv(history, path, epoch_offset=0):
    """Save a Keras history object to CSV with standardised column names."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h = history.history
    epochs = range(1 + epoch_offset, len(h["loss"]) + 1 + epoch_offset)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_accuracy",
                           "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        for i, epoch in enumerate(epochs):
            writer.writerow({
                "epoch":          epoch,
                "train_loss":     round(h["loss"][i], 6),
                "train_accuracy": round(h["accuracy"][i], 6),
                "val_loss":       round(h["val_loss"][i], 6),
                "val_accuracy":   round(h["val_accuracy"][i], 6),
            })
    print(f"  History CSV  : {path}  ({len(h['loss'])} epochs)")


def export_tflite(model, out_path: Path):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    tflite_bytes = converter.convert()
    out_path.write_bytes(tflite_bytes)
    print(f"  TFLite model : {out_path}  ({len(tflite_bytes) / 1024 / 1024:.1f} MB)")


def save_report(history1, history2, class_names, out_dir: Path):
    report = {
        "num_classes": len(class_names),
        "phase1": {
            "epochs_ran": len(history1.history["loss"]),
            "best_val_accuracy": float(max(history1.history["val_accuracy"])),
        },
        "phase2": {
            "epochs_ran": len(history2.history["loss"]),
            "best_val_accuracy": float(max(history2.history["val_accuracy"])),
        },
    }
    path = out_dir / "retrain_report.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"  Report       : {path}")


def main():
    configure_gpu()

    print("\n" + "=" * 60)
    print("Fruits360 – Fixed Retraining (Cucumber 6 bias fix)")
    print("=" * 60)

    CKPT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # ── 1. class weights ───────────────────────────────────────────────────────
    print("\n[1/5] Computing class weights …")
    folder_to_group, group_names = load_class_groups()
    grouped = folder_to_group is not None

    if grouped:
        print(f"  Mode: GROUPED ({len(group_names)} ingredient classes)")
        _, folder_class_names = compute_class_weights(TRAIN_DIR)
        class_weights = compute_class_weights_grouped(
            TRAIN_DIR, folder_class_names, folder_to_group, group_names
        )
        class_names = group_names
    else:
        print("  Mode: ALL CLASSES (no class_groups.json found)")
        class_weights, class_names = compute_class_weights(TRAIN_DIR)
        folder_class_names = class_names

    num_classes = len(class_names)

    # ── 2. datasets ────────────────────────────────────────────────────────────
    print("\n[2/5] Building datasets …")
    train_ds, val_ds = build_datasets(folder_class_names)
    if grouped:
        train_ds, val_ds = apply_group_remapping(
            train_ds, val_ds, folder_class_names, folder_to_group, group_names
        )
    print(f"  Training batches  : {len(train_ds)}")
    print(f"  Validation batches: {len(val_ds)}")

    # ── 3. model ───────────────────────────────────────────────────────────────
    print("\n[3/5] Building model (EfficientNetB0 + in-model augmentation) …")
    model, base_model = build_model(num_classes)
    model.summary(line_length=80)

    # ── 4. phase 1 – train only the new top layers ────────────────────────────
    print(f"\n[4/5] Phase 1 – frozen backbone, up to {PHASE1_EPOCHS} epochs …")
    ckpt_path = CKPT_DIR / "fruits360_fixed.weights.h5"  # weights-only, Keras 3 convention
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(ckpt_path),
        class_weight=class_weights,
    )

    history_dir = PROJECT_ROOT / "history"
    save_history_csv(history1, history_dir / "weighted_phase1.csv")

    # Reload the best checkpoint before fine-tuning
    model.load_weights(str(ckpt_path))

    # ── 5. phase 2 – partial unfreeze of EfficientNetB0 ───────────────────────
    print(
        f"\n[5/5] Phase 2 – unfreeze top {UNFREEZE_LAYERS} backbone layers, "
        f"up to {PHASE2_EPOCHS} epochs …"
    )
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-5),  # very low LR for fine-tuning
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        callbacks=make_callbacks(ckpt_path, patience_stop=5, patience_lr=2),
        class_weight=class_weights,
    )

    model.load_weights(str(ckpt_path))

    # Save phase 2 with epoch numbers continuing from phase 1
    phase1_epochs_ran = len(history1.history["loss"])
    save_history_csv(history2, history_dir / "weighted_phase2.csv",
                     epoch_offset=phase1_epochs_ran)

    # Combined file: all epochs in one place for easy comparison
    combined_path = history_dir / "weighted_combined.csv"
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "train_loss", "train_accuracy",
                           "val_loss", "val_accuracy", "phase"]
        )
        writer.writeheader()
        for i, (loss, acc, vloss, vacc) in enumerate(zip(
            history1.history["loss"], history1.history["accuracy"],
            history1.history["val_loss"], history1.history["val_accuracy"],
        )):
            writer.writerow({"epoch": i + 1, "train_loss": round(loss, 6),
                             "train_accuracy": round(acc, 6),
                             "val_loss": round(vloss, 6),
                             "val_accuracy": round(vacc, 6), "phase": 1})
        for i, (loss, acc, vloss, vacc) in enumerate(zip(
            history2.history["loss"], history2.history["accuracy"],
            history2.history["val_loss"], history2.history["val_accuracy"],
        )):
            writer.writerow({"epoch": phase1_epochs_ran + i + 1,
                             "train_loss": round(loss, 6),
                             "train_accuracy": round(acc, 6),
                             "val_loss": round(vloss, 6),
                             "val_accuracy": round(vacc, 6), "phase": 2})
    print(f"  History CSV  : {combined_path}  (all {phase1_epochs_ran + len(history2.history['loss'])} epochs)")

    # ── evaluate ───────────────────────────────────────────────────────────────
    print("\nFinal evaluation on validation set …")
    val_loss, val_acc = model.evaluate(val_ds, verbose=1)
    print(f"  Validation accuracy : {val_acc * 100:.2f}%")
    print(f"  Validation loss     : {val_loss:.4f}")

    # ── export ─────────────────────────────────────────────────────────────────
    print("\nExporting …")
    keras_path = CKPT_DIR / "fruits360_classifier.keras"  # native Keras 3 format
    model.save(str(keras_path))
    print(f"  Keras model  : {keras_path}")

    tflite_path = MODEL_DIR / "fruits360_classifier.tflite"
    export_tflite(model, tflite_path)

    labels_path = MODEL_DIR / "labels.txt"
    labels_path.write_text("\n".join(class_names))
    print(f"  Labels file  : {labels_path}  ({num_classes} classes)")

    save_report(history1, history2, class_names, MODEL_DIR)

    best_acc = max(
        max(history1.history["val_accuracy"]),
        max(history2.history["val_accuracy"]),
    )
    print("\n" + "=" * 60)
    print(f"Retraining complete. Best val accuracy: {best_acc * 100:.2f}%")
    print("\nNext steps:")
    print("  1. Copy models/fruits360_classifier.tflite to your Raspberry Pi")
    print("  2. Copy models/labels.txt to your Raspberry Pi")
    print("  3. Run: python pi_app.py --mode oak-d --confidence 0.75")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
