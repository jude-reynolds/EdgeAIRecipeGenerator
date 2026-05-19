import argparse
import csv
import os
import sys
from pathlib import Path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from app.data_utils import build_fruits360_datasets


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
    print(f"History saved: {path}  ({len(h['loss'])} epochs)")


def build_classifier(input_shape=(100, 100, 3), num_classes=260):
    base_model = EfficientNetB0(include_top=False, input_shape=input_shape, weights='imagenet')
    base_model.trainable = False

    inputs = layers.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Train Fruits-360 classifier')
    parser.add_argument('--data-dir', type=str, default='data', help='Fruits-360 branch root directory')
    parser.add_argument('--branch', choices=['100x100', 'original-size'], default='100x100', help='Fruits-360 branch to use')
    parser.add_argument('--output-dir', type=str, default='checkpoints', help='Model checkpoint output')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--prepare-data', action='store_true', help='Only prepare data and exit')
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    image_size = (100, 100) if args.branch == '100x100' else (224, 224)
    train_ds, val_ds, class_names = build_fruits360_datasets(
        args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
    )

    if args.prepare_data:
        print('Data preparation complete. Classes:', len(class_names))
        return

    model = build_classifier(input_shape=(*image_size, 3), num_classes=len(class_names))

    checkpoint_path = os.path.join(args.output_dir, 'fruits360_classifier.h5')
    callbacks = [
        ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_accuracy'),
        EarlyStopping(patience=5, monitor='val_accuracy', restore_best_weights=True),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    model.save(checkpoint_path)
    print('Saved trained model to', checkpoint_path)

    history_dir = Path(args.output_dir).parent / "history"
    save_history_csv(history, history_dir / "unweighted.csv")


if __name__ == '__main__':
    main()
