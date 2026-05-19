from pathlib import Path
import tensorflow as tf
import zipfile
import tempfile
import os


def build_fruits360_datasets(data_dir, batch_size=32, image_size=(100, 100)):
    data_path = Path(data_dir)
    
    # Handle zip files
    if str(data_path).lower().endswith('.zip'):
        if not data_path.exists():
            raise FileNotFoundError(f'Zip file not found: {data_path}')
        
        # Extract to temporary directory
        temp_dir = tempfile.mkdtemp()
        print(f'Extracting zip to temporary directory: {temp_dir}')
        
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        data_dir = Path(temp_dir)
    else:
        data_dir = Path(data_dir)
        if not data_dir.exists():
            raise FileNotFoundError('Fruits-360 dataset directory not found: ' + str(data_dir))

    if (data_dir / 'Training').exists() and (data_dir / 'Test').exists():
        train_dir = data_dir / 'Training'
        val_dir = data_dir / 'Test'
    elif (data_dir / 'train').exists() and (data_dir / 'test').exists():
        train_dir = data_dir / 'train'
        if (data_dir / 'validation').exists():
            val_dir = data_dir / 'validation'
        elif (data_dir / 'Validation').exists():
            val_dir = data_dir / 'Validation'
        else:
            val_dir = data_dir / 'test'
    else:
        raise FileNotFoundError('Could not locate Fruits-360 training and validation directories under ' + str(data_dir))

    print(f'Loading training data from: {train_dir}')
    print(f'Loading validation data from: {val_dir}')

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=123,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    class_names = train_ds.class_names
    AUTOTUNE = tf.data.AUTOTUNE

    # Don't cache full dataset - use prefetch only to stream data efficiently
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
