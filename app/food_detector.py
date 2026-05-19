import numpy as np
import tensorflow as tf
import cv2


class FoodDetector:
    def __init__(self, model_path: str, labels_path: str = None, tflite: bool = False):
        self.tflite = tflite
        self.labels = self._load_labels(labels_path)

        if tflite:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
        else:
            self.model = tf.keras.models.load_model(model_path)
            self.input_shape = self.model.input_shape[1:3]

    def _load_labels(self, path):
        if path and tf.io.gfile.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return None

    def _preprocess(self, frame, size=(224, 224)):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, size)
        # Keep [0, 255] range — model was trained on raw pixel values (no /255 normalisation)
        image = image.astype(np.float32)
        return np.expand_dims(image, axis=0)

    def predict(self, frame):
        if self.tflite:
            input_tensor = self._preprocess(frame, tuple(self.input_details[0]['shape'][1:3]))
            self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        else:
            input_tensor = self._preprocess(frame, self.input_shape)
            output_data = self.model.predict(input_tensor)

        index = int(np.argmax(output_data[0]))
        confidence = float(np.max(output_data[0]))
        label = self.labels[index] if self.labels and index < len(self.labels) else str(index)
        return {'label': label, 'confidence': confidence}
