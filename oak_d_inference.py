#!/usr/bin/env python3
"""
OAK-D Camera Inference on Raspberry Pi 5 using TensorFlow Lite.
Runs real-time fruit classification and integrates with recipe API.
Usage: python3 oak_d_inference.py [--model path/to/model.tflite]
"""

import cv2
import numpy as np
import depthai as dai
import tensorflow as tf
import os
import sys
from pathlib import Path
import json
from datetime import datetime

class FruitsClassifier:
    def __init__(self, model_path, labels_path):
        """Initialize TFLite model and OAK-D pipeline."""
        
        # Load TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.input_shape = self.input_details[0]['shape']
        print(f"Model input shape: {self.input_shape}")
        
        # Load labels
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]
        
        print(f"Loaded {len(self.labels)} fruit classes")
        
        # Initialize OAK-D pipeline
        self.pipeline = dai.Pipeline()
        self._setup_oak_d()
        
        # Start pipeline (this connects to device internally)
        self.pipeline.start()
        self.device = None  # Pipeline manages device connection
        
    def _setup_oak_d(self):
        """Configure OAK-D pipeline with depthai 3.6.1 API."""
        
        # Create color camera node with CAM_A socket (modern non-deprecated syntax)
        cam_rgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        
        # Request output at full resolution for display (400p)
        cap = dai.ImgFrameCapability()
        cap.size.fixed((400, 400))
        cap.fps.fixed(30.0)
        cap.resizeMode = dai.ImgResizeMode.CROP
        
        # Get output queue for display (full resolution)
        video_out = cam_rgb.requestOutput(cap, True).createOutputQueue()
        self.video_queue = video_out
    
    def classify_frame(self, frame, blocked_classes=None):
        """Run inference on frame, optionally blocking certain classes."""
        
        if blocked_classes is None:
            blocked_classes = []
        
        # Resize to model input size (100x100)
        frame_resized = cv2.resize(frame, (self.input_shape[2], self.input_shape[1]))
        
        # Cast to float32 but keep [0, 255] range — model was trained on raw pixel values
        frame_float = frame_resized.astype(np.float32)

        # Set input tensor
        self.interpreter.set_tensor(self.input_details[0]['index'],
                                    np.expand_dims(frame_float, axis=0))
        
        # Run inference
        self.interpreter.invoke()
        
        # Get output
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        
        # Get top predictions (sorted by confidence)
        sorted_indices = np.argsort(output_data[0])[::-1]
        
        # Find first non-blocked class
        for class_id in sorted_indices:
            label = self.labels[class_id]
            if label not in blocked_classes:
                confidence = output_data[0][class_id]
                return label, confidence, class_id
        
        # Fallback: return highest confidence even if blocked
        confidence = np.max(output_data)
        class_id = np.argmax(output_data[0])
        return self.labels[class_id], confidence, class_id
    
    def run_inference_loop(self, confidence_threshold=0.7, blocked_classes=None):
        """Main inference loop.
        
        Args:
            confidence_threshold: Minimum confidence for detection
            blocked_classes: List of class names to skip (e.g., ["Cucumber 6"])
        """
        
        if blocked_classes is None:
            blocked_classes = []
        
        print("\n" + "="*70)
        print("OAK-D INFERENCE - Fruits360 Classifier")
        print("="*70)
        print("Press 'q' to quit, 's' to save detection")
        print("Point camera at fruits for classification")
        if blocked_classes:
            print(f"Blocked classes: {', '.join(blocked_classes)}")
        print("="*70 + "\n")
        
        detections = []
        last_detection = None
        detection_cooldown = 0
        
        try:
            while self.pipeline.isRunning():
                # Get RGB frame
                rgb_data = self.video_queue.get()
                frame = rgb_data.getCvFrame()
                
                # Classify (with blocked classes)
                label, confidence, class_id = self.classify_frame(frame, blocked_classes)
                
                # Only record new detections if enough frames have passed
                if detection_cooldown > 0:
                    detection_cooldown -= 1
                
                # Filter by confidence and cooldown
                if confidence > confidence_threshold and detection_cooldown == 0:
                    # Only record if different fruit or high confidence jump
                    if last_detection is None or last_detection['fruit'] != label:
                        detection = {
                            'timestamp': datetime.now().isoformat(),
                            'fruit': label,
                            'confidence': float(confidence),
                            'class_id': int(class_id)
                        }
                        detections.append(detection)
                        last_detection = detection
                        detection_cooldown = 15  # Wait 15 frames before next detection
                        
                        print(f"[DETECTED] {label} ({confidence:.2%})")
                
                # Display on frame
                text = f"{label}: {confidence:.1%}"
                color = (0, 255, 0) if confidence > confidence_threshold else (0, 0, 255)
                cv2.putText(frame, text, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show frame
                cv2.imshow("Fruits360 OAK-D", frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save detections to JSON
                    output_file = 'detections.json'
                    with open(output_file, 'w') as f:
                        json.dump(detections, f, indent=2)
                    print(f"\nSaved {len(detections)} detections to {output_file}\n")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            self.pipeline.stop()
            
            print("\n" + "="*70)
            print(f"INFERENCE COMPLETE - Detected {len(detections)} unique fruits")
            print("="*70 + "\n")
            
            return detections

def main():
    parser = __import__('argparse').ArgumentParser(description='OAK-D Fruit Classifier')
    parser.add_argument('--model', default='models/fruits360_classifier.tflite',
                       help='Path to TFLite model')
    parser.add_argument('--labels', default='models/labels.txt',
                       help='Path to labels file')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Confidence threshold')
    
    args = parser.parse_args()
    
    # Validate files
    if not os.path.exists(args.model):
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)
    if not os.path.exists(args.labels):
        print(f"ERROR: Labels not found: {args.labels}")
        sys.exit(1)
    
    # Run classifier
    classifier = FruitsClassifier(args.model, args.labels)
    detections = classifier.run_inference_loop(args.confidence)

if __name__ == '__main__':
    main()
