from pathlib import Path
import cv2
import argparse
import numpy as np
import os
import torch
from collections import defaultdict

from PySide6.QtCore import QThread, Signal
 
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator



#from ultralytics.yolo.v8.detect.predict import Detections

import tensorflow as tf
from tensorflow.keras.models import load_model

import torch #========================================> GPU IMPORTANT <========


class ActionDetectionThread(QThread):
    detected_actions_list = Signal(object)
    progress_update = Signal(int)

    def __init__(self, video_keypoints, black_frames, video_frames, detections):
        super().__init__()

        self.video_keypoints = video_keypoints
        self.black_frames = black_frames
        self.video_frames = video_frames
        self.detections = detections

        self.detected_actions = []
        
        script_dir = Path(__file__).parent  # Get script's folder
        model_path = script_dir.parent / "assets" / "action_recognition_model.h5"
        self.action_recognition_model = load_model(model_path)

 # Recompile model (optional but can help)
        ############

        self.temp_sequence = []
        self.buffer_size = 30
        self.actions_list = np.array(['Extending Right Arm', 'Standing', 'Sitting'])
        self.recent_action = None
        self.translate_action_results = None
        self.person_keypoints = defaultdict(lambda: [])
        self._running = True
    
    def preprocess_keypoints(self, keypoints_dict):
        """
        Convert keypoints dictionary into a properly shaped numpy array for inference.
        Missing keypoints (zeros) remain unchanged.
        """
        processed_data = {}
        for person_id, keypoints in keypoints_dict.items():
            keypoints = keypoints.flatten()  # Convert (17, 2) to (34,)
            processed_data[person_id] = keypoints
        return processed_data

    def draw_action_text(self, frame, black_frames, detections, actions):
        """
        Draw bounding boxes with action labels on each frame.
        """
        for person_id, bbox in detections.items():
            x, y, w, h = map(int, bbox)  # Ensure coordinates are integers
            action_text = f"Action: {actions.get(person_id, 'Unknown')}"
            
            # Ensure text position is within frame bounds
            text_x, text_y = x, max(y - 10, 10)  # Prevent negative y
            
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, black_frames


    def predict_actions(self, frames_keypoints):
        """
        Predict actions based on sequential keypoints data.
        Returns a dictionary with person ID as keys and their predicted actions as values.
        """
        predictions = {}

        for person_id, keypoints_sequence in frames_keypoints.items():
            # Ensure keypoints_sequence contains exactly 30 valid frames
            if len(keypoints_sequence) != self.buffer_size:
                print(f"Warning: Person {person_id} does not have {self.buffer_size} valid frames. Skipping prediction.")
                predictions[person_id] = "No Action"
                continue

            # Convert keypoints sequence to numpy array, handling missing or malformed data
            cleaned_sequence = []
            for i, keypoints in enumerate(keypoints_sequence):
                if isinstance(keypoints, (list, np.ndarray)) and np.array(keypoints).shape == (34,):
                    cleaned_sequence.append(np.array(keypoints))
                else:
                    print(f"Warning: Frame {i} for person {person_id} has invalid keypoints. Replacing with zeros.")
                    cleaned_sequence.append(np.zeros(34))  # Fill invalid keypoints with zeros

            keypoints_array = np.array(cleaned_sequence)  # Ensure uniform shape (30, 34)

            # Final shape validation
            if keypoints_array.shape != (self.buffer_size, 34):
                print(f"Error: Unexpected shape {keypoints_array.shape} for person {person_id}. Skipping prediction.")
                predictions[person_id] = "No Action"
                continue

            input_data = np.expand_dims(keypoints_array, axis=0)  # Ensure shape is (1, 30, 34)

            try:
                # Convert to tensor for GPU processing
                input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

                # Run inference
                prediction = self.action_recognition_model.predict(input_tensor)[0]  # Assuming single output per person
                predicted_action = np.argmax(prediction)  # Get the action INDEX

                self.translate_action_results = self.actions_list[predicted_action]  # Get action name
                predictions[person_id] = self.translate_action_results

            except Exception as e:
                print(f"Error predicting action for person {person_id}: {e}")
                predictions[person_id] = "No Action"  # Handle prediction errors gracefully

        return predictions

    
    def run(self):
        
        """
        Process video frames and return a list of frames with bounding boxes and action labels.
        """
        frame_count = len(self.video_keypoints) - 1

        for frame_idx, frame_keypoints in enumerate(self.video_keypoints):
            processed_keypoints = self.preprocess_keypoints(frame_keypoints)
            
            for person_id, keypoints in processed_keypoints.items():
                self.person_keypoints[person_id].append(keypoints)
                
                # Keep only the latest 30 frames
                if len(self.person_keypoints[person_id]) > self.buffer_size:
                    self.person_keypoints[person_id].pop(0)
            
            # Predict actions once we have enough frames
            action_predictions = self.predict_actions(self.person_keypoints)

            self.detected_actions.append(action_predictions)
            
            progress = int((frame_idx / frame_count) * 100)
            self.progress_update.emit(progress)
            

        self.detected_actions_list.emit(self.detected_actions)

    def stop(self):
        self._running = False
        self.wait()