import cv2
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap

from trackers.pose_detection import PoseDetection
from utils.drawing_utils import DrawingUtils
from utils.tools import Tools
from utils.cvfpscalc import CvFpsCalc

class CameraFeed:
    def __init__(self, label, white_frame_label, main_window):
        self.pose_detection = PoseDetection(humanDetectionModel='yolov8m.pt',
                                            humanDetectConf=0.4,
                                            humanPoseModel='yolov8m-pose.pt',
                                            humanPoseConf=0.4
                                            )
        
        # Get FPS
        self.getFPS = CvFpsCalc(buffer_len=10)
        
        self.drawing_utils = DrawingUtils()
        self.tools_utils = Tools()
        
        self.isRecording = False
        self.folder_count = 0
        self.frame_count = 0
        self.alreadyChecked = False
        self.countdown = 0  # Add countdown attribute
    
        self.label = label
        self.white_frame_label = white_frame_label
        self.main_window = main_window
        self.cap = None
        self.timer = QTimer()
        self.countdown_timer = QTimer()  # Add a separate timer for the countdown

        self.timer.timeout.connect(self.update_frame)
        self.countdown_timer.timeout.connect(self.update_countdown)  # Connect the countdown timer to the update_countdown method


    def start_camera(self, index):
        self.cap = cv2.VideoCapture(index) 
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 0) 

        self.timer.start(10)  # Keep the camera feed timer at 10 milliseconds
        self.countdown_timer.start(1000)  # Set the countdown timer to 1000 milliseconds (1 second)

    def update_frame(self): #GET THE FRAME HERE
        ret, output_frame = self.cap.read()
        color = 255 #255 - WHITE, 0 - BLACK
        
        returned_frame = output_frame
        
        returned_frame, normalized_keypoints, bbox = self.pose_detection.getHumanPoseKeypoints(frame=output_frame)
        
        #==Conditionals for VISUALIZATIONS
        if self.main_window.showCameraLandmarksChkBox.isChecked():
            self.drawing_utils.drawPoseLandmarks(frame = returned_frame,
                                                keypoints = normalized_keypoints)
        
        if self.main_window.showCameraBoundingBoxChkBox.isChecked():
            self.drawing_utils.draw_bounding_box(frame=returned_frame,
                                                 box=bbox)
        
        if self.main_window.show_skeleton_camera.isChecked():
            self.drawing_utils.draw_keypoints_and_skeleton(frame=returned_frame,
                                                           keypoints=normalized_keypoints)
        
        processed_frame = returned_frame
        
        if self.main_window.darkMode_whiteframe.isChecked():
            color = 0
        else:
            color = 255
        white_frame = color * np.ones_like(processed_frame)
        
        self.drawing_utils.drawPoseLandmarks(frame = white_frame,
                                                keypoints = normalized_keypoints)
        
        if self.main_window.show_whiteframe_boundingbox.isChecked():
            self.drawing_utils.draw_bounding_box(frame=white_frame,
                                                 box=bbox)
        
        if self.main_window.show_skeleton_white_frame.isChecked():
            self.drawing_utils.draw_keypoints_and_skeleton(frame=white_frame,
                                                 keypoints=normalized_keypoints)
        


        #===FOR LOGGING
        
        
        
        if self.main_window.recording_button.text() == "STOP\nRECORDING":
            if self.main_window.status_label.text() == "RECORDING":
                chosen_directory = self.main_window.directoryLineEdit.text()
                chosen_action = self.main_window.action_comboBox.currentText()
                destination_directory = os.path.join(chosen_directory, chosen_action)
        
                self.folder_count = self.tools_utils.count_folders(directory=destination_directory)
                
                # Ensure the folder exists before recording keypoints
                if not os.path.isdir(os.path.join(destination_directory, str(self.folder_count))):
                    os.mkdir(os.path.join(destination_directory, str(self.folder_count)))
                
                self.record_and_save_keypoints(normalized_keypoints=normalized_keypoints,
                                            frame_num=self.frame_count)
            
                self.frame_count += 1

                if self.frame_count % int(self.main_window.sequence_slider.value()) == 0:
                    self.main_window.status_label.setText("NOT RECORDING")
                    self.frame_count = 0
                    self.folder_count += 1
                    os.mkdir(os.path.join(destination_directory, str(self.folder_count)))
                    self.countdown = int(self.main_window.interval_slider.value())  # Set countdown
                    self.countdown_timer.start(1000)  # Restart the countdown timer

            if self.countdown == 0:
                self.main_window.status_label.setText("RECORDING")

        elif self.main_window.recording_button.text() == "START\nRECORDING":
           self.frame_count = 0
            
        #===FOR SHOWING IN LABELS
        if ret:
            # Display countdown
            if self.countdown > 0:
                countdown_text = f"{self.countdown}"
                text_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)[0]
                text_x = (processed_frame.shape[1] - text_size[0]) // 2
                text_y = (processed_frame.shape[0] + text_size[1]) // 2
                cv2.putText(processed_frame, countdown_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), 10, cv2.LINE_AA)
            
            # Convert the frame to QImage
            height, width, channel = processed_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(processed_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)

            # Generate a white frame
            
            white_q_img = QImage(white_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            white_pixmap = QPixmap.fromImage(white_q_img)
            scaled_white_pixmap = white_pixmap.scaled(self.white_frame_label.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            self.white_frame_label.setPixmap(scaled_white_pixmap)
            
            fps = self.getFPS.get()
            self.main_window.fps_label.setText(str(fps))
            
    def update_countdown(self):
        if self.countdown > 0:
            self.countdown -= 1
        else:
            self.countdown_timer.stop()  # Stop the countdown timer when countdown reaches 0
            

    def stop_camera(self):
        self.timer.stop()
        self.countdown_timer.stop()  # Stop the countdown timer
        if self.cap is not None:
            self.cap.release()
            
        # Clear the labels and set the text
        self.main_window.camera_feed.clear()
        self.main_window.camera_feed.setText("Camera stopped. No feed available.")
        self.main_window.camera_feed.setAlignment(Qt.AlignCenter)
        
        self.main_window.white_frame_feed.clear()
        self.main_window.white_frame_feed.setText("White frame stopped. No feed available.")
        self.main_window.white_frame_feed.setAlignment(Qt.AlignCenter)

    
    def record_and_save_keypoints(self, normalized_keypoints, frame_num):
        flattenedList = normalized_keypoints.flatten()
        chosen_directory = self.main_window.directoryLineEdit.text()
        chosen_action = self.main_window.action_comboBox.currentText()
        destination_directory = os.path.join(chosen_directory, chosen_action)
        no_of_sequences = self.main_window.sequence_slider.value()
        
        if not os.path.isdir(destination_directory):
            QMessageBox.critical(self.main_window, "Error", "The specified directory does not exist. Check the chosen directory.")
            self.main_window.toggle_button()
            return
        
        final_destination_directory = os.path.join(destination_directory, str(self.folder_count))

        # NEW Export keypoints
        npy_path = os.path.join(final_destination_directory, str(frame_num))
        np.save(npy_path, flattenedList)

    def countdownEnd(self):
        self.frame_count = 0
        self.main_window.status_label.setText("RECORDING")