import cv2

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap



from utils import (VideoProcessorThread, VideoPlayerThread)
from trackers import (PoseDetection,
                    ActionDetectionThread)

class CenterVideo:
    def __init__(self, main_window):
        self.main_window = main_window
        
        self.human_detect_model = "yolov8m.pt"
        self.human_detect_conf = 0.5
        self.human_pose_model = "yolov8m-pose.pt"
        self.human_pose_conf = 0.5
        self.iou_value = 0.3

        self.returned_frames = []
        self.humanDetectionResults = []
        self.action_results_list = []

        self.humanPoseDetectionResults = None
        self.isImportDone = False
        self.videoHeight = None
        self.videoWidth = None
        self.number_of_frames = 0

        self.video_player_thread = None

        self.directory = None
    
        self.detection = PoseDetection(humanDetectionModel=self.human_detect_model,
                                                    humanDetectConf= self.human_detect_conf,
                                                    humanPoseModel= self.human_pose_model,
                                                    humanPoseConf= self.human_pose_conf
                                                    )
        

    #When browse for center video is clicked
    def browse_video(self):
        self.directory, _ = QFileDialog.getOpenFileName(self.main_window, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov *.wmv)")
        if self.directory:
            self.returned_frames = []
            self.humanDetectionResults = []
            self.action_results_list = []
            self.main_window.videoDirectory_center.setText(f"{self.directory}")
            self.start_video_processing(self.directory)
            self.main_window.is_center_video_ready = False

            self.main_window.video_player_thread_preview = None
            self.main_window.human_pose_results_center = None
            self.main_window.human_detect_results_center = None
            self.main_window.action_results_list_center = None
            self.main_window.import_video_button_center.setEnabled(False)
    
    def start_video_processing(self, video_path):
        self.main_window.status_label_center.setText("[ PROCESSING VIDEO... PLEASE WAIT ]")

        self.video_processor = VideoProcessorThread(video_path, resize_frames=False,
                                                    isFront=False,
                                                    human_detection_model=self.human_detect_model,
                                                    human_detection_confidence=self.human_detect_conf,
                                                    human_pose_model=self.human_pose_model,
                                                    human_pose_confidence=self.human_pose_conf,
                                                    main_window=self.main_window)
        
        self.video_processor.human_detect_results.connect(self.update_detection_results)
        self.video_processor.human_pose_results.connect(self.update_pose_detection_results)
        self.video_processor.progress_update.connect(self.update_progress_bar)

        #Start the operation
        self.video_processor.start()
        
    def update_detection_results(self, results_list):
        self.humanDetectionResults = results_list
        self.main_window.human_detect_results_center = results_list
        
        print("HUMAN DETECT RESULTS:", len(results_list))
    
    def update_pose_detection_results(self, pose_results):
        self.humanPoseDetectionResults = pose_results
        self.main_window.human_pose_results_center = pose_results
        self.identify_actions()

        print("POSE RESULTS:", len(pose_results))
    
    def identify_actions(self):
        self.main_window.status_label_center.setText("[ IDENTIFYING ACTIONS...]")

        self.action_detection_thread = ActionDetectionThread(video_keypoints=self.humanPoseDetectionResults,
                                                            black_frames=self.main_window.center_white_frames_preview,
                                                            video_frames=self.returned_frames,
                                                            detections=self.humanDetectionResults)
        
        self.action_detection_thread.detected_actions_list.connect(self.update_action_results)
        self.action_detection_thread.progress_update.connect(self.update_progress_bar)  
        self.action_detection_thread.start()

    def update_action_results(self, actions):
        self.action_results_list = actions
        self.main_window.action_results_list_center = actions
        self.main_window.status_label_center.setText("[ ACTIONS IDENTIFICATION, DONE! ]")
        
        self.main_window.is_center_video_ready = True

        self.main_window.activate_analytics(True)
        
        self.main_window.import_video_button_center.setEnabled(True)

        cap = cv2.VideoCapture(self.directory)

        #Get the frame count
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        #Get the video length
        # Get the frame rate (frames per second)
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the video length in seconds
        video_length = frame_count / fps
        
        # Convert video length to hours, minutes, and seconds
        hours = int(video_length // 3600)
        minutes = int((video_length % 3600) // 60)
        seconds = int(video_length % 60)
        time_formatted = f"{hours:02}:{minutes:02}:{seconds:02}"

        self.main_window.center_frame_count.setText(str(frame_count))
        
        self.main_window.center_video_length.setText(time_formatted)
        

        if self.main_window.is_center_video_ready and self.main_window.is_front_video_ready:
            self.main_window.activate_analytics(True)
            
        else:
            self.main_window.activate_analytics(False)
        
        print("CENTER VIDEO READY: ", self.main_window.is_center_video_ready)
        print("FRONT VIDEO READY: ", self.main_window.is_front_video_ready)

        print("ACTION_RESULT_LIST_FRONT:, ", self.main_window.action_result_list_front)
        print("ACTION_RESULT_LIST_CENTER:, ", self.main_window.action_result_list_center)
        
        self.video_processor.stop()
        self.video_processor = None
    
    def update_progress_bar(self,value):
        self.main_window.importProgressBar_center.setValue(value)

    def closeEvent(self, event):
        self.video_processor.stop()
        self.white_frame_generator.stop()
        event.accept()
    

    #=== FOR UPDATING FRAMES:

    def update_frame(self, center_frame):
        if len(center_frame.shape) == 3:
            center_video_height, center_video_width, _ = center_frame.shape
        else:
            center_video_height, center_video_width = center_frame.shape

        bytes_per_line = 3 * center_video_width
        q_img_center = QImage(center_frame.data, center_video_width, center_video_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage to the QLabel with aspect ratio maintained and white spaces
        pixmap_center = QPixmap.fromImage(q_img_center)
        scaled_pixmap_center = pixmap_center.scaled(self.main_window.video_preview_label_center.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.main_window.video_preview_label_center.setPixmap(scaled_pixmap_center)


    def cleanup_thread(self):
        """Called when the thread finishes to release memory."""
        self.video_player_thread.deleteLater()
        self.video_player_thread = None