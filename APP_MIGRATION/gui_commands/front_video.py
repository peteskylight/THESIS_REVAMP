import cv2

from PySide6.QtWidgets import QFileDialog
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap



from utils import (VideoProcessorThread, VideoPlayerThread)
from trackers import (PoseDetection,
                    ActionDetectionThread)

class FrontVideo:
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
        self.front_video_processor = None
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
            
            self.main_window.video_player_thread_preview = None

            self.main_window.videoDirectory_front.setText(f"{self.directory}")
            
            self.main_window.is_front_video_ready = False
            self.main_window.human_pose_results_front = None
            self.main_window.human_detect_results_front = None
            self.main_window.action_results_list_front = None
            
            self.main_window.import_video_button_front.setEnabled(False)

            self.start_video_processing(self.directory)

    
    def start_video_processing(self, video_path):
        self.main_window.status_label_front.setText("[ PROCESSING VIDEO... PLEASE WAIT ]")

        self.front_video_processor = VideoProcessorThread(video_path, resize_frames=False,
                                                    isFront=True,
                                                    human_detection_model=self.human_detect_model,
                                                    human_detection_confidence=self.human_detect_conf,
                                                    human_pose_model=self.human_pose_model,
                                                    human_pose_confidence=self.human_pose_conf,
                                                    main_window=self.main_window)
        
        self.front_video_processor.human_detect_results.connect(self.update_detection_results)
        self.front_video_processor.human_pose_results.connect(self.update_pose_detection_results)
        self.front_video_processor.progress_update.connect(self.update_progress_bar)

        #Start the operation
        self.front_video_processor.start()
        
    def update_detection_results(self, results_list):
        self.humanDetectionResults = results_list
        self.main_window.human_detect_results_front = results_list

        print("HUMAN DETECT RESULTS:", len(results_list))
    
    def update_pose_detection_results(self, pose_results):
        self.humanPoseDetectionResults = pose_results
        self.main_window.human_pose_results_front = pose_results
        self.identify_actions()

        print("POSE RESULTS:", len(pose_results))
    
    def identify_actions(self):
        self.main_window.status_label_front.setText("[ IDENTIFYING ACTIONS...]")

        self.action_detection_thread = ActionDetectionThread(video_keypoints=self.humanPoseDetectionResults,
                                                            black_frames=self.main_window.front_white_frames_preview,
                                                            video_frames=self.returned_frames,
                                                            detections=self.humanDetectionResults)
        
        self.action_detection_thread.detected_actions_list.connect(self.update_action_results)
        self.action_detection_thread.progress_update.connect(self.update_progress_bar)  
        self.action_detection_thread.start()

    def update_action_results(self, actions):
        self.action_results_list = actions
        self.main_window.action_results_list_front = actions
        self.main_window.status_label_front.setText("[ ACTIONS IDENTIFICATION, DONE! ]")
        self.main_window.is_front_video_ready = True

        self.main_window.import_video_button_front.setEnabled(True)

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

        self.main_window.front_frame_count.setText(str(frame_count))

        self.main_window.front_video_length.setText(time_formatted)
        
        if self.main_window.is_center_video_ready and self.main_window.is_front_video_ready:
            self.main_window.activate_analytics(True)
        else:
            self.main_window.activate_analytics(False)
        
        self.front_video_processor.stop()
        self.front_video_processor = None
    
    def update_progress_bar(self,value):
        self.main_window.importProgressBar_front.setValue(value)

    def closeEvent(self, event):
        self.front_video_processor.stop()
        self.white_frame_generator.stop()
        event.accept()
    
    #=== FOR UPDATING FRAMES:

    def update_frame(self, front_frame):
        if len(front_frame.shape) == 3:
            front_video_height, front_video_width, _ = front_frame.shape
        else:
            front_video_height, front_video_width = front_frame.shape

        initial_row_height = int(front_video_height * (1/16))  # Bottom 4 rows (adjust as needed)

        cv2.line(img=front_frame, pt1=(0, initial_row_height), pt2=(front_frame.shape[1], initial_row_height),color = (0,255,0), thickness=2)

        bytes_per_line = 3 * front_video_width
        q_img_front = QImage(front_frame.data, front_video_width, front_video_height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage to the QLabel with aspect ratio maintained and white spaces
        pixmap_front = QPixmap.fromImage(q_img_front)
        scaled_pixmap_front = pixmap_front.scaled(self.main_window.video_preview_label_front.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        self.main_window.video_preview_label_front.setPixmap(scaled_pixmap_front)



    def cleanup_thread(self):
        """Called when the thread finishes to release memory."""
        self.video_player_thread.deleteLater()
        self.video_player_thread = None