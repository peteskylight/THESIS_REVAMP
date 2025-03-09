import cv2
from PySide6.QtCore import QThread, Signal
import numpy as np

from ultralytics import YOLO
import queue
import threading
from PySide6.QtGui import QImage, QPixmap

from pathlib import Path

class VideoUtils:
    
    def __init__(self) -> None:
        pass

    def read_video(self, video_path, resize_frames): #LIST
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if resize_frames:
                resized_frame = cv2.resize(frame, (1088, 608))
                frames.append(resized_frame)
            else:
                frames.append(frame)
        cap.release()
        return frames
    #Old Code
    
    def save_video(self, output_video_frames, output_video_path, monitorFrames=False):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
        
        for frame in output_video_frames:
            out.write(frame)
            if monitorFrames:
                cv2.imshow("Monitor Frames", frame)
                cv2.waitKey(10)
        out.release()
    
    def generate_white_frame(self, height, width):
        white_frame = np.ones((height, width, 3), dtype=np.uint8) * 255 # Create a white frame (all pixel values set to 255)
        return white_frame

class WhiteFrameGenerator(QThread):
    progress_update = Signal(object)
    return_white_frames = Signal(object)
    
    def __init__(self, number_of_frames, width, height):
        super().__init__()
        self.number_of_frames = number_of_frames
        self.videoWidth = width
        self.videoHeight = height
        self._running = True
    def run(self):
        white_frames = []
        current_frame = 0
        total_frames_length = self.number_of_frames
        for frame in range(total_frames_length):
            white_frame = np.ones((self.videoHeight, self.videoWidth, 3), dtype=np.uint8) * 0 
            white_frames.append(white_frame)
            current_frame += 1
            progress = int(current_frame/total_frames_length *100)
            
            self.progress_update.emit(progress)
            
            del white_frame
            del progress
            
        self.return_white_frames.emit(white_frames)
        del white_frames
    def stop(self):
        self._running = False
        self.wait()


class VideoProcessorThread(QThread):
    """
    This class is used to process the video in a thread.
    It will process the video frame by frame and emit signals with the results.
    What I mean with processing is to detect humans and their keypoints frame by frame. 
    Appending each result in the corresponding list.
    Emmits signals with the results.

    Here is the pseudo code:
    1. Read the video
    2. Get the total frames
    3. Loop through the video frames
       (Inside the loop)
        4. Get the frame
        5. Detect humans in the frame
        detected_human = [{Person1: 4 Corners; Person2: 4corners}]

        6. Append the results in the human_detect_results_list - GLOBAL

        7. Detect the keypoints in the frame based on the results of the cropped image in the human_detect_results_list

        8. Append the results in the human_pose_results_list
        
        9. Update the progress
    (Outside the loop)
    10. Emit the signals with the results

    """
    #Initiate the signals

    human_detect_results = Signal(object)
    human_pose_results = Signal(object)
    progress_update = Signal(object)

    def __init__(self, video_path,
                 resize_frames = False,
                 isFront = True,
                 human_detection_model = None,
                 human_detection_confidence = 0.5,
                 human_pose_model = None,
                 human_pose_confidence = 0.5,
                 main_window = None):

        super().__init__()

        self.isFront = isFront
        self.human_detection_model = YOLO(human_detection_model)  
        self.human_detection_confidence = human_detection_confidence
        self.human_pose_model = YOLO(human_pose_model)
        self.human_pose_confidence = human_pose_confidence
        self.main_window = main_window

        self.human_detect_results_list = []
        self.human_pose_results_list = []

        self.video_path = video_path
        self.resize_frames = resize_frames

        self.initial_row_height = None

        self._running = True

    #Mask for front camera
    def create_roi_mask(self, frame, row_height):
        height, width, _ = frame.shape
        mask = np.zeros((height, width), dtype=np.uint8)
        roi = (0, row_height+10, width, height)  # Bottom to middle
        cv2.rectangle(mask, (roi[0], roi[1]), (roi[2], roi[3]), 255, -1)
        cv2.line(img=frame, pt1=(0, row_height), pt2=(frame.shape[1], row_height),color = (0,255,0), thickness=2)
        return mask
    
    #Human Detection
    def human_detect(self, frame):
        # Declare student_dict early
        student_dict = {}

        # Add conditional to create mask or not
        if self.isFront:
            roi_mask = self.create_roi_mask(frame, self.initial_row_height)
            masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            results = self.human_detection_model.track(masked_frame,
                                                    conf=self.human_detection_confidence,
                                                    persist=True,
                                                    classes=0,
                                                    iou=0.3,
                                                    agnostic_nms=True)[0]
        else:
            results = self.human_detection_model.track(frame,
                                                    conf=self.human_detection_confidence,
                                                    persist=True,
                                                    classes=0,
                                                    iou=0.3,
                                                    agnostic_nms=True)[0]

        id_name_dict = results.names

        for result in results:
            boxes = result.boxes
            for box in boxes:
                if box.id is not None and box.xyxy is not None and box.cls is not None:
                    track_id = int(box.id.tolist()[0])
                    track_result = box.xyxy.tolist()[0]
                    object_cls_id = box.cls.tolist()[0]
                    object_cls_name = id_name_dict.get(object_cls_id, "unknown")

                    if object_cls_name == "person":
                        student_dict[track_id] = track_result
                else:
                    print("One of the attributes is None:", box.id, box.xyxy, box.cls)
                    continue  # Avoids UnboundLocalError

        return student_dict  # ✅ Always return a dictionary

    
    #Human Pose Detection
    def human_pose_detect(self, frame, human_results):
        keypoints_dict = {}  # Initialize here to store keypoints for all detections in the frame

        for track_id, bbox in human_results.items():
            # Extract the bounding box coordinates
            x1, y1, x2, y2 = map(int, bbox)
            cropped_image = frame[y1:y2, x1:x2]

            try:
                # Perform pose detection on the cropped image
                results = self.human_pose_model(cropped_image, self.human_pose_confidence)
                for result in results:
                    if result.keypoints:
                        keypoints_normalized = np.array(result.keypoints.xyn.cpu().numpy()[0])
                        keypoints_dict[track_id] = keypoints_normalized
                    else:
                        print(f"Error processing track ID {track_id}: {e}")
            except Exception as e:
                print(f"Error processing track ID {track_id}: {e}")
                keypoints_dict[track_id] = None
        
        return keypoints_dict

    def extend_frame_width(self,frame: np.ndarray, extension: int = 300) -> np.ndarray:
        """
        Extends the width of a given frame by adding black padding on both sides.

        Parameters:
        frame (np.ndarray): The input frame (1920x1080 expected).
        extension (int): The number of pixels to extend on each side. Default is 500.

        Returns:
        np.ndarray: The processed frame with extended width.
        """
        height, width, channels = frame.shape
        
        # Ensure the input frame is 1920x1080
        if width != 1920 or height != 1080:
            raise ValueError("Expected a 1920x1080 frame.")
        
        # Create a new black frame with extended width
        new_width = width + (2 * extension)
        extended_frame = np.zeros((height, new_width, channels), dtype=np.uint8)
        
        # Place the original frame in the center
        extended_frame[:, extension:extension + width] = frame
        
        return extended_frame
    
    #Main function to run in a thread
    def run(self):
        #Get the video
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        current_frame = 0

        #Variables for the human detection model
        
        while self._running:
            ret, retrieved_frame = cap.read() #Get the frame

            
            
            #Safety net if there is no frame anymore
            if not ret:
                break
            
            #Extend the frame
            frame = self.extend_frame_width(frame=retrieved_frame,
                                            extension=300)
            
            #Get the shape of the frame
            height, width, _ = frame.shape

            #Get the initial row height
            self.initial_row_height = int(height * (1/16))  # Bottom 4 rows (adjust as needed)
            
            human_detect_results = self.human_detect(frame)
            human_pose_detect_results = self.human_pose_detect(frame, human_detect_results)

            self.human_detect_results_list.append(human_detect_results)
            self.human_pose_results_list.append(human_pose_detect_results)

            current_frame += 1
            progress = int(current_frame/total_frames *100)
            self.progress_update.emit(progress)
            
            del frame #Just to delete unnecessary data

        self.human_detect_results.emit(self.human_detect_results_list)
        self.human_pose_results.emit(self.human_pose_results_list)

        #Just to delete unnecessary data
        del self.human_detect_results_list
        del self.human_pose_results_list

        cap.release() #Release the video

    def stop(self):
        self._running = False
        self.wait()

############################################################################################################


class VideoPlayerThread(QThread):
    '''
    The technique used in playing the video is to preload the frames in a queue and then play them in a loop.
    This is done to ensure that the video is played smoothly without any lagging.
    The technique is called "Frame Buffering WITH Lazy Lodaing
    Producer-Consumer Pattern (Using queue.Queue)
        Producer: The preload_frames() function continuously reads frames in the background and stores them in a queue (queue.Queue).
        Consumer: The run() method fetches preloaded frames from the queue and sends them to the UI for display.
    Advantage: This prevents the UI thread from waiting for frame decoding, making playback smooth.
         Lazy Loading (On-Demand Processing)

    Instead of storing all frames in RAM, we load only the next few frames (maxsize=10 in the queue).
    When a frame is needed, it's already decoded and ready for display.

    '''

    
    frames_signal = Signal(object)
    #cv2.setNumThreads(1)  # Disable OpenCV multi-threading to reduce CPU usage

    def __init__(self, center_video_path, front_video_path, main_window):
        super().__init__()

        self.main_window = main_window

        self.center_video_path = center_video_path
        self.front_video_path = front_video_path

        #Separate captures for each video

        self.center_cap = cv2.VideoCapture(self.center_video_path)
        self.front_cap = cv2.VideoCapture(self.front_video_path)

        #For all
        self.running = True
        self.paused = False

        self.max_size = 30
        #Keep preloaded frames here

        self.center_video_frame_queue = queue.Queue(maxsize=self.max_size)  
        self.center_video_black_frame_queue = queue.Queue(maxsize=self.max_size) 
        self.front_video_frame_queue = queue.Queue(maxsize=self.max_size)  
        self.front_video_black_frame_queue = queue.Queue(maxsize=self.max_size)  

        self.current_center_video_frame_index = 0
        self.current_front_video_frame_index = 0
        self.current_frame_index = 0

        self.target_frame_index = 0 # Set target frame index
        self.preload_thread = threading.Thread(target=self.preload_frames, daemon=True)
        self.preload_thread.start()  # Start background frame loader

        self.skeleton_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), 
            (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 12), 
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]


    def write_action_text(self, frame, black_frame, detections, actions):
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
            cv2.putText(black_frame, action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, black_frame

    def drawing_bounding_box(self, video_frame, results):

        black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        item_count = 0

        
        
        for track_id, bbox in results.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(video_frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(black_frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(black_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        return video_frame, black_frame
    
    def drawing_keypoints(self, keypoints_dict, detection, black_frame, video_frame):
        for track_id in keypoints_dict:
            if track_id in detection:
                keypoints = keypoints_dict[track_id]
                bbox = detection[track_id]
                bbox_x, bbox_y, bbox_w, bbox_h = bbox

                cropped_frame = black_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)]
                video_cropped_frame = video_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)]

                for keypoint in keypoints:
                    x = int(keypoint[0] * cropped_frame.shape[1])
                    y = int(keypoint[1] * cropped_frame.shape[0])
                    cv2.circle(cropped_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
                    cv2.circle(video_cropped_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

                # Draw skeleton
                for pair in self.skeleton_pairs:
                    if pair[0] < len(keypoints) and pair[1] < len(keypoints):
                        pt1 = keypoints[pair[0]]
                        pt2 = keypoints[pair[1]]
                        #print(f"pt1: {pt1}, pt2: {pt2}")
                        # # Debug statement
                        if isinstance(pt1, np.ndarray) and isinstance(pt2, np.ndarray):
                            x1 = int(pt1[0] * cropped_frame.shape[1])
                            y1 = int(pt1[1] * cropped_frame.shape[0])
                            x2 = int(pt2[0] * cropped_frame.shape[1])
                            y2 = int(pt2[1] * cropped_frame.shape[0])
                            
                            if 0.1 <= x1 < cropped_frame.shape[1] and 0.1 <= y1 < cropped_frame.shape[0] and 0.1 <= x2 < cropped_frame.shape[1] and 0.1 <= y2 < cropped_frame.shape[0]:
                                cv2.line(cropped_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                                cv2.line(video_cropped_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        else:
                            print(f"Invalid keypoints for track ID {track_id}: pt1={pt1}, pt2={pt2}")
                    else:
                        print(f"Keypoints array too small for track ID {track_id}: {keypoints}")

                # Place the cropped frame back into the original frame

                black_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)] = cropped_frame
                video_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)] = video_cropped_frame
        
        return video_frame, black_frame

    def preload_frames(self):
        results = None
        keypoints = None
        center_video_black_frame = None
        front_video_black_frame = None

        front_video_results = self.main_window.human_detect_results_front
        front_video_keypoints = self.main_window.human_pose_results_front
        front_video_actions = self.main_window.action_results_list_front

        center_video_results = self.main_window.human_detect_results_center
        center_video_keypoints = self.main_window.human_pose_results_center
        center_video_actions = self.main_window.action_results_list_center

        """
        Background thread to keep decoding frames ahead of playback
        """

        while self.running:
            if not self.paused and (not self.center_video_frame_queue.full()):
                front_ret, front_frame = self.front_cap.read()
                center_ret, center_frame = self.center_cap.read()

                if (not front_ret) or (not center_ret):
                    self.center_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video if it ends
                    self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_index = 0
                    self.target_frame_index = 0 
                    continue
                
                # Draw bounding boxes and keypoints only for the current frame index
                '''
                LOL EQUAL LANG NAMAN EH! NAGLOLOKOHAN LANG TAYO RITO AHAHHAHHAHA
                '''

                if self.current_frame_index == self.target_frame_index:  # Draw only for target frame index #
                    center_frame, center_video_black_frame = self.drawing_bounding_box(video_frame=center_frame,
                                                                results=center_video_results[int(self.current_frame_index)])
                    
                    center_frame, center_video_black_frame = self.drawing_keypoints(keypoints_dict=center_video_keypoints[int(self.current_frame_index)],
                                                                detection=center_video_results[int(self.current_frame_index)],
                                                                black_frame=center_video_black_frame,
                                                                video_frame=center_frame)
                    center_frame, center_video_black_frame = self.write_action_text(frame=center_frame,
                                                                                    black_frame=center_video_black_frame,
                                                                                    detections=center_video_results[int(self.current_frame_index)],
                                                                                    actions=center_video_actions[int(self.current_frame_index)])

                    
                    front_frame, front_video_black_frame = self.drawing_bounding_box(video_frame=front_frame,
                                                                results=front_video_results[int(self.current_frame_index)])
                    
                    front_frame, front_video_black_frame = self.drawing_keypoints(keypoints_dict=front_video_keypoints[int(self.current_frame_index)],
                                                                detection=front_video_results[int(self.current_frame_index)],
                                                                black_frame=front_video_black_frame,
                                                                video_frame=front_frame)
                    
                    front_frame, front_video_black_frame = self.write_action_text(frame=front_frame,
                                                                                  black_frame=front_video_black_frame,
                                                                                  detections=front_video_results[int(self.current_frame_index)],
                                                                                  actions=front_video_actions[int(self.current_frame_index)])

                else:
                    # If not the target frame, skip drawing
                    center_frame = center_frame  # Just pass the frame unchanged
                    center_video_black_frame = center_video_black_frame  # Just pass the black_frame unchanged
                    front_frame = front_frame
                    front_video_black_frame = front_video_black_frame
                
                # Store the frame with the drawn bounding box and keypoints
                if center_frame is not None and center_video_black_frame is not None:
                    self.center_video_frame_queue.put(center_frame)
                    self.center_video_black_frame_queue.put(center_video_black_frame)

                if front_frame is not None and front_video_black_frame is not None:
                    self.front_video_frame_queue.put(front_frame)
                    self.front_video_black_frame_queue.put(front_video_black_frame)


                self.current_frame_index += 1
                self.target_frame_index +=1

        # self.center_cap.release()
        # self.front_cap.release()

    def run(self):
        """Main playback loop, sending frames from queue to UI"""
        while self.running:
            if not self.paused and not self.center_video_frame_queue.empty():
                center_video_frame = self.center_video_frame_queue.get()  # Get frame from queue
                center_black_frame = self.center_video_black_frame_queue.get()

                front_video_frame = self.front_video_frame_queue.get()  # Get frame from queue
                front_black_frame = self.front_video_black_frame_queue.get()

                center_frame = None
                front_frame = None

                front_frame = front_video_frame
                center_frame = center_video_frame
            

                # height, width, channels = frame.shape
                # bytes_per_line = channels * width
                # qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.frames_signal.emit([center_frame, front_frame])  # Send frame to UI

                cv2.waitKey(1000//30)  # Play at ~30 FPS


    def stop(self):
        """Stops video playback and terminates threads."""
        self.running = False  # Stop preload loop
        self.quit()
        self.wait()
        if self.preload_thread.is_alive():
            self.preload_thread.join()  # Ensure the thread exits cleanly

        self.center_cap.release()
        self.front_cap.release()


    def pause(self, status):
        self.paused = status
        if status:
            self.main_window.play_pause_button_video_preview.setText("PLAY PREVIEW")
        else:
            self.main_window.play_pause_button_video_preview.setText("PAUSE PREVIEW")


class SeekingVideoPlayerThread(QThread):
    '''
    The technique used in playing the video is to preload the frames in a queue and then play them in a loop.
    This is done to ensure that the video is played smoothly without any lagging.
    The technique is called "Frame Buffering WITH Lazy Lodaing
    Producer-Consumer Pattern (Using queue.Queue)
        Producer: The preload_frames() function continuously reads frames in the background and stores them in a queue (queue.Queue).
        Consumer: The run() method fetches preloaded frames from the queue and sends them to the UI for display.
    Advantage: This prevents the UI thread from waiting for frame decoding, making playback smooth.
         Lazy Loading (On-Demand Processing)

    Instead of storing all frames in RAM, we load only the next few frames (maxsize=10 in the queue).
    When a frame is needed, it's already decoded and ready for display.

    '''

    
    frames_signal = Signal(object)
    #cv2.setNumThreads(1)  # Disable OpenCV multi-threading to reduce CPU usage

    def __init__(self, center_video_path, front_video_path, main_window):
        super().__init__()

        self.main_window = main_window

        self.center_video_path = center_video_path
        self.front_video_path = front_video_path

        #Separate captures for each video

        self.center_cap = cv2.VideoCapture(self.center_video_path)
        self.front_cap = cv2.VideoCapture(self.front_video_path)

        #For all
        self.running = True
        self.paused = False

        self.max_size = 15

        #Keep preloaded frames here

        self.classroom_heatmap_frames_queue = queue.Queue(maxsize=self.max_size)

        self.original_center_video_frame_queue = queue.Queue(maxsize=self.max_size)  
        self.center_video_frame_queue = queue.Queue(maxsize=self.max_size)  
        self.center_video_black_frame_queue = queue.Queue(maxsize=self.max_size) 

        self.original_front_video_frame_queue = queue.Queue(maxsize=self.max_size)  
        self.front_video_frame_queue = queue.Queue(maxsize=self.max_size)  
        self.front_video_black_frame_queue = queue.Queue(maxsize=self.max_size)  

        self.current_center_video_frame_index = 0
        self.current_front_video_frame_index = 0
        self.current_frame_index = 0

        self.target_frame_index = 0 # Set target frame index
        self.preload_thread = threading.Thread(target=self.preload_frames, daemon=True)
        self.preload_thread.start()  # Start background frame loader

        #For getting the seatplan image hehe
        script_dir = Path(__file__).parent  # Get script's folder
        image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
        self.seat_plan_picture = cv2.imread(image_path)
        self.seat_plan_picture_previous_frame = None
        self.isFirstFrame =True

        self.skeleton_pairs = [
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), 
            (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 12), 
            (11, 13), (12, 14), (13, 15), (14, 16)
        ]


        # Define 4 corresponding points from the center video frame to the heatmap


        self.src_pts_center = np.array([
                            [655,121],  # Top-left
                            [1939,169],  # Top-right
                            [105,674],  # Bottom-left
                            [2477,722]   # Bottom-right
                        ], dtype=np.float32)

        self.dst_pts_center = np.array([
                            [0,0],  # Top-left
                            [1357,0],  # Top-right
                            [0,293],  # Bottom-left
                            [1355, 293]   # Bottom-right
                        ], dtype=np.float32)

        self.H_center, _ = cv2.findHomography(self.src_pts_center, self.dst_pts_center, cv2.RANSAC)

        # Define homography for front video
        src_pts_front = np.array([
                            [471, 161],  # Top-left
                            [2001, 27],  # Top-right
                            [31, 896],  # Bottom-left
                            [2516, 792]   # Bottom-right
                        ], dtype=np.float32)

        dst_pts_front = np.array([
                        [0,293],  # Top-left
                        [1355, 293],  # Top-right
                        [3,593],  # Bottom-left
                        [1353,591]   # Bottom-right
                    ], dtype=np.float32)

        self.H_front, _ = cv2.findHomography(src_pts_front, dst_pts_front, cv2.RANSAC)


    

    def write_action_text(self, frame, black_frame, detections, actions):
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
            cv2.putText(black_frame, action_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame, black_frame

    def drawing_bounding_box(self, video_frame, results):

        black_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        item_count = 0

        
        
        for track_id, bbox in results.items():
            x1, y1, x2, y2 = bbox
            cv2.putText(video_frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(video_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            cv2.putText(black_frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] - 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)
            cv2.rectangle(black_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        
        return video_frame, black_frame
    
    def drawing_keypoints(self, keypoints_dict, detection, black_frame, video_frame):
        for track_id in keypoints_dict:
            if track_id in detection:
                keypoints = keypoints_dict[track_id]
                bbox = detection[track_id]
                bbox_x, bbox_y, bbox_w, bbox_h = bbox

                cropped_frame = black_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)]
                video_cropped_frame = video_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)]

                for keypoint in keypoints:
                    x = int(keypoint[0] * cropped_frame.shape[1])
                    y = int(keypoint[1] * cropped_frame.shape[0])
                    cv2.circle(cropped_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)
                    cv2.circle(video_cropped_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

                # Draw skeleton
                for pair in self.skeleton_pairs:
                    if pair[0] < len(keypoints) and pair[1] < len(keypoints):
                        pt1 = keypoints[pair[0]]
                        pt2 = keypoints[pair[1]]
                        #print(f"pt1: {pt1}, pt2: {pt2}")
                        # # Debug statement
                        if isinstance(pt1, np.ndarray) and isinstance(pt2, np.ndarray):
                            x1 = int(pt1[0] * cropped_frame.shape[1])
                            y1 = int(pt1[1] * cropped_frame.shape[0])
                            x2 = int(pt2[0] * cropped_frame.shape[1])
                            y2 = int(pt2[1] * cropped_frame.shape[0])
                            
                            if 0.1 <= x1 < cropped_frame.shape[1] and 0.1 <= y1 < cropped_frame.shape[0] and 0.1 <= x2 < cropped_frame.shape[1] and 0.1 <= y2 < cropped_frame.shape[0]:
                                cv2.line(cropped_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                                cv2.line(video_cropped_frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                        else:
                            print(f"Invalid keypoints for track ID {track_id}: pt1={pt1}, pt2={pt2}")
                    else:
                        print(f"Keypoints array too small for track ID {track_id}: {keypoints}")

                # Place the cropped frame back into the original frame

                black_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)] = cropped_frame
                video_frame[int(bbox_y):int(bbox_h), int(bbox_x):int(bbox_w)] = video_cropped_frame
        
        return video_frame, black_frame
    
    # def drawing_classroom_heatmap(self, frame, results):
    #     # Use a cached heatmap to avoid recomputing everything
    #     if self.isFirstFrame:
    #         self.heatmap_image = self.seat_plan_picture.copy()
    #         self.isFirstFrame = False
    #     else:
    #         self.heatmap_image = frame.copy()

    #     # Process people in parallel
    #     radius = 150
    #     gradient_circle = self.create_gradient_circle(radius, (0, 0, 255), 16)  # Precompute once

    #     for person_id, bbox in results.items():
    #         center = (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))
    #         self.overlay_image_alpha(self.heatmap_image, gradient_circle, (center[0] - radius, center[1] - radius))

    #     self.seat_plan_picture_previous_frame = self.heatmap_image
    #     return self.heatmap_image

    def drawing_classroom_heatmap(self, frame, results, H_transform):
        """
        Uses homography to transform detected person locations onto the heatmap.
        Then, overlays a gradient heatmap circle at the transformed location.

        - H_transform: Homography matrix for mapping detections to the heatmap.
        """

        # Use a cached heatmap to avoid recomputing everything
        if self.isFirstFrame:
            self.heatmap_image = self.seat_plan_picture.copy()
            self.isFirstFrame = False
        else:
            self.heatmap_image = frame.copy()

        #Resize in order to make good homography Transformation
        #self.heatmap_image = self.extend_frame_width()

        # Define the radius and precompute the gradient circle
        radius = 150
        gradient_circle = self.create_gradient_circle(radius, (0, 0, 255), 16)

        for person_id, bbox in results.items():
            x1, y1, x2, y2 = bbox  # Bounding box coordinates

            # Compute the center of the whole bounding box
            center_point = np.array([[[(x1 + x2) / 2, (y1 + y2) / 2]]], dtype=np.float32)

            # Apply homography transformation
            transformed_point = cv2.perspectiveTransform(center_point, H_transform)
            mapped_x, mapped_y = transformed_point[0][0]  # Extract transformed coordinates

            # Overlay the heatmap gradient at the transformed location
            self.overlay_image_alpha(self.heatmap_image, gradient_circle, (int(mapped_x - radius), int(mapped_y - radius)))

        # Cache the last processed heatmap frame
        self.seat_plan_picture_previous_frame = self.heatmap_image

        return self.heatmap_image


        # Function to create a gradient circle
    def create_gradient_circle(self, radius, color, max_alpha):
        Y, X = np.ogrid[:2*radius, :2*radius]
        center = radius
        dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)

        alpha = np.clip(max_alpha - (max_alpha * (dist_from_center / radius)), 0, max_alpha).astype(np.uint8)

        # Vectorized assignment
        gradient_circle = np.zeros((2*radius, 2*radius, 4), dtype=np.uint8)
        gradient_circle[..., :3] = color  # Set RGB channels
        gradient_circle[..., 3] = alpha  # Set alpha channel

        return gradient_circle



    # Function to overlay image with alpha
    def overlay_image_alpha(self, img, img_overlay, pos):
        """
        Blends an overlay image (e.g., heatmap) onto another image with transparency.
        """
        x, y = pos
        h, w = img_overlay.shape[:2]

        # Compute overlap regions
        y1, y2 = max(0, y), min(img.shape[0], y + h)
        x1, x2 = max(0, x), min(img.shape[1], x + w)

        y1o, y2o = max(0, -y), min(h, img.shape[0] - y)
        x1o, x2o = max(0, -x), min(w, img.shape[1] - x)

        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        # Alpha blending
        img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
        alpha = img_overlay_crop[..., 3:4] / 255.0  # Keep alpha as (h, w, 1) for broadcasting
        img[y1:y2, x1:x2, :3] = (alpha * img_overlay_crop[..., :3] +
                                (1 - alpha) * img[y1:y2, x1:x2, :3]).astype(np.uint8)



    def extend_frame_width(self,frame: np.ndarray, extension: int = 300) -> np.ndarray:
        """
        Extends the width of a given frame by adding black padding on both sides.

        Parameters:
        frame (np.ndarray): The input frame (1920x1080 expected).
        extension (int): The number of pixels to extend on each side. Default is 500.

        Returns:
        np.ndarray: The processed frame with extended width.
        """
        height, width, channels = frame.shape
        
        # Ensure the input frame is 1920x1080
        if width != 1920 or height != 1080:
            raise ValueError("Expected a 1920x1080 frame.")
        
        # Create a new black frame with extended width
        new_width = width + (2 * extension)
        extended_frame = np.zeros((height, new_width, channels), dtype=np.uint8)
        
        # Place the original frame in the center
        extended_frame[:, extension:extension + width] = frame
        
        return extended_frame

    def preload_frames(self):
        results = None
        keypoints = None
        center_video_black_frame = None
        front_video_black_frame = None
        front_classroom_heatmap = None
        center_classroom_heatmap = None

        front_video_results = self.main_window.human_detect_results_front
        front_video_keypoints = self.main_window.human_pose_results_front
        front_video_actions = self.main_window.action_results_list_front

        center_video_results = self.main_window.human_detect_results_center
        center_video_keypoints = self.main_window.human_pose_results_center
        center_video_actions = self.main_window.action_results_list_center

        """
        Background thread to keep decoding frames ahead of playback
        """

        while self.running:
            if not self.paused and (not self.center_video_frame_queue.full()):
                front_ret, retrieved_front_frame = self.front_cap.read()
                center_ret, retrieved_center_frame = self.center_cap.read()

                front_frame = self.extend_frame_width(frame=retrieved_front_frame,
                                                      extension=300)
                
                center_frame = self.extend_frame_width(frame=retrieved_center_frame,
                                                       extension=300)
            
 
                original_front_frame = front_frame.copy()
                original_center_frame = center_frame.copy()
                
                min_value, max_value = self.main_window.timeFrameRangeSlider.value()

                #Make sure that the videos are starting based on the set range sliders
                if self.isFirstFrame:
                    self.center_cap.set(cv2.CAP_PROP_POS_FRAMES, min_value)
                    self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, min_value)
                    self.isFirstFrame = False
                    self.seat_plan_picture_previous_frame = self.seat_plan_picture.copy()

                if (not front_ret) or (self.current_frame_index > (max_value-1)):
                    self.center_cap.set(cv2.CAP_PROP_POS_FRAMES, min_value)  # Restart video if it ends
                    self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, min_value)

                    self.seat_plan_picture_previous_frame = self.seat_plan_picture.copy()
                    self.isFirstFrame = True

                    self.current_frame_index = min_value
                    self.target_frame_index = min_value
                    continue
                
                # Draw bounding boxes and keypoints only for the current frame index
                '''
                LOL EQUAL LANG NAMAN EH! NAGLOLOKOHAN LANG TAYO RITO AHAHHAHHAHA
                '''

                if self.current_frame_index == self.target_frame_index:  # Draw only for target frame index #
                    #DEBUGGING
                    print("CURRENT INDEX: ", self.current_center_video_frame_index)
                    print("LENGTH OF LIST", len(center_video_results))
                    center_frame, center_video_black_frame = self.drawing_bounding_box(video_frame=center_frame,
                                                                results=center_video_results[int(self.current_frame_index)])
                    
                    center_frame, center_video_black_frame = self.drawing_keypoints(keypoints_dict=center_video_keypoints[int(self.current_frame_index)],
                                                                detection=center_video_results[int(self.current_frame_index)],
                                                                black_frame=center_video_black_frame,
                                                                video_frame=center_frame)
                    center_frame, center_video_black_frame = self.write_action_text(frame=center_frame,
                                                                                    black_frame=center_video_black_frame,
                                                                                    detections=center_video_results[int(self.current_frame_index)],
                                                                                    actions=center_video_actions[int(self.current_frame_index)])

                    
                    front_frame, front_video_black_frame = self.drawing_bounding_box(video_frame=front_frame,
                                                                results=front_video_results[int(self.current_frame_index)])
                    
                    front_frame, front_video_black_frame = self.drawing_keypoints(keypoints_dict=front_video_keypoints[int(self.current_frame_index)],
                                                                detection=front_video_results[int(self.current_frame_index)],
                                                                black_frame=front_video_black_frame,
                                                                video_frame=front_frame)
                    
                    front_frame, front_video_black_frame = self.write_action_text(frame=front_frame,
                                                                                  black_frame=front_video_black_frame,
                                                                                  detections=front_video_results[int(self.current_frame_index)],
                                                                                  actions=front_video_actions[int(self.current_frame_index)])
                    heatmap_frame_threshold = 30
                    if self.current_frame_index % heatmap_frame_threshold == 0:
                        front_classroom_heatmap = self.drawing_classroom_heatmap(
                            frame=self.seat_plan_picture_previous_frame,
                            results=front_video_results[int(self.current_frame_index)],
                            H_transform=self.H_front  # Apply front video homography
                        )

                        center_classroom_heatmap = self.drawing_classroom_heatmap(
                            frame=front_classroom_heatmap,
                            results=center_video_results[int(self.current_frame_index)],
                            H_transform=self.H_center  # Apply center video homography
                        )

                        # Store the heatmap in the queue
                        self.classroom_heatmap_frames_queue.put(center_classroom_heatmap)
                    else:
                        if self.isFirstFrame:
                            self.classroom_heatmap_frames_queue.put(self.seat_plan_picture)
                        else:
                            self.classroom_heatmap_frames_queue.put(self.seat_plan_picture_previous_frame)

                else:
                    # If not the target frame, skip drawing
                    self.center_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                    self.front_cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                    self.current_frame_index = self.target_frame_index
                
                # Store the frame with the drawn bounding box and keypoints
                if center_frame is not None and center_video_black_frame is not None:
                    self.original_center_video_frame_queue.put(original_center_frame)
                    self.center_video_frame_queue.put(center_frame)
                    self.center_video_black_frame_queue.put(center_video_black_frame)

                    


                if front_frame is not None and front_video_black_frame is not None:
                    self.original_front_video_frame_queue.put(original_front_frame)
                    self.front_video_frame_queue.put(front_frame)
                    self.front_video_black_frame_queue.put(front_video_black_frame)

                
                self.current_frame_index += 1
                self.target_frame_index +=1

        # self.center_cap.release()
        # self.front_cap.release()

    def run(self):
        """Main playback loop, sending frames from queue to UI"""
        while self.running:
            if not self.paused and not self.center_video_frame_queue.empty():
                center_video_frame = self.center_video_frame_queue.get()  # Get frame from queue
                center_black_frame = self.center_video_black_frame_queue.get()
                original_center_frame = self.original_center_video_frame_queue.get()

                front_video_frame = self.front_video_frame_queue.get()  # Get frame from queue
                front_black_frame = self.front_video_black_frame_queue.get()
                original_front_frame = self.original_front_video_frame_queue.get()

                classroom_heatmap = self.classroom_heatmap_frames_queue.get()

                # if self.current_frame_index == self.target_frame_index:  
                #     self.frames_signal.emit([center_video_frame, front_video_frame])

                center_frame = None
                front_frame = None

                if self.main_window.keypointsOnlyChkBox_front_analytics.isChecked():
                    self.main_window.originalVideoOnlyChkBox_front_analytics.setEnabled(False)
                    front_frame = front_black_frame
                else:
                    self.main_window.originalVideoOnlyChkBox_front_analytics.setEnabled(True)
                    front_frame = front_video_frame

                if self.main_window.keypointsOnlyChkBox_Center_analytics.isChecked():
                    self.main_window.originalVideoOnlyChkBox_center_analytics.setEnabled(False)
                    center_frame = center_black_frame
                else:
                    self.main_window.originalVideoOnlyChkBox_center_analytics.setEnabled(True)
                    center_frame = center_video_frame

                if self.main_window.originalVideoOnlyChkBox_front_analytics.isChecked():
                    self.main_window.keypointsOnlyChkBox_front_analytics.setEnabled(False)
                    front_frame = original_front_frame
                else:
                    self.main_window.keypointsOnlyChkBox_front_analytics.setEnabled(True)
                    front_frame = front_video_frame
                
                if self.main_window.originalVideoOnlyChkBox_center_analytics.isChecked():
                    self.main_window.keypointsOnlyChkBox_Center_analytics.setEnabled(False)
                    center_frame = original_center_frame
                else:
                    self.main_window.keypointsOnlyChkBox_Center_analytics.setEnabled(True)
                    center_frame = center_video_frame

                self.frames_signal.emit([center_frame, front_frame, classroom_heatmap])  # Send frameS to UI displayer/updater

                
                
                cv2.waitKey(1000//30)  # Play at ~30 FPS

    def stop(self):
        self.running = False
        self.quit()
        self.wait()

    def pause(self, status):
        self.paused = status
        if status:
            self.main_window.play_pause_button_video_preview.setText("PLAY PREVIEW")
        else:
            self.main_window.play_pause_button_video_preview.setText("PAUSE PREVIEW")

