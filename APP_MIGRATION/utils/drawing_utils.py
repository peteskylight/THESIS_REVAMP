import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator

import random
from PySide6.QtCore import QThread, Signal
import numpy as np
from ultralytics.utils.plotting import Annotator

class DrawingUtils:
    def __init__(self) -> None:
        pass

    def draw_bounding_box(self, frame, box):
        # Draw bounding box
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cv2.rectangle(frame, (x, y), (w, h), (0, 255, 0), 2)
        cv2.putText(frame, "Tester", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def draw_bounding_box_import(self, frame, bbox, track_id):
        # Draw bounding box
        x1, y1, x2, y2 = bbox
        cv2.putText(frame, f"Student ID: {track_id}", (int(bbox[0]), int(bbox[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    
    def drawPoseLandmarks(self, frame, keypoints):
        for keypoint in keypoints:
            x = int(keypoint[0] * frame.shape[1])
            y = int(keypoint[1] * frame.shape[0])
            cv2.circle(frame, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
        
        return frame
    
    
    def draw_keypoints_and_skeleton(self,frame, keypoints):
        skeleton_pairs = [
        (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 6), 
        (5, 7), (6, 8), (7, 9), (8, 10), (5, 11), (6, 12), (11, 12), 
        (11, 13), (12, 14), (13, 15), (14, 16)
        ]
    
        # Draw skeleton
        for pair in skeleton_pairs:
            pt1 = keypoints[pair[0]]
            pt2 = keypoints[pair[1]]
            x1 = int(pt1[0] * frame.shape[1])
            y1 = int(pt1[1] * frame.shape[0])
            x2 = int(pt2[0] * frame.shape[1])
            y2 = int(pt2[1] * frame.shape[0])
            if 0.1 <= x1 < frame.shape[1] and 0.1 <= y1 < frame.shape[0] and 0.1 <= x2 < frame.shape[1] and 0.1 <= y2 < frame.shape[0]:
                cv2.line(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        
        return frame



from superqt import QRangeSlider
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout
from PySide6.QtGui import QPainter, QPixmap, QImage
from PySide6.QtCore import Qt

class ThumbnailRangeSlider(QRangeSlider):


    '''
    THIS FUNCTION IS JUST EME
    '''
    def __init__(self, video_path, num_thumbnails=10, *args, **kwargs):
        """
        Custom QRangeSlider that overlays video thumbnails in the selected range.

        :param video_path: Path to the video file.
        :param num_thumbnails: Number of thumbnails to extract.
        """
        super().__init__(*args, **kwargs)
        self.video_path = video_path
        self.num_thumbnails = num_thumbnails
        self.thumbnails = self.extract_video_thumbnails()

    def extract_video_thumbnails(self):
        """ Extracts evenly spaced thumbnails from the video. """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Cannot open video")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = total_frames // self.num_thumbnails
        thumbnails = []

        for i in range(self.num_thumbnails):
            frame_index = i * frame_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                height, width, channel = frame.shape
                bytes_per_line = channel * width
                qimg = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                thumbnails.append(QPixmap.fromImage(qimg))

        cap.release()
        return thumbnails

    def paintEvent(self, event):
        """ Custom painting of the slider including video thumbnails. """
        super().paintEvent(event)  # Call base class painting
        
        painter = QPainter(self)
        slider_rect = self.rect()

        # Get min/max handle positions
        min_pos, max_pos = self.value()
        range_min, range_max = self.minimum(), self.maximum()

        # Convert values to pixel positions
        min_x = int(slider_rect.width() * (min_pos - range_min) / (range_max - range_min))
        max_x = int(slider_rect.width() * (max_pos - range_min) / (range_max - range_min))

        # Draw Thumbnails in the Selected Range
        if self.thumbnails:
            num_thumbnails = len(self.thumbnails)
            thumb_width = (max_x - min_x) // num_thumbnails  # Adjust thumbnail width dynamically

            for i in range(num_thumbnails):
                x_pos = min_x + i * thumb_width
                thumb = self.thumbnails[i].scaled(thumb_width, 30, Qt.KeepAspectRatio)  # Resize thumbnail
                painter.drawPixmap(x_pos, slider_rect.height() // 2 - 15, thumb)

        painter.end()