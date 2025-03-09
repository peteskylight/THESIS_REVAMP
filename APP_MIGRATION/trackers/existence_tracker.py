
import cv2

class ExistenceTracker:
    def __init__(self, main_window, tracked_humans):
        self.main_window = main_window
        self.tracked_humans = tracked_humans

        self.total_durations = {}

    
    def calculate_total_duration(self, videoDirectorylineEdit):
        #Get the fps of the video in the directory line edit
        video_path = videoDirectorylineEdit.text()
        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)

        total_durations = {}
        for human_id, frames in self.tracked_humans.items():
            if frames['start_frame'] is not None and frames['end_frame'] is not None:
                duration_frames = frames['end_frame'] - frames['start_frame'] + 1
                duration_seconds = duration_frames / fps
                if human_id in total_durations:
                    total_durations[human_id] += duration_seconds
                else:
                    total_durations[human_id] = duration_seconds

        # Print the total duration for each tracked human
        for human_id, duration_seconds in total_durations.items():
            print(f"Human {human_id} existed for {duration_seconds:.2f} seconds throughout the video.")

