import cv2
import numpy as np
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox

from pathlib import Path


class AnalyticsTab:
    def __init__(self, main_window):
        self.main_window = main_window
        script_dir = Path(__file__).parent  # Get script's folder
        image_path = script_dir.parent / "assets" / "SEAT PLAN.png"
        self.seat_plan_picture = cv2.imread(str(image_path))
        self.action_selector = None
        
        self.action_labels = ['All Actions', 'Extending Right Arm', 'Standing', 'Sitting']
        self.human_detect_results_front = []
        self.human_detect_results_center = []
        self.action_results_front = []  # New list to store detected actions
        self.action_results_center = []  # New list to store detected actions

        


    def toggle_analytics_tab(self):
        # Toggle the visibility of the tab
        if self.analytics_tab_index == -1 or not self.MainTab.isTabVisible(self.analytics_tab_index):
            # Add the tab back
            self.MainTab.addTab(self.analytics_tab, self.analytics_tab_title)
            self.analytics_tab_index = self.MainTab.indexOf(self.analytics_tab)
            self.MainTab.setCurrentIndex(self.analytics_tab_index)

            # Create QComboBox only if it doesn't exist
            if self.action_selector is None:
                self.action_selector = QComboBox()
                self.action_selector.addItems(self.action_labels)  # Now this works
                self.action_selector.currentIndexChanged.connect(self.update_selected_action)
                self.action_selector.setFixedSize(120, 30)
                
                # Add combo box to the main window
                self.main_window.layout().addWidget(self.action_selector)
            
            self.selected_action = 'All Actions'  # Default selection
        else:
            # Remove the tab
            self.MainTab.removeTab(self.analytics_tab_index)
            self.analytics_tab_index = -1

    def update_selected_action(self):
        """Updates the selected action from the combo box and refreshes the heatmap."""
        if self.action_selector:
            self.selected_action = self.action_selector.currentText()
            print(f"Selected Action: {self.selected_action}")  # Debugging print

            # Get the latest heatmap frame
            heatmap_frame = self.get_latest_heatmap_frame()  

            if heatmap_frame is not None:
                print("[DEBUG] Heatmap frame received, updating...")
                
                self.update_heatmap(
                    heatmap_frame, 
                    self.selected_action, 
                    self.human_detect_results_front,  
                    self.human_detect_results_center,
                    self.action_results_front,  # Pass action results
                    self.action_results_center
                )
            else:
                print("[DEBUG] No heatmap frame available.")


    def update_heatmap(self, frame, selected_action=None):
        """Updates the heatmap based on the selected action and integrates action detection results."""
        
        if frame is None:
            print("[DEBUG] No frame provided for heatmap update.")
            return

        # Debugging: Check stored action results before filtering
        print(f"[DEBUG] Stored action_results_list_front before update: {self.action_results_list_front}")
        print(f"[DEBUG] Stored action_results_list_center before update: {self.action_results_list_center}")

        # Ensure action results are valid lists before proceeding
        if self.action_results_list_front is None:
            self.action_results_list_front = []
        
        if self.action_results_list_center is None:
            self.action_results_list_center = []

        # Filter data based on the selected action
        filtered_data = []
        if selected_action:
            for frame_results in self.action_results_list_front:
                for track_id, action in frame_results.items():
                    if action == selected_action:
                        filtered_data.append(track_id)

            for frame_results in self.action_results_list_center:
                for track_id, action in frame_results.items():
                    if action == selected_action:
                        filtered_data.append(track_id)

            print(f"[DEBUG] Filtered Data for action '{selected_action}': {filtered_data}")

        # If no data matches the selected action, show debug message
        if not filtered_data:
            print(f"[DEBUG] No data matched the selected action.")

        # Generate heatmap based on the selected action
        frame = self.generate_heatmap_based_on_action(frame, selected_action)

        # Store the latest heatmap frame
        self.main_window.heatmap_frame = frame.copy()

        # Convert frame to QImage for displaying in QLabel
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage to the QLabel with aspect ratio maintained
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.main_window.heatmap_present_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.main_window.heatmap_present_label.setPixmap(scaled_pixmap)





    def generate_heatmap_based_on_action(self, frame, selected_action):
        """Generates a heatmap based on the selected action."""
        print("[DEBUG] Action Results (Front):", self.action_results_list_front)
        print("[DEBUG] Action Results (Center):", self.action_results_list_center)

        filtered_data = []

        # Combine detection and action results
        if self.human_detect_results_front and self.action_results_list_front:
            for detection, action in zip(self.human_detect_results_front, self.action_results_list_front):
                if action and action.get("action") and (action["action"].lower() == selected_action.lower() or selected_action == "All Actions"):
                    detection["action"] = action["action"]  # Ensure action is linked
                    filtered_data.append(detection)

        if self.human_detect_results_center and self.action_results_list_center:
            for detection, action in zip(self.human_detect_results_center, self.action_results_list_center):
                if action and action.get("action") and (action["action"].lower() == selected_action.lower() or selected_action == "All Actions"):
                    detection["action"] = action["action"]  # Ensure action is linked
                    filtered_data.append(detection)

        print(f"[DEBUG] Filtered Data for action '{selected_action}':", filtered_data)

        if not filtered_data:
            print("[DEBUG] No data matched the selected action.")
            return frame  # Return original frame if no matches

        return self.generate_heatmap(filtered_data)





    def generate_full_heatmap(self):
        """Generates the full heatmap and stores it in heatmap_frame."""
        heatmap = np.zeros((480, 640, 3), dtype=np.uint8)  # Replace with actual heatmap logic
        self.main_window.heatmap_frame = heatmap  # Store the generated heatmap
        return heatmap

    def generate_filtered_heatmap(self, action):
        """Returns the heatmap image filtered for the specific action."""
        # Placeholder for heatmap filtering logic based on action
        return np.zeros((480, 640, 3), dtype=np.uint8)  # Example black image
    
    def remove_extended_width(self, frame: np.ndarray, extension: int = 500) -> np.ndarray:
        """
        Removes the extended width from a frame by cropping the black padding.

        Parameters:
        frame (np.ndarray): The input frame with extended width.
        extension (int): The number of pixels to remove from each side. Default is 500.

        Returns:
        np.ndarray: The cropped frame without the extended width.
        """
        height, width, channels = frame.shape
        expected_width = 1920 + (2 * extension)
        
        # Ensure the input frame has the expected extended width
        if width != expected_width or height != 1080:
            raise ValueError("Expected a frame with extended width.")
        
        # Crop the frame to remove the black padding
        cropped_frame = frame[:, extension:extension + 1920]
        
        return cropped_frame

    def update_frame_for_center_video_label(self, frame, center_starting_y, front_starting_y):
        if frame is not None:
            # Remove extended width directly inside this function
            extension = 300
            if len(frame.shape) == 3:
                frame_height, frame_width, channels = frame.shape
            else:
                frame_height, frame_width = frame.shape

            if frame_width > extension * 2:
                frame = frame[:, extension:-extension]  # Remove extended width from both sides

            # Recalculate frame dimensions after cropping width
            if len(frame.shape) == 3:
                frame_height, frame_width, _ = frame.shape
            else:
                frame_height, frame_width = frame.shape

            h = front_starting_y - center_starting_y

            # Ensure the cropping region is within the frame's bounds
            center_starting_y = max(0, center_starting_y)  # Ensure y is not negative
            front_starting_y = min(frame_height, front_starting_y)  # Ensure front_starting_y does not exceed frame height

            if h <= 0:
                raise ValueError("Invalid cropping parameters: front_starting_y must be greater than center_starting_y.")


            # Convert the cropped frame to QImage
            # Crop the frame
            cropped_frame = np.ascontiguousarray(frame[center_starting_y:front_starting_y, 0:frame_width])

            # Convert the cropped frame to QImage
            height, width, channel = cropped_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(cropped_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()


            # Set the QImage to the QLabel with aspect ratio maintained
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.center_video_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.center_video_preview_label.setPixmap(scaled_pixmap)


    def update_frame_for_front_video_label(self, frame, starting_y, whole_classroom_height):
        if frame is not None:
            # Remove extended width directly inside this function
            extension = 300
            if len(frame.shape) == 3:
                frame_height, frame_width, channels = frame.shape
            else:
                frame_height, frame_width = frame.shape

            if frame_width > extension * 2:
                frame = frame[:, extension:-extension]  # Remove extended width from both sides

            # Recalculate frame dimensions after cropping width
            if len(frame.shape) == 3:
                frame_height, frame_width, _ = frame.shape
            else:
                frame_height, frame_width = frame.shape

            # Define the starting y coordinate and height for cropping
            y = starting_y  # Starting y-coordinate for cropping
            h = whole_classroom_height - y  # Base height on the difference to avoid manual subtraction

            # Ensure the cropping region is within the frame's bounds
            if y + h > frame_height:
                y = max(0, frame_height - h)  # Adjust y to keep the cropping region within bounds
                h = min(h, frame_height)      # Adjust h if necessary

            # Crop the frame
            # Crop the frame
            cropped_frame = np.ascontiguousarray(frame[y:y + h, 0:frame_width])

            # Convert the cropped frame to QImage
            height, width, channel = cropped_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(cropped_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()


            # Set the QImage to the QLabel with aspect ratio maintained and white spaces
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(self.main_window.front_video_preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.main_window.front_video_preview_label.setPixmap(scaled_pixmap)


    def update_heatmap(self, frame, selected_action=None):
        """Updates the heatmap based on the selected action and integrates action detection results."""
        
        if frame is None:
            print("[DEBUG] No frame provided for heatmap update.")
            return

        # Debugging: Check stored action results before filtering
        print(f"[DEBUG] Stored action_results_list_front before update: {self.action_results_list_front}")
        print(f"[DEBUG] Stored action_results_list_center before update: {self.action_results_list_center}")

        # Ensure action results are valid lists before proceeding
        if self.action_results_list_front is None:
            self.action_results_list_front = []
        
        if self.action_results_list_center is None:
            self.action_results_list_center = []

        # Filter data based on the selected action
        filtered_data = []
        if selected_action:
            for frame_results in self.action_results_list_front:
                for track_id, action in frame_results.items():
                    if action == selected_action:
                        filtered_data.append(track_id)

            for frame_results in self.action_results_list_center:
                for track_id, action in frame_results.items():
                    if action == selected_action:
                        filtered_data.append(track_id)

            print(f"[DEBUG] Filtered Data for action '{selected_action}': {filtered_data}")

        # If no data matches the selected action, show debug message
        if not filtered_data:
            print(f"[DEBUG] No data matched the selected action.")

        # Generate heatmap based on the selected action
        frame = self.generate_heatmap_based_on_action(frame, selected_action)

        # Store the latest heatmap frame
        self.main_window.heatmap_frame = frame.copy()

        # Convert frame to QImage for displaying in QLabel
        height, width = frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

        # Set the QImage to the QLabel with aspect ratio maintained
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.main_window.heatmap_present_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.main_window.heatmap_present_label.setPixmap(scaled_pixmap)


    def create_heatmap_from_data(self, filtered_data):
        """Generates a heatmap image from filtered detection data."""
        if not filtered_data:
            print("[DEBUG] No data available for heatmap generation.")
            return np.zeros((480, 640, 3), dtype=np.uint8)  # Blank heatmap

        heatmap = np.zeros((480, 640), dtype=np.float32)

        for person in filtered_data:
            x, y, _, _ = person.get("bbox", [0, 0, 0, 0])
            if 0 <= x < 640 and 0 <= y < 480:
                heatmap[y, x] += 1  # Increase intensity at detected locations

        heatmap = np.uint8(255 * (heatmap / heatmap.max())) if heatmap.max() > 0 else heatmap
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap_colored