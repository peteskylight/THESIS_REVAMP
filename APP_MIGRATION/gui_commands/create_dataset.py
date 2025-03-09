import os
import psutil
import GPUtil
import time

from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem
from PySide6.QtCore import QTime

from pygrabber.dshow_graph import FilterGraph
from utils.camera import CameraFeed

class CreateDataset:
    def __init__(self, main_window):
        self.main_window = main_window
        self.camera_feed_instance = CameraFeed(self.main_window.camera_feed, self.main_window.white_frame_feed, self.main_window)
        self.fps_limit = 18  # Set FPS limit
        self.frame_interval = 1.0 / self.fps_limit  # Time per frame

    def start_camera(self):
        selected_index = self.main_window.cameraComboBox.currentIndex()
        self.camera_feed_instance.start_camera(selected_index)
        
        FRONT_CAMERA_RTSP_URL = "rtsp://admin:Bennett2432@192.168.1.64:554/stream"
        self.camera_feed_instance.start_camera(FRONT_CAMERA_RTSP_URL)
        
        self.enforce_fps_limit()

    def stop_camera(self):
        self.camera_feed_instance.stop_camera()

    def enforce_fps_limit(self):
        last_time = time.time()
        while self.camera_feed_instance.running:
            current_time = time.time()
            elapsed_time = current_time - last_time

            if elapsed_time < self.frame_interval:
                time.sleep(self.frame_interval - elapsed_time)

            last_time = time.time()

    def list_available_cameras(self):
        graph = FilterGraph()
        available_cameras = graph.get_input_devices()
        return available_cameras

    def populate_camera_combo_box(self):
        available_cameras = self.list_available_cameras()
        for camera in available_cameras:
            self.main_window.cameraComboBox.addItem(camera)

    def update_usage(self):
        cpu_usage = psutil.cpu_percent()
        ram_usage = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu_usage = gpus[0].load * 100 if gpus else 0
        
        self.main_window.cpu_label.setText(f"{cpu_usage} %")
        self.main_window.ram_label.setText(f"{ram_usage} %")
        self.main_window.gpu_label.setText(f"{gpu_usage:.2f} %")

    def update_time(self):
        current_time = QTime.currentTime()
        self.main_window.timeLCD.display(current_time.toString("hh:mm:ss"))
    
    def open_file_explorer(self):
        directory = QFileDialog.getExistingDirectory(self.main_window, "Select Directory")
        if directory:
            self.main_window.directoryLineEdit.setText(directory)
            self.scan_directory(directory)

    def scan_directory(self, directory):
        self.main_window.action_comboBox.clear()
        for folder_name in os.listdir(directory):
            folder_path = os.path.join(directory, folder_name)
            if os.path.isdir(folder_path):
                self.main_window.action_comboBox.addItem(folder_name)

    def add_folder(self):
        directory = self.main_window.directoryLineEdit.text()
        folder_name = self.main_window.action_comboBox.currentText()

        if not os.path.isdir(directory):
            QMessageBox.critical(self.main_window, "Error", "The specified directory does not exist.")
            return

        new_folder_path = os.path.join(directory, folder_name)
        if os.path.exists(new_folder_path):
            QMessageBox.critical(self.main_window, "Error", "The action folder already exists.")
            return

        try:
            os.makedirs(new_folder_path)
            QMessageBox.information(self.main_window, "Success", "Action folder created successfully.")
            self.scan_directory(directory)
        except Exception as e:
            QMessageBox.critical(self.main_window, "Error", f"An error occurred: {e}")

    def delete_folder(self):
        directory = self.main_window.directoryLineEdit.text()
        folder_name = self.main_window.action_comboBox.currentText()

        if not os.path.isdir(directory):
            QMessageBox.critical(self.main_window, "Error", "The specified directory does not exist.")
            return

        folder_path = os.path.join(directory, folder_name)
        if not os.path.exists(folder_path):
            QMessageBox.critical(self.main_window, "Error", "The folder does not exist.")
            return

        reply = QMessageBox.question(self.main_window, "Confirm Deletion", f"Are you sure you want to delete the action folder '{folder_name}'?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.Yes:
            try:
                os.rmdir(folder_path)
                QMessageBox.information(self.main_window, "Success", "Folder deleted successfully.")
                self.scan_directory(directory)
            except Exception as e:
                QMessageBox.critical(self.main_window, "Error", f"An error occurred: {e}")
    
    def showActionsToTable(self):
        folder_path = self.main_window.directoryLineEdit.text()
        folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
        self.main_window.action_table.setRowCount(len(folders))
        self.main_window.action_table.setColumnCount(2)

        for row, folder in enumerate(folders):
            folder_item = QTableWidgetItem(folder)
            subfolder_count = len([f for f in os.listdir(os.path.join(folder_path, folder)) if os.path.isdir(os.path.join(folder_path, folder, f))])
            subfolder_item = QTableWidgetItem(str(subfolder_count))
            self.main_window.action_table.setItem(row, 0, folder_item)
            self.main_window.action_table.setItem(row, 1, subfolder_item)