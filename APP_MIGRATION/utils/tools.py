import os
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
import argparse

class Tools():
    def __init__(self):
        pass
        
    
    def getAbsPath(self, filePath):
        abs_path = os.path.abspath(filePath)
        print(abs_path)
        return abs_path

    def convert_slashes(self, directory_path):
        return directory_path.replace("\\", "/")

    def parse_arguments() -> argparse.Namespace: # For Camera
        parser=argparse.ArgumentParser(description="YOLOv8 Live")
        parser.add_argument(
            "--webcam-resolution",
            default=[1280,720], #default must be 1280, 720
            nargs=2,
            type=int
        )
        args = parser.parse_args()
        return args
    
    def count_folders(self, directory): #Get the number of folders inside a folder
        folder_count = 0
        try:
            for name in os.listdir(directory):
                folder_path = os.path.join(directory, name)
                if os.path.isdir(folder_path):
                    folder_count += 1
        except FileNotFoundError:
            print("The specified directory does not exist.")
        except Exception as e:
            print(f"An error occurred: {e}")
        
        if folder_count == 0:
            return 0
        else:
            return folder_count-1
    

