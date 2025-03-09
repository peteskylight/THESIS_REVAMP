import sys
import cv2
import os
import numpy as np
import psutil
import GPUtil

from PySide6.QtWidgets import (QApplication,
                                QMainWindow,
                                QVBoxLayout,
                                QFileDialog,
                                QMessageBox,
                                QTableWidgetItem,
                                QWidget,
                                QButtonGroup,
                                QComboBox)

from PySide6.QtCore import QRect, QCoreApplication, QMetaObject, QTimer, QTime, Qt, QDate
from PySide6.QtGui import QImage, QPixmap

from superqt import QRangeSlider

from ultralytics import YOLO

from utils.drawing_utils import DrawingUtils

from utils import (Tools,
                   VideoUtils,
                   VideoPlayerThread,
                   SeekingVideoPlayerThread)

from gui_codes import Ui_MainWindow
from gui_commands import (CenterVideo,
                          FrontVideo,
                          CreateDataset,
                          AnalyticsTab)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        '''
        GLOBAL VARIABLES INSTANCES
        '''
        self.FrontVideo = FrontVideo(main_window=self)
        self.CreateDataset = CreateDataset(main_window=self)
        self.AnalyticsTab = AnalyticsTab(main_window=self)
        self.CenterVideo = CenterVideo(main_window=self)

        self.drawing_utils = DrawingUtils()
        self.tools_utils = Tools()
        self.video_utils = VideoUtils()

        self.setupUi(self)

        #Set Default Panel:
        self.stackedPanels.setCurrentIndex(1)

        '''
        Connecting buttons to redirect to the stacked panels
        PANEL INDEX:
        HOME - 1
        CREATE DATASET - 2
        ANALYTICS - 3
        IMPORT - 4
        '''

        self.Home_button.clicked.connect(self.showHomePanel)
        self.Import_button.clicked.connect(self.showImportPanel)


        '''
        IMPORTING TAB CODE SECTION
        '''

        #FRONT VIDEO
        self.import_video_button_front.clicked.connect(self.FrontVideo.browse_video)

        



    
    def showHomePanel(self): #FOR SHOWING HOME PANEL
        self.stackedPanels.setCurrentIndex(1)

    def showImportPanel(self):
        self.stackedPanels.setCurrentIndex(4)
    
    def showAnalyticsPanel(self):
        self.stackedPanels.setCurrentIndex(3)

