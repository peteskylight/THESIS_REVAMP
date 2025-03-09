from PySide6.QtWidgets import (QApplication,
                                QMainWindow,
                                QVBoxLayout,
                                QFileDialog,
                                QMessageBox,
                                QTableWidgetItem,
                                QWidget,
                                QButtonGroup,
                                QComboBox)


from gui_codes import Ui_MainWindow

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)

        self.Home_button.clicked.connect(self.showHomePanel)
        self.Import_button.clicked.connect(self.showImportPanel)
    
    def showHomePanel(self):
        self.stackedWindows.setCurrentIndex(1)

    def showImportPanel(self):
        self.stackedWindows.setCurrentIndex(4)

