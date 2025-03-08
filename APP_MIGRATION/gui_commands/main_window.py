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