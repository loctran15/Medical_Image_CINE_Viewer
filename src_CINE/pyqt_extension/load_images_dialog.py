import os.path
import sys
from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtWidgets import QHBoxLayout, QComboBox, QLabel, \
    QScrollArea, QPushButton, QListWidget, QGridLayout, QVBoxLayout, \
    QSizePolicy, QListWidgetItem, QDialog, QLineEdit
from PyQt6.QtCore import Qt, QSize
import re

from typing import Tuple, Any, Optional

import sys

from src_CINE.io_image.image_enum import ImageFileType, ImageType


class LoadImagesDialog(QDialog):
    keyRelease = QtCore.pyqtSignal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Load Images")
        self.window_width, self.window_height = 1200, 800
        self.setMinimumSize(self.window_width, self.window_height)

        self.layout = QHBoxLayout()
        self.setLayout(self.layout)

        self.num_image_indexes: list[int] = []
        self.file_list = []

        self.initUI()

        self.updateButtonStatus()
        self.setButtonConnections()

    def keyReleaseEvent(self, event):
        super(LoadImagesDialog, self).keyPressEvent(event)
        self.keyRelease.emit(event.key())

    @property
    def num_images(self) -> int:
        return len(self.num_image_indexes)

    @num_images.setter
    def num_images(self, num_images):
        if (self.listWidgetRight is not None):
            self.listWidgetRight.phase_indexes = [i + 1 for i in range(num_images)]
        self.num_image_indexes = [i + 1 for i in range(num_images)]

    def initUI(self):
        self.setStyleSheet('''
        QWidget {
            font-size: 10px;
        }
        QPushButton {
            font-size: 10px;
            width: 100px;
            height: 25px;
        }
        ''')

        subLayouts = {}

        subLayouts['LeftColumn'] = QGridLayout()
        subLayouts['RightColumn'] = QVBoxLayout()
        self.layout.addLayout(subLayouts['LeftColumn'], 1)
        self.layout.addLayout(subLayouts['RightColumn'], 1)

        self.buttons = {}
        self.buttons['>>'] = QPushButton('&>>')
        self.buttons['>'] = QPushButton('>')
        self.buttons['<'] = QPushButton('<')
        self.buttons['<<'] = QPushButton('&<<')
        self.buttons['Load'] = QPushButton('Load')

        for k in self.buttons:
            self.buttons[k].setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding)

        """
        First Column
        """
        self.listWidgetLeft = QListWidget()
        subLayouts['LeftColumn'].addWidget(self.listWidgetLeft, 1, 0, 6, 4)

        subLayouts['LeftColumn'].setRowStretch(5, 1)
        subLayouts['LeftColumn'].addWidget(self.buttons['>>'], 1, 5, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        subLayouts['LeftColumn'].addWidget(self.buttons['<'], 2, 5, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        subLayouts['LeftColumn'].addWidget(self.buttons['>'], 3, 5, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        subLayouts['LeftColumn'].addWidget(self.buttons['<<'], 4, 5, 1, 1, alignment=Qt.AlignmentFlag.AlignTop)
        subLayouts['LeftColumn'].addWidget(self.buttons['Load'], 5, 5, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)

        """
        Second Column
        """
        # self.listWidgetRight = QListWidget()
        self.listWidgetRight = ListDragDropWidget()
        hLayout = QHBoxLayout()
        subLayouts['RightColumn'].addLayout(hLayout)

        hLayout.addWidget(self.listWidgetRight, 4)

    def setButtonConnections(self):
        self.listWidgetLeft.itemSelectionChanged.connect(self.updateButtonStatus)
        self.listWidgetRight.itemSelectionChanged.connect(self.updateButtonStatus)
        self.keyRelease.connect(self.on_key)

        self.buttons['>'].clicked.connect(self.buttonAddClicked)
        self.buttons['<'].clicked.connect(self.buttonRemoveClicked)
        self.buttons['>>'].clicked.connect(self.buttonAddAllClicked)
        self.buttons['<<'].clicked.connect(self.buttonRemoveAllClicked)
        self.buttons['Load'].clicked.connect(self.buttonLoadClicked)

    def buttonAddClicked(self):
        row = self.listWidgetLeft.currentRow()
        self.listWidgetRightAdd(row)

    def listWidgetRightAdd(self, list_widget_left_row: Optional[int] = None, item: Optional[str] = None):
        if (list_widget_left_row and item):
            print("list_widget_left_row and item cannot exist at the same time")
        if (list_widget_left_row is not None):
            rowStringItem = self.listWidgetLeft.takeItem(list_widget_left_row)
            item = rowStringItem.text()

        self.listWidgetRight.modAddItem(item)

    def buttonRemoveClicked(self):
        row = self.listWidgetRight.currentRow()
        self.listWidgetRightRemove(row)

    def listWidgetRightRemove(self, list_widget_right_row: int, is_complete_remove: bool = False):
        rowItem = self.listWidgetRight.item(list_widget_right_row)
        rowItemWidget: QCustomQWidget = self.listWidgetRight.itemWidget(rowItem)
        rowStringItem = rowItemWidget.filePath.text()
        self.listWidgetRight.takeItem(list_widget_right_row)
        if (not is_complete_remove):
            self.listWidgetLeft.addItem(rowStringItem)

    def buttonAddAllClicked(self):
        for i in range(self.listWidgetLeft.count()):
            self.listWidgetRightAdd(list_widget_left_row=0)

    def buttonRemoveAllClicked(self):
        for i in range(self.listWidgetRight.count()):
            self.listWidgetRightRemove(list_widget_right_row=0)

    def buttonLoadClicked(self):
        self.file_list = self.get_file_list()
        self.close()

    def updateButtonStatus(self):

        self.buttons['>'].setDisabled(not bool(self.listWidgetLeft.selectedItems()) or self.listWidgetLeft.count() == 0)
        self.buttons['<'].setDisabled(
            not bool(self.listWidgetRight.selectedItems()) or self.listWidgetRight.count() == 0)

    def get_file_list(self) -> list:
        file_list = []
        imageFileMap = {"DICOM": ImageFileType.DICOM, "NIFTY": ImageFileType.NIFTY, "TIFF": ImageFileType.TIFF}
        for i in range(self.listWidgetRight.count()):
            rowItem = self.listWidgetRight.item(i)
            rowWidgetItem: QCustomQWidget = self.listWidgetRight.itemWidget(rowItem)
            path = rowWidgetItem.filePath.text()
            imageFileType = imageFileMap[
                rowWidgetItem.imageFileTypeBox.itemText(rowWidgetItem.imageFileTypeBox.currentIndex())]
            # phase_index = rowWidgetItem.phaseIndexBox.itemText(rowWidgetItem.imageFileTypeBox.currentIndex())
            phase_index = rowWidgetItem.phaseIndexBox.currentText()
            if (phase_index == "-"):
                phase_index = i + 1
            else:
                phase_index = int(phase_index)
            item = (path, imageFileType, phase_index)
            file_list.append(item)
        return file_list

    def on_key(self, key):
        if key == Qt.Key.Key_Backspace:
            self.listWidgetRightRemove(list_widget_right_row=self.listWidgetRight.currentRow(), is_complete_remove=True)


class ListDragDropWidget(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.resize(600, 600)
        self.phase_indexes = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()

            links = []
            for url in event.mimeData().urls():
                if url.isLocalFile():
                    links.append(str(url.toLocalFile()))
                else:
                    links.append(str(url.toString()))
            for link in links:
                self.modAddItem(link)
        else:
            event.ignore()

    def modAddItem(self, path: str):
        widget = QCustomQWidget()
        widget.setFilePath(path)
        widget.initImageFileType()
        widget.initPhaseIndexes(self.phase_indexes)

        WidgetItem = QListWidgetItem(self)

        WidgetItem.setSizeHint(QSize(0, 60))

        self.addItem(WidgetItem)
        self.setItemWidget(WidgetItem, widget)


class QCustomQWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(QCustomQWidget, self).__init__(parent)

        self.setMaximumSize(550, 60)
        self.QHBoxLayout = QHBoxLayout()

        self.filePath = QLineEdit()
        self.filePath.setMaximumSize(450, 60)
        self.filePath.setStyleSheet("padding: 6px")

        self.scroll = QScrollArea()
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.filePath)
        self.scroll.setMaximumSize(450, 60)

        self.imageFileTypeBox = QComboBox()
        self.imageFileTypeBox.addItems(["DICOM", "NIFTY", "TIFF"])
        self.imageFileTypeBox.setMaximumSize(100, 60)
        self.phaseIndexBox = QComboBox()
        self.phaseIndexBox.setMaximumSize(50, 60)
        self.QHBoxLayout.addWidget(self.scroll)
        self.QHBoxLayout.addWidget(self.imageFileTypeBox)
        self.QHBoxLayout.addWidget(self.phaseIndexBox)

        self.setLayout(self.QHBoxLayout)
        self.scroll.horizontalScrollBar().setValue(100)

    def setFilePath(self, path: str):
        path = os.path.normpath(path)
        # self.path = path
        # short_path = os.path.join(*path.split(os.sep)[-3:])
        self.filePath.setText(path)

    def initImageFileType(self):
        path = self.filePath.text()
        file_name = os.path.basename(path)
        if bool(re.match(".*\.(nii|nii\.gz)$", file_name)):
            self.imageFileTypeBox.setCurrentIndex(1)
        elif bool(re.match(".*\.(tiff|tif)$", file_name)):
            self.imageFileTypeBox.setCurrentIndex(2)
        else:
            self.imageFileTypeBox.setCurrentIndex(0)

    def initPhaseIndexes(self, number_list: list[int]):
        self.phaseIndexBox.addItems(["-"] + [str(i) for i in number_list])


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    myApp = LoadImagesDialog()
    myApp.show()

    try:
        sys.exit(app.exec())
    except SystemExit:
        print('Closing Window...')
