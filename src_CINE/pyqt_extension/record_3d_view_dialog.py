from PyQt6.QtWidgets import QApplication, QLineEdit, QWidget, QFormLayout, QPushButton, QDialog
from PyQt6.QtGui import QIntValidator, QDoubleValidator, QFont
from PyQt6.QtCore import Qt
import sys


class record_3d_view_dialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.images_folder_name = QLineEdit()
        self.images_folder_name.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.images_folder_name.setFont(QFont("Arial", 14))
        self.images_folder_name.setText("3d_view_images")

        self.video_name = QLineEdit()
        self.video_name.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.video_name.setFont(QFont("Arial", 14))
        self.video_name.setText("3d_view.avi")

        self.fps = QLineEdit()
        self.fps.setValidator(QIntValidator())
        self.fps.setFont(QFont("Arial", 14))
        self.fps.setText("10")

        self.all_phase_button = QPushButton("enter")
        self.layout = QFormLayout()
        self.layout.addRow("images folder name", self.images_folder_name)
        self.layout.addRow("video name", self.video_name)
        self.layout.addRow("fps", self.fps)
        self.layout.addRow(self.all_phase_button)

        self.all_phase_button.released.connect(self.enter_handler)
        self.setLayout(self.layout)

    def enter_handler(self):
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = record_3d_view_dialog()
    win.show()
    sys.exit(app.exec())
