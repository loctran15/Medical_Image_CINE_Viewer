# Form implementation generated from reading ui file 'MainWindow.ui'
#
# Created by: PyQt6 UI code generator 6.2.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtOpenGLWidgets import QOpenGLWidget


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1392, 1010)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.view_3 = QOpenGLWidget(self.centralwidget)
        self.view_3.setGeometry(QtCore.QRect(280, 450, 521, 411))
        self.view_3.setObjectName("view_3")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(842, 449, 531, 441))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.view_4 = QtWidgets.QGraphicsView(self.layoutWidget)
        self.view_4.setObjectName("view_4")
        self.verticalLayout_2.addWidget(self.view_4)
        self.view_4_scroll_bar = QtWidgets.QScrollBar(self.layoutWidget)
        self.view_4_scroll_bar.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.view_4_scroll_bar.setObjectName("view_4_scroll_bar")
        self.verticalLayout_2.addWidget(self.view_4_scroll_bar)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(10, 20, 261, 71))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.gridLayout = QtWidgets.QGridLayout(self.layoutWidget1)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.show_button = QtWidgets.QPushButton(self.layoutWidget1)
        self.show_button.setObjectName("show_button")
        self.gridLayout.addWidget(self.show_button, 0, 0, 1, 1)
        self.reset_button = QtWidgets.QPushButton(self.layoutWidget1)
        self.reset_button.setObjectName("reset_button")
        self.gridLayout.addWidget(self.reset_button, 0, 1, 1, 1)
        self.load_label_button = QtWidgets.QPushButton(self.layoutWidget1)
        self.load_label_button.setObjectName("load_label_button")
        self.gridLayout.addWidget(self.load_label_button, 1, 0, 1, 1)
        self.load_volume_button = QtWidgets.QPushButton(self.layoutWidget1)
        self.load_volume_button.setObjectName("load_volume_button")
        self.gridLayout.addWidget(self.load_volume_button, 1, 1, 1, 1)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(280, 20, 551, 391))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.view_1 = QtWidgets.QGraphicsView(self.layoutWidget2)
        self.view_1.setObjectName("view_1")
        self.horizontalLayout.addWidget(self.view_1)
        self.view_1_scroll_bar = QtWidgets.QScrollBar(self.layoutWidget2)
        self.view_1_scroll_bar.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.view_1_scroll_bar.setObjectName("view_1_scroll_bar")
        self.horizontalLayout.addWidget(self.view_1_scroll_bar)
        self.layoutWidget3 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget3.setGeometry(QtCore.QRect(840, 20, 531, 421))
        self.layoutWidget3.setObjectName("layoutWidget3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget3)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.view_2 = QtWidgets.QGraphicsView(self.layoutWidget3)
        self.view_2.setObjectName("view_2")
        self.verticalLayout.addWidget(self.view_2)
        self.view_2_scroll_bar = QtWidgets.QScrollBar(self.layoutWidget3)
        self.view_2_scroll_bar.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.view_2_scroll_bar.setObjectName("view_2_scroll_bar")
        self.verticalLayout.addWidget(self.view_2_scroll_bar)
        self.layoutWidget4 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget4.setGeometry(QtCore.QRect(280, 870, 541, 25))
        self.layoutWidget4.setObjectName("layoutWidget4")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.layoutWidget4)
        self.horizontalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.view_3_back_button = QtWidgets.QToolButton(self.layoutWidget4)
        self.view_3_back_button.setObjectName("view_3_back_button")
        self.horizontalLayout_2.addWidget(self.view_3_back_button)
        self.view_3_play_button = QtWidgets.QToolButton(self.layoutWidget4)
        self.view_3_play_button.setObjectName("view_3_play_button")
        self.horizontalLayout_2.addWidget(self.view_3_play_button)
        self.view_3_next_button = QtWidgets.QToolButton(self.layoutWidget4)
        self.view_3_next_button.setObjectName("view_3_next_button")
        self.horizontalLayout_2.addWidget(self.view_3_next_button)
        self.view_3_slider = QtWidgets.QSlider(self.layoutWidget4)
        self.view_3_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.view_3_slider.setObjectName("view_3_slider")
        self.horizontalLayout_2.addWidget(self.view_3_slider)
        self.view_3_LCD_number = QtWidgets.QLCDNumber(self.layoutWidget4)
        self.view_3_LCD_number.setObjectName("view_3_LCD_number")
        self.horizontalLayout_2.addWidget(self.view_3_LCD_number)
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 100, 261, 491))
        self.tabWidget.setMinimumSize(QtCore.QSize(261, 491))
        self.tabWidget.setMaximumSize(QtCore.QSize(261, 491))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.layoutWidget5 = QtWidgets.QWidget(self.tab)
        self.layoutWidget5.setGeometry(QtCore.QRect(0, 310, 262, 25))
        self.layoutWidget5.setObjectName("layoutWidget5")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget5)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.layoutWidget5)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.opacity_slider = QtWidgets.QSlider(self.layoutWidget5)
        self.opacity_slider.setMinimumSize(QtCore.QSize(121, 0))
        self.opacity_slider.setMaximumSize(QtCore.QSize(121, 22))
        self.opacity_slider.setOrientation(QtCore.Qt.Orientation.Horizontal)
        self.opacity_slider.setObjectName("opacity_slider")
        self.horizontalLayout_3.addWidget(self.opacity_slider)
        self.opacity_LCD_number = QtWidgets.QLCDNumber(self.layoutWidget5)
        self.opacity_LCD_number.setMinimumSize(QtCore.QSize(64, 0))
        self.opacity_LCD_number.setMaximumSize(QtCore.QSize(64, 16777215))
        self.opacity_LCD_number.setObjectName("opacity_LCD_number")
        self.horizontalLayout_3.addWidget(self.opacity_LCD_number)
        self.layoutWidget6 = QtWidgets.QWidget(self.tab)
        self.layoutWidget6.setGeometry(QtCore.QRect(0, 10, 251, 221))
        self.layoutWidget6.setObjectName("layoutWidget6")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.layoutWidget6)
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.label_AO_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_AO_checkbox.setObjectName("label_AO_checkbox")
        self.verticalLayout_4.addWidget(self.label_AO_checkbox)
        self.label_LV_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_LV_checkbox.setObjectName("label_LV_checkbox")
        self.verticalLayout_4.addWidget(self.label_LV_checkbox)
        self.label_LVM_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_LVM_checkbox.setObjectName("label_LVM_checkbox")
        self.verticalLayout_4.addWidget(self.label_LVM_checkbox)
        self.label_LA_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_LA_checkbox.setObjectName("label_LA_checkbox")
        self.verticalLayout_4.addWidget(self.label_LA_checkbox)
        self.label_RV_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_RV_checkbox.setObjectName("label_RV_checkbox")
        self.verticalLayout_4.addWidget(self.label_RV_checkbox)
        self.label_RA_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_RA_checkbox.setObjectName("label_RA_checkbox")
        self.verticalLayout_4.addWidget(self.label_RA_checkbox)
        self.horizontalLayout_5.addLayout(self.verticalLayout_4)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.label_LAA_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_LAA_checkbox.setObjectName("label_LAA_checkbox")
        self.verticalLayout_6.addWidget(self.label_LAA_checkbox)
        self.label_SVC_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_SVC_checkbox.setObjectName("label_SVC_checkbox")
        self.verticalLayout_6.addWidget(self.label_SVC_checkbox)
        self.label_IVC_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_IVC_checkbox.setObjectName("label_IVC_checkbox")
        self.verticalLayout_6.addWidget(self.label_IVC_checkbox)
        self.label_PA_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_PA_checkbox.setObjectName("label_PA_checkbox")
        self.verticalLayout_6.addWidget(self.label_PA_checkbox)
        self.label_PV_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_PV_checkbox.setObjectName("label_PV_checkbox")
        self.verticalLayout_6.addWidget(self.label_PV_checkbox)
        self.label_WH_checkbox = QtWidgets.QCheckBox(self.layoutWidget6)
        self.label_WH_checkbox.setObjectName("label_WH_checkbox")
        self.verticalLayout_6.addWidget(self.label_WH_checkbox)
        self.horizontalLayout_5.addLayout(self.verticalLayout_6)
        self.layoutWidget_3 = QtWidgets.QWidget(self.tab)
        self.layoutWidget_3.setGeometry(QtCore.QRect(0, 390, 263, 25))
        self.layoutWidget_3.setObjectName("layoutWidget_3")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout(self.layoutWidget_3)
        self.horizontalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.x_tracking_label = QtWidgets.QLabel(self.layoutWidget_3)
        self.x_tracking_label.setObjectName("x_tracking_label")
        self.horizontalLayout_7.addWidget(self.x_tracking_label)
        self.x_tracking_LCD_number = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.x_tracking_LCD_number.setObjectName("x_tracking_LCD_number")
        self.horizontalLayout_7.addWidget(self.x_tracking_LCD_number)
        self.y_tracking_label = QtWidgets.QLabel(self.layoutWidget_3)
        self.y_tracking_label.setObjectName("y_tracking_label")
        self.horizontalLayout_7.addWidget(self.y_tracking_label)
        self.y_tracking_LCD_number_2 = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.y_tracking_LCD_number_2.setObjectName("y_tracking_LCD_number_2")
        self.horizontalLayout_7.addWidget(self.y_tracking_LCD_number_2)
        self.z_tracking_label = QtWidgets.QLabel(self.layoutWidget_3)
        self.z_tracking_label.setObjectName("z_tracking_label")
        self.horizontalLayout_7.addWidget(self.z_tracking_label)
        self.z_tracking_LCD_number = QtWidgets.QLCDNumber(self.layoutWidget_3)
        self.z_tracking_LCD_number.setObjectName("z_tracking_LCD_number")
        self.horizontalLayout_7.addWidget(self.z_tracking_LCD_number)
        self.layoutWidget_4 = QtWidgets.QWidget(self.tab)
        self.layoutWidget_4.setGeometry(QtCore.QRect(0, 350, 251, 25))
        self.layoutWidget_4.setObjectName("layoutWidget_4")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.layoutWidget_4)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_tracking_label = QtWidgets.QLabel(self.layoutWidget_4)
        self.label_tracking_label.setObjectName("label_tracking_label")
        self.horizontalLayout_8.addWidget(self.label_tracking_label)
        self.label_tracking_LCD_number = QtWidgets.QLCDNumber(self.layoutWidget_4)
        self.label_tracking_LCD_number.setObjectName("label_tracking_LCD_number")
        self.horizontalLayout_8.addWidget(self.label_tracking_LCD_number)
        self.intensity_tracking_label = QtWidgets.QLabel(self.layoutWidget_4)
        self.intensity_tracking_label.setObjectName("intensity_tracking_label")
        self.horizontalLayout_8.addWidget(self.intensity_tracking_label)
        self.intensity_tracking_LCD_number = QtWidgets.QLCDNumber(self.layoutWidget_4)
        self.intensity_tracking_LCD_number.setObjectName("intensity_tracking_LCD_number")
        self.horizontalLayout_8.addWidget(self.intensity_tracking_LCD_number)
        self.layoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.layoutWidget_2.setGeometry(QtCore.QRect(0, 430, 121, 26))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.layoutWidget_2)
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.fps_label = QtWidgets.QLabel(self.layoutWidget_2)
        self.fps_label.setObjectName("fps_label")
        self.horizontalLayout_6.addWidget(self.fps_label)
        self.fps_spinbox = QtWidgets.QSpinBox(self.layoutWidget_2)
        self.fps_spinbox.setObjectName("fps_spinbox")
        self.horizontalLayout_6.addWidget(self.fps_spinbox)
        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.tab)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(0, 240, 251, 41))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.size_plot_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.size_plot_button.setObjectName("size_plot_button")
        self.horizontalLayout_10.addWidget(self.size_plot_button)
        self.record_button = QtWidgets.QPushButton(self.horizontalLayoutWidget_2)
        self.record_button.setObjectName("record_button")
        self.horizontalLayout_10.addWidget(self.record_button)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.layoutWidget7 = QtWidgets.QWidget(self.tab_2)
        self.layoutWidget7.setGeometry(QtCore.QRect(0, 10, 261, 100))
        self.layoutWidget7.setObjectName("layoutWidget7")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.layoutWidget7)
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.segmentation_CMACS_button = QtWidgets.QPushButton(self.layoutWidget7)
        self.segmentation_CMACS_button.setObjectName("segmentation_CMACS_button")
        self.verticalLayout_5.addWidget(self.segmentation_CMACS_button)
        self.segmentation_deep_heart_button = QtWidgets.QPushButton(self.layoutWidget7)
        self.segmentation_deep_heart_button.setObjectName("segmentation_deep_heart_button")
        self.verticalLayout_5.addWidget(self.segmentation_deep_heart_button)
        self.segmentation_registration_button = QtWidgets.QPushButton(self.layoutWidget7)
        self.segmentation_registration_button.setObjectName("segmentation_registration_button")
        self.verticalLayout_5.addWidget(self.segmentation_registration_button)
        self.segmentation_save_label_button = QtWidgets.QPushButton(self.tab_2)
        self.segmentation_save_label_button.setGeometry(QtCore.QRect(0, 430, 251, 32))
        self.segmentation_save_label_button.setMinimumSize(QtCore.QSize(251, 32))
        self.segmentation_save_label_button.setMaximumSize(QtCore.QSize(251, 32))
        self.segmentation_save_label_button.setObjectName("segmentation_save_label_button")
        self.layoutWidget8 = QtWidgets.QWidget(self.tab_2)
        self.layoutWidget8.setGeometry(QtCore.QRect(30, 130, 210, 20))
        self.layoutWidget8.setObjectName("layoutWidget8")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.layoutWidget8)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.segmentation_all_phases_radio_button = QtWidgets.QRadioButton(self.layoutWidget8)
        self.segmentation_all_phases_radio_button.setObjectName("segmentation_all_phases_radio_button")
        self.horizontalLayout_4.addWidget(self.segmentation_all_phases_radio_button)
        self.segmentation_current_phase_radio_button = QtWidgets.QRadioButton(self.layoutWidget8)
        self.segmentation_current_phase_radio_button.setObjectName("segmentation_current_phase_radio_button")
        self.horizontalLayout_4.addWidget(self.segmentation_current_phase_radio_button)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.tabWidget.addTab(self.tab_3, "")
        self.status_text_browser = QtWidgets.QTextBrowser(self.centralwidget)
        self.status_text_browser.setGeometry(QtCore.QRect(10, 670, 261, 221))
        self.status_text_browser.setObjectName("status_text_browser")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(10, 610, 253, 41))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.level_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.level_label.setObjectName("level_label")
        self.horizontalLayout_9.addWidget(self.level_label)
        self.level_spinbox = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.level_spinbox.setObjectName("level_spinbox")
        self.horizontalLayout_9.addWidget(self.level_spinbox)
        self.window_label = QtWidgets.QLabel(self.horizontalLayoutWidget)
        self.window_label.setObjectName("window_label")
        self.horizontalLayout_9.addWidget(self.window_label)
        self.window_spinbox = QtWidgets.QDoubleSpinBox(self.horizontalLayoutWidget)
        self.window_spinbox.setObjectName("window_spinbox")
        self.horizontalLayout_9.addWidget(self.window_spinbox)
        self.view_3.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget.raise_()
        self.layoutWidget.raise_()
        self.status_text_browser.raise_()
        self.tabWidget.raise_()
        self.horizontalLayoutWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1392, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.show_button.setText(_translate("MainWindow", "Show"))
        self.reset_button.setText(_translate("MainWindow", "Reset"))
        self.load_label_button.setText(_translate("MainWindow", "Load Label"))
        self.load_volume_button.setText(_translate("MainWindow", "Load Volume"))
        self.view_3_back_button.setText(_translate("MainWindow", "..."))
        self.view_3_play_button.setText(_translate("MainWindow", "..."))
        self.view_3_next_button.setText(_translate("MainWindow", "..."))
        self.label.setText(_translate("MainWindow", " Opacity   "))
        self.label_AO_checkbox.setText(_translate("MainWindow", "AO"))
        self.label_LV_checkbox.setText(_translate("MainWindow", "LV"))
        self.label_LVM_checkbox.setText(_translate("MainWindow", "LVM"))
        self.label_LA_checkbox.setText(_translate("MainWindow", "LA"))
        self.label_RV_checkbox.setText(_translate("MainWindow", "RV"))
        self.label_RA_checkbox.setText(_translate("MainWindow", "RA"))
        self.label_LAA_checkbox.setText(_translate("MainWindow", "LAA"))
        self.label_SVC_checkbox.setText(_translate("MainWindow", "SVC"))
        self.label_IVC_checkbox.setText(_translate("MainWindow", "IVC"))
        self.label_PA_checkbox.setText(_translate("MainWindow", "PA"))
        self.label_PV_checkbox.setText(_translate("MainWindow", "PV"))
        self.label_WH_checkbox.setText(_translate("MainWindow", "WH"))
        self.x_tracking_label.setText(_translate("MainWindow", " x"))
        self.y_tracking_label.setText(_translate("MainWindow", "y"))
        self.z_tracking_label.setText(_translate("MainWindow", "z"))
        self.label_tracking_label.setText(_translate("MainWindow", " Label"))
        self.intensity_tracking_label.setText(_translate("MainWindow", "Intensity"))
        self.fps_label.setText(_translate("MainWindow", " FPS"))
        self.size_plot_button.setText(_translate("MainWindow", "Size Plot"))
        self.record_button.setText(_translate("MainWindow", "record 3d view"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "label"))
        self.segmentation_CMACS_button.setText(_translate("MainWindow", "CMACS"))
        self.segmentation_deep_heart_button.setText(_translate("MainWindow", "Deep Heart"))
        self.segmentation_registration_button.setText(_translate("MainWindow", "Registration"))
        self.segmentation_save_label_button.setText(_translate("MainWindow", "Save Label"))
        self.segmentation_all_phases_radio_button.setText(_translate("MainWindow", "All phases"))
        self.segmentation_current_phase_radio_button.setText(_translate("MainWindow", "Current phase"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Segmentation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Page"))
        self.level_label.setText(_translate("MainWindow", " Level"))
        self.window_label.setText(_translate("MainWindow", " Window"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec())
