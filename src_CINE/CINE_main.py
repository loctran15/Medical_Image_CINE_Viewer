import os.path
import time
from pathlib import Path
import sys

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from UI.MainWindow import Ui_MainWindow
from pyqt_extension.line import Edited_Canvas
from vispy import app, scene, color, visuals
from vispy.scene.widgets.viewbox import ViewBox
from typing import Tuple, Union, Dict, Optional, Any
from vispy.visuals.isosurface import IsosurfaceVisual
from vispy.scene.visuals import Isosurface
from vispy.visuals.filters import ShadingFilter

import numpy as np

from io_image.image import Image
from io_image.image_enum import ImageType, ImageFileType, ColorType
from io_image.reader import ImageReader
import imageio
from src_CINE.Dicom_eval import cine_size_plot

import cv2

from pyqt_extension.load_images_dialog import LoadImagesDialog
from src_CINE.log_status import logger

from functools import partial

from pyqt_extension.record_3d_view_dialog import record_3d_view_dialog


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # setup UI
        self.setupUi(self)

        # setup views
        # view 1
        self.canvas1 = Edited_Canvas(keys='interactive', size=(500, 400), show=True)
        self.view_1.setLayout(QVBoxLayout())
        self.view_1_scroll_bar.valueChanged.connect(self.view_1_scroll_bar_valueChanged_handler)
        self.view_1.layout().addWidget(self.canvas1.native)
        self.canvas1.vertical_line_moved_signal.connect(self.view_4_scroll_bar.setValue)
        self.canvas1.horizon_line_moved_signal.connect(self.view_2_scroll_bar.setValue)
        # view 2
        self.canvas2 = Edited_Canvas(keys='interactive', size=(500, 400), show=True)
        self.view_2.setLayout(QVBoxLayout())
        self.view_2_scroll_bar.valueChanged.connect(self.view_2_scroll_bar_valueChanged_handler)
        self.view_2.layout().addWidget(self.canvas2.native)
        self.canvas2.vertical_line_moved_signal.connect(self.view_4_scroll_bar.setValue)
        self.canvas2.horizon_line_moved_signal.connect(self.view_1_scroll_bar.setValue)
        # view 3
        self.canvas3 = scene.SceneCanvas(keys='interactive', size=(500, 400), show=True)
        self.view_3.setLayout(QVBoxLayout())
        self.view_3.layout().addWidget(self.canvas3.native)
        # view 4
        self.canvas4 = Edited_Canvas(keys='interactive', size=(500, 400), show=True)
        self.view_4.setLayout(QVBoxLayout())
        self.view_4_scroll_bar.valueChanged.connect(self.view_4_scroll_bar_valueChanged_handler)
        self.view_4.layout().addWidget(self.canvas4.native)
        self.canvas4.vertical_line_moved_signal.connect(self.view_2_scroll_bar.setValue)
        self.canvas4.horizon_line_moved_signal.connect(self.view_1_scroll_bar.setValue)

        # set up 3d_view
        # play_LCD_number
        self.view_3_LCD_number.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        # slider
        self.view_3_slider.setTracking(False)
        self.view_3_slider.valueChanged.connect(self.view_3_slider_valueChanged_handler)
        self.view_3_slider.sliderMoved.connect(self.view_3_slider_moved_handler)
        # next button/ back button
        self.view_3_next_button.clicked.connect(self.view_3_next_button_clicked_handler)
        self.view_3_back_button.clicked.connect(self.view_3_back_button_clicked_handler)
        # play button
        self.view_3_play_button.setChecked(True)
        self.view_3_play_button.clicked.connect(self.view_3_play_button_clicked_handler)
        self.view_3_play_button.setCheckable(True)
        # record 3d views (whole cardiac cycle)
        self.record_button.pressed.connect(self.record_3d_view)

        # opacity
        self.opacity_slider.setMaximum(10)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setSingleStep(1)
        self.opacity_LCD_number.setSegmentStyle(QLCDNumber.SegmentStyle.Flat)
        self.opacity_slider.valueChanged.connect(self.opacity_slider_valueChanged_handler)
        self.opacity_slider.sliderReleased.connect(self.opacity_slider_released_handler)
        self.opacity_slider.setValue(10)

        # size
        self.size_plot_button.pressed.connect(self.show_size_plot_handler)

        # show button
        self.show_button.pressed.connect(self.loading_3d_view)

        self.is_visual_dict = {'WH': False, 'AO': False, 'LV': False, 'LA': False, 'RV': False,
                               'RA': False, 'LAA': False, 'SVC': False, 'IVC': False, 'PA': False,
                               'PV': False, 'LVM': False}

        self.setup_checkboxes(self.is_visual_dict)

        # reset button
        self.reset_button.pressed.connect(self.reset)

        # checkboxes
        self.label_AO_checkbox.stateChanged.connect(self.AO_checkbox_stateChanged_handler)
        self.label_LV_checkbox.stateChanged.connect(self.LV_checkbox_stateChanged_handler)
        self.label_LVM_checkbox.stateChanged.connect(self.LVM_checkbox_stateChanged_handler)
        self.label_LA_checkbox.stateChanged.connect(self.LA_checkbox_stateChanged_handler)
        self.label_RV_checkbox.stateChanged.connect(self.RV_checkbox_stateChanged_handler)
        self.label_RA_checkbox.stateChanged.connect(self.RA_checkbox_stateChanged_handler)
        self.label_LAA_checkbox.stateChanged.connect(self.LAA_checkbox_stateChanged_handler)
        self.label_SVC_checkbox.stateChanged.connect(self.SVC_checkbox_stateChanged_handler)
        self.label_IVC_checkbox.stateChanged.connect(self.IVC_checkbox_stateChanged_handler)
        self.label_PA_checkbox.stateChanged.connect(self.PA_checkbox_stateChanged_handler)
        self.label_PV_checkbox.stateChanged.connect(self.PV_checkbox_stateChanged_handler)
        self.label_WH_checkbox.stateChanged.connect(self.WH_checkbox_stateChanged_handler)

        # log text browser
        self.status_text_browser.setReadOnly(True)
        logger_handler = logger.QTextEditLogger(self.status_text_browser)
        self.logger = logger.get_logger(qtlog_handler=logger_handler)

        # Qtimer
        self.timer = QTimer()
        self.timer.timeout.connect(self.timer_timeout_handler)

        # fps
        self.fps_spinbox.valueChanged.connect(self.fps_spinbox_valueChanged_handler)
        self.fps_spinbox.setValue(10)

        self.load_label_button.pressed.connect(partial(self.load_files, ImageType.LABEL))
        self.load_volume_button.pressed.connect(partial(self.load_files, ImageType.GRAYSCALE))

        # properties
        # level and window
        self.level_spinbox.setMinimum(-2048)
        self.level_spinbox.setMaximum(2048)
        self.level_spinbox.setValue(0)
        self.level_spinbox.setSingleStep(100)
        self.level_spinbox.valueChanged.connect(self.level_spinbox_valueChanged_handler)
        self.window_spinbox.setMinimum(0)
        self.window_spinbox.setMaximum(5000)
        self.window_spinbox.setSingleStep(200)
        self.window_spinbox.setValue(5000)
        self.window_spinbox.valueChanged.connect(self.window_spinbox_valueChanged_handler)

        self.current_phase_index: Optional[
            int] = -1  # start from 0. -1 means have not been initialized or not a sequence of phases
        self.label_list: Optional[list[Image]] = []
        self.volume_list: Optional[list[Image]] = []

        # view properties
        # view 1
        self.viewbox1 = None
        # children of viewbox 1
        self.gray_scale_viewbox1 = None
        self.label_viewbox1 = None
        # view 2
        self.viewbox2 = None
        # children of viewbox 2
        self.gray_scale_viewbox2 = None
        self.label_viewbox2 = None
        # view 4
        self.viewbox4 = None
        # children of viewbox 4
        self.gray_scale_viewbox4 = None
        self.label_viewbox4 = None

        # check whether the 3d view was setup in each image
        self.is_3d_view_loaded = False
        # 3d view
        # viewboxes of the 3d view
        self.viewboxes: Optional[list[ViewBox]] = None

        @self.canvas3.events.mouse_move.connect
        def on_mouse_move(event):
            if (self.is_3d_view_loaded):
                self.update_light()

        # camera
        self.camera1 = None
        self.camera2 = None
        self.camera3 = None
        self.camera4 = None

    def show_size_plot_handler(self):
        if (len(self.label_list) != 0):
            cine_size_plot.plot(self.label_list)
            cine_size_plot.show()

    def AO_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['AO'] = True
            self.logger.info("hello")
        else:
            self.is_visual_dict['AO'] = False

    def LV_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['LV'] = True
        else:
            self.is_visual_dict['LV'] = False

    def LVM_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['LVM'] = True
        else:
            self.is_visual_dict['LVM'] = False

    def LA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['LA'] = True
        else:
            self.is_visual_dict['LA'] = False

    def RV_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['RV'] = True
        else:
            self.is_visual_dict['RV'] = False

    def RA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['RA'] = True
        else:
            self.is_visual_dict['RA'] = False

    def LAA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['LAA'] = True
        else:
            self.is_visual_dict['LAA'] = False

    def SVC_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['SVC'] = True
        else:
            self.is_visual_dict['SVC'] = False

    def IVC_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['IVC'] = True
        else:
            self.is_visual_dict['IVC'] = False

    def PA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['PA'] = True
        else:
            self.is_visual_dict['PA'] = False

    def PV_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['PV'] = True
        else:
            self.is_visual_dict['PV'] = False

    def WH_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visual_dict['WH'] = True
        else:
            self.is_visual_dict['WH'] = False

    def setup_checkboxes(self, is_visual_dict: dict[str, bool]):
        """
        initialize check boxes based on is_visual_dict
        Args:
            is_visual_dict: an dictionary where keys are structures and values are boolean

        Returns: None

        """
        for name, value in is_visual_dict.items():
            eval(f"self.label_{name}_checkbox").setChecked(value)

    def opacity_slider_valueChanged_handler(self, value):
        """
        when a user drags the slide bar, the number displayed on opacity_LCD_number changes.
        Args:
            value: value from 0 to 10 is equivalent to opacity from 0 to 1

        Returns: None

        """
        alpha = value / 10
        self.opacity_LCD_number.display(alpha)

    def opacity_slider_released_handler(self):
        """
        when a user release the mouse, all label images RGBA data changes to the updated
        alpha. Then the 2d views of the current phase was shown. The 3d view remains the same
        Returns: None

        """

        # get the alpha value
        alpha = self.opacity_LCD_number.value()
        # iterate over images in self.label_list, if it is not none the update the RGBA data
        if (self.label_list):
            for image in self.label_list:
                if (image):
                    image.set_RGBA_data(alpha=alpha)
            # show the 2d views of the current phase
            self.show_2d_views(grayscale_image=self.volume_list[self.current_phase_index],
                               label_image=self.label_list[self.current_phase_index],
                               is_reload=True)

    # load data
    def load_files(self, imageType: ImageType):
        """
        load images to self.label_list or self.volume_list
        Args:
            imageType: ImageType enum indicates whether a user want to load grayscale(s) or mask(s)

        Returns: None

        """

        # choose which dialog based on ImageType
        dialog: LoadImagesDialog = LoadImagesDialog()
        dialog.setWindowTitle("Load Label(s) dialog") if imageType == ImageType.LABEL else dialog.setWindowTitle(
            "Load Volume(s) dialog")

        dialog.num_images = len(self.label_list)

        # display the chosen dialog
        dialog.exec()

        # user's inputs
        tuple_path_list = dialog.file_list
        if (len(tuple_path_list) == 0):
            return

        paths = []
        image_file_types = []
        phase_indexes = []
        for item in tuple_path_list:
            paths.append(item[0])
            image_file_types.append(item[1])
            phase_indexes.append(item[2])

        if (imageType == ImageType.LABEL):
            images = ImageReader.read_case(paths=paths, image_file_types=image_file_types,
                                           phase_indexes=phase_indexes, image_type=imageType)
            if (len(self.label_list) == 0 or len(images) > len(self.label_list)):
                self.label_list = images
            else:
                for image in images:
                    self.label_list[image.phase - 1] = image

        elif (imageType == ImageType.GRAYSCALE):
            images = ImageReader.read_case(paths=paths, image_file_types=image_file_types,
                                           phase_indexes=phase_indexes, image_type=imageType)
            if (len(self.volume_list) == 0 or len(images) > len(self.volume_list)):
                self.volume_list = images
            else:
                for image in images:
                    self.volume_list[image.phase - 1] = image

        # if this is the first time we load labels/volumes, assign current_phase_index with index 0
        if (self.current_phase_index == -1):
            self.current_phase_index = 0

        # assign values to self.volume_list or self.label_list such that
        # their lengths are equal
        if (len(self.volume_list) == 0 and len(self.label_list) != 0):
            self.volume_list = [None] * len(self.label_list)
        elif (len(self.volume_list) != 0 and len(self.label_list) == 0):
            self.label_list = [None] * len(self.volume_list)
        elif (len(self.volume_list) != len(self.label_list)):
            raise Exception("the number of volume and label do not match")

        # initialize RGBA data of all labels in self.label_list
        for image in self.label_list:
            if (image):
                image.set_RGBA_data(alpha=self.opacity_LCD_number.value())

        # initialize the slider of view 3 based on the number of images in self.volume_list/label_list
        self.view_3_slider.setMinimum(0)
        self.view_3_slider.setMaximum(max(len(self.volume_list), len(self.label_list)) - 1)
        self.view_3_slider.setTickInterval(1)

        self.show_2d_views(frame_indexes=(30, 30, 30), grayscale_image=self.volume_list[self.current_phase_index],
                           label_image=self.label_list[self.current_phase_index],
                           is_reload=True)

        # initialize value displayed on view_3_LCD
        if (self.view_3_LCD_number.value() == 0):
            self.view_3_LCD_number.display(1)

        # initialize viewboxes (each of which contains a 3d mask image)
        if (imageType == ImageType.LABEL):
            self.viewboxes = [scene.widgets.ViewBox(parent=self.canvas3.scene) for _ in
                              range(len(self.label_list))]

    def show_2d_views(self, frame_indexes: Optional[Tuple[int]] = None, grayscale_image: Optional[Image] = None,
                      label_image: Optional[Image] = None,
                      is_reload: bool = True):
        """
        show view 1,2 and 4.
        """
        if (frame_indexes == None):
            frame_index_view_1 = self.view_1_scroll_bar.value()
            frame_index_view_2 = self.view_2_scroll_bar.value()
            frame_index_view_4 = self.view_4_scroll_bar.value()
        else:
            frame_index_view_1, frame_index_view_2, frame_index_view_4 = frame_indexes

        self.show_view_1(frame_index_view_1, grayscale_image, label_image, is_reload)
        self.show_view_2(frame_index_view_2, grayscale_image, label_image, is_reload)
        self.show_view_4(frame_index_view_4, grayscale_image, label_image, is_reload)

        # set the values of scroll bars to match with the frame_indexes.
        # because the change in scroll bars will call show_2d_views function so
        # to avoid infinite loop, the below statements are executed only when is_reload == True
        if (is_reload):
            self.view_1_scroll_bar.setValue(frame_index_view_1)
            self.view_2_scroll_bar.setValue(frame_index_view_2)
            self.view_4_scroll_bar.setValue(frame_index_view_4)

    def show_view_1(self, frame_index: int, grayscale_image: Optional[Image] = None,
                    label_image: Optional[Image] = None,
                    is_reload: bool = False):
        """

        Args:
            label_image:
            grayscale_image:
            frame_index: starts from 0
            is_reload: if is_reload == True, the new viewbox, camera and scroll_bar of view 1,2, and 4 are
            re-initialized.

        Returns: None

        """
        if (grayscale_image is None and label_image is None):
            return

        if (is_reload):
            self.viewbox1 = scene.widgets.ViewBox()
            self.canvas1.set_view(self.viewbox1)
            if (label_image and grayscale_image and label_image.shape != grayscale_image.shape):
                print("label and volume shape mismatch!")
                return
            if (grayscale_image):
                self.gray_scale_viewbox1 = scene.visuals.Image(grayscale_image.array_data[frame_index, :, :],
                                                               cmap="grays",
                                                               parent=self.viewbox1.scene)
                self.gray_scale_viewbox1.visible = True

                self.canvas1.set_limit(0, grayscale_image.shape[1] - 1, 0, grayscale_image.shape[2] - 1)
                self.view_1_scroll_bar.setMinimum(0)
                self.view_1_scroll_bar.setMaximum(grayscale_image.shape[0] - 1)
            if (label_image):
                self.label_viewbox1 = scene.visuals.Image(label_image.RGBA_data[frame_index, :, :],
                                                          parent=self.viewbox1.scene)
                self.label_viewbox1.transform = visuals.transforms.STTransform(translate=(0, 0, -0.5))
                self.label_viewbox1.visible = True
                self.canvas1.set_limit(0, label_image.shape[1] - 1, 0, label_image.shape[2] - 1)
                self.view_1_scroll_bar.setMinimum(0)
                self.view_1_scroll_bar.setMaximum(label_image.shape[0] - 1)

            camera = scene.PanZoomCamera(aspect=1)
            camera.flip = (0, 1, 0)
            camera.zoom(200, center=(0.05, 0.05))
            self.canvas1.set_camera(camera)
        else:
            if (label_image):
                self.label_viewbox1.set_data(label_image.RGBA_data[frame_index, :, :])
                self.label_viewbox1.update()
                if (not self.label_viewbox1.visible):
                    self.label_viewbox1.visible = True
            else:
                # if the label_image is None but the self.label_viewbox1 are initialized, to avoid display the
                # last label image, set the visibility of self.label_viewbox1 to False.
                if (self.label_viewbox1):
                    self.label_viewbox1.visible = False

            if (grayscale_image):
                max_value = np.max(grayscale_image.array_data)
                min_value = np.min(grayscale_image.array_data)
                self.gray_scale_viewbox1.set_data(grayscale_image.array_data[frame_index, :, :])
                self.gray_scale_viewbox1.clim = (min_value, max_value)
                self.gray_scale_viewbox1.update()
                if (not self.gray_scale_viewbox1.visible):
                    self.gray_scale_viewbox1.visible = True
            else:
                if (self.gray_scale_viewbox1):
                    self.gray_scale_viewbox1.visible = False

        self.canvas1.scene.update()

    def show_view_2(self, frame_index: int, grayscale_image: Optional[Image] = None,
                    label_image: Optional[Image] = None,
                    is_reload: bool = False):
        """

        Args:
            label_image:
            grayscale_image:
            frame_index: starts from 0
            is_reload: if is_reload == True, the new viewbox, camera and scroll_bar of view 1,2, and 4 are
            re-initialized.

        Returns: None

        """
        if (grayscale_image is None and label_image is None):
            return

        if (is_reload):
            self.viewbox2 = scene.widgets.ViewBox()
            self.canvas2.set_view(self.viewbox2)
            if (label_image and grayscale_image and label_image.shape != grayscale_image.shape):
                print("label and volume shape mismatch!")
                return
            if (grayscale_image):
                self.gray_scale_viewbox2 = scene.visuals.Image(grayscale_image.array_data[:, frame_index, :],
                                                               cmap="grays",
                                                               parent=self.viewbox2.scene)
                self.gray_scale_viewbox2.visible = True

                self.canvas2.set_limit(0, grayscale_image.shape[2] - 1, 0, grayscale_image.shape[0] - 1)
                self.view_2_scroll_bar.setMinimum(0)
                self.view_2_scroll_bar.setMaximum(grayscale_image.shape[1] - 1)
            if (label_image):
                self.label_viewbox2 = scene.visuals.Image(label_image.RGBA_data[:, frame_index, :],
                                                          parent=self.viewbox2.scene)
                self.label_viewbox2.transform = visuals.transforms.STTransform(translate=(0, 0, -0.5))
                self.label_viewbox2.visible = True
                self.canvas2.set_limit(0, label_image.shape[2] - 1, 0, label_image.shape[0] - 1)
                self.view_2_scroll_bar.setMinimum(0)
                self.view_2_scroll_bar.setMaximum(label_image.shape[1] - 1)

            camera = scene.PanZoomCamera(aspect=1)
            camera.flip = (0, 1, 0)
            camera.zoom(200, center=(0.05, 0.2))
            self.canvas2.set_camera(camera)
        else:
            if (label_image):
                self.label_viewbox2.set_data(label_image.RGBA_data[:, frame_index, :])
                self.label_viewbox2.update()
                if (not self.label_viewbox2.visible):
                    self.label_viewbox2.visible = True
            else:
                if (self.label_viewbox2):
                    self.label_viewbox2.visible = False

            if (grayscale_image):
                max_value = np.max(grayscale_image.array_data)
                min_value = np.min(grayscale_image.array_data)
                self.gray_scale_viewbox2.set_data(grayscale_image.array_data[:, frame_index, :])
                self.gray_scale_viewbox2.clim = (min_value, max_value)
                self.gray_scale_viewbox2.update()
                if (not self.gray_scale_viewbox2.visible):
                    self.gray_scale_viewbox2.visible = True
            else:
                if (self.gray_scale_viewbox2):
                    self.gray_scale_viewbox2.visible = False

        self.canvas2.scene.update()

    def show_view_4(self, frame_index: int, grayscale_image: Optional[Image] = None,
                    label_image: Optional[Image] = None,
                    is_reload: bool = False):
        """

        Args:
            label_image:
            grayscale_image:
            frame_index: starts from 0
            is_reload: if is_reload == True, the new viewbox, camera and scroll_bar of view 1,2, and 4 are
            re-initialized.

        Returns: None

        """
        if (grayscale_image is None and label_image is None):
            return

        if (is_reload):
            self.viewbox4 = scene.widgets.ViewBox()
            self.canvas4.set_view(self.viewbox4)
            if (label_image and grayscale_image and label_image.shape != grayscale_image.shape):
                print("label and volume shape mismatch!")
                return
            if (grayscale_image):
                self.gray_scale_viewbox4 = scene.visuals.Image(grayscale_image.array_data[:, :, frame_index],
                                                               cmap="grays",
                                                               parent=self.viewbox4.scene)
                self.gray_scale_viewbox4.visible = True

                self.canvas4.set_limit(0, grayscale_image.shape[1] - 1, 0, grayscale_image.shape[0] - 1)
                self.view_4_scroll_bar.setMinimum(0)
                self.view_4_scroll_bar.setMaximum(grayscale_image.shape[2] - 1)
            if (label_image):
                self.label_viewbox4 = scene.visuals.Image(label_image.RGBA_data[:, :, frame_index],
                                                          parent=self.viewbox4.scene)
                self.label_viewbox4.transform = visuals.transforms.STTransform(translate=(0, 0, -0.5))
                self.label_viewbox4.visible = True
                self.canvas4.set_limit(0, label_image.shape[1] - 1, 0, label_image.shape[0] - 1)
                self.view_4_scroll_bar.setMinimum(0)
                self.view_4_scroll_bar.setMaximum(label_image.shape[2] - 1)

            camera = scene.PanZoomCamera(aspect=1)
            camera.flip = (0, 1, 0)
            camera.zoom(200, center=(0.05, 0.2))
            self.canvas4.set_camera(camera)
        else:
            if (label_image):
                self.label_viewbox4.set_data(label_image.RGBA_data[:, :, frame_index])
                self.label_viewbox4.update()
                if (not self.label_viewbox4.visible):
                    self.label_viewbox4.visible = True
            else:
                if (self.label_viewbox4):
                    self.label_viewbox4.visible = False

            if (grayscale_image):
                max_value = np.max(grayscale_image.array_data)
                min_value = np.min(grayscale_image.array_data)
                self.gray_scale_viewbox4.set_data(grayscale_image.array_data[:, :, frame_index])
                self.gray_scale_viewbox4.clim = (min_value, max_value)
                self.gray_scale_viewbox4.update()
                if (not self.gray_scale_viewbox4.visible):
                    self.gray_scale_viewbox4.visible = True
            else:
                if (self.gray_scale_viewbox4):
                    self.gray_scale_viewbox4.visible = False

        self.canvas4.scene.update()

    def show_view_3(self, image: Image = None, is_reload: bool = True):
        """

        Args:
            image: must be a label
            is_reload: if True, the camera will be reset.

        Returns:

        """
        if (image is None):
            try:
                last_widget = self.canvas3.central_widget._widgets[-1]
                self.canvas3.central_widget.remove_widget(last_widget)
            except:
                pass
            return

        if (image.imageType != ImageType.LABEL):
            print("only show masks in view 3!")
            return
        # set up the camera
        if (is_reload):
            fov = 60.
            self.camera3 = scene.cameras.ArcballCamera(fov=fov, name='Arcball',
                                                       center=(
                                                           image.shape[0] / 2, image.shape[1] / 2, image.shape[2] / 2),
                                                       distance=400)
        self.viewboxes[image.phase - 1].camera = self.camera3
        try:
            last_widget = self.canvas3.central_widget._widgets[-1]
            self.canvas3.central_widget.remove_widget(last_widget)
        except:
            pass

        self.canvas3.central_widget.add_widget(self.viewboxes[image.phase - 1])
        # self.canvas3.scene.update()
        self.canvas3.show()
        self.update_light()

    # TODO: fix the bug when drag horizon/vertical line
    def view_1_scroll_bar_valueChanged_handler(self, value):
        # if there is no data loaded into the software, the movement of slider does not have
        # any effect
        if (len(self.label_list) == 0 and len(self.volume_list) == 0):
            return

        label_image = self.label_list[self.current_phase_index]
        volume_image = self.volume_list[self.current_phase_index]
        self.show_view_1(value, volume_image, label_image, is_reload=False)
        self.canvas2.set_horizon_pos(value)
        self.canvas4.set_horizon_pos(value)

    def view_2_scroll_bar_valueChanged_handler(self, value):
        # if there is no data loaded into the software, the movement of slider does not have
        # any effect
        if (len(self.label_list) == 0 and len(self.volume_list) == 0):
            return

        label_image = self.label_list[self.current_phase_index]
        volume_image = self.volume_list[self.current_phase_index]
        self.show_view_2(value, volume_image, label_image, is_reload=False)
        self.canvas1.set_horizon_pos(value)
        self.canvas4.set_horizon_pos(value)

    def view_4_scroll_bar_valueChanged_handler(self, value):
        # if there is no data loaded into the software, the movement of slider does not have
        # any effect
        if (len(self.label_list) == 0 and len(self.volume_list) == 0):
            return

        label_image = self.label_list[self.current_phase_index]
        volume_image = self.volume_list[self.current_phase_index]
        self.show_view_4(value, volume_image, label_image, is_reload=False)
        self.canvas1.set_horizon_pos(value)
        self.canvas2.set_horizon_pos(value)

    def view_3_slider_valueChanged_handler(self, value):
        # if there is no data loaded into the software, the movement of slider does not have
        # any effect
        if (len(self.label_list) == 0 and len(self.volume_list) == 0):
            return
        # value starts from 0
        self.current_phase_index = value
        is_reload = (not self.gray_scale_viewbox1 and self.volume_list[self.current_phase_index]) or \
                    (not self.label_viewbox1 and self.label_list[self.current_phase_index])
        self.show_2d_views(grayscale_image=self.volume_list[self.current_phase_index],
                           label_image=self.label_list[self.current_phase_index],
                           is_reload=is_reload)
        # show the 3d view only if all labels are loaded (image3DScene is setup)
        if (self.is_3d_view_loaded):
            is_reload = bool(not self.camera3)
            self.show_view_3(image=self.label_list[self.current_phase_index], is_reload=is_reload)
        self.view_3_LCD_number.display(value + 1)

    def view_3_slider_moved_handler(self, value):
        self.view_3_LCD_number.display(value + 1)

    def view_3_next_button_clicked_handler(self):
        self.current_phase_index += 1
        self.current_phase_index = self.current_phase_index % (self.view_3_slider.maximum() + 1)
        self.view_3_slider.setValue(self.current_phase_index)

    def view_3_back_button_clicked_handler(self):
        self.current_phase_index -= 1
        self.current_phase_index = self.current_phase_index % (self.view_3_slider.maximum() + 1)
        self.view_3_slider.setValue(self.current_phase_index)

    def view_3_play_button_clicked_handler(self, value):
        if (self.view_3_play_button.isChecked()):
            self.timer.start()
        else:
            self.timer.stop()

    def update_light(self, view: Optional[ViewBox] = None):
        """
        update the light on the current view only
        """
        if not view:
            view = self.viewboxes[self.current_phase_index]
        if (self.camera3 is not None):
            transform = self.camera3.transform
            # for viewbox in self.viewboxes:
            for child in view.scene.children:
                if ((isinstance(child, Isosurface) or isinstance(child, IsosurfaceVisual)) and child.visible):
                    dir = np.concatenate(((0, 0, -1), [0]))
                    light_dir = transform.map(dir)[:3]
                    if (not np.array_equal(light_dir, child.shading_filter.light_dir)):
                        child.shading_filter.light_dir = light_dir

    def loading_3d_view(self):
        """
        set up the 3d view(image3DScene) of all labels in self.label_list
        Returns: None

        """
        if (len(self.label_list) == 0):
            print("no label found! cannot loading 3d view")
            return

        for image in self.label_list:
            # if the image is None, continue the loop. otherwise, update or create a 3dscene of the image
            if (image is None):
                continue
            if (not image.image3DScene):
                image.generate_3d_scene(parent=self.viewboxes[image.phase - 1].scene,
                                        is_visual_dict=self.is_visual_dict,
                                        alpha=self.opacity_LCD_number.value())
                is_reload = True
            else:
                image.image3DScene.update_scene(is_visual_dict=self.is_visual_dict,
                                                alpha=self.opacity_LCD_number.value())
                is_reload = False

        self.is_3d_view_loaded = True
        self.show_view_3(image=self.label_list[self.current_phase_index], is_reload=is_reload)

    def record_3d_view(self):
        """
        record video of the 3d scene
        Returns: None

        """

        # select the out directory
        dialog = QFileDialog()
        selected_dir = dialog.getExistingDirectory(self, 'Select a directory')

        # user's input
        setting_dialog = record_3d_view_dialog()
        setting_dialog.exec()
        images_folder_name = setting_dialog.images_folder_name.text()
        video_name = setting_dialog.video_name.text()
        fps = int(setting_dialog.fps.text())

        file_dir = os.path.join(selected_dir, images_folder_name)

        # create folder if not exist
        Path(file_dir).mkdir(parents=True, exist_ok=True)

        # write images
        if (self.is_3d_view_loaded):
            for index, viewbox in enumerate(self.viewboxes):
                if (viewbox):
                    self.view_3_slider.setValue(index)
                    pixmap = self.view_3.grab()
                    file_path = os.path.join(file_dir, f"phase_{str(index + 1).zfill(2)}.png")
                    pixmap.save(file_path, quality=100)
        time.sleep(2)

        # read the images and convert them into avi file.
        img_array = []
        for index in range(len(self.viewboxes)):
            path = os.path.join(file_dir, f"phase_{str(index + 1).zfill(2)}.png")
            img = imageio.imread(path)
            img_array.append(img)

        out_path = os.path.join(os.path.dirname(file_dir), video_name)
        out = imageio.get_writer(out_path, fps=fps, codec='mjpeg', quality=10)
        for i in range(len(img_array)):
            out.append_data(img_array[i])

        out.close()

        # open the avi file. Work only on windows
        # os.startfile(out_path)
        # timer slots

    def timer_timeout_handler(self):
        value = (self.current_phase_index + 1) % (self.view_3_slider.maximum() + 1)
        self.view_3_slider.setValue(value)

    def reset(self) -> None:
        """
        reset the software, remove all of the images (grayscale images or masks) from the software.
        """
        # properties
        self.current_phase_index: Optional[
            int] = -1  # start from 0. -1 means have not been initialized or not a sequence of phases
        # reset the label_list and volume_list to empty lists
        self.label_list: Optional[list[Image]] = []
        self.volume_list: Optional[list[Image]] = []

        # remove 3d images from the 3d scene
        try:
            last_widget = self.canvas3.central_widget._widgets[-1]
            self.canvas3.central_widget.remove_widget(last_widget)
        except:
            pass

        self.viewboxes: Optional[list[ViewBox]] = None
        # remove images from 2d scenes
        self.canvas1.reset()
        self.viewbox1 = None
        self.gray_scale_viewbox1 = None
        self.label_viewbox1 = None
        self.canvas2.reset()
        self.viewbox2 = None
        self.gray_scale_viewbox2 = None
        self.label_viewbox2 = None
        self.canvas4.reset()
        self.viewbox4 = None
        self.gray_scale_viewbox4 = None
        self.label_viewbox4 = None

        self.is_3d_view_loaded = False

        # reset camera to None
        self.camera1 = None
        self.camera2 = None
        self.camera3 = None
        self.camera4 = None

        self.view_3_slider.setValue(0)
        self.view_3_LCD_number.display(1)
        self.view_3_slider.IsEnabled = False

    def fps_spinbox_valueChanged_handler(self, value):
        time = int(1000 / value)
        self.timer.setInterval(time)

    def level_spinbox_valueChanged_handler(self, level):
        for image in self.volume_list:
            if (image is not None):
                image.set_window_level(self.window_spinbox.value(), level)
        if (len(self.volume_list) != 0):
            self.show_2d_views(grayscale_image=self.volume_list[self.current_phase_index],
                               label_image=self.label_list[self.current_phase_index], is_reload=True)

    def window_spinbox_valueChanged_handler(self, window):
        for image in self.volume_list:
            if (image is not None):
                image.set_window_level(window, self.level_spinbox.value())
        if (len(self.volume_list) != 0):
            self.show_2d_views(grayscale_image=self.volume_list[self.current_phase_index],
                               label_image=self.label_list[self.current_phase_index], is_reload=True)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("CTA App")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
