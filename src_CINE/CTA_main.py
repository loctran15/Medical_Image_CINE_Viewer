# 1/20/2020
import datetime
from PyQt5.QtCore import *
import vispy.visuals
from PyQt5.QtWidgets import *
from vispy import app, scene, color, visuals
from vispy.visuals.transforms import STTransform
import os
import numpy as np
import SimpleITK as sitk
from Dicom_helper.helper_funcs import label_dict
import time
import re
from pyqt_extension import Plot
from pyqt_extension.line import EditLineVisual, Edited_Canvas
from pyqt_extension.load_label_dialog import LoadLabelDialog
from functools import partial
from subprocess import run, PIPE
from UI.MainWindow import Ui_MainWindow
from scipy import ndimage as ndi
from file_loader.file_loader import LoadDataWorker, read_label, filter, read_dicom_case, read_nifti
from registration.regis import RegistrationAllWorker, get_registration
from Dicom_helper import coefficient_of_variance
from src_CMACS.main_CL_CINE import get_CMACS_seg
from src_DeepHeart.infer import main_cine
from pathlib import Path

DATE = datetime.date.today()
now = datetime.datetime.now()
TIME = current_time = now.strftime("%H-%M-%S")
DATASET_NAME = ""
DEEPHEART_CHECKPOINT = "checkpoints_CL_CMACS_RT0_CV0"  # "checkpoints_CL_CMACS_RT8_75SS_CV0" #"checkpoints_ML_bbox_CV0" #"checkpoints_CL_CMACS_RT0_CV0"
CMACS_VERSION = ""
REGISTRATION_METHOD = "reg_all"  # "reg_neighbor"


class RenderWorker(QRunnable):
    def __init__(self, fn, current_phase_index, *args, **kwargs):
        super(RenderWorker, self).__init__()
        self.fn = fn
        self.current_phase_index = current_phase_index
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        self.fn(*self.args, self.current_phase_index)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        # setup UI
        self.setupUi(self)

        # properties
        self.files_list_address_dict = {"label": list(), "gray_scale": list()}
        self.current_phase_index = 0

        # segmentation_method
        self.segmentation_method = ""  # "CMACS","DeepHeart","Regis"

        self.is_visible_parts = {'WHS': False, 'AA': False, 'LV': False, 'LA': False, 'RV': False, 'RA': False,
                                 'LAA': False, 'SVC': False, 'IVC': False, 'PA': False, 'PV': False, 'LVM': False}
        self.segment_visuals_dict = {'WHS': list(), 'AA': list(), 'LV': list(), 'LA': list(), 'RV': list(),
                                     'RA': list(), 'LAA': list(), 'SVC': list(), 'IVC': list(), 'PA': list(),
                                     'PV': list(), 'LVM': list()}

        self.plotting_dataset = {'phase index': None, 'WHS': None, 'AA': None, 'LV': None, 'LA': None, 'RV': None,
                                 'RA': None, 'LAA': None, 'SVC': None, 'IVC': None, 'PA': None, 'PV': None, 'LVM': None}

        # the number of segment_position should be equal to the number of label in a list
        self.segment_parts_changed = True

        self.viewboxes_list = []
        self.viewbox1 = None
        self.viewbox2 = None
        self.viewbox3 = None
        self.view_3_current_camera = None

        self.view_1_current_frame = 30
        self.view_2_current_frame = 30
        self.view_4_current_frame = 30

        # initial = 0 whenever we load new grayscale images or reset the program
        self.initial = 0
        self.labels_list = None
        self.gray_scales_list = None

        # label and grayscale related variable
        self.is_labels_list_changed = True
        self.is_gray_scales_list_changed = True

        # shape of either label or grayscale image
        self.image_shape = None

        # keep track of the last phase
        self.last_phase_index = 0
        self.last_phase_shown_index = 0

        # bounding box
        self.bounding_box = None

        # views
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
        self.canvas = scene.SceneCanvas(keys='interactive', size=(500, 400), show=True)
        self.view_3.setLayout(QVBoxLayout())
        self.view_3.layout().addWidget(self.canvas.native)
        # view 4
        self.canvas4 = Edited_Canvas(keys='interactive', size=(500, 400), show=True)
        self.view_4.setLayout(QVBoxLayout())
        self.view_4_scroll_bar.valueChanged.connect(self.view_4_scroll_bar_valueChanged_handler)
        self.view_4.layout().addWidget(self.canvas4.native)
        self.canvas4.vertical_line_moved_signal.connect(self.view_2_scroll_bar.setValue)
        self.canvas4.horizon_line_moved_signal.connect(self.view_1_scroll_bar.setValue)

        @self.canvas.events.mouse_move.connect
        def on_mouse_move(event):
            if (self.view_3_current_camera is not None):
                transform = self.view_3_current_camera.transform
                for segment in self.segment_visuals_dict.keys():
                    if (self.segment_visuals_dict[segment][self.last_phase_index] is not None and
                            self.segment_visuals_dict[segment][self.last_phase_index].visible == True):
                        dir = np.concatenate((self.initial_light_dir, [0]))
                        self.segment_visuals_dict[segment][self.last_phase_index].light_dir = transform.map(dir)[:3]

        # play_LCD_number
        self.view_3_LCD_number.setSegmentStyle(QLCDNumber.Flat)
        # slider
        self.view_3_slider.setTracking(False)
        self.view_3_slider.valueChanged.connect(self.view_3_slider_valueChanged_handler)
        # nextButton
        self.view_3_next_button.clicked.connect(self.view_3_next_button_clicked_handler)
        self.view_3_back_button.clicked.connect(self.view_3_back_button_clicked_handler)

        # play button
        self.view_3_play_button.setChecked(True)
        self.view_3_play_button.clicked.connect(self.view_3_play_button_clicked_handler)
        self.view_3_play_button.setCheckable(True)

        # load nifti button
        self.load_nifti_button.pressed.connect(partial(self.load_files, "nifti"))

        # load dicom button
        self.load_dicom_button.pressed.connect(partial(self.load_files, "dicom"))

        # load label button
        self.load_label_button.pressed.connect(partial(self.load_files, "label"))

        # show button
        self.show_button.pressed.connect(self.show_button_pressed_handler)

        # segmentation button
        self.segmentation_CMACS_button.pressed.connect(self.segmentation_CMACS_button_clicked_handler)
        # save label segmentation button
        self.segmentation_save_label_button.pressed.connect(self.segmentation_save_label_button_pressed_handler)
        # segmentation using deep heart button
        self.segmentation_deep_heart_button.pressed.connect(self.segmentation_deep_heart_button_pressed_handler)
        # segmentation all phases radio button
        self.segmentation_all_phases_radio_button.pressed.connect(
            self.segmentation_all_phases_radio_button_pressed_handler)
        # segmentation current phase button
        self.segmentation_current_phase_radio_button.pressed.connect(
            self.segmentation_current_phase_radio_button_pressed_handler)

        # label_check_all_button functions like save all of the label nii to a pre-defined file
        self.label_check_all_button.pressed.connect(self.label_check_all_button_pressed_handler)

        # by default, segmentation_current_phase_radio_button is checked
        self.segmentation_current_phase_radio_button.setChecked(True)

        # registration
        self.segmentation_registration_button.pressed.connect(self.segmentation_registration_button_pressed_handler)

        # checkboxes
        self.label_AA_checkbox.stateChanged.connect(self.AA_checkbox_stateChanged_handler)
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
        self.label_WHS_checkbox.stateChanged.connect(self.WHS_checkbox_stateChanged_handler)
        # initialize the checkboxes to checked
        self.label_AA_checkbox.setChecked(False)
        self.label_LV_checkbox.setChecked(False)
        self.label_LVM_checkbox.setChecked(False)
        self.label_LA_checkbox.setChecked(False)
        self.label_RV_checkbox.setChecked(False)
        self.label_RA_checkbox.setChecked(False)
        self.label_LAA_checkbox.setChecked(False)
        self.label_SVC_checkbox.setChecked(False)
        self.label_IVC_checkbox.setChecked(False)
        self.label_PA_checkbox.setChecked(False)
        self.label_PV_checkbox.setChecked(False)
        self.label_WHS_checkbox.setChecked(False)

        # opacity
        self.opacity_slider.setMaximum(10)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setSingleStep(1)
        self.opacity_LCD_number.setSegmentStyle(QLCDNumber.Flat)
        # initialize opacity value
        self.opacity_slider.valueChanged.connect(self.opacity_slider_valueChanged_handler)

        # text browser
        self.status_text_browser.setReadOnly(True)

        # Qtimer
        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.timer_timeout_handler)

        # multithreading
        self.threadpool = QThreadPool()
        self.threadpool.setMaxThreadCount(3)

        # load mainwindow ui
        self.show()
        vispy.app.run()

    # TODO: TEMPORARY
    def label_check_all_button_pressed_handler(self):
        if (self.segmentation_method == "DeepHeart"):
            OUTPUT_DIR = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/DeepHeart/{DEEPHEART_CHECKPOINT}/Labels/{DATE}/{TIME}/"
        elif (self.segmentation_method == "CMACS"):
            OUTPUT_DIR = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/CMACS/{CMACS_VERSION}/Labels/{DATE}/{TIME}/"
        else:
            OUTPUT_DIR = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/Draft/Labels/{DATE}/{TIME}/"
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        print("DEBUG: label_check_all_button_pressed_handler was clicked")

        for index in range(len(self.labels_list)):
            if (self.labels_list[index] is not None):
                out = sitk.GetImageFromArray(np.copy(self.labels_list[index]))
                sitk.WriteImage(out, os.path.join(OUTPUT_DIR, "label_" + str(index + 1) + ".nii.gz"))

    def read_data(self, gray_scales_list, labels_list, directories, file_type):
        if (file_type == "nifti" or file_type == "dicom"):
            self.gray_scales_list = gray_scales_list
        elif (file_type == "label"):
            self.labels_list = labels_list

        for directory in directories:
            self.display_on_text_browser("loaded: " + directory)

        if (self.initial == 0):
            self.initialize_segment_visuals_dict(phase_number=max(len(self.files_list_address_dict["label"]),
                                                                  len(self.files_list_address_dict["gray_scale"])))
        if (file_type == "nifti" or file_type == "dicom"):
            self.image_shape = list(np.shape(gray_scales_list[0]))
            self.display_on_text_browser(
                "number of volumes loaded: " + str(len(self.files_list_address_dict["gray_scale"])))
            self.display_on_text_browser("volume shape: " + str(np.shape(gray_scales_list[0])))
        elif (file_type == "label"):
            self.image_shape = list(np.shape(labels_list[0]))
            self.display_on_text_browser("number of volumes loaded: " + str(len(self.files_list_address_dict["label"])))
            self.display_on_text_browser("volume shape: " + str(np.shape(labels_list[0])))

    # handle load files
    def load_files(self, files_type):
        self.clear_text_browser()
        self.display_on_text_browser("loading " + files_type)
        label_dialog = None
        if (files_type == "label"):
            label_dialog = LoadLabelDialog(self)
            label_dialog.exec()
            current_directory = label_dialog.current_directory
        else:
            # create a instance of dialog class
            dialog = QFileDialog(self, "open File")
            # we can select any kind of file in the dialog
            dialog.setFileMode(QFileDialog.AnyFile)
            # set the chosen file as a current directory
            current_directory = dialog.getExistingDirectory(self, 'open File')
            self.display_on_text_browser("directory: " + current_directory)
            self.display_on_text_browser("file type: " + files_type)

        # get all the files' address in the current directory
        if (files_type == "nifti" or files_type == "dicom"):
            self.files_list_address_dict["gray_scale"] = self.get_files_address(current_directory, files_type)
            self.is_gray_scales_list_changed = True
            self.is_labels_list_changed = False
            self.display_on_text_browser("number of phases: " + str(len(self.files_list_address_dict["gray_scale"])))

        elif (files_type == "label" and label_dialog.type_name == "all phases"):
            self.files_list_address_dict["label"] = self.get_files_address(current_directory, files_type)
            self.is_labels_list_changed = True
            self.is_gray_scales_list_changed = False
            self.display_on_text_browser("number of phases: " + str(len(self.files_list_address_dict["label"])))

        elif files_type == "label" and label_dialog.type_name == "current phase":
            self.initial = 0
            self.initialize_labels_list(len(self.gray_scales_list))
            self.labels_list[self.current_phase_index] = read_label(current_directory)[0]
            self.bounding_box = self.get_bounding_box(self.labels_list[self.current_phase_index])
            self.is_labels_list_changed = True
            self.is_gray_scales_list_changed = False
            return

        self.initial = 0

        if (self.is_gray_scales_list_changed and self.initial == 0):
            self.initialize_gray_scales_list(len(self.files_list_address_dict["gray_scale"]))
        if (self.is_labels_list_changed and self.initial == 0):
            self.initialize_labels_list(len(self.files_list_address_dict["label"]))

        # worker = LoadDataWorker(self.read_data, self.files_list_address_dict, files_type, self.gray_scales_list,
        #                         self.labels_list)
        # self.threadpool.start(worker)
        if files_type == "dicom" or files_type == "nifty":
            directories = self.files_list_address_dict['gray_scale']
        elif files_type == "label":
            directories = self.files_list_address_dict['label']

        for index, address in enumerate(directories):
            if (files_type == "dicom"):
                self.gray_scales_list[index], _ = read_dicom_case(address)
            elif (files_type == "nifty"):
                self.gray_scales_list[index], _ = read_nifti(address)
            elif (files_type == "label"):
                self.labels_list[index], _ = read_label(address)

        self.read_data(self.gray_scales_list, self.labels_list, self.files_list_address_dict, files_type)

    def get_files_address(self, directory, files_type):
        global DATASET_NAME

        files_list_dict = {"label": list(), "gray_scale": list()}
        if (files_type == "nifti"):
            DATASET_NAME = directory.split("/")[-1]
            for root, dirs, files in os.walk(directory):
                for file in files:
                    link = os.path.join(directory, file)
                    if (re.split('[\' .,()_]', file)[-1] == "gz"):
                        files_list_dict["gray_scale"].append(link)
            # when we get addresses inside a folder using os.walk, the order of addresses is random, we need to sort them
            files_list_dict["gray_scale"] = self.sort_files_list(files_list_dict["gray_scale"], files_type)
            return files_list_dict["gray_scale"]
        elif (files_type == "dicom"):
            DATASET_NAME = directory.split("/")[-1]
            for root, dirs, files in os.walk(directory):
                for file in dirs:
                    link = os.path.join(directory, file)
                    files_list_dict["gray_scale"].append(link)
            files_list_dict["gray_scale"] = self.sort_files_list(files_list_dict["gray_scale"], files_type)
            return files_list_dict["gray_scale"]
        elif (files_type == "label"):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    link = os.path.join(directory, file)
                    if (re.split('[\' .,()_]', file)[-1] == "gz"):
                        files_list_dict["label"].append(link)
            # when we get addresses inside a folder using os.walk, the order of addresses is random, we need to sort them
            files_list_dict["label"] = self.sort_files_list(files_list_dict["label"], files_type)
            return files_list_dict["label"]

    def sort_files_list(self, unordered_files_list, file_type):
        unordered_files_list = filter(unordered_files_list, file_type)
        sorted_files_list = [0] * len(unordered_files_list)
        for file_path in unordered_files_list:
            print(re.split('[\' .,()_]', file_path))
            if (file_type == "label"):
                sorted_files_list[int(re.split('[\' .,()_]', file_path)[-3]) - 1] = file_path
            elif (file_type == "nifti"):
                sorted_files_list[int(re.split('[\' .,()_]', file_path)[-3]) - 1] = file_path
            elif (file_type == "dicom"):
                sorted_files_list[int(re.split('[\' .,()_]', file_path)[-1]) - 1] = file_path
        return sorted_files_list

    # show_button_pressed_handler
    def show_button_pressed_handler(self):
        if (len(self.files_list_address_dict["label"]) != len(self.files_list_address_dict["gray_scale"]) and len(
                self.files_list_address_dict["label"]) != 0 and len(self.files_list_address_dict["gray_scale"]) != 0):
            raise Exception("Sorry, the number of labels does not match the number of gray scale images")
        # update later: handle a case when users don't input gray_scale images
        # update later: handle a case when users don't input label images
        # update later: handle a case when shape of gray_scale does not match shape of label images and return the mismatched files.
        # it takes 5-10 seconds to render each phase, what I would like to do is to render all phases when user hits the show button
        if (self.is_labels_list_changed or self.segment_parts_changed):
            self.setup_plotting_dataset()
            self.render_all_phases(self.labels_list)
            self.segment_parts_changed = False

        # set up the slider range
        if (self.initial == 0):
            self.opacity_slider.setValue(1)
            self.setup_view_3_slider_range(self.files_list_address_dict)
            self.set_scroll_bar_range()
            self.setup_label_volumes_list(alpha=self.opacity_LCD_number.value())

        self.show_view_3(phase_index=self.last_phase_index)

    def setup_plotting_dataset(self):
        for key in self.plotting_dataset:
            if (key == "phase index"):
                self.plotting_dataset[key] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            else:
                self.plotting_dataset[key] = [None] * max(len(self.files_list_address_dict["label"]),
                                                          len(self.files_list_address_dict["gray_scale"]))

    def initialize_labels_list(self, n_elements):
        self.labels_list = [None] * n_elements

    def initialize_gray_scales_list(self, n_elements):
        self.gray_scales_list = [None] * n_elements

    # render all the phases (labels)
    def render_all_phases(self, labels_list):
        # create a list of viewboxes.
        self.display_on_text_browser("rendering: ")
        # temparory self.segment_parts_changed
        if (self.initial == 0 or self.segment_parts_changed):
            self.viewboxes_list = list()
            for i in range(
                    max(len(self.files_list_address_dict["label"]), len(self.files_list_address_dict["gray_scale"]))):
                self.viewboxes_list.append(scene.widgets.ViewBox(parent=self.canvas.scene))

        if (labels_list is not None and (self.is_labels_list_changed or self.segment_parts_changed)):
            # render each phase into each viewbox
            # execute only when we already input the label images

            for phase_index in range(len(self.viewboxes_list)):
                if (self.labels_list[phase_index] is not None):
                    worker = RenderWorker(self.render_phase_into_viewbox, phase_index)
                    self.threadpool.start(worker)

            if (self.threadpool.waitForDone()):
                if (self.initial == 0):
                    self.set_view_3_camera(self.labels_list)

    def set_view_3_camera(self, vols_list):
        fov = 60.
        vol = None
        for element in vols_list:
            if (element is not None):
                vol = element
        self.view_3_current_camera = scene.cameras.ArcballCamera(fov=fov, name='Arcball',
                                                                 center=(
                                                                     vol.shape[0] / 2, vol.shape[1] / 2,
                                                                     vol.shape[2] / 2),
                                                                 distance=400)
        self.segment_visuals_dict["AA"][self.last_phase_index].light_dir = (0, -1, 0)
        self.initial_light_dir = self.view_3_current_camera.transform.imap(
            self.segment_visuals_dict["AA"][self.last_phase_index].light_dir)[:3]

    def render_phase_into_viewbox(self, phase_index, alpha=1):
        vol = self.labels_list[phase_index]
        print(phase_index)

        WHS = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["WH"])
        WHS[z, x, y] = 1
        self.plotting_dataset['WHS'][phase_index] = len(z)

        AA = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["AO"])
        AA[z, x, y] = 1
        self.plotting_dataset['AA'][phase_index] = len(z)

        LV = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["LV"])
        LV[z, x, y] = 1
        self.plotting_dataset['LV'][phase_index] = len(z)

        LA = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["LA"])
        LA[z, x, y] = 1
        self.plotting_dataset['LA'][phase_index] = len(z)

        RV = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["RV"])
        RV[z, x, y] = 1
        self.plotting_dataset['RV'][phase_index] = len(z)

        RA = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["RA"])
        RA[z, x, y] = 1
        self.plotting_dataset['RA'][phase_index] = len(z)

        LAA = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["LAA"])
        LAA[z, x, y] = 1
        self.plotting_dataset['LAA'][phase_index] = len(z)

        SVC = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["SVC"])
        SVC[z, x, y] = 1
        self.plotting_dataset['SVC'][phase_index] = len(z)

        IVC = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["IVC"])
        IVC[z, x, y] = 1
        self.plotting_dataset['IVC'][phase_index] = len(z)

        PA = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["PA"])
        PA[z, x, y] = 1
        self.plotting_dataset['PA'][phase_index] = len(z)

        PV = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["PV"])
        PV[z, x, y] = 1
        self.plotting_dataset['PV'][phase_index] = len(z)

        LVM = np.zeros(vol.shape, dtype=np.uint16)
        z, x, y = np.where(vol == label_dict["LVM"])
        LVM[z, x, y] = 1
        self.plotting_dataset['LVM'][phase_index] = len(z)

        viewBox = self.viewboxes_list[phase_index]
        light_color = color.Color('white', alpha=0.9)

        self.segment_visuals_dict["AA"][phase_index] = scene.visuals.Isosurface(AA, level=AA.max() / 4.,
                                                                                color=(1, 1, 0, 1), shading='smooth',
                                                                                parent=viewBox.scene)

        self.segment_visuals_dict["AA"][phase_index].ambient_light_color = color.Color('white')
        self.segment_visuals_dict["AA"][phase_index].visible = self.is_visible_parts['AA']

        self.segment_visuals_dict["LV"][phase_index] = scene.visuals.Isosurface(LV, level=LV.max() / 4.,
                                                                                color=(1, 0, 1, 1), shading='smooth',
                                                                                parent=viewBox.scene)
        self.segment_visuals_dict["LV"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["LV"][phase_index].visible = self.is_visible_parts['LV']

        self.segment_visuals_dict["LVM"][phase_index] = scene.visuals.Isosurface(LVM, level=LVM.max() / 4.,
                                                                                 color=(0, 0.5, 0, 1), shading='smooth',
                                                                                 parent=viewBox.scene)

        self.segment_visuals_dict["LVM"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["LVM"][phase_index].visible = self.is_visible_parts['LVM']

        self.segment_visuals_dict["LA"][phase_index] = scene.visuals.Isosurface(LA, level=LA.max() / 4.,
                                                                                color=(1, 0.4, 0.4, 1),
                                                                                shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["LA"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["LA"][phase_index].visible = self.is_visible_parts['LA']

        self.segment_visuals_dict["RV"][phase_index] = scene.visuals.Isosurface(RV, level=RV.max() / 4.,
                                                                                color=(0, 1, 1, 1), shading='smooth',
                                                                                parent=viewBox.scene)

        self.segment_visuals_dict["RV"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["RV"][phase_index].visible = self.is_visible_parts['RV']

        self.segment_visuals_dict["RA"][phase_index] = scene.visuals.Isosurface(RA, level=RA.max() / 4.,
                                                                                color=(0.4, 0.75, 0.4, 1),
                                                                                shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["RA"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["RA"][phase_index].visible = self.is_visible_parts['RA']

        self.segment_visuals_dict["LAA"][phase_index] = scene.visuals.Isosurface(LAA, level=LAA.max() / 4.,
                                                                                 color=(0.75, 0.4, 0.4, 1),
                                                                                 shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["LAA"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["LAA"][phase_index].visible = self.is_visible_parts['LAA']

        self.segment_visuals_dict["SVC"][phase_index] = scene.visuals.Isosurface(SVC, level=SVC.max() / 4.,
                                                                                 color=(0.4, 0.4, 0.75, 1),
                                                                                 shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["SVC"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["SVC"][phase_index].visible = self.is_visible_parts['SVC']

        self.segment_visuals_dict["IVC"][phase_index] = scene.visuals.Isosurface(IVC, level=IVC.max() / 4.,
                                                                                 color=(0.4, 0.5, 0.4, 1),
                                                                                 shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["IVC"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["IVC"][phase_index].visible = self.is_visible_parts['IVC']

        self.segment_visuals_dict["PA"][phase_index] = scene.visuals.Isosurface(PA, level=PA.max() / 4.,
                                                                                color=(0.4, 0.4, 1, 1),
                                                                                shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["PA"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["PA"][phase_index].visible = self.is_visible_parts['PA']

        self.segment_visuals_dict["PV"][phase_index] = scene.visuals.Isosurface(PV, level=PV.max() / 4.,
                                                                                color=(0, 0.5, 0.5, 1),
                                                                                shading='smooth', parent=viewBox.scene)

        self.segment_visuals_dict["PV"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["PV"][phase_index].visible = self.is_visible_parts['PV']

        self.segment_visuals_dict["WHS"][phase_index] = scene.visuals.Isosurface(WHS, level=WHS.max() / 4.,
                                                                                 color=(0, 1, 0, 1), shading='smooth',
                                                                                 parent=viewBox.scene)

        self.segment_visuals_dict["WHS"][phase_index].ambient_light_color = light_color
        self.segment_visuals_dict["WHS"][phase_index].visible = self.is_visible_parts['WHS']

    def segmentation_save_label_button_pressed_handler(self):
        if (self.segmentation_method == "DeepHeart"):
            default_dir = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/DeepHeart/{DEEPHEART_CHECKPOINT}/Debug/"
        elif (self.segmentation_method == "CMACS"):
            default_dir = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/CMACS/{CMACS_VERSION}/Debug/"
        else:
            default_dir = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/Draft/Debug/"
            Path(default_dir).mkdir(parents=True, exist_ok=True)
        # create a instance of dialog class
        dialog = QFileDialog(self, "open File")
        # we can select any kind of file in the dialog
        dialog.setFileMode(QFileDialog.AnyFile)
        # set the chosen file as a current directory
        current_directory = dialog.getExistingDirectory(self, 'open File', default_dir)
        # get all the files' address in the current directory

        for key in self.plotting_dataset:
            if (key != "phase index"):
                # change from mm3 to ml
                self.plotting_dataset[key] = np.array(self.plotting_dataset[key]) / 1000
                # Plot.save_plot(self.plotting_dataset['phase index'], self.plotting_dataset[key], key, current_directory)

        OUTPUT_DIR = current_directory + f"/Size/{DATE}/{TIME}/"
        Plot.save_dataset_csv(self.plotting_dataset, OUTPUT_DIR,
                              filename=f"{DATASET_NAME}_size_Data_{DEEPHEART_CHECKPOINT}")

        # Coefficient of Variation dataset
        CV_data = coefficient_of_variance.get_CV_dataset(self.labels_list, self.gray_scales_list)

        OUTPUT_DIR = current_directory + f"/Coefficient_Variation/{DATE}/{TIME}/"

        Plot.save_dataset_csv(CV_data, OUTPUT_DIR, filename=f"{DATASET_NAME}_CV_Data_{DEEPHEART_CHECKPOINT}")

    # initialize segment visual dict
    def initialize_segment_visuals_dict(self, phase_number):
        for key in self.segment_visuals_dict:
            self.segment_visuals_dict[key] = [None] * phase_number

    # set up label volume
    def setup_label_volumes_list(self, alpha=1):
        self.vol_label_list = [None] * max(len(self.files_list_address_dict["gray_scale"]),
                                           len(self.files_list_address_dict["label"]))
        for i in range(len(self.vol_label_list)):
            if (self.labels_list is None or self.labels_list[i] is None):
                continue
            else:
                self.vol_label_list[i] = np.empty((self.image_shape[0], self.image_shape[1], self.image_shape[2], 4),
                                                  dtype=np.float)
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["AO"])] = np.array(
                    [1, 1, 0, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["LV"])] = np.array(
                    [1, 0, 1, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["LVM"])] = np.array(
                    [0, 0.5, 0, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["LA"])] = np.array(
                    [1, 0.4, 0.4, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["RV"])] = np.array(
                    [0, 1, 1, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["RA"])] = np.array(
                    [0.4, 0.75, 0.4, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["LAA"])] = np.array(
                    [0.75, 0.4, 0.4, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["SVC"])] = np.array(
                    [0.4, 0.4, 0.75, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["IVC"])] = np.array(
                    [0.4, 0.5, 0.4, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["PA"])] = np.array(
                    [0.4, 0.4, 1, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["PV"])] = np.array(
                    [0, 0.5, 0.5, alpha])
                self.vol_label_list[i][np.where(self.labels_list[i] == label_dict["WH"])] = np.array(
                    [0, 1, 0, alpha])
        self.show_views_124(self.last_phase_index)

    # show current phase on the view #3
    def show_view_3(self, phase_index, outdated_viewbox=None):

        # render phase when the requirement meets. otherwise, only show view 1,2,4
        if (self.labels_list is not None):
            self.viewboxes_list[phase_index].camera = self.view_3_current_camera
            try:
                if (outdated_viewbox is None):
                    self.canvas.central_widget.remove_widget(self.last_shown_viewbox)
                else:
                    self.canvas.central_widget.remove_widget(outdated_viewbox)
            except:
                pass
            finally:
                self.canvas.central_widget.add_widget(self.viewboxes_list[phase_index])
                self.canvas.scene.update()
                self.last_shown_viewbox = self.viewboxes_list[phase_index]

        # keep track of the last phaserun
        self.last_phase_index = phase_index

        self.show_views_124(phase_index=phase_index)

    # show gray_scale and labels on view 1,2 and 4
    def show_views_124(self, phase_index):

        self.show_view_1(phase_index, self.view_1_current_frame, self.vol_label_list[phase_index])
        self.show_view_2(phase_index, self.view_2_current_frame, self.vol_label_list[phase_index])
        self.show_view_4(phase_index, self.view_4_current_frame, self.vol_label_list[phase_index])

        # overcome error when set initial value is 0
        if (self.initial == 0):
            # set the initial variable to 1
            self.initial = 1

            self.view_1_scroll_bar.setValue(30)
            self.view_2_scroll_bar.setValue(30)
            self.view_4_scroll_bar.setValue(30)

    def show_view_1(self, phase_index, frame_index, vol_label):
        if (self.initial == 0):
            self.viewbox1 = scene.widgets.ViewBox()
            if (self.gray_scales_list is not None):
                self.gray_scale_viewbox1 = scene.visuals.Image(self.gray_scales_list[phase_index][frame_index, :, :],
                                                               cmap="grays",
                                                               parent=self.viewbox1.scene)
            if (vol_label is not None):
                self.label_viewbox1 = scene.visuals.Image(vol_label[frame_index, :, :],
                                                          parent=self.viewbox1.scene)
                self.label_viewbox1.transform = visuals.transforms.STTransform(translate=(0, 0, -0.5))
            self.canvas1.set_view(self.viewbox1)
            self.canvas1.set_limit(0, self.image_shape[1] - 1, 0, self.image_shape[2] - 1)
            self.camera1 = scene.PanZoomCamera(aspect=1)
            self.camera1.flip = (0, 1, 0)
            self.camera1.zoom(200, center=(0.05, 0.05))
            self.canvas1.set_camera(self.camera1)
        else:
            if (vol_label is not None):
                self.label_viewbox1.set_data(vol_label[frame_index, :, :])
                self.label_viewbox1.visible = True
                self.label_viewbox1.update()
            else:
                try:
                    self.label_viewbox1.visible = False
                except:
                    pass
            if (self.gray_scales_list is not None):
                self.gray_scale_viewbox1.set_data(self.gray_scales_list[phase_index][frame_index, :, :])
                self.gray_scale_viewbox1.update()

        self.canvas1.scene.update()

    def show_view_2(self, phase_index, frame_index, vol_label):
        if (self.initial == 0):
            self.viewbox2 = scene.widgets.ViewBox()
            if (self.gray_scales_list is not None):
                self.gray_scale_viewbox2 = scene.visuals.Image(self.gray_scales_list[phase_index][:, frame_index, :],
                                                               cmap="grays",
                                                               parent=self.viewbox2.scene)
            if (vol_label is not None):
                self.label_viewbox2 = scene.visuals.Image(vol_label[:, frame_index, :],
                                                          parent=self.viewbox2.scene)
                self.label_viewbox2.transform = visuals.transforms.STTransform(translate=(0, 0, -0.5))
            self.canvas2.set_view(self.viewbox2)
            self.canvas2.set_limit(0, self.image_shape[2] - 1, 0, self.image_shape[0] - 1)
            self.camera2 = scene.PanZoomCamera(aspect=1)
            self.camera2.flip = (0, 1, 0)
            self.camera2.zoom(200, center=(0.05, 0.2))
            self.canvas2.set_camera(self.camera2)
        else:
            if (vol_label is not None):
                self.label_viewbox2.set_data(vol_label[:, frame_index, :])
                self.label_viewbox2.visible = True
            else:
                try:
                    self.label_viewbox2.visible = False
                except:
                    pass
            if (self.gray_scales_list is not None):
                self.gray_scale_viewbox2.set_data(self.gray_scales_list[phase_index][:, frame_index, :])

        self.canvas2.scene.update()

    def show_view_4(self, phase_index, frame_index, vol_label):
        if (self.initial == 0):
            self.viewbox4 = scene.widgets.ViewBox()
            if (self.gray_scales_list is not None):
                self.gray_scale_viewbox4 = scene.visuals.Image(self.gray_scales_list[phase_index][:, :, frame_index],
                                                               cmap="grays",
                                                               parent=self.viewbox4.scene)
            if (vol_label is not None):
                self.label_viewbox4 = scene.visuals.Image(vol_label[:, :, frame_index],
                                                          parent=self.viewbox4.scene)
                self.label_viewbox4.transform = visuals.transforms.STTransform(translate=(0, 0, -0.5))
            self.canvas4.set_view(self.viewbox4)
            self.canvas4.set_limit(0, self.image_shape[1] - 1, 0, self.image_shape[0] - 1)
            self.camera4 = scene.PanZoomCamera(aspect=1)
            self.camera4.flip = (0, 1, 0)
            self.camera4.zoom(200, center=(0.05, 0.2))
            self.canvas4.set_camera(self.camera4)

        else:
            if (vol_label is not None):
                self.label_viewbox4.set_data(vol_label[:, :, frame_index])
                self.label_viewbox4.visible = True
            else:
                try:
                    self.label_viewbox4.visible = False
                except:
                    pass
            if (self.gray_scales_list is not None):
                self.gray_scale_viewbox4.set_data(self.gray_scales_list[phase_index][:, :, frame_index])

            self.canvas4.scene.update()

    def get_segmentation_CMACS(self, data, phase_index):
        self.segmentation_method = "CMACS"
        phase_index = phase_index + 1
        BBOX = 1
        REG_LIB = 'deeds'  # 'correg' # 'deeds'
        FUSION = 'mv'  # 'staple', 'mv'
        dll_path = 'D:/Medical_Image_Analyzer/CT_SEG/CMACS/'
        moving_path = 'D:/Medical_Image_Analyzer/CT_SEG/_Templates'
        debug_path = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/CMACS/{CMACS_VERSION}/Debug/Labels/{phase_index}/{DATE}/{TIME}/"
        RES = 1  # Random Walk Resolution
        GT = 0  # 1 if ground truth is available, otherwise, 0
        PRE_RES = 2  # Preprocessing Resolution
        REG_RES = 1  # Registratrion at 1mm or 2mm?

        final_label_rw = get_CMACS_seg(data, BBOX, moving_path, debug_path, dll_path, GT, RES, REG_LIB, PRE_RES,
                                       REG_RES, FUSION)
        self.bounding_box = self.get_bounding_box(final_label_rw)
        return final_label_rw

    def get_segmentation_CMACS_EXE(self, data, phase_index):
        self.segmentation_method = "CMACS"
        RES = 1  # Random Walk Resolution
        print(data.shape)

        print("---RELEASE MODE---")

        zs = data.shape[0]
        xs = data.shape[1]
        ys = data.shape[2]
        BBOX = 1
        REG_LIB = '5lv'  # 'correg' # '5lv'
        PRE_RES = 2  # Preprocessing Resolution
        FUSION = 'mv'  # 'staple', 'mv'
        dll_path = 'D:/Medical_Image_Analyzer/CT_SEG/CMACS/'
        moving_path = 'D:/Medical_Image_Analyzer/CT_SEG/_Templates/'
        debug_path = "./"
        input_volume = data.tostring()
        p = run(['../CT_SEG/CMACS_EXE/cmacs.exe',
                 str(zs), str(xs),
                 moving_path,
                 FUSION, debug_path, str(0),
                 str(RES), REG_LIB, str(PRE_RES), dll_path, str(BBOX)],
                input=input_volume, stdout=PIPE)
        stdout_result = p.stdout
        final_label_rw = np.fromstring(stdout_result, dtype=np.uint16)
        final_label_rw = np.reshape(final_label_rw, (int(zs), int(xs), int(xs)))
        self.bounding_box = self.get_bounding_box(final_label_rw)
        return final_label_rw

    def get_bounding_box(self, label):
        bbox = np.zeros(label.shape, dtype=np.uint8)
        z, x, y = np.where((label == label_dict["WH"]) | (label == label_dict["LV"]) | (label == label_dict["LVM"]) | (
                label == label_dict["LA"]) | (label == label_dict["RV"]) | (label == label_dict["RA"]) | (
                                   label == label_dict["AO"]))
        bbox[z, x, y] = 1
        return bbox

    def get_segmentation_deep_heart_EXE(self, data):
        self.segmentation_method = "DeepHeart"
        data = data.astype(dtype=np.int16)
        print(data.shape)
        zs = data.shape[0]
        xs = data.shape[1]
        ys = data.shape[2]
        start = time.time()
        config_path = "../CT_SEG/DEEPHEART_EXE"
        input_volume = data.tostring()
        p = run(['../CT_SEG/DEEPHEART_EXE/deepheart.exe', config_path,
                 str(zs), str(xs), str(ys)],
                input=input_volume, stdout=PIPE)
        stdout_result = p.stdout[0:len(input_volume)]
        pred_label = np.fromstring(stdout_result, dtype=np.uint16)
        print(pred_label.shape)
        pred_label = np.reshape(pred_label, data.shape)
        end = time.time()
        print("Total time: " + str(end - start) + " (sec)")
        return pred_label

    def get_segmentation_deep_heart(self, data, phase_index):
        self.segmentation_method = "DeepHeart"
        phase_index = phase_index + 1
        data = data.astype(dtype=np.int16)
        print(data.shape)
        zs = data.shape[0]
        xs = data.shape[1]
        ys = data.shape[2]
        start = time.time()
        config_path = "../src_DeepHeart/resources"
        checkpoint = "checkpoints_CL_CMACS_RT0_CV0"
        debug_path = f"D:/DATA/SQUEEZ/out/{DATASET_NAME}/DEEPHEART/{DEEPHEART_CHECKPOINT}/Debug/Labels/{phase_index}/{DATE}/{TIME}/"
        chkpt_path = f"../src_DeepHeart/chkpt/{checkpoint}"
        label_mask = main_cine(f_vol_1mm=data, config_path=config_path, chkpt_path=chkpt_path, BBOX=1, DEBUG=1,
                               debug_path=debug_path)
        end = time.time()
        print("Total time: " + str(end - start) + " (sec)")
        return label_mask

    # view 3 slots
    # play_sliders slots
    def view_3_slider_valueChanged_handler(self, value):
        value = value % max(len(self.files_list_address_dict["gray_scale"]), len(self.files_list_address_dict["label"]))
        phase_index = value
        self.current_phase_index = phase_index
        self.show_view_3(phase_index)

        self.view_3_LCD_number.display(phase_index + 1)

    def setup_view_3_slider_range(self, files_list_address_dict):
        self.view_3_slider.setMinimum(0)
        self.view_3_slider.setMaximum(
            max(len(files_list_address_dict["gray_scale"]), len(files_list_address_dict["label"])) - 1)
        self.view_3_slider.setTickInterval(1)

    # next/back button slots
    def view_3_next_button_clicked_handler(self):
        value = self.last_phase_index + 1
        self.view_3_slider.setValue(
            value % max(len(self.files_list_address_dict["gray_scale"]), len(self.files_list_address_dict["label"])))

    def view_3_back_button_clicked_handler(self):
        value = self.last_phase_index - 1
        if (value < 0):
            value = max(len(self.files_list_address_dict["gray_scale"]),
                        len(self.files_list_address_dict["label"])) - abs(value)
        self.view_3_slider.setValue(
            value % max(len(self.files_list_address_dict["gray_scale"]), len(self.files_list_address_dict["label"])))

    # play button slot
    def view_3_play_button_clicked_handler(self):
        if (self.view_3_play_button.isChecked()):
            self.timer.start()
        else:
            self.timer.stop()

    # timer slots
    def timer_timeout_handler(self):
        value = self.last_phase_index + 1
        self.view_3_slider.setValue(
            value % max(len(self.files_list_address_dict["gray_scale"]), len(self.files_list_address_dict["label"])))

    # view 1,2,4 related slots
    def set_scroll_bar_range(self):
        self.view_1_scroll_bar.setMinimum(0)
        self.view_1_scroll_bar.setMaximum(self.image_shape[0] - 1)
        self.view_2_scroll_bar.setMinimum(0)
        self.view_2_scroll_bar.setMaximum(self.image_shape[1] - 1)
        self.view_4_scroll_bar.setMinimum(0)
        self.view_4_scroll_bar.setMaximum(self.image_shape[2] - 1)

    def view_1_scroll_bar_valueChanged_handler(self, value):
        self.show_view_1(self.last_phase_index, value, self.vol_label_list[self.last_phase_index])
        self.canvas2.set_horizon_pos(value)
        self.canvas4.set_horizon_pos(value)
        self.view_1_current_frame = value

    def view_2_scroll_bar_valueChanged_handler(self, value):
        self.show_view_2(self.last_phase_index, value, self.vol_label_list[self.last_phase_index])
        self.canvas1.set_horizon_pos(value)
        self.canvas4.set_vertical_pos(value)
        self.view_2_current_frame = value

    def view_4_scroll_bar_valueChanged_handler(self, value):
        self.show_view_4(self.last_phase_index, value, self.vol_label_list[self.last_phase_index])
        self.canvas1.set_vertical_pos(value)
        self.canvas2.set_vertical_pos(value)
        self.view_4_current_frame = value

    def opacity_slider_valueChanged_handler(self, value):
        alpha = value / 10
        self.opacity_LCD_number.display(alpha)
        self.setup_label_volumes_list(alpha)

    # checkboxes slots
    def AA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['AA'] = True
        else:
            self.is_visible_parts['AA'] = False
        self.segment_parts_changed = True

    def LV_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['LV'] = True
        else:
            self.is_visible_parts['LV'] = False
        self.segment_parts_changed = True

    def LVM_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['LVM'] = True
        else:
            self.is_visible_parts['LVM'] = False
        self.segment_parts_changed = True

    def LA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['LA'] = True
        else:
            self.is_visible_parts['LA'] = False
        self.segment_parts_changed = True

    def RV_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['RV'] = True
        else:
            self.is_visible_parts['RV'] = False
        self.segment_parts_changed = True

    def RA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['RA'] = True
        else:
            self.is_visible_parts['RA'] = False
        self.segment_parts_changed = True

    def LAA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['LAA'] = True
        else:
            self.is_visible_parts['LAA'] = False
        self.segment_parts_changed = True

    def SVC_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['SVC'] = True
        else:
            self.is_visible_parts['SVC'] = False
        self.segment_parts_changed = True

    def IVC_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['IVC'] = True
        else:
            self.is_visible_parts['IVC'] = False
        self.segment_parts_changed = True

    def PA_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['PA'] = True
        else:
            self.is_visible_parts['PA'] = False
        self.segment_parts_changed = True

    def PV_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['PV'] = True
        else:
            self.is_visible_parts['PV'] = False
        self.segment_parts_changed = True

    def WHS_checkbox_stateChanged_handler(self, state):
        if (state == 2):
            self.is_visible_parts['WHS'] = True
        else:
            self.is_visible_parts['WHS'] = False
        self.segment_parts_changed = True

    # segmentation button slot
    def segmentation_CMACS_button_clicked_handler(self):
        phase_index = self.last_phase_index
        if (self.labels_list is None):
            # make sure number of labels is equal to number of gray scales
            self.initialize_labels_list(len(self.files_list_address_dict["gray_scale"]))
        if (self.viewboxes_list is None):
            # make sure we initialize the viewboxes_list if the variable is none
            for i in range(
                    max(len(self.files_list_address_dict["label"]), len(self.files_list_address_dict["gray_scale"]))):
                self.viewboxes_list.append(scene.widgets.ViewBox(parent=self.canvas.scene))
        start = time.time()
        if (self.segmentation_all_phases_radio_button.isChecked()):
            for phase_index in range(len(self.viewboxes_list)):
                self.labels_list[phase_index] = self.get_segmentation_CMACS(self.gray_scales_list[phase_index],
                                                                            phase_index)
        else:
            self.labels_list[phase_index] = self.get_segmentation_CMACS(self.gray_scales_list[phase_index], phase_index)
        done = time.time()
        print("DEBUG: Time Interval Segmentation CMACS " + str(done - start))
        outdated_viewbox = None
        if (self.labels_list[phase_index] is not None):
            self.viewboxes_list[phase_index] = scene.widgets.ViewBox(parent=self.canvas.scene)

        self.initial = 0
        self.is_labels_list_changed = True
        self.show_button_pressed_handler()

    def segmentation_deep_heart_button_pressed_handler(self):
        phase_index = self.last_phase_index
        if (self.labels_list is None):
            # make sure number of labels is equal to number of gray scales
            self.initialize_labels_list(len(self.files_list_address_dict["gray_scale"]))
        if (self.viewboxes_list is None):
            # make sure we initbvialize the viewboxes_list if the variable is none
            for i in range(
                    max(len(self.files_list_address_dict["label"]), len(self.files_list_address_dict["gray_scale"]))):
                self.viewboxes_list.append(scene.widgets.ViewBox(parent=self.canvas.scene))
        start = time.time()
        if (self.segmentation_all_phases_radio_button.isChecked()):
            for phase_index in range(len(self.viewboxes_list)):
                self.labels_list[phase_index] = self.get_segmentation_deep_heart(self.gray_scales_list[phase_index],
                                                                                 phase_index)
        else:
            self.labels_list[phase_index] = self.get_segmentation_deep_heart(self.gray_scales_list[phase_index],
                                                                             phase_index)
        done = time.time()
        print("DEBUG: Time Interval Segmentation DEEPHEART " + str(done - start))
        outdated_viewbox = None
        if (self.labels_list[phase_index] is not None):
            self.viewboxes_list[phase_index] = scene.widgets.ViewBox(parent=self.canvas.scene)

        self.initial = 0
        self.is_labels_list_changed = True
        self.show_button_pressed_handler()

    def segmentation_registration_button_pressed_handler(self):
        # registration_method:
        first_label_index = 0
        phase_index = self.last_phase_index
        m_vol_grayscale = None
        m_vol_label = None
        if (self.labels_list is None):
            # need to segmentation first
            raise Exception("need to do segmentation first")
        for index in range(len(self.labels_list)):
            if (self.labels_list[index] is not None):
                first_label_index = index
        start = time.time()
        if (self.segmentation_all_phases_radio_button.isChecked()):
            if (REGISTRATION_METHOD == "reg_neighbor"):
                worker = RegistrationNbWorker(self.set_labels_list, self.display_on_text_browser, self.labels_list,
                                              self.gray_scales_list, first_label_index, self.bounding_box)
                self.threadpool.start(worker)
            elif (REGISTRATION_METHOD == "reg_all"):
                worker = RegistrationAllWorker(self.set_labels_list, self.display_on_text_browser, self.labels_list,
                                               self.gray_scales_list, first_label_index, self.bounding_box)
                self.threadpool.start(worker)
        else:
            if (self.labels_list[phase_index] is not None):
                raise Exception("already label the phase")
            else:
                for i in range(len(self.labels_list)):
                    if (self.labels_list[i] is not None):
                        m_vol_grayscale = self.gray_scales_list[i]
                        m_vol_label = self.labels_list[i]
                        break
                self.labels_list[phase_index], index = get_registration(m_vol_grayscale, m_vol_label,
                                                                        self.gray_scales_list[phase_index],
                                                                        self.labels_list[phase_index],
                                                                        index=self.current_phase_index)
        if (self.threadpool.waitForDone()):
            done = time.time()
            print("DEBUG: Time Interval registration " + str(done - start))
            outdated_viewbox = None
            if (self.labels_list[phase_index] is not None):
                self.viewboxes_list[phase_index] = scene.widgets.ViewBox(parent=self.canvas.scene)
            self.initial = 0
            self.is_labels_list_changed = True
            self.show_button_pressed_handler()

    def refine(self, volume=None, mask=None, thres=None):
        z, x, y = np.where(volume < thres)
        mask[z, x, y] = 0

        KERNEL_SIZE = 2
        r2 = np.arange(-KERNEL_SIZE, KERNEL_SIZE + 1) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        s_e_sphere2 = (dist2 <= KERNEL_SIZE ** 2).astype(np.int)
        mask = ndi.binary_erosion(mask, structure=s_e_sphere2).astype(dtype=np.uint8)

        # Get largest object
        label_objects, nb_labels = ndi.label(mask)
        sizes = np.bincount(label_objects.ravel())
        sorted_sizes = np.sort(sizes)[::-1]
        if len(sizes) > 2:
            max_sz = np.max(sorted_sizes[1:-1])
            mask_sizes = (sizes == max_sz)
        elif len(sizes) == 2:
            mask_sizes = (sizes == sorted_sizes[1])
        mask = mask_sizes[label_objects].astype(dtype=np.int)

        KERNEL_SIZE = 2
        r2 = np.arange(-KERNEL_SIZE, KERNEL_SIZE + 1) ** 2
        dist2 = r2[:, None, None] + r2[:, None] + r2
        s_e_sphere2 = (dist2 <= KERNEL_SIZE ** 2).astype(np.int)
        mask = ndi.binary_dilation(mask, structure=s_e_sphere2).astype(dtype=np.uint8)

        return mask.astype(dtype=np.uint8)

    def set_labels_list(self, labels_list):
        self.labels_list = labels_list

    def segmentation_all_phases_radio_button_pressed_handler(self):
        self.segmentation_current_phase_radio_button.setChecked(False)

    def segmentation_current_phase_radio_button_pressed_handler(self):
        self.segmentation_all_phases_radio_button.setChecked(False)

    def display_on_text_browser(self, string):
        self.status_text_browser.insertPlainText("_ " + string + "\n")

    def clear_text_browser(self):
        self.status_text_browser.clear()


if __name__ == '__main__':
    app = QApplication([])
    app.setApplicationName("App")
    window = MainWindow()
    app.exec_()
