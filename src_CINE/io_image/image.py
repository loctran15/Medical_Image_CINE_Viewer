from enum import Enum, auto
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing import Tuple, Union, Dict, Optional, Any
import numpy.typing as npt
import SimpleITK as sitk
from pydicom.dataset import Dataset
import numpy as np
from numba import jit

import os
from pathlib import Path

from .exporter import ImageExporter
from .image_enum import *

from vispy.scene.widgets.viewbox import ViewBox
from .scene import Image3DScene
from vispy.color.color_array import Color


@dataclass
class Image:
    path: str
    imageFileType: ImageFileType
    imageType: ImageType
    ios: Optional[int] = field(default=1)
    array_data: Optional[np.ndarray] = field(default=None, repr=False)
    dataset: Any = field(default=None, repr=False)
    phase: Optional[int] = field(default=None)
    RGBA_data: Optional[np.ndarray] = field(default=None, repr=False)
    image3DScene: Optional[Image3DScene] = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self.path = os.path.normpath(self.path)
        if (not isinstance(self.array_data, (np.ndarray, np.generic)) and self.dataset != None):
            if self.imageFileType == ImageFileType.DICOM:
                raise Exception("missing array_data, please use reader.read_dicom")
            elif self.imageFileType == ImageFileType.TIFF:
                self.array_data = sitk.GetArrayFromImage(self.dataset)
        elif (not isinstance(self.array_data, (np.ndarray, np.generic)) and self.dataset == None):
            raise Exception("missing both array_data and dataset fields")

    def set_window_level(self, window: int, level: int):
        if (self.imageType == ImageType.GRAYSCALE):
            max = level + window / 2
            min = level - window / 2
            array_data = sitk.GetArrayFromImage(self.dataset)
            self.array_data = array_data.clip(min, max)
        else:
            print("cannot perform set_window_level method on grayscale images")

    def set_RGBA_data(self, alpha: int = 1):
        if self.imageType == ImageType.GRAYSCALE:
            return
        else:
            if len(self.shape) == 3:
                self.RGBA_data = np.zeros(
                    (self.array_data.shape[0], self.array_data.shape[1], self.array_data.shape[2], 4), dtype=np.float_)
                self.RGBA_data[np.where(self.array_data == label_dict["AO"])] = np.array([1, 1, 0, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["LV"])] = np.array([1, 0, 1, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["LVM"])] = np.array([0, 0.5, 0, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["LA"])] = np.array([1, 0.4, 0.4, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["RV"])] = np.array([0, 1, 1, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["RA"])] = np.array([0.4, 0.75, 0.4, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["LAA"])] = np.array([0.75, 0.4, 0.4, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["SVC"])] = np.array([0.4, 0.4, 0.75, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["IVC"])] = np.array([0.4, 0.5, 0.4, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["PA"])] = np.array([0.4, 0.4, 1, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["PV"])] = np.array([0, 0.5, 0.5, alpha])
                self.RGBA_data[np.where(self.array_data == label_dict["WH"])] = np.array([0, 1, 0, alpha])

    @property
    def shape(self) -> tuple:
        return self.array_data.shape

    # TODO: how to identify whether an image has color or not
    @property
    def color(self) -> Optional[ColorType]:
        return None

    @property
    def size(self) -> Optional[Size]:
        sz = Size()
        if self.imageType == ImageType.GRAYSCALE:
            return None
        else:
            for key in RGB_color_dict.keys():
                z, _, _ = np.where(self.array_data == label_dict[key])
                setattr(sz, key, len(z))

            return sz

    def export(self, out_path: str, file_name: str, out_type: ImageFileType):
        if (out_type == ImageFileType.DICOM):
            exporter = ImageExporter(image=self, out_path=out_path, file_name=file_name, out_type=out_type)
            exporter.export()
        elif (out_type == ImageFileType.NIFTY):
            exporter = ImageExporter(image=self, out_path=out_path, file_name=file_name, out_type=out_type)
            exporter.export()
        elif (out_type == ImageFileType.ROI):
            exporter = ImageExporter(image=self, out_path=out_path, file_name=file_name, out_type=out_type)
            exporter.export()
        elif (out_type == ImageFileType.TIFF):
            exporter = ImageExporter(image=self, out_path=out_path, file_name=file_name, out_type=out_type)
            exporter.export()
        else:
            print(f"{out_type} is not supported!")

    def generate_3d_scene(self, parent: ViewBox, is_visual_dict: Optional[dict[str, bool]] = None,
                          alpha: Optional[int] = 1):
        if (not parent):
            print("unable to generate 3d scene without parent")
            return

        if (self.imageType == ImageType.LABEL):
            self.image3DScene = Image3DScene(parent)
            self.image3DScene.setup_scene(image_array=self.array_data, parent=parent,
                                          is_visual_dict=is_visual_dict,
                                          alpha=alpha)
        elif (self.imageType == ImageType.GRAYSCALE):
            print("grayscale image 3d scene not supported yet!")
