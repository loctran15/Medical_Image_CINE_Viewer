import SimpleITK as sitk

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Protocol

import os
from pathlib import Path

from .image_enum import *

import numpy as np

from src_CINE.Dicom_helper.helper_funcs import label_dict

from pyqtgraph.exporters.CSVExporter import CSVExporter as BaseCSVExporter
from pyqtgraph.graphicsItems.PlotItem import PlotItem
from pyqtgraph.parametertree.Parameter import Parameter
from PyQt6 import QtCore

translate = QtCore.QCoreApplication.translate

color_map = {
    "BOX": [255, 0, 0],
    "WH": [0, 255, 0],
    "LUNG": [0, 0, 255],
    "LVM": [0, 127, 0],
    "LV": [255, 0, 255],
    "AO": [255, 255, 0],
    "LIVER": [191, 191, 0],
    "DAS": [127, 127, 0],
    "RV": [0, 255, 255],
    "CW": [0, 191, 191],
    "PV": [0, 127, 127],
    "LA": [255, 95, 95],
    "LAA": [191, 95, 95],
    "RA": [95, 191, 95],
    "IVC": [95, 127, 95],
    "PA": [95, 95, 255],
    "SVC": [95, 95, 191],
    "SPINE": [95, 95, 127]
}


class Image(Protocol):
    dir: str
    data: np.ndarray
    dimension: int
    color: ColorType
    imageFileType: ImageFileType


class ImageExporter(ABC):
    out_dir: str
    file_name: str
    out_type: ImageFileType
    image: Image

    @abstractmethod
    def export(self):
        ...

    @abstractmethod
    def convert(self) -> np.ndarray:
        ...


# TODO: write dicom. https://stackoverflow.com/questions/14350675/create-pydicom-file-from-numpy-array
# @dataclass
# class DicomExporter(ImageExporter):
#     tag: Dict[str, str] = None
#
#     def export(self):
#         pass
#
#     def convert(self) -> np.ndarray:
#         pass

@dataclass
class NiftiExporter(ImageExporter):
    def __post_init__(self):
        if (not self.file_name.endswith(".nii.gz")):
            self.file_name = self.file_name + ".nii.gz"

    def export(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        nii_array = self.convert()
        out = sitk.GetImageFromArray(nii_array)
        out.SetSpacing([1, 1, 1])
        sitk.WriteImage(out, os.path.join(self.out_dir, self.file_name))
        print(f"saved to {self.out_dir}/{self.file_name}")

    def convert(self) -> np.ndarray:
        if (self.image.imageFileType == ImageFileType.TIFF and self.image.color == ColorType.RGB):
            zs, xs, ys, _ = self.image.shape
            nii_array = np.zeros((zs, xs, ys), dtype=np.uint8)
            if (self.image.shape == 4):
                for segment, color in color_map.items():
                    z, x, y = np.where((nii_array[:, :, :, 0] == color[0]) & (nii_array[:, :, :, 1] == color[1]) & (
                            nii_array[:, :, :, 2] == color[2]))
                    nii_array[z, x, y] = label_dict[segment]
            return nii_array

        raise NotImplementedError


# @dataclass
# class ROIExporter(ImageExporter):
#
#     def __post_init__(self):
#         if(not self.file_name.endswith(".roi")):
#             self.file_name = self.file_name + ".roi"
#
#     def export(self):
#         Path(self.out_dir).mkdir(parents=True, exist_ok=True)
#
#     def convert(self) -> np.ndarray:
#         ...

@dataclass
class TiffExporter(ImageExporter):
    def __post_init__(self):
        if (not self.file_name.endswith(".tiff") and not self.file_name.endswith(".tif")):
            self.file_name = self.file_name + ".tiff"

    def export(self):
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        tiff_array = self.convert()
        out = sitk.GetImageFromArray(tiff_array)
        out.SetSpacing([1, 1, 1])
        sitk.WriteImage(out, os.path.join(self.out_dir, self.file_name))
        print(f"saved to {self.out_dir}/{self.file_name}")

    def convert(self) -> np.ndarray:
        if (self.image.imageFileType == ImageFileType.NIFTY and self.image.color == ColorType.RGB):
            tiff_array = np.zeros((self.image.shape, 3), dtype=np.uint8)
            for segment in color_map.keys():
                z, x, y = np.where(tiff_array == label_dict[segment])
                tiff_array[z, x, y] = color_map[segment]
            return tiff_array

        raise NotImplementedError


class CSVExporter(BaseCSVExporter):
    Name = "modified CSV exporter"

    def __init__(self, item):
        BaseCSVExporter.__init__(self, item)
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'separator', 'title': translate("Exporter", 'separator'), 'type': 'list', 'value': 'comma',
             'limits': ['comma', 'tab']},
            {'name': 'precision', 'title': translate("Exporter", 'precision'), 'type': 'int', 'value': 10,
             'limits': [0, None]},
            {'name': 'columnMode', 'title': translate("Exporter", 'columnMode'), 'type': 'list',
             'limits': ['(x,y,y,y) for all plots', '(x,y) per plot']}
        ])

    def export(self, fileName=None):
        if not isinstance(self.item, PlotItem):
            raise Exception("Must have a PlotItem selected for CSV export.")

        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return

        data = []
        header = []

        appendAllX = self.params['columnMode'] == '(x,y) per plot'

        for i, c in enumerate(self.item.curves):
            cd = c.getData()
            if cd[0] is None:
                continue
            data.append(cd)
            if hasattr(c, 'implements') and c.implements('plotData') and c.name() is not None:
                name = c.name().replace('"', '""')
                xName, yName = '"' + "phase" + '"', '"' + name + '"'
            else:
                xName = 'x%04d' % i
                yName = 'y%04d' % i
            if appendAllX or i == 0:
                header.extend([xName, yName])
            else:
                header.extend([yName])

        if self.params['separator'] == 'comma':
            sep = ','
        else:
            sep = '\t'

        with open(fileName, 'w') as fd:
            fd.write(sep.join(map(str, header)) + '\n')
            i = 0
            numFormat = '%%0.%dg' % self.params['precision']
            numRows = max([len(d[0]) for d in data])
            for i in range(numRows):
                for j, d in enumerate(data):
                    # write x value if this is the first column, or if we want
                    # x for all rows
                    if appendAllX or j == 0:
                        if d is not None and i < len(d[0]):
                            fd.write(numFormat % d[0][i] + sep)
                        else:
                            fd.write(' %s' % sep)

                    # write y value
                    if d is not None and i < len(d[1]):
                        fd.write(numFormat % d[1][i] + sep)
                    else:
                        fd.write(' %s' % sep)
                fd.write('\n')
