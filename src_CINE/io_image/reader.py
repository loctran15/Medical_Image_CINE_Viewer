import SimpleITK as sitk

from abc import ABC, abstractmethod
from dataclasses import dataclass

import os
from pathlib import Path
import glob
import re

from typing import Protocol, Dict, Union, Optional, Any
import numpy.typing as npt
import numpy as np
from skimage.transform import resize
import pydicom as dicom

from .image_enum import ImageType, ImageFileType, ColorType
from src_CINE.Dicom_helper.helper_funcs import label_dict
from .image import Image


class ImageReader:

    @classmethod
    def generate_image_reader(cls, image_file_type: ImageFileType) -> Any:
        if (image_file_type == ImageFileType.NIFTY):
            return cls.read_nifti
        elif (image_file_type == ImageFileType.DICOM):
            return cls.read_dicom
        elif (image_file_type == ImageFileType.TIFF):
            return cls.read_tiff

    @staticmethod
    def read_nifti(path: str, image_type: ImageType, phase: Optional[int] = 1) -> Image:
        path = os.path.normpath(path)
        img = sitk.ReadImage(path)
        img_array = sitk.GetArrayViewFromImage(img)
        return Image(path=path, dataset=img, array_data=img_array, imageType=image_type,
                     imageFileType=ImageFileType.NIFTY, phase=phase)

    @staticmethod
    def read_tiff(path, image_type: ImageType, phase: Optional[int] = 1) -> Image:
        path = os.path.normpath(path)
        img = sitk.ReadImage(path, imageIO="TIFFImageIO")
        try:
            description = img.GetMetaData("ImageDescription")
            Vscale = description.split("_")[0].split(":")[1]
        except:
            Vscale = 1
        orig_image_array = sitk.GetArrayFromImage(img)
        zs, xs, ys, ds = orig_image_array.shape
        zs, xs, ys, ds = zs, int(xs / int(Vscale)), int(ys / int(Vscale)), ds
        img_array = resize(orig_image_array, (zs, xs, ys, ds), order=1, mode='reflect', preserve_range=True,
                           anti_aliasing=0)
        return Image(path=path, dataset=img, array_data=img_array, imageType=image_type,
                     imageFileType=ImageFileType.TIFF, phase=phase)

    # TODO: color dicom image? iso = 0?
    @staticmethod
    def read_dicom(path: Union[str, list[str]], image_type: ImageType, iso: int = 1,
                   phase: Optional[int] = 1) -> Image:
        """
        the function is capable of reading 2D and 3D
        """
        path = os.path.normpath(path)
        if (len(path) == 1 and os.path.isfile(path)):
            path = os.path.normpath(path)
            dataset = dicom.dcmread(path)
            try:
                n_slice = dataset[0x0028, 0x0008].value  # Number of Frames
            except:
                n_slice = 1
            try:
                slsp = dataset[0x7005, 0x1047].value
            except:
                slsp = dataset[0x7005, 0x1022].value
            try:
                rescale_intercept = dataset[0x0028, 0x1052].value  # Rescale Intercept
                rescale_slope = dataset[0x0028, 0x1053].value  # Rescale Slope
            except:
                rescale_intercept = 0
                rescale_slope = 1
            try:
                pxsz = dataset[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
            except:
                pxsz = None

            array_data = dataset.pixel_array.astype(float)
            array_data = array_data * rescale_slope + rescale_intercept

        if (os.path.isdir(path)):
            # filter and sort dcm files
            orig_dcm_file_paths = []
            orig_dcm_index = []
            for root, dirs, file in os.walk(path):
                for item in file:
                    if (item.endswith('.dcm')):
                        orig_dcm_file_paths.append(os.path.join(root, item))
                        orig_dcm_index.append(int(item.split(".")[-2]))
            initial_index = min(orig_dcm_index)
            sorted_dcm_file_paths = [None for i in range(len(orig_dcm_file_paths))]
            # print(str(sorted_dcm_file_paths) + "Dicom files found")
            # the way to sort image
            for file_path in orig_dcm_file_paths:
                index = int(file_path.split(".")[-2])
                sorted_dcm_file_paths[index - initial_index] = file_path

            img_num_lst = []
            slloc_lst = []
            n_slice = len(sorted_dcm_file_paths)
            for fn in sorted_dcm_file_paths:
                dataset = dicom.read_file(fn)
                # Read the first dicom to get the image size
                row = dataset[0x0028, 0x0010].value
                col = dataset[0x0028, 0x0011].value
                slloc = abs(float(dataset[0x0020, 0x0032].value[2]))  # dcm_file[0x0020, 0x1041].value
                slloc_lst.append(slloc)
                rescale_intercept = dataset[0x0028, 0x1052].value
                rescale_slope = dataset[0x0028, 0x1053].value
                try:
                    slth = dataset[0x0018, 0x0050].value
                    pxsz = dataset[0x0028, 0x0030].value
                except:
                    try:
                        slth = dataset[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0018, 0x0050].value
                        pxsz = dataset[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
                    except:
                        slth = 1
                        pxsz = 1
                try:
                    slsp = dataset[0x7005, 0x1047].value
                except:
                    try:
                        slsp = dataset[0x7005, 0x1022].value
                    except:
                        slsp = slth

            array_data = np.ndarray(shape=(len(sorted_dcm_file_paths), row, col), dtype=np.float64)

            # Sort the dicoms and assign it to numpy array
            # Find right index for Image Number
            for fn in sorted_dcm_file_paths:
                dcm_file = dicom.read_file(fn)
                image_number = dcm_file[0x0020, 0x0013].value
                img_num_lst.append(image_number)
            min_idx = min(img_num_lst)
            for fn in sorted_dcm_file_paths:
                dcm_file = dicom.read_file(fn)
                image_number = dcm_file[0x0020, 0x0013].value
                pixel_data = dcm_file.pixel_array.astype(float)
                array_data[image_number - min_idx, :, :] = pixel_data * rescale_slope + rescale_intercept

        # for 2d CT images
        if (len(slloc_lst) != 0):
            slsp = abs(slloc_lst[-2] - slloc_lst[-1])

        x_spacing_orig, y_spacing_orig = float(pxsz[0]), float(pxsz[1])

        if (iso != -1):
            x_isotropic = iso / x_spacing_orig  # 1 mm

        SIZE = int(array_data.shape[1] / x_isotropic + 0.5)  # 1.0 mm

        if (n_slice == 1):
            array_data = resize(array_data, (SIZE, SIZE), order=1, mode='reflect', preserve_range=True,
                                anti_aliasing=0)
        else:
            z_spacing_orig = float(slsp)
            if (iso != -1):
                z_isotropic = iso / z_spacing_orig
            ZDIM = int(array_data.shape[0] / z_isotropic + 0.5)
            array_data = resize(array_data, (ZDIM, SIZE, SIZE), order=1, mode='reflect', preserve_range=True,
                                anti_aliasing=0)

        return Image(path=path, dataset=dataset, array_data=array_data, imageType=image_type,
                     imageFileType=ImageFileType.DICOM, phase=phase)

    @staticmethod
    def read_case(paths: list[str], image_file_types: list[ImageFileType], phase_indexes: list[int],
                  image_type: ImageType,
                  iso: int = 1) -> list[Image]:
        images = []
        for i, path in enumerate(paths):
            image_file_type = image_file_types[i]
            phase = phase_indexes[i]
            path = os.path.normpath(path)

            imageReader = ImageReader.generate_image_reader(image_file_type)
            images.append(imageReader(path, image_type, phase))

        images = sorted(images, key=lambda image: image.phase)

        return images


def main():
    pass


if __name__ == "__main__":
    main()
