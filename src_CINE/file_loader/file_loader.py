from PyQt5.QtCore import *
from multiprocessing import Pool, Process, Lock
import numpy as np
import SimpleITK as sitk
from src_CINE.Dicom_helper import dcm_reader
import re

class LoadDataWorker(QRunnable):
    def __init__(self, fn, files_list_address_dict, files_type, gray_scales_list, labels_list, *args, **kwargs):
        super(LoadDataWorker, self).__init__()
        self.fn = fn
        self.files_list_address_dict = files_list_address_dict
        self.files_type = files_type
        self.gray_scales_list = gray_scales_list
        self.labels_list = labels_list
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''
        directories = None
        if (self.files_type == "dicom" or self.files_type == "nifti"):
            self.gray_scales_list, directories = load_data(self.files_list_address_dict['gray_scale'], self.files_type)
        elif (self.files_type == "label"):
            self.labels_list, directories = load_data(self.files_list_address_dict['label'], self.files_type)

        self.fn(self.gray_scales_list, self.labels_list, directories, self.files_type)


def load_data(files_list_addresses, files_type):
    print("load_data")
    processes = []
    if (files_type.lower() == "dicom"):
        with Pool() as p:
            packed_data = p.map(read_dicom_case, files_list_addresses)
            gray_scales_list = [a[0] for a in packed_data]
            directories = [a[1] for a in packed_data]
        return gray_scales_list, directories
    elif (files_type.lower() == "nifti"):
        with Pool() as p:
            packed_data = p.map(read_nifti, files_list_addresses)
            gray_scales_list = [a[0] for a in packed_data]
            directories = [a[1] for a in packed_data]
        return gray_scales_list, directories
    elif (files_type.lower() == "label"):
        with Pool() as p:
            packed_data = p.map(read_label, files_list_addresses)
            labels_list = [a[0] for a in packed_data]
            directories = [a[1] for a in packed_data]
        return labels_list, directories

def read_dicom_case(address):
    if(address is None):
        return None, address
    vol = dcm_reader.read_cases(address)
    return np.copy(vol), address


def read_nifti(address):
    if (address is None):
        return None, address
    gold_itk = sitk.ReadImage(address)
    vol = sitk.GetArrayViewFromImage(gold_itk)
    return np.copy(vol), address


def read_label(address):
    if (address is None):
        return None, address
    gold_itk = sitk.ReadImage(address, sitk.sitkInt16)
    vol = sitk.GetArrayViewFromImage(gold_itk)
    return np.copy(vol), address

def filter(file_names, file_type):
    if(file_type == "label"):
        return file_names
    filtered_list = []
    for name in file_names:
        if (re.split('[\' .,()_]', name)[-1].isdigit()):
            filtered_list.append(name)

    return filtered_list

def sort_files_list(unordered_files_list, file_type):
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