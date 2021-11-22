from scipy.stats import variation
import numpy as np
from src_CINE.Dicom_helper.helper_funcs import label_dict
import os
from src_CINE.file_loader import file_loader
import re
from src_CINE.Dicom_eval.excel import dicts_to_excel
import datetime
from shutil import copyfile


def get_CV_dataset(labels, grayscales):
    LV_CV = []
    LA_CV = []
    RV_CV = []
    RA_CV = []
    LVM_CV = []
    WH_CV = []
    for index in range(len(labels)):
        label = labels[index]
        grayscale = grayscales[index]

        if(label is None or grayscale is None):
            LV_CV.append(-1)
            LA_CV.append(-1)
            RV_CV.append(-1)
            RA_CV.append(-1)
            LVM_CV.append(-1)
            WH_CV.append(-1)
            continue

        z, x, y = np.where(label == label_dict["LV"])
        LV = grayscale[z,x,y].flatten()
        z, x, y = np.where(label == label_dict["LA"])
        LA = grayscale[z, x, y].flatten()
        z, x, y = np.where(label == label_dict["RV"])
        RV = grayscale[z, x, y].flatten()
        z, x, y = np.where(label == label_dict["RA"])
        RA = grayscale[z, x, y].flatten()
        z, x, y = np.where(label == label_dict["LVM"])
        LVM = grayscale[z, x, y].flatten()

        z, x, y = np.where((label == label_dict["WH"]) | (label == label_dict["LVM"]) |
                           (label == label_dict["LV"]) | (label == label_dict["AO"]) |
                           (label == label_dict["RV"]) | (label == label_dict["LA"]) |
                           (label == label_dict["LAA"]) | (label == label_dict["RA"]) |
                           (label == label_dict["PA"]) | (label == label_dict["SVC"]))
        WH = grayscale[z, x, y].flatten()

        LV_CV.append(variation(LV))
        LA_CV.append(variation(LA))
        RV_CV.append(variation(RV))
        RA_CV.append(variation(RA))
        LVM_CV.append(variation(LVM))
        WH_CV.append(variation(WH))
    data_dict = {
        'LV': LV_CV,
        'LA': LA_CV,
        'RV': RV_CV,
        'RA': RA_CV,
        'LVM': LVM_CV,
        'WH': WH_CV
    }

    return data_dict


def segmented_path(segmentation_type):
    path = None
    if (segmentation_type.startswith("reg_all")):
        path = "regis/reg_all/" + segmentation_type[8:]
    elif (segmentation_type == "manual_label"):
        path = "manual_label"
    else:
        if (segmentation_type.startswith("Deep_Heart")):
            path = "Deep_Heart/" + segmentation_type[11:]
        elif (segmentation_type.startswith("CMACS")):
            path = "CMACS/" + segmentation_type[6:]

    assert path is not None, "invalid segmentation type"

    return path

if __name__ == '__main__':
    CASE_DIRS = ["D:\DATA\SQUEEZ1\eval\SQUEEZ_0003_1\SQUEEZ_0003_1_1mm_nii"]
    case_names = [os.path.normpath(case_dir).split("\\")[-1] for case_dir in CASE_DIRS]
    RES = 1
    now = datetime.datetime.now()
    date = now.strftime("%Y-%m-%d-%H-%M")
    EXCEL_NAMES = [f"CV_{case_names[0]}_{RES}mm_manual_label_{date}"]
    SEGMENT_NAMES = ["LV", "LA", "RV", "RA", "LVM", "WH"]
    SEGMENTATION_TYPES = ["manual_label"]

    """
    SEGMENTATION_TYPES
    1. "manual_label"                                           for for manual label
    2. "Deep_Heart" + "_" + check point                         for deep heart segmentation
    3. "CMACS" + "_" + "default_parameters"                     for CMACS segmentation using default parameters
    4. "reg_all" + "_" + either option 1,2 or 3 + "_" + m_phase for using registration for segmentation
    """

    print("CASE_DIRS:", CASE_DIRS)
    print("EXCEL_NAMES:", EXCEL_NAMES)
    print("SEGMENT_NAMES:", SEGMENT_NAMES)
    print("SEGMENTATION_TYPES:", SEGMENTATION_TYPES)

    a = "X"
    while (a != "N" and a != "Y"):
        a = input("CONTINUE? (Y/N): ")
        if (a.lower() == "y"):
            break

        elif (a.lower() == "n"):
            raise Exception('correct your input!')

    assert len(CASE_DIRS) == len(
        EXCEL_NAMES), f"the number of element in CASE_DIRS is {len(CASE_DIRS)} while in EXCEL_NAMES is {len(EXCEL_NAMES)}"

    assert len(CASE_DIRS) == len(
        SEGMENTATION_TYPES), f"the number of element in CASE_DIRS is {len(CASE_DIRS)} while in SEGMENTATION_TYPES is {len(SEGMENTATION_TYPES)}"

    for index in range(len(CASE_DIRS)):
        N_PHASES = len(os.listdir(CASE_DIRS[index]))
        case_dir = CASE_DIRS[index]
        excel_name = EXCEL_NAMES[index]
        segmentation_key = SEGMENTATION_TYPES[index]
        if (segmentation_key == "manual_label"):
            segmentation_key = "Label.nii.gz"
        sorted_grayscale_phase_addresses = [None for i in range(N_PHASES)]
        segment_size_dataset_dict = {}
        sorted_mask_phase_addresses = [None for i in range(N_PHASES)]
        phase_names = [None for i in range(N_PHASES)]

        for phase in os.listdir(case_dir):
            phase_index = phase.rsplit(".", 1)[1]
            if (phase_index.isdigit()):
                phase_index = int(phase_index)
                for file in os.listdir(os.path.join(case_dir, phase)):
                    if("Image.nii.gz" in file):
                        sorted_grayscale_phase_addresses[phase_index - 1] = os.path.join(case_dir,phase, file)
                        phase_names[phase_index - 1] = phase

        grayscales_list, grayscale_phases_addresses = file_loader.load_data(sorted_grayscale_phase_addresses, "nifti")

        for phase in os.listdir(case_dir):
            phase_index = phase.rsplit(".", 1)[1]
            if (phase_index.isdigit()):
                phase_index = int(phase_index)
                for file in os.listdir(os.path.join(case_dir, phase)):
                    if (segmentation_key in file):
                        sorted_mask_phase_addresses[phase_index - 1] = os.path.join(case_dir, phase, file)

        masks_list, mask_phases_addresses = file_loader.load_data(sorted_mask_phase_addresses, "nifti")

        CV_dataset_dict = get_CV_dataset(masks_list, grayscales_list)
        CV_dataset_dict['study id'] = phase_names

        DATE = now.strftime("%Y-%m-%d")
        segmented_path = segmented_path(SEGMENTATION_TYPES[index])
        out_dir = os.path.dirname(case_dir) + "/out" + f"/{RES}mm" + f"/{segmented_path}" + f"/debug/Coefficient_Variation/{DATE}/"
        out_dir = os.path.normpath(out_dir)
        dicts_to_excel([CV_dataset_dict],["CV"],out_dir=out_dir,excel_name=excel_name)