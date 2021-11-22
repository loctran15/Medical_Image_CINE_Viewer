
import numpy as np
from src_CINE.Dicom_helper.helper_funcs import label_dict
import os
from src_CINE.file_loader import file_loader
import re
from src_CINE.Dicom_eval.excel import dicts_to_excel
import datetime
from shutil import copyfile

def get_segment_size(mask, segment_names):
    segment_size_dict = {}
    for segment_name in segment_names:
        if(segment_name == "WH"):
            z, x, y = np.where((mask == label_dict["WH"]) | (mask == label_dict["LVM"]) |
                               (mask == label_dict["LV"]) | (mask == label_dict["AO"]) |
                               (mask == label_dict["RV"]) | (mask == label_dict["LA"]) |
                               (mask == label_dict["LAA"]) | (mask == label_dict["RA"]) |
                               (mask == label_dict["PA"]) | (mask == label_dict["SVC"]))
        else:
            z, x, y = np.where(mask == label_dict[segment_name])
        segment_size_dict[segment_name] = len(z) / 1000
    return segment_size_dict


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
    EXCEL_NAMES = [f"size_{case_names[0]}_{RES}mm_manual_label_{date}"]
    SEGMENT_NAMES = ["LV", "LA", "RV", "RA", "LVM", "WH"]
    SEGMENTATION_TYPES = ["manual_label"]

    """
    SEGMENTATION_TYPES
    1. "manual_label"                                           for for manual label
    2. "Deep_Heart" + "_" + check point                         for deep heart segmentation
    3. "CMACS" + "_" + "default_parameters"                     for CMACS segmentation using default parameters
    4. "reg_all" + "_" + either option 1,2 or 3 + "_" + m_phase for using registration for segmentation
    """

    print("CASE_DIRS:",CASE_DIRS)
    print("EXCEL_NAMES:", EXCEL_NAMES)
    print("SEGMENT_NAMES:", SEGMENT_NAMES)
    print("SEGMENTATION_TYPES:", SEGMENTATION_TYPES)

    a = "X"
    while(a != "N" and a != "Y"):
        a = input("CONTINUE? (Y/N): ")
        if(a.lower() == "y"):
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
        segment_size_dataset_dict = {}
        for segment_name in SEGMENT_NAMES:
            segment_size_dataset_dict[segment_name] = [None for i in range(N_PHASES)]
        sorted_mask_phase_addresses = [None for i in range(N_PHASES)]
        phase_names = [None for i in range(N_PHASES)]

        for phase in os.listdir(case_dir):
            phase_index = phase.rsplit(".", 1)[1]
            if(phase_index.isdigit()):
                phase_index = int(phase_index)
                for file in os.listdir(os.path.join(case_dir,phase)):
                    if(segmentation_key in file):
                        sorted_mask_phase_addresses[phase_index - 1] = os.path.join(case_dir,phase,file)
                        mask, _ = file_loader.read_label(sorted_mask_phase_addresses[phase_index - 1])
                        segment_size_dict = get_segment_size(mask, SEGMENT_NAMES)
                        for segment_name in SEGMENT_NAMES:
                            segment_size_dataset_dict[segment_name][phase_index - 1] = segment_size_dict[segment_name]
                        phase_names[phase_index - 1] = phase

        segment_size_dataset_dict['study id'] = phase_names
        DATE = now.strftime("%Y-%m-%d")
        segmented_path = segmented_path(SEGMENTATION_TYPES[index])
        out_dir = os.path.dirname(case_dir) + "/out" + f"/{RES}mm" + f"/{segmented_path}" + f"/debug/size/{DATE}/"
        out_dir = os.path.normpath(out_dir)
        dicts_to_excel([segment_size_dataset_dict],["size"],out_dir=out_dir,excel_name=excel_name)

