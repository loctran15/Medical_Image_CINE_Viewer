from scipy.stats import variation
import numpy as np
from src_CINE.Dicom_helper.helper_funcs import label_dict
import os
from src_CINE.file_loader import file_loader
import re
from src_CINE.Dicom_eval.excel import dicts_to_excel
import glob
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

#TODO: write main function
if __name__ == '__main__':

    # NOTE: output file's dirs depends on CASE_DIRS, and MASK_DIRS
    RES = 1
    CASE_DIRS = ["D:/DATA/CTA_CINE/original_CTA_CINE/CT5000818554_20210701_20P","D:/DATA/CTA_CINE/original_CTA_CINE/CT5000818554_20210701_33P"]
    case_names = [os.path.normpath(case_dir).split("\\")[-1] for case_dir in CASE_DIRS]
    #temp_dirs = [f"D:/DATA/CTA_CINE/eval/{case_name}/out/1mm/regis/Reg_all/CMACS_default_parameters_26/labels/" for case_name in case_names]
    temp_dirs = [f"D:/DATA/CTA_CINE/eval/CT5000818554_20210701_20P/out/1mm/regis/Reg_all/Deep_Heart_checkpoints_ML_CINE_15/labels/", "D:/DATA/CTA_CINE/eval/CT5000818554_20210701_33P/out/1mm/regis/Reg_all/Deep_Heart_checkpoints_ML_CINE_26/labels/"]
    MASK_DIRS = [max(glob.glob(os.path.join(os.path.normpath(temp_dir), '*/*/')), key=os.path.getmtime) for temp_dir in temp_dirs]
    dates = [mask_dir.strip("\\").split("\\")[-2] for mask_dir in MASK_DIRS]
    times = [mask_dir.strip("\\").split("\\")[-1] for mask_dir in MASK_DIRS]
    EXCEL_NAMES = [
                   f"CV_{case_names[0]}_{RES}mm_Reg_all_Deep_Heart_checkpoints_ML_CINE_15_{dates[0]}-{times[0]}",f"CV_{case_names[1]}_{RES}mm_Reg_all_Deep_Heart_checkpoints_ML_CINE_26_{dates[0]}-{times[0]}"
                   ]
    SEGMENT_NAMES = ["LV", "LA", "RV", "RA", "LVM", "WH"]

    print("CASE_DIRS:", CASE_DIRS)
    print("MASK_DIRS:", MASK_DIRS)
    print("EXCEL_NAMES:", EXCEL_NAMES)
    print("SEGMENT_NAMES:", SEGMENT_NAMES)

    a = "X"
    while (a != "N" and a != "Y"):
        a = input("CONTINUE? (Y/N): ")

    if (a.lower() == "n"):
        raise Exception('correct your input!')

    assert len(CASE_DIRS) == len(MASK_DIRS), f"the number of element in CASE_DIRS is {CASE_DIRS} while in MASK_DIRS is {MASK_DIRS}"
    assert len(CASE_DIRS) == len(
        EXCEL_NAMES), f"the number of element in CASE_DIRS is {CASE_DIRS} while in EXCEL_NAMES is {EXCEL_NAMES}"
    for index in range(len(CASE_DIRS)):
    #for index in [len(CASE_DIRS) - 1]:
        info_txt_dir = None
        N_PHASES = len(os.listdir(CASE_DIRS[index]))
        case_dir = CASE_DIRS[index]
        mask_dir = MASK_DIRS[index]
        excel_name = EXCEL_NAMES[index]
        sorted_grayscale_phase_addresses = [None for i in range(N_PHASES)]
        sorted_mask_phase_addresses = [None for i in range(N_PHASES)]
        phase_names = [None for i in range(N_PHASES)]
        for entry in os.listdir(case_dir):
            if (entry.split(".")[-1].isdigit()):
                sorted_grayscale_phase_addresses[int(entry.split(".")[-1]) - 1] = os.path.join(case_dir, entry)
                phase_names[int(entry.split(".")[-1]) - 1] = entry

        grayscales_list, grayscales_phases_addresses = file_loader.load_data(sorted_grayscale_phase_addresses, "dicom")

        for entry in os.listdir(mask_dir):
            if (entry.endswith("info.txt")):
                info_txt_dir = os.path.join(mask_dir, entry)
            elif (re.split("[._]",entry)[-3].isdigit()):
                sorted_mask_phase_addresses[int(re.split("[._]",entry)[-3]) - 1] = os.path.join(mask_dir, entry)

        masks_list, mask_phases_addresses = file_loader.load_data(sorted_mask_phase_addresses, "label")
        CV_dataset_dict = get_CV_dataset(masks_list,grayscales_list)
        CV_dataset_dict['study id'] = phase_names
        DATE = mask_dir.strip("\\").split("\\")[-2]
        TIME = mask_dir.strip("\\").split("\\")[-1]
        out_dir = "\\".join(mask_dir.strip("\\").split("\\")[:-3]) + f"\\debug\\CV\\{DATE}\\"
        dicts_to_excel([CV_dataset_dict],["CV"],out_dir=out_dir,excel_name=excel_name)

        if (info_txt_dir):
            copyfile(info_txt_dir, os.path.join(out_dir, "info.txt"))