
import numpy as np
from src_CINE.Dicom_helper.helper_funcs import label_dict
import os
from src_CINE.file_loader import file_loader
import re
from src_CINE.Dicom_eval.excel import dicts_to_excel
import glob
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


if __name__ == '__main__':
    #NOTE: output file's dirs depends on MASK_DIRS
    """
    CASES_DIR = "D:/DATA/CTA_CINE/original_CTA_CINE/"
    CASE_DIRS = [os.path.join(CASES_DIR, dir_name) for dir_name in os.listdir(CASES_DIR) if dir_name.startswith("CT5000205012_20140710")]
    MASK_DIRS = [f"D:/DATA/CTA_CINE/eval/{dir_name}_2mm/out/regis/Reg_all/Deep_Heart_checkpoints_ML_CINE_15/labels/2021-06-24/08-38-49" for
                 dir_name in os.listdir(CASES_DIR) if dir_name.startswith("CT5000205012_20140710")]
    EXCEL_NAMES = [f"size_{dir_name}_Reg_all_Deep_Heart_checkpoints_ML_CINE_15_2021-06-24-08-38-49" for dir_name in
                   os.listdir(CASES_DIR) if dir_name.startswith("CT5000205012_20140710")]
    SEGMENT_NAMES = ["LV","LA","RV","RA","LVM","WH"]

    SEGMENT_NAMES = ["LV", "LA", "RV", "RA", "LVM", "WH"]
    """
    RES = 1
    CASE_DIRS = ["D:/DATA/CTA_CINE/original_CTA_CINE/CT5000818554_20210701_20P","D:/DATA/CTA_CINE/original_CTA_CINE/CT5000818554_20210701_33P"]
    case_names = [os.path.normpath(case_dir).split("\\")[-1] for case_dir in CASE_DIRS]
    #temp_dirs = [f"D:/DATA/CTA_CINE/eval/{case_name}/out/1mm/regis/Reg_all/CMACS_default_parameters_26/labels/" for case_name in case_names]
    temp_dirs = [f"D:/DATA/CTA_CINE/eval/CT5000818554_20210701_20P/out/1mm/regis/Reg_all/Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_15/labels/", "D:/DATA/CTA_CINE/eval/CT5000818554_20210701_33P/out/1mm/regis/Reg_all/Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_26/labels/"]
    MASK_DIRS = [max(glob.glob(os.path.join(os.path.normpath(temp_dir), '*/*/')), key=os.path.getmtime) for temp_dir in temp_dirs]
    dates = [mask_dir.strip("\\").split("\\")[-2] for mask_dir in MASK_DIRS]
    times = [mask_dir.strip("\\").split("\\")[-1] for mask_dir in MASK_DIRS]
    EXCEL_NAMES = [
                   f"size_{case_names[0]}_{RES}mm_Reg_all_Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_15_{dates[0]}-{times[0]}",f"size_{case_names[1]}_{RES}mm_Reg_all_Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_26_{dates[0]}-{times[0]}"
                   ]
    SEGMENT_NAMES = ["LV", "LA", "RV", "RA", "LVM", "WH"]

    print("CASE_DIRS:",CASE_DIRS)
    print("MASK_DIRS:",MASK_DIRS)
    print("EXCEL_NAMES:", EXCEL_NAMES)
    print("SEGMENT_NAMES:", SEGMENT_NAMES)

    a = "X"
    while(a != "N" and a != "Y"):
        a = input("CONTINUE? (Y/N): ")


    if (a.lower() == "n"):
        raise Exception('correct your input!')

    assert len(MASK_DIRS) == len(
        EXCEL_NAMES), f"the number of element in MASK_DIRS is {len(MASK_DIRS)} while in EXCEL_NAMES is {len(EXCEL_NAMES)}"

    for index in range(len(MASK_DIRS)):
        N_PHASES = len(os.listdir(CASE_DIRS[index]))
        mask_dir = MASK_DIRS[index]
        excel_name = EXCEL_NAMES[index]
        segment_size_dataset_dict = {}
        for segment_name in SEGMENT_NAMES:
            segment_size_dataset_dict[segment_name] = [None for i in range(N_PHASES)]
        sorted_grayscale_phase_addresses = [None for i in range(N_PHASES)]
        sorted_mask_phase_addresses = [None for i in range(N_PHASES)]
        phase_names = [None for i in range(N_PHASES)]
        info_txt_dir = ""
        for entry in os.listdir(mask_dir):
            if (entry.endswith("info.txt")):
                info_txt_dir = os.path.join(mask_dir, entry)
            elif (re.split("[._]",entry)[-3].isdigit()):
                sorted_mask_phase_addresses[int(re.split("[._]",entry)[-3]) - 1] = os.path.join(mask_dir, entry)
                mask, _ = file_loader.read_label(os.path.join(mask_dir, entry))
                segment_size_dict = get_segment_size(mask, SEGMENT_NAMES)
                for segment_name in SEGMENT_NAMES:
                    segment_size_dataset_dict[segment_name][int(re.split("[._]",entry)[-3]) - 1] = segment_size_dict[segment_name]
                phase_names[int(re.split("[._]",entry)[-3]) - 1] = entry

        segment_size_dataset_dict['study id'] = phase_names
        DATE = mask_dir.strip("\\").split("\\")[-2]
        TIME = mask_dir.strip("\\").split("\\")[-1]
        out_dir = "\\".join(mask_dir.strip("\\").split("\\")[:-3]) + f"/debug/size/{DATE}/"
        dicts_to_excel([segment_size_dataset_dict],["size"],out_dir=out_dir,excel_name=excel_name)

        if(info_txt_dir):
            copyfile(info_txt_dir,os.path.join(out_dir, "info.txt"))