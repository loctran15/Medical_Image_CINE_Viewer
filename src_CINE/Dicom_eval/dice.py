__author__ = "Vy Bui"
__email__ = "01bui@cua.edu"

import os
import numpy as np
import pandas as pd
from medpy.metric import *
import SimpleITK as sitk
import datetime
from src_CINE.Dicom_helper.helper_funcs import label_dict
now = datetime.datetime.now()

#NOTE: output file's dirs depends on CASE_DIRS and METHOD_NAMES
CASE_DIRS = ["D:/DATA/SQUEEZ/eval/SQUEEZ_0002_2/SQUEEZ_0002_2_1mm_nii","D:/DATA/SQUEEZ/eval/SQUEEZ_0002_2/SQUEEZ_0002_2_1mm_nii","D:/DATA/SQUEEZ/eval/SQUEEZ_0002_2/SQUEEZ_0002_2_1mm_nii",
             "D:/DATA/SQUEEZ/eval/SQUEEZ_0005_1/SQUEEZ_0005_1_1mm_nii","D:/DATA/SQUEEZ/eval/SQUEEZ_0005_1/SQUEEZ_0005_1_1mm_nii","D:/DATA/SQUEEZ/eval/SQUEEZ_0005_1/SQUEEZ_0005_1_1mm_nii",] #modify before using
METHOD_NAMES = ["Reg_all_CMACS_default_parameters_15","Reg_all_Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_15","Reg_all_Deep_Heart_checkpoints_ML_CINE_15",
                "Reg_all_CMACS_default_parameters_15","Reg_all_Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_15","Reg_all_Deep_Heart_checkpoints_ML_CINE_15"]

assert len(CASE_DIRS) == len(METHOD_NAMES), f"the length of currdirs is {CASE_DIRS} while the length of method_list is {METHOD_NAMES}"

id = 0
for currdir in CASE_DIRS[id:]:
    if(METHOD_NAMES[id].startswith("Deep_Heart")):
        out_dir_extension = METHOD_NAMES[id][:10] +"/"+ METHOD_NAMES[id][11:] +  "/debug/dice/"
    elif (METHOD_NAMES[id].startswith("CMACS")):
        out_dir_extension = METHOD_NAMES[id][:5] + "/" + METHOD_NAMES[id][6:] + "/debug/dice/"
    elif (METHOD_NAMES[id].startswith("Reg")):
        out_dir_extension = "regis" + "/" + METHOD_NAMES[id][:7] + "/" + METHOD_NAMES[id][8:] + "/debug/dice/"

    xout_dir = "/".join(currdir.split("/")[:-1]) + "/out/" + out_dir_extension + now.strftime("%Y-%m-%d") + "/"
    if not os.path.exists(xout_dir):
        os.makedirs(xout_dir)
    name = "dice_" + "_".join(currdir.split("/")[-1].split("_")[:3]) +"_" + METHOD_NAMES[id] + "_" #excel file name
    folder_name = METHOD_NAMES[id] + "_evaluation"

    writer = pd.ExcelWriter(xout_dir + name + now.strftime("%Y-%m-%d-%H-%M") + '.xlsx')
    print(currdir)
    print(xout_dir + name + now.strftime("%Y-%m-%d-%H-%M") + '.xlsx')
    dirs = os.listdir(currdir)
    dirs.sort(key = lambda x:int(x.split('.')[-1]))
    TOTAL = len(dirs)
    # Empty numpy arrays to hold the results
    dice_result = np.zeros((TOTAL, 6))  # Dice
    hd_results = np.zeros((TOTAL, 5))  # Haurdoff distance
    msd_results = np.zeros((TOTAL, 5))  # Mean surface distance
    case_num = 0
    study_id = []
    gold_dir = "nan"
    for p in dirs:
        nhlbi_dir = ""
        for root, dirs, file in os.walk(os.path.join(currdir, p)):
            for item in file:
                label_file_name = METHOD_NAMES[id] + "_label.nii.gz"
                if item == label_file_name: #modify before using
                    nhlbi_dir = os.path.join(root, item)
                elif item.endswith('_Label.nii.gz'): # Point to ground truth #modify before using
                    out_dir = root
                    gold_dir = os.path.join(root, item)
        if(nhlbi_dir == ""):
            break
        print(gold_dir)
        study_id.append(p)
        gold_itk = sitk.ReadImage(gold_dir, sitk.sitkUInt16)
        gold_volume = sitk.GetArrayViewFromImage(gold_itk)
        nhlbi_itk = sitk.ReadImage(nhlbi_dir, sitk.sitkUInt16)
        nhlbi_volume = sitk.GetArrayViewFromImage(nhlbi_itk)

        print(gold_volume.shape, nhlbi_volume.shape)

        """Get 7 structures in stacom ground truth data"""
        gold_4c = np.zeros(gold_volume.shape, dtype=np.uint16)
        gold_RV = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["RV"])
        gold_RV[z, x, y] = 1
        gold_4c[z, x, y] = 1

        gold_LA = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["LA"])
        gold_LA[z, x, y] = 1
        gold_4c[z, x, y] = 1

        gold_RA = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["RA"])
        gold_RA[z, x, y] = 1
        gold_4c[z, x, y] = 1

        gold_MYO = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["LVM"])
        gold_MYO[z, x, y] = 1
        gold_4c[z, x, y] = 1

        gold_LV = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["LV"])
        gold_LV[z, x, y] = 1
        gold_4c[z, x, y] = 1

        gold_AO = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["AO"])
        gold_AO[z, x, y] = 1

        gold_PA = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where(gold_volume == label_dict["PA"])
        gold_PA[z, x, y] = 1
        gold_4c[z, x, y] = 1

        gold_WH = np.zeros(gold_volume.shape, dtype=np.uint16)
        z, x, y = np.where((gold_volume == label_dict["WH"]) | (gold_volume == label_dict["LVM"]) |
                           (gold_volume == label_dict["LV"]) | (gold_volume == label_dict["AO"]) |
                           (gold_volume == label_dict["RV"]) | (gold_volume == label_dict["LA"]) |
                           (gold_volume == label_dict["LAA"]) | (gold_volume == label_dict["RA"]) |
                           (gold_volume == label_dict["PA"]) | (gold_volume == label_dict["SVC"]))
        gold_WH[z, x, y] = 1
        gold_4c[z, x, y] = 1

        """Get 7 structures in segmented data"""
        nhlbi_4c = np.zeros(nhlbi_volume.shape, dtype=np.uint16)

        nhlbi_RV = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["RV"])
        nhlbi_RV[z, x, y] = 1
        nhlbi_4c[z, x, y] = 1

        nhlbi_LA = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["LA"])
        nhlbi_LA[z, x, y] = 1
        nhlbi_4c[z, x, y] = 1

        nhlbi_RA = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["RA"])
        nhlbi_RA[z, x, y] = 1
        nhlbi_4c[z, x, y] = 1

        nhlbi_MYO = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["LVM"])
        nhlbi_MYO[z, x, y] = 1
        nhlbi_4c[z, x, y] = 1

        nhlbi_LV = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["LV"])
        nhlbi_LV[z, x, y] = 1
        nhlbi_4c[z, x, y] = 1

        nhlbi_AO = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["AO"])
        nhlbi_AO[z, x, y] = 1

        nhlbi_PA = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where(nhlbi_volume == label_dict["PA"])
        nhlbi_PA[z, x, y] = 1

        nhlbi_WH = np.zeros(nhlbi_volume.shape, dtype=np.uint16)
        z, x, y = np.where((nhlbi_volume == label_dict["WH"]) | (nhlbi_volume == label_dict["LVM"]) |
                           (nhlbi_volume == label_dict["LV"]) | (nhlbi_volume == label_dict["AO"]) |
                           (nhlbi_volume == label_dict["RV"]) | (nhlbi_volume == label_dict["LA"]) |
                           (nhlbi_volume == label_dict["LAA"]) | (nhlbi_volume == label_dict["RA"]) |
                           (nhlbi_volume == label_dict["PA"]) | (nhlbi_volume == label_dict["SVC"]))
        nhlbi_WH[z, x, y] = 1

        z, x, y = np.where(gold_PA > 0)
        nhlbi_PA[:np.min(z), :, :] = 0
        nhlbi_PA[:, :, :np.min(y)] = 0

        label_mask_cardiac = np.zeros(nhlbi_volume.shape, dtype=np.uint8)
        idx = np.where(nhlbi_MYO == 1)
        label_mask_cardiac[idx] = label_dict["LVM"]
        idx = np.where(nhlbi_AO == 1)
        label_mask_cardiac[idx] = label_dict["AO"]
        idx = np.where(nhlbi_LV == 1)
        label_mask_cardiac[idx] = label_dict["LV"]
        idx = np.where(nhlbi_RA == 1)
        label_mask_cardiac[idx] = label_dict["RA"]
        idx = np.where(nhlbi_PA > 0)
        label_mask_cardiac[idx] = label_dict["PA"]
        idx = np.where(nhlbi_RV > 0)
        label_mask_cardiac[idx] = label_dict["RV"]
        idx = np.where(nhlbi_LA > 0)
        label_mask_cardiac[idx] = label_dict["LA"]
        idx = np.where(nhlbi_WH > 0)
        label_mask_cardiac[idx] = label_dict["WH"]
        final_seg_itk = sitk.GetImageFromArray(label_mask_cardiac)

        if not os.path.exists(out_dir + "/" + folder_name):
            os.makedirs(out_dir + "/" + folder_name)

        sitk.WriteImage(final_seg_itk, out_dir + "/" + folder_name + '/F_cardiac_eval.nii.gz')

        label_mask_cardiac = np.zeros(nhlbi_volume.shape, dtype=np.uint8)
        idx = np.where(gold_MYO == 1)
        label_mask_cardiac[idx] = label_dict["LVM"]
        idx = np.where(gold_AO == 1)
        label_mask_cardiac[idx] = label_dict["AO"]
        idx = np.where(gold_LV == 1)
        label_mask_cardiac[idx] = label_dict["LV"]
        idx = np.where(gold_RA == 1)
        label_mask_cardiac[idx] = label_dict["RA"]
        idx = np.where(gold_PA > 0)
        label_mask_cardiac[idx] = label_dict["PA"]
        idx = np.where(gold_RV > 0)
        label_mask_cardiac[idx] = label_dict["RV"]
        idx = np.where(gold_LA > 0)
        label_mask_cardiac[idx] = label_dict["LA"]
        final_seg_itk = sitk.GetImageFromArray(label_mask_cardiac)
        sitk.WriteImage(final_seg_itk, out_dir + "/" + folder_name + '/F_cardiac_truth.nii.gz')

        ####################################################################################################################
        print("---Getting Dice---")
        #dice_result[case_num, 0] = binary.dc(nhlbi_AO, gold_AO)
        dice_result[case_num, 0] = binary.dc(nhlbi_LV, gold_LV)
        dice_result[case_num, 1] = binary.dc(nhlbi_LA, gold_LA)
        dice_result[case_num, 2] = binary.dc(nhlbi_RV, gold_RV)
        dice_result[case_num, 3] = binary.dc(nhlbi_RA, gold_RA)
        dice_result[case_num, 4] = binary.dc(nhlbi_MYO, gold_MYO)
        dice_result[case_num, 5] = binary.dc(nhlbi_WH, gold_WH)
        #dice_result[case_num, 6] = binary.dc(nhlbi_PA, gold_PA)
        print(dice_result[case_num, :])

        print("---Getting HD---")
        #hd_results[case_num, 0] = binary.hd(nhlbi_AO, gold_AO)
        hd_results[case_num, 0] = binary.hd(nhlbi_LV, gold_LV)
        hd_results[case_num, 1] = binary.hd(nhlbi_LA, gold_LA)
        hd_results[case_num, 2] = binary.hd(nhlbi_RV, gold_RV)
        hd_results[case_num, 3] = binary.hd(nhlbi_RA, gold_RA)
        hd_results[case_num, 4] = binary.hd(nhlbi_MYO, gold_MYO)
        #hd_results[case_num, 5] = binary.hd(nhlbi_WH, gold_WH)
        #hd_results[case_num, 6] = binary.hd(nhlbi_PA, gold_PA)
        print(hd_results[case_num, :])

        print("---Getting MSD---")
        #msd_results[case_num, 0] = binary.assd(nhlbi_AO, gold_AO)
        msd_results[case_num, 0] = binary.assd(nhlbi_LV, gold_LV)
        msd_results[case_num, 1] = binary.assd(nhlbi_LA, gold_LA)
        msd_results[case_num, 2] = binary.assd(nhlbi_RV, gold_RV)
        msd_results[case_num, 3] = binary.assd(nhlbi_RA, gold_RA)
        msd_results[case_num, 4] = binary.assd(nhlbi_MYO, gold_MYO)
        #msd_results[case_num, 5] = binary.assd(nhlbi_WH, gold_WH)
        #msd_results[case_num, 6] = binary.assd(nhlbi_PA, gold_PA)
        print(msd_results[case_num, :])
        case_num += 1
    id += 1
    # Graft our results matrix into pandas data frames
    dice_result_pd = pd.DataFrame(data=dice_result, index=study_id, columns=['LV', 'LA', 'RV', 'RA', 'LVM', 'WH'])
    hd_results_pd = pd.DataFrame(data=hd_results, index=study_id, columns=['LV', 'LA', 'RV', 'RA', 'LVM'])
    msd_results_pd = pd.DataFrame(data=msd_results, index=study_id, columns=['LV', 'LA', 'RV', 'RA', 'LVM'])

    dice_result_pd.to_excel(writer, 'Dice', startcol=0)
    hd_results_pd.to_excel(writer, 'HD', startcol=0)
    msd_results_pd.to_excel(writer, 'MSD', startcol=0)

    writer.save()
