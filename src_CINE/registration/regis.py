from PyQt5.QtCore import *
from multiprocessing import Pool
import numpy as np
from src_CINE.Dicom_helper.helper_funcs import label_dict
from src_CINE.Dicom_helper import save_mask
from itertools import repeat
from scipy import ndimage as ndi
import os
from src_CMACS.cmacs_bbox_detection import bbox_detection
from src_CMACS import cmacs_mcs
from src_CINE.file_loader import file_loader
import datetime
import time
import SimpleITK as sitk
from pathlib import Path
from skimage import segmentation, filters




Z, X, Y = 0, 0, 0


class RegistrationAllWorker(QRunnable):
    def __init__(self, fn, display_on_text_browser, labels_list, grayscales_list, moving_vol_index, bounding_box, *args,
                 **kwargs):
        super(RegistrationAllWorker, self).__init__()
        self.bounding_box = None
        self.display_on_text_browser = display_on_text_browser
        self.fn = fn
        self.labels_list = np.copy(labels_list)
        self.moving_vol_index = moving_vol_index
        self.args = args
        self.kwargs = kwargs
        self.grayscales_list = grayscales_list

    @pyqtSlot()
    def run(self):
        self.labels_list, indexes = get_registrations(self.labels_list, self.grayscales_list, self.moving_vol_index,
                                                      self.bounding_box)
        self.fn(self.labels_list)
        for index in indexes:
            self.display_on_text_browser(f"registered from {self.moving_vol_index + 1} to {index + 1}")


def get_registrations(labels_list, grayscales_list, moving_vol_index, bounding_box):
    indexes = list(range(len(labels_list)))
    with Pool() as p:
        packed_data = p.starmap(get_registration,
                                zip(repeat(grayscales_list[moving_vol_index]), repeat(labels_list[moving_vol_index]),
                                    grayscales_list, labels_list, indexes, repeat(bounding_box)))
        labels_list = [a[0] for a in packed_data]
        indexes = [a[1] for a in packed_data]
    return labels_list, indexes


def get_registration(m_vol_grayscale, m_vol_label, f_vol_grayscale, f_vol_label, m_index, f_index, debug_path, bounding_box=None, deform_field_out = True, BBOX = 1, RES = 1, refined = True):
    print(f"registering from phase {m_index + 1} to phase {f_index + 1}")
    if (f_vol_label is not None):
        if(deform_field_out):
            prerefined_labels_debug_path = debug_path.split("\\")
            prerefined_labels_debug_path[-3] = "pre-refined_labels"
            prerefined_labels_debug_path = "\\".join(prerefined_labels_debug_path)
            save_mask.save_mask(f_vol_label, f"phase_{(f_index + 1)}", prerefined_labels_debug_path)
        # return np.copy(f_vol_label), f_index
    else:
        if(BBOX == 1):
            if (bounding_box is not None):
                f_bbox = bounding_box
                m_bbox0 = bounding_box
            else:
                f_bbox, _, _ = bbox_detection(volume_dir=f_vol_grayscale, check_dir=3, debug_path=None, DEBUG=0)

                m_bbox0, _, _ = bbox_detection(volume_dir=m_vol_grayscale, check_dir=3, debug_path=None, DEBUG=0)


        reg_mask, warp1, rescale_fvol, fw_sub1,deform_field = cmacs_mcs.multi_registration_CINE(m_vol_orig=m_vol_grayscale,
                                                                              f_vol_orig=f_vol_grayscale,
                                                                              f_bbox=f_bbox, m_bbox=f_bbox,
                                                                              m_seg=m_vol_label, index=f_index,
                                                                              dll_path="D:/Medical_Image_Analyzer/CT_SEG/CMACS/", RES=RES,
                                                                              DLL="deeds", BBOX=BBOX, DEBUG=1, debug_path = f"{debug_path}/{f_index + 1}/")
        if (deform_field_out):
            mask = np.zeros(f_vol_grayscale.shape, dtype=np.uint8)
            z, x, y = np.where(f_bbox > 0)
            mask[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = reg_mask
            prerefined_labels_debug_path = debug_path.split("\\")
            prerefined_labels_debug_path[-3] = "pre-refined_labels"
            prerefined_labels_debug_path = "\\".join(prerefined_labels_debug_path)
            save_mask.save_mask(mask, f"phase_{(f_index + 1)}", prerefined_labels_debug_path)

            z, x, y = np.where(f_bbox > 0)
            deformation_field_numpy_path = debug_path.split("\\")
            deformation_field_numpy_path[-3] = "deformation_field\\temp_numpy_data"
            deformation_field_numpy_path = "\\".join(deformation_field_numpy_path)

            deformation_field_numpy_path = f"{deformation_field_numpy_path}\\df_from_{m_index + 1}_to_{(f_index + 1)}.npy"
            df_field = np.zeros((f_vol_grayscale.shape[0],f_vol_grayscale.shape[1],f_vol_grayscale.shape[2],3), dtype=np.float32)
            print(f"phase: {f_index + 1}, deform_field shape: {deform_field.shape}, reg_mask shape: {reg_mask.shape}")
            df_field[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = deform_field
            Path(os.path.split(deformation_field_numpy_path)[0]).mkdir(parents=True, exist_ok=True)
            np.save(deformation_field_numpy_path, df_field)
            print("saved to:", deformation_field_numpy_path)

    if(refined):
        if (f_vol_label is not None):
            mask = np.copy(f_vol_label)
        else:
            mask = np.zeros(f_vol_grayscale.shape, dtype=np.uint8)
            z, x, y = np.where(f_bbox > 0)
            mask[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = reg_mask

        refined_mask = np.zeros(f_vol_grayscale.shape, dtype=np.uint8)

        for segment_value in sorted(label_dict.values()):
            binary_mask = np.zeros(mask.shape, dtype=np.uint8)
            val_list = list(label_dict.values())
            key_list = list(label_dict.keys())
            position = val_list.index(segment_value)
            if (key_list[position] == "BOX"):
                """z, x, y = np.where((mask == label_dict["BOX"]) | (mask == label_dict["WH"]))
                binary_mask[z, x, y] = 1
                refined_mask[z, x, y] = segment_value"""
                continue
            if (key_list[position] == "LVM"):
                z, x, y = np.where((mask == label_dict["LVM"]) | (mask == label_dict["LV"]))
                binary_mask[z, x, y] = 1
            elif (key_list[position] == "WH"):
                z, x, y = np.where(
                    (mask == label_dict["WH"]) | (mask == label_dict["LVM"]) | (mask ==label_dict["LV"]) | (mask == label_dict["AO"]) | (
                            mask == label_dict["RV"]) | (mask == label_dict["LA"]) | (mask == label_dict["LAA"]) | (mask == label_dict["RA"]) | (
                            mask == label_dict["PA"]) | (mask == label_dict["SVC"]))
                binary_mask[z, x, y] = 1
            else:
                z, x, y = np.where(mask == segment_value)
                binary_mask[z, x, y] = 1

            if (key_list[position] == "LVM"):
                binary_mask = refine(f_vol_grayscale, binary_mask, -90)
            elif (key_list[position] == "LV"):
                case_name = debug_path.split("\\")[4]
                case_name = case_name.split("_")[0]
                debug_dir_PP = os.path.join("D:\DATA\MR-CT_comparison_cases",case_name,"1st_attempt",str(f_index + 1))
                binary_mask = refine(f_vol_grayscale, binary_mask, -90, "LV",refined_mask, debug_dir_PP)
            elif (len(set(binary_mask.flatten())) != 2):
                continue
            elif (key_list[position] == "WH"):
                binary_mask = refine(f_vol_grayscale, binary_mask, -400)
            elif (key_list[position] == "LA" or key_list[position] == "AO"):
                binary_mask = refine(f_vol_grayscale, binary_mask, 200)
            else:
                binary_mask = refine(f_vol_grayscale, binary_mask, -90)
            print("refined: " + key_list[position] + "segment value: " + str(segment_value))

            z, x, y = np.where(binary_mask == 1)
            refined_mask[z, x, y] = segment_value

        return np.copy(refined_mask), f_index
    else:
        if (f_vol_label is not None):
            mask = np.copy(f_vol_label)
        else:
            mask = np.zeros(f_vol_grayscale.shape, dtype=np.uint8)
            z, x, y = np.where(f_bbox > 0)
            mask[np.min(z):np.max(z) + 1, np.min(x):np.max(x) + 1, np.min(y):np.max(y) + 1] = reg_mask
        return np.copy(mask), f_index

def refine(volume=None, bin_mask=None, thres=None, label_name=None, seg_mask=None, debug_dir = None):
    step = 1
    z, x, y = np.where(volume < thres)
    bin_mask[z, x, y] = 0

    if (label_name == "LV"):
        z, x, y =  np.where(seg_mask==5)
        bin_mask[z, x, y] = 1
        if (debug_dir):
            save_mask.save_mask(bin_mask, f"{step}_LV_and_LVM_mask", debug_dir)
            step += 1
        #threshold the ring using threshold_otsu
        otsu_threshold_val = filters.threshold_otsu(volume[z, x, y])
        otsu_threshold_val = min(0.875 * otsu_threshold_val, 200)
        print("otsu_threshold_val: ", otsu_threshold_val)
        if (debug_dir):
            with open(f'{debug_dir}//threshold_val.txt', 'w') as f:
                f.write(f"threshold value: {otsu_threshold_val}")
        z, x, y = np.where(volume <= otsu_threshold_val)
        bin_mask[z, x, y] = 0
        if (debug_dir):
            save_mask.save_mask(bin_mask, f"{step}_threshold_mask", debug_dir)
            step += 1



    # Get largest object
    label_objects, nb_labels = ndi.label(bin_mask)
    sizes = np.bincount(label_objects.ravel())
    sorted_sizes = np.sort(sizes)[::-1]
    if len(sizes) > 2:
        max_sz = np.max(sorted_sizes[1:-1])
        mask_sizes = (sizes == max_sz)
        bin_mask = mask_sizes[label_objects].astype(dtype=np.int)
    elif len(sizes) == 2:
        mask_sizes = (sizes == sorted_sizes[1])
        bin_mask = mask_sizes[label_objects].astype(dtype=np.int)
    if (debug_dir):
        save_mask.save_mask(bin_mask, f"{step}_largest_object_mask", debug_dir)
        step += 1

    #Apply filters on the binary mask

    #remove small-isolated dots
    filter_kernel = np.zeros((5, 5, 5), dtype=np.float32)
    filter_kernel[1:4, 1:4, 1:4] = 0.75
    filter_kernel[2, 2, 2] = 1
    filtered_mask = ndi.convolve(bin_mask.astype(np.float32), filter_kernel, mode="nearest")
    z, x, y = np.where(filtered_mask < 0.3 * np.sum(filter_kernel))
    bin_mask[z, x, y] = 0
    if (debug_dir):
        save_mask.save_mask(bin_mask, f"{step}_removed_dots_mask", debug_dir)
        step += 1

    if (label_name == "LV"):
        #fill small holes and add some 1 pixels to the right of the array (axial view)
        filter_kernel = np.ones((5,5,5),dtype=np.float32)
        filter_kernel[:,:,:3] = 0.125
        filtered_mask = ndi.convolve(bin_mask.astype(np.float32), filter_kernel, mode="nearest")
        z,x,y = np.where(filtered_mask >= np.sum(filter_kernel)/2)
        filtered_mask = np.zeros(bin_mask.shape, dtype=np.int)
        filtered_mask[z,x,y] = 1
        bin_mask = bin_mask + filtered_mask
        bin_mask = (bin_mask >= 1).astype(np.int)
        if (debug_dir):
            save_mask.save_mask(bin_mask, f"{step}_additional_dots_mask", debug_dir)
            step += 1

        #restrict the expansion of the LV label. New LV mask will be inside (LVM + pre-refined LV) label
        if(seg_mask.any()):
            z,x,y = np.where(seg_mask != 5)
            bin_mask[z,x,y] = 0
            if (debug_dir):
                save_mask.save_mask(bin_mask, f"{step}_restricted_mask", debug_dir)
                step += 1

    #closing
    KERNEL_SIZE = 3
    r2 = np.arange(-KERNEL_SIZE, KERNEL_SIZE + 1) ** 2
    dist2 = r2[:, None, None] + r2[:, None] + r2
    s_e_sphere2 = (dist2 <= KERNEL_SIZE ** 2).astype(np.int)
    bin_mask = ndi.binary_closing(bin_mask, structure=s_e_sphere2).astype(dtype=np.uint8)
    if (debug_dir):
        save_mask.save_mask(bin_mask, f"{step}_closing_mask", debug_dir)
        step += 1

    # opening
    bin_mask = ndi.binary_opening(bin_mask, structure=np.ones((5, 5, 5))).astype(dtype=np.uint8)
    if (debug_dir):
        save_mask.save_mask(bin_mask, f"{step}_opening_mask", debug_dir)
        step += 1

    #fill_holes
    bin_mask = ndi.binary_fill_holes(bin_mask).astype(dtype=np.uint8)
    if (debug_dir):
        save_mask.save_mask(bin_mask, f"{step}_filled_holes_mask", debug_dir)
        step += 1


    if (debug_dir):
        save_mask.save_mask(bin_mask, f"{step}_final_mask", debug_dir)
        step += 1

    return bin_mask.astype(dtype=np.uint8)



if __name__ == "__main__":
    #regis all script
    # NOTE: output file's dirs depends on CASE_DIRS, REGIS_TYPE, moving_phase_index and segmentation_method
    DATE = datetime.date.today()
    now = datetime.datetime.now()
    RES = 1
    TIME = current_time = now.strftime("%H-%M-%S")
    REGIS_TYPE = "Reg_all"
    CASES_DIR = "D:/DATA/CTA_CINE/original_CTA_CINE"
    #CASES_DIRS = [os.path.join(CASES_DIR,CASE_NAME) for CASE_NAME in os.listdir(CASES_DIR) if CASE_NAME.startswith(("CT5000701682_20191024","CT5000398057_20160616"))]
    CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000666588_20190627",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000645713_20190409",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000637145_20190314",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000564630_20180503",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000469497_20170406",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000398057_20160616",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000701682_20191024",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000305661_20150709",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000305393_20150708",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000302027_20150625",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000603980_20181018_1",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000276823_20150521",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000289271_20150507",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000633055_20190215",
                  r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000289264_20150507",
               ]
    M_VOL_MASK_ADDRESSES = [r"D:\DATA\CTA_CINE\eval\CT5000666588_20190627\out\1mm\CMACS\default_parameters\labels\2021-06-09\08-06-04\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000645713_20190409\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000637145_20190314\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000564630_20180503\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000469497_20170406\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000398057_20160616\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000701682_20191024\out\1mm\CMACS\default_parameters\labels\2021-06-15\10-37-11\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000305661_20150709\out\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000305393_20150708\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000302027_20150625\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000603980_20181018_1\out\1mm\CMACS\default_parameters\labels\2021-06-15\10-37-11\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000276823_20150521\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000289271_20150507\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000633055_20190215\out\1mm\CMACS\default_parameters\labels\2021-06-15\10-37-11\phase_15.nii.gz",
                            r"D:\DATA\CTA_CINE\eval\CT5000289264_20150507\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz"
                            ]

    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000645713_20190409"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000645713_20190409\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz"]
    # #
    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000637145_20190314"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000637145_20190314\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz"]
    # #
    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000564630_20180503"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000564630_20180503\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz"]
    # #
    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000469497_20170406"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000469497_20170406\out\1mm\CMACS\default_parameters\labels\2021-05-29\10-07-28\phase_15.nii.gz"]
    # #
    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000701682_20191024"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000701682_20191024\out\1mm\CMACS\default_parameters\labels\2021-06-15\10-37-11\phase_15.nii.gz"]
    # #
    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000398057_20160616"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000398057_20160616\out\1mm\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz"]
    # #
    # CASES_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000305661_20150709"]
    # M_VOL_MASK_ADDRESSES = [
    #     r"D:\DATA\CTA_CINE\eval\CT5000305661_20150709\out\CMACS\default_parameters\labels\2021-06-08\09-36-41\phase_15.nii.gz"]

    moving_phase_indexs = [15,15,15,15,15,15,15,15,15,15,15,15,15,15,15]  # starts from 1
    segmentation_method = f"CMACS_default_parameters"
    info = False #output info.txt file for addtional information about the dataset
    f_phase_indexes = [8,7,9,8,9,8,9,9,7,9,9,9,8,8,9]
    #f_phase_indexes = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15]
    for index in range(len(CASES_DIRS)):
        moving_phase_index = moving_phase_indexs[index]
        segmentation_position_name = segmentation_method + "_" + str(moving_phase_index)
        case_dir = CASES_DIRS[index]
        n_phases = len(os.listdir(case_dir))
        m_vol_mask_address = M_VOL_MASK_ADDRESSES[index]
        assert os.path.split(os.path.normpath(case_dir))[-1] in m_vol_mask_address, "paths do not match"
        DATASET_NAME = os.path.split(os.path.normpath(case_dir))[-1]
        sorted_phases_addresses = [None for i in range(n_phases)]
        indexes = list(range(n_phases))
        for entry in os.listdir(case_dir):
            if (entry.split(".")[-1].isdigit()):
                sorted_phases_addresses[int(entry.split(".")[-1]) - 1] = os.path.join(case_dir,entry)

        grayscales_list, phases_addresses = file_loader.load_data(sorted_phases_addresses, "dicom")

        gold_itk = sitk.ReadImage(m_vol_mask_address, sitk.sitkInt16)
        m_vol_mask = sitk.GetArrayViewFromImage(gold_itk)
        masks_list = [None for i in range(n_phases)]
        masks_list[moving_phase_index - 1] = m_vol_mask
        bounding_box = None
        reg_debug_path = f"D:/DATA/CTA_CINE/eval/{DATASET_NAME}/out/{RES}mm/regis/{REGIS_TYPE}/{segmentation_position_name}/debug/labels/{DATE}/{TIME}"
        reg_debug_path = os.path.normpath(reg_debug_path)
        start = time.time()
        # with Pool() as p:
        #     packed_data = p.starmap(get_registration,
        #                             zip(repeat(grayscales_list[moving_phase_index - 1]),
        #                                 repeat(m_vol_mask),
        #                                 grayscales_list, masks_list,repeat(moving_phase_index - 1) ,indexes,repeat(reg_debug_path),repeat(bounding_box)))
        #     masks_list = [a[0] for a in packed_data]
        #     indexes = [a[1] for a in packed_data]

        """for i in indexes:
            get_registration(grayscales_list[moving_phase_index - 1],m_vol_mask,grayscales_list[i],masks_list[i],indexes[i],reg_debug_path, bounding_box)"""
        f_index = f_phase_indexes[index] - 1
        mask, index = get_registration(grayscales_list[moving_phase_index - 1], m_vol_mask, grayscales_list[f_index], masks_list[f_index], 14,
                         indexes[f_index], reg_debug_path, bounding_box,deform_field_out=False)
        phase_address = sorted_phases_addresses[f_index].split("\\")[-1]
        temp_list = phase_address.rsplit(".", 1)
        phase_address = f"{temp_list[0]}.{int(temp_list[1]):02d}"
        #out_dir = f"D:/DATA/CTA_CINE/eval/{DATASET_NAME}/{DATASET_NAME}_{RES}mm_nii/{phase_address}/"
        DATASET_NAME = DATASET_NAME.split("_")[0]
        out_dir = f"D:/DATA/MR-CT_comparison_cases/{DATASET_NAME}/{phase_address}"
        file_name = f"{REGIS_TYPE}_{segmentation_position_name}_label_1st_attempt"
        save_mask.save_mask(mask, file_name, out_dir)

        end = time.time()
        print("time interval: ", end - start)

        # for ind in range(len(masks_list)):
        #     mask = masks_list[ind]
        #     phase_index = indexes[ind] + 1
        #     out_dir = f"D:/DATA/CTA_CINE/eval/{DATASET_NAME}/out/{RES}mm/regis/{REGIS_TYPE}/{segmentation_position_name}/labels/{DATE}/{TIME}/"
        #     file_name = f"phase_{phase_index:02d}"
        #     save_mask.save_mask(mask, file_name, out_dir)
        #
        # if (info):
        #     # LinearWrap parameters
        #     # out_txt = "default"
        #
        #     out_txt = ""
        #     grid_spacing = np.asarray([7, 6, 5, 4], dtype=np.float32)
        #     search_radius = np.asarray([5, 4, 3, 2], dtype=np.float32)
        #     quantisation = np.asarray([4, 3, 2, 1], dtype=np.float32)
        #     levels = 4
        #     qc = 2
        #     alpha = 0.1
        #     out_txt = f"linearWrap:\n\tgrid_spacing: {grid_spacing}\n\tsearch_radius: {search_radius}\n\tquantisation: {quantisation}\n\tlevels: {levels}\n\tqc: {qc}\n\talpha: {alpha}\n\n\n"
        #
        #     # DeedsWrap parameters
        #     qc = 1
        #     alpha = 0.1
        #     levels = 5
        #     grid_spacing = np.asarray([8, 6, 4, 2], dtype=np.float32)
        #     search_radius = np.asarray([8, 6, 4, 2], dtype=np.float32)
        #     quantisation = np.asarray([7.5, 1.0, 1.0, 1.0], dtype=np.float32)
        #     out_txt = out_txt + f"DeedsWrap:\n\tgrid_spacing: {grid_spacing}\n\tsearch_radius: {search_radius}\n\tquantisation: {quantisation}\n\tlevels: {levels}\n\tqc: {qc}\n\talpha: {alpha}"
        #
        #     out_txt_dir = os.path.join(out_dir, "info.txt")
        #     f = open(out_txt_dir, "w")
        #     f.write(out_txt)
        #     f.close()
        #
        # for ind in range(len(masks_list)):
        #     mask = masks_list[ind]
        #     phase_address = sorted_phases_addresses[ind].split("\\")[-1]
        #     temp_list = phase_address.rsplit(".", 1)
        #     phase_address = f"{temp_list[0]}.{int(temp_list[1]):02d}"
        #     out_dir = f"D:/DATA/CTA_CINE/eval/{DATASET_NAME}/{DATASET_NAME}_{RES}mm_nii/{phase_address}/"
        #     #file_name = f"{REGIS_TYPE}_{segmentation_position_name}_label"
        #     file_name = f"{REGIS_TYPE}_{segmentation_position_name}_label_1try"
        #     save_mask.save_mask(mask, file_name, out_dir)
