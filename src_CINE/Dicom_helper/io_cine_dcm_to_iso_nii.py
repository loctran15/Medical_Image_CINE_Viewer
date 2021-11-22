import os
import SimpleITK as sitk
import numpy as np
from dcm_helper import dcm_reader
from skimage.transform import resize
from pathlib import Path

if __name__ == "__main__":
    #CASE_DIRS = ['D:/DATA/CTA_CINE/original/CT5000436748_20161117_HUDSON-KIMBERLEY-KAY']
    # CASES_DIR = "D:/DATA/CTA_CINE/original_CTA_CINE/"
    # CASE_DIRS = [os.path.join(CASES_DIR,dir_name) for dir_name in os.listdir(CASES_DIR) if dir_name.startswith("CT5000701682_20191024")]
    #CASE_DIRS = [r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000818554_20210701_33P",
    #             r"D:\DATA\CTA_CINE\original_CTA_CINE\CT5000818554_20210701_20P"]
    CASE_DIRS = ["D:\DATA\SQUEEZ\SQUEEZ_0003_1"]

    for case_dir in CASE_DIRS:
        print("working case: ", case_dir)
        orig_phase_names = os.listdir(case_dir)
        TOTAL = len(orig_phase_names)
        ISO = 1 #1mm
        multi_study_check = 0
        case_dir_name = os.path.split(os.path.normpath(case_dir))[-1]
        for phase_num in range(0, TOTAL, 1):
            orig_read_dir = os.path.join(case_dir, orig_phase_names[phase_num])
            print(orig_read_dir)
            temp_list = orig_phase_names[phase_num].rsplit(".",1)
            phase_name = f"{temp_list[0]}.{int(temp_list[1]):02d}"
            out_dir = f'D:/DATA/SQUEEZ1/eval/{case_dir_name}/{case_dir_name}_{ISO}mm_nii/{phase_name}/'
            orig_dcm_paths = [os.path.join(orig_read_dir,name) for name in os.listdir(orig_read_dir) if name.endswith(".dcm")]
            assert orig_dcm_paths != [], f"no files was found in {orig_read_dir}"
            orig_dcm_paths = sorted(orig_dcm_paths,key= lambda name: int(name.split(".")[-2]))
            print(f"{len(orig_dcm_paths)} files found")

            orig_volume, pxsz, slsp, slth, img_num_lst, slloc_lst = dcm_reader(orig_dcm_paths)

            #for 2d CT images
            if(len(slloc_lst) != 0):
                slsp = slloc_lst[-2] - slloc_lst[-1]

            x_spacing_orig, y_spacing_orig, z_spacing_orig = float(pxsz[0]), float(pxsz[1]), float(slsp)
            print(x_spacing_orig, y_spacing_orig, z_spacing_orig)

            x_isotropic = ISO / x_spacing_orig  # 1 mm
            z_isotropic = ISO / z_spacing_orig

            SIZE = int(orig_volume.shape[1] / x_isotropic + 0.5)  # 1.0 mm
            ZDIM = int(orig_volume.shape[0] / z_isotropic + 0.5)

            data = resize(orig_volume, (ZDIM, SIZE, SIZE), order=1, mode='reflect', preserve_range=True,
                              anti_aliasing=0)
            data = data.astype(np.int16)
            box_volume_itk = sitk.GetImageFromArray(data.astype(np.int16))

            Path(out_dir).mkdir(parents=True, exist_ok=True)
            sitk.WriteImage(box_volume_itk, out_dir + orig_phase_names[phase_num] + '_Image.nii.gz')
