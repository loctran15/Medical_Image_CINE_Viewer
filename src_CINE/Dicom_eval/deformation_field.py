import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path
from src_CINE.file_loader import file_loader
from src_CINE.Dicom_helper.helper_funcs import label_dict

def save_displacement_fields(deform_field, out_dir, file_name):
    if (not os.path.exists(out_dir)):
        os.makedirs(out_dir)
    path_file = os.path.join(out_dir,file_name)
    deform_field = deform_field.astype(np.float64)
    displacement_field_image = sitk.GetImageFromArray(deform_field, isVector=True)
    #sitk.WriteImage(displacement_field_image, "D:/DATA/TEST/regis/regis_all/CMACS_15/deformation_field/v_u_w.nii.gz")
    displacement_field_transform = sitk.DisplacementFieldTransform(displacement_field_image)
    sitk.WriteTransform(displacement_field_transform, path_file)
    print(f"saved to {path_file}")

if __name__ == "__main__":
    segment = "LVM" #LVM or WH
    out_dirs = [r"D:\DATA\CTA_CINE\eval\CT5000818554_20210701_33P\out\1mm\regis\Reg_all\Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_26\debug\deformation_field\fields\\" + str(segment)]
    masks_dirs = [r"D:\DATA\CTA_CINE\eval\CT5000818554_20210701_33P\out\1mm\regis\Reg_all\Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_26\debug\pre-refined_labels\2021-07-04\10-27-34"]
    temp_np_data_dirs = [r"D:\DATA\CTA_CINE\eval\CT5000818554_20210701_33P\out\1mm\regis\Reg_all\Deep_Heart_checkpoints_CL_CMACS_RT0_CV0_26\debug\deformation_field\temp_numpy_data\2021-07-04\10-27-34"]


    for out_dir,masks_dir,temp_np_data_dir in zip(out_dirs,masks_dirs,temp_np_data_dirs):
        m_phase_list = range(1, len(os.listdir(masks_dir)) + 1, 1)  # start from 1 to 20, included
        f_phase_list = range(2, len(os.listdir(masks_dir)) + 2, 1)
        for i in range(len(m_phase_list)):
            m_phase = m_phase_list[i]
            f_phase = f_phase_list[i]
            if(m_phase == len(m_phase_list)):
                f_phase = 1
            m_seg_pad, _ = file_loader.read_label(f"{masks_dir}/phase_{m_phase}.nii.gz")

            if (segment == "WH"):
                z, x, y = np.where(m_seg_pad == 0)
            else:
                z,x,y = np.where(m_seg_pad != label_dict[segment])

            if(m_phase == 25):
                m_deform = np.load(
                    f"{temp_np_data_dir}/df_from_26_to_{m_phase}.npy")
                deform_field = 0 - m_deform
                deform_field[z, x, y] = np.zeros((3,))
                save_displacement_fields(deform_field, out_dir, f"deform_from_{m_phase}_to_{f_phase}.tfm")
            elif(m_phase == 26):
                f_deform = np.load(
                    f"{temp_np_data_dir}/df_from_26_to_{f_phase}.npy")
                deform_field = f_deform
                deform_field[z, x, y] = np.zeros((3,))
                save_displacement_fields(deform_field, out_dir, f"deform_from_{m_phase}_to_{f_phase}.tfm")
            else:
                m_deform = np.load(
                    f"{temp_np_data_dir}/df_from_26_to_{m_phase}.npy")
                f_deform = np.load(f"{temp_np_data_dir}/df_from_26_to_{f_phase}.npy")
                deform_field = f_deform - m_deform
                deform_field[z, x, y] = np.zeros((3,))
                # save
                save_displacement_fields(deform_field, out_dir, f"deform_from_{m_phase}_to_{f_phase}.tfm")



