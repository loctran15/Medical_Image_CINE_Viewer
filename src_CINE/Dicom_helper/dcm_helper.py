__author__ = "Vy Bui"
__email__ = "01bui@cua.edu"

import numpy as np
import pydicom as dicom

def dcm_reader(dcm_files):
    img_num_lst = []
    slloc_lst = []
    if len(dcm_files) > 1:
        for fn in dcm_files:
            dcm_file = dicom.read_file(fn)
            # Read the first dicom to get the image size
            row = dcm_file[0x0028, 0x0010].value
            col = dcm_file[0x0028, 0x0011].value
            slloc = abs(float(dcm_file[0x0020, 0x0032].value[2])) #dcm_file[0x0020, 0x1041].value
            slloc_lst.append(slloc)
            rescale_intercept = dcm_file[0x0028, 0x1052].value
            rescale_slope = dcm_file[0x0028, 0x1053].value
            try:
                slth = dcm_file[0x0018, 0x0050].value
                pxsz = dcm_file[0x0028, 0x0030].value
            except:
                try:
                    slth = dcm_file[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0018, 0x0050].value
                    pxsz = dcm_file[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
                except:
                    slth = 1
                    pxsz = 1
            try:
                slsp = dcm_file[0x7005, 0x1047].value
            except:
                try:
                    slsp = dcm_file[0x7005, 0x1022].value
                except:
                    slsp = slth

        dataset = np.ndarray(shape=(len(dcm_files), row, col), dtype=np.float64)

        # Sort the dicoms and assign it to numpy array
        # Find right index for Image Number
        for fn in dcm_files:
            dcm_file = dicom.read_file(fn)
            image_number = dcm_file[0x0020, 0x0013].value
            #cmt_lst = dcm_file[0x0020, 0x4000].value
            img_num_lst.append(image_number)
        min_idx = min(img_num_lst)
        for fn in dcm_files:
            dcm_file = dicom.read_file(fn)
            image_number = dcm_file[0x0020, 0x0013].value
            dcm_pixel_data = dcm_file.pixel_array.astype(float)
            dataset[image_number - min_idx, :, :] = dcm_pixel_data*rescale_slope + rescale_intercept
            #dataset[image_number - min_idx, :, :] =((dcm_pixel_data - dcm_pixel_data.min()) * (1 / (dcm_pixel_data.max() - dcm_pixel_data.min()) * 255)).astype('uint8')
    else: #3D
        for fn in dcm_files:
            # Read the first dicom to get the image size
            dcm_file = dicom.read_file(fn)
            rows = dcm_file[0x0028, 0x0010].value #Rows
            cols = dcm_file[0x0028, 0x0011].value #Columns
            slices = dcm_file[0x0028, 0x0008].value #Number of Frames

            try:
                slsp = dcm_file[0x7005, 0x1047].value
            except:
                slsp = dcm_file[0x7005, 0x1022].value
            try:
                rescale_intercept = dcm_file[0x0028, 0x1052].value  # Rescale Intercept
                rescale_slope = dcm_file[0x0028, 0x1053].value  # Rescale Slope
            except:
                rescale_intercept = 0
                rescale_slope = 1
            try:
                slth = dcm_file[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0018, 0x0050].value
                pxsz = dcm_file[0x5200, 0x9229][0][0x0028, 0x9110][0][0x0028, 0x0030].value
            except:
                slth = None
                pxsz = None

        dataset = np.ndarray(shape=(slices, rows, cols), dtype=np.float32)

        # Sort the dicoms and assign it to numpy array
        for fn in dcm_files:
            #print fn
            dcm_file = dicom.read_file(fn)
            dcm_pixel_data = dcm_file.pixel_array.astype(float)
            dataset = dcm_pixel_data*rescale_slope + rescale_intercept
    return [dataset, pxsz, slsp, slth, img_num_lst, slloc_lst]