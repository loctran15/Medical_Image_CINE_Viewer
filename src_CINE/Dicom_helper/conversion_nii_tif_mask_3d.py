import numpy as np
from src_CINE.Dicom_helper.helper_funcs import label_dict
from save_mask import save_mask
import SimpleITK as sitk
from skimage.transform import resize
from pathlib import Path
import os
from scipy import ndimage as ndi
from skimage import morphology
from src_CMACS.cmacs_bbox_detection import bbox_detection

#3d tiff or nii.gz

color_map = {
    "BOX"  : [255,  0,  0],
    "WH":    [  0,255,  0],
    "LUNG" : [  0,  0,255],
    "LVM"  : [  0,127,  0],
    "LV"   : [255,  0,255],
    "AO"   : [255,255,  0],
    "LIVER": [191,191,  0],
    "DAS"  : [127,127,  0],
    "RV"   : [  0,255,255],
    "CW"   : [  0,191,191],
    "PV"   : [  0,127,127],
    "LA"   : [255, 95, 95],
    "LAA"  : [191, 95, 95],
    "RA"   : [ 95,191, 95],
    "IVC"  : [ 95,127, 95],
    "PA"   : [ 95, 95,255],
    "SVC"  : [ 95, 95,191],
    "SPINE": [ 95, 95,127]
}

def nii_to_tiff(mask_array):
    #RGB image
    tiff_mask_array = np.zeros((*mask_array.shape,3),dtype=np.uint8)
    if(len(mask_array.shape) == 3):
        for segment in color_map.keys():
            z,x,y = np.where(mask_array == label_dict[segment])
            tiff_mask_array[z,x,y] = color_map[segment]
    return tiff_mask_array

def tiff_to_nii(mask_array):
    zs,xs,ys,_ = mask_array.shape
    nii_mask_array = np.zeros((zs,xs,ys),dtype=np.uint8)
    if (len(mask_array.shape) == 4):
        for segment,color in color_map.items():
            z, x, y = np.where((mask_array[:,:,:,0] == color[0]) & (mask_array[:,:,:,1] == color[1]) & (mask_array[:,:,:,2] == color[2]))
            nii_mask_array[z, x, y] = label_dict[segment]
    return nii_mask_array


def read_tiff(path):
    image = sitk.ReadImage(path, imageIO="TIFFImageIO")
    try:
        description = image.GetMetaData("ImageDescription")
        Vscale = description.split("_")[0].split(":")[1]
    except:
        Vscale = 1
    orig_image_array = sitk.GetArrayFromImage(image)
    zs, xs, ys, ds = orig_image_array.shape
    zs, xs, ys, ds = zs, int(xs / int(Vscale)), int(ys / int(Vscale)), ds
    image_array = resize(orig_image_array, (zs, xs, ys, ds), order=1, mode='reflect', preserve_range=True, anti_aliasing=0)
    return image_array

def write_nii_mask(file_name,out_dir,mask_array):
    #imageio.mvolwrite(path,mask_array)

    save_mask(mask_array, file_name, out_dir = out_dir)





