import SimpleITK as sitk
import os
import numpy as np
from pathlib import Path

def save_mask(mask, file_name, out_dir = "D:/DATA/TEST/regis"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if(not file_name.endswith(".nii.gz")):
        file_name = file_name + ".nii.gz"
    mask = mask.astype(np.uint16)
    out = sitk.GetImageFromArray(mask)
    out.SetSpacing([1, 1, 1])
    sitk.WriteImage(out, os.path.join(out_dir, file_name))
    print(f"saved to {out_dir}/{file_name}")