import pandas as pd
from scipy.stats import variation
import numpy as np
from .helper_funcs import label_dict

def get_CV_dataset(labels, grayscales):
    LV_CV = []
    LA_CV = []
    RV_CV = []
    RA_CV = []
    LVM_CV = []
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

        LV_CV.append(variation(LV))
        LA_CV.append(variation(LA))
        RV_CV.append(variation(RV))
        RA_CV.append(variation(RA))
        LVM_CV.append(variation(LVM))

    data_dict = {
        'LV': LV_CV,
        'LA': LA_CV,
        'RV': RV_CV,
        'RA': RA_CV,
        'LVM': LVM_CV
    }

    return data_dict

#TODO: write main function
