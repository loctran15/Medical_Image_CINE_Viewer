__author__ = "Vy Bui"
__email__ = "01bui@cua.edu"

import numpy as np
import os
from skimage.transform import resize
from dcm_helper import dcm_reader


def read_cases(path, ISO=1):  # Use this to read data for segmentation

    print(path)

    """
    Read Original Volume
    """
    # For several folders in one study
    orig_dcm_files = []
    orig_dcm_index = []
    for root, dirs, file in os.walk(path):
        for item in file:
            if (item.endswith('.dcm')):
                orig_dcm_files.append(os.path.join(root, item))
                orig_dcm_index.append(int(item.split(".")[-2]))
    initial_index = min(orig_dcm_index)
    num_orig_files = len(orig_dcm_files)
    print(str(num_orig_files) + " files found")
    sorted_orig_dcm_files = [None] * len(orig_dcm_files)
    for file in orig_dcm_files:
        index = int(file.split(".")[-2])
        sorted_orig_dcm_files[index - initial_index] = file

    """dataset, pxsz, slice_overlap, slth, ds, frames = dcm_reader(orig_dcm_files)

    print (dataset.shape)
    slice_overlap = float(slice_overlap)
    slsp = slth - slice_overlap

    if slsp == 0:
    slsp = slice_overlap"""

    orig_volume, pxsz, slsp, slth, img_num_lst, slloc_lst = dcm_reader(sorted_orig_dcm_files)

    # for 2d CT images
    if (len(slloc_lst) != 0):
        slsp = abs(slloc_lst[-2] - slloc_lst[-1])

    slth = abs(slth)

    # print('Slice Thickness: ' + str(slth))
    # print('Slice Spacing: ' + str(slsp))
    x_spacing_orig, y_spacing_orig, z_spacing_orig = float(pxsz[0]), float(pxsz[1]), float(slsp)

    x_isotropic = ISO / x_spacing_orig  # 1 mm
    z_isotropic = ISO / z_spacing_orig

    SIZE = int(orig_volume.shape[1] / x_isotropic + 0.5)  # 1.0 mm
    ZDIM = int(orig_volume.shape[0] / z_isotropic + 0.5)

    print("original shape: ", orig_volume.shape)
    data = resize(orig_volume, (ZDIM, SIZE, SIZE), order=1, mode='reflect', preserve_range=True, anti_aliasing=0)
    data = data.astype(np.int16)
    print("resized shape: ", orig_volume.shape)
    return data


def read_seg(path):  # Use this to read data for evaluation
    paths = os.listdir(path)

    p_gold = []
    p_vitrea = []
    p_nhlbi = []

    """
    Get all cases under each folder
    Make sure cases study numbers are matched for p_gold, p_vitrea, p_nhlbi
    """
    for p in paths:
        this_p = path + '/' + p
        all_folds = os.listdir(this_p)
        for fol in all_folds:
            last_p = this_p + '/' + fol
            # If gold volume
            if 'gold' in last_p:
                p_gold.append(last_p)
            # Vitrea volume
            elif 'vitrea' in last_p:
                p_vitrea.append(last_p)
            # NHLBI volume
            elif 'nhlbi' in last_p:
                p_nhlbi.append(last_p)

    """
    Throw error if study numbers in 3 sets are not in order
    """
    print("Checking study number...")
    for item in range(len(p_nhlbi)):
        gold_item = p_gold[item].split("CT")
        gold_num = gold_item[-1][0:10]
        nhlbi_item = p_nhlbi[item].split("CT")
        nhlbi_num = nhlbi_item[-1][0:10]
        vitrea_item = p_vitrea[item].split("CT")
        vitrea_num = vitrea_item[-1][0:10]
        if gold_num == nhlbi_num and nhlbi_num == vitrea_num:
            continue
        else:
            print("Study number in 3 sets are not in order. Debug me!")
            return

    """
    Read DICOM in gold, nhlbi, vitrea folders and convert to binary mask
    """
    print("Concat seg in gold...")
    for p in p_gold:
        print(p)
        gold_dcm_files = []
        for root, dirs, file in os.walk(os.path.join(path, p)):
            if file != [] and file[0].endswith('.dcm'):
                for item in file:
                    gold_dcm_files.append(os.path.join(root, item))
        gold_dataset, _, _, _, _, _ = dcm_reader(gold_dcm_files)
        gold_binary_volume = np.zeros(gold_dataset.shape, dtype=np.uint)
        gold_binary_volume[gold_dataset > -1000] = 1
        gold_binary_volumes = gold_binary_volume
        count = 1
        break
    if count == 1:
        for p in p_gold[1:]:
            print(p)
            gold_dcm_files = []
            for root, dirs, file in os.walk(os.path.join(path, p)):
                if file != [] and file[0].endswith('.dcm'):
                    for item in file:
                        gold_dcm_files.append(os.path.join(root, item))
            gold_dataset, _, _, _, _, _ = dcm_reader(gold_dcm_files)
            gold_binary_volume = np.zeros(gold_dataset.shape, dtype=np.uint)
            gold_binary_volume[gold_dataset > -1000] = 1
            gold_binary_volumes = np.concatenate((gold_binary_volumes, gold_binary_volume))

    print("Concat seg in nhlbi...")
    for p in p_nhlbi:
        print(p)
        nhlbi_dcm_files = []
        for root, dirs, file in os.walk(os.path.join(path, p)):
            if file != [] and file[0].endswith('.dcm'):
                for item in file:
                    nhlbi_dcm_files.append(os.path.join(root, item))
        nhlbi_dataset, _, _, _, _, _ = dcm_reader(nhlbi_dcm_files)
        nhlbi_binary_volume = np.zeros(nhlbi_dataset.shape, dtype=np.uint)
        nhlbi_binary_volume[nhlbi_dataset > -1000] = 1
        nhlbi_binary_volumes = nhlbi_binary_volume
        count = 2
        break
    if count == 2:
        for p in p_nhlbi[1:]:
            print(p)
            nhlbi_dcm_files = []
            for root, dirs, file in os.walk(os.path.join(path, p)):
                if file != [] and file[0].endswith('.dcm'):
                    for item in file:
                        nhlbi_dcm_files.append(os.path.join(root, item))
            nhlbi_dataset, _, _, _, _, _ = dcm_reader(nhlbi_dcm_files)
            nhlbi_binary_volume = np.zeros(nhlbi_dataset.shape, dtype=np.uint)
            nhlbi_binary_volume[nhlbi_dataset > -1000] = 1
            nhlbi_binary_volumes = np.concatenate((nhlbi_binary_volumes, nhlbi_binary_volume))

    print("Concat seg in vitrea...")
    for p in p_vitrea:
        print(p)
        vitrea_dcm_files = []
        for root, dirs, file in os.walk(os.path.join(path, p)):
            if file != [] and file[0].endswith('.dcm'):
                for item in file:
                    vitrea_dcm_files.append(os.path.join(root, item))
        vitrea_dataset, _, _, _, _, _ = dcm_reader(vitrea_dcm_files)
        vitrea_binary_volume = np.zeros(vitrea_dataset.shape, dtype=np.uint)
        vitrea_binary_volume[vitrea_dataset > -1000] = 1
        vitrea_binary_volumes = vitrea_binary_volume
        count = 3
        break
    if count == 3:
        for p in p_vitrea[1:]:
            print(p)
            vitrea_dcm_files = []
            for root, dirs, file in os.walk(os.path.join(path, p)):
                if file != [] and file[0].endswith('.dcm'):
                    for item in file:
                        vitrea_dcm_files.append(os.path.join(root, item))
            vitrea_dataset, _, _, _, _, _ = dcm_reader(vitrea_dcm_files)
            vitrea_binary_volume = np.zeros(vitrea_dataset.shape, dtype=np.uint)
            vitrea_binary_volume[vitrea_dataset > -1000] = 1
            vitrea_binary_volumes = np.concatenate((vitrea_binary_volumes, vitrea_binary_volume))

    return [gold_binary_volumes, nhlbi_binary_volumes, vitrea_binary_volumes]


def read_seg_by_case(path, case_num):  # Use this to read data for evaluation
    paths = os.listdir(path)

    p_gold = []
    p_vitrea = []
    p_nhlbi = []

    """
    Get all cases under each folder
    Make sure cases study numbers are matched for p_gold, p_vitrea, p_nhlbi
    """
    for p in paths:
        this_p = path + '/' + p
        all_folds = os.listdir(this_p)
        for fol in all_folds:
            last_p = this_p + '/' + fol
            # If gold volume
            if 'gold' in last_p:
                p_gold.append(last_p)
            # Vitrea volume
            elif 'vitrea' in last_p:
                p_vitrea.append(last_p)
            # NHLBI volume
            elif 'nhlbi' in last_p:
                p_nhlbi.append(last_p)

    """
    Throw error if study numbers in 3 sets are not in order
    """
    print("Checking study number...")
    for item in range(len(p_nhlbi)):
        gold_item = p_gold[item].split("CT")
        gold_num = gold_item[-1][0:10]
        nhlbi_item = p_nhlbi[item].split("CT")
        nhlbi_num = nhlbi_item[-1][0:10]
        vitrea_item = p_vitrea[item].split("CT")
        vitrea_num = vitrea_item[-1][0:10]
        if gold_num == nhlbi_num and nhlbi_num == vitrea_num:
            continue
        else:
            print("Study number in 3 sets are not in order. Debug me!")
            return

    """
    Read DICOM in gold, nhlbi, vitrea folders and convert to binary mask
    """
    p = p_nhlbi[case_num]
    print(p)
    nhlbi_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                nhlbi_dcm_files.append(os.path.join(root, item))
    nhlbi_dataset, xy_spacing_orig, slice_overlap, slice_thickness, _, _ = dcm_reader(nhlbi_dcm_files)

    slice_overlap = float(slice_overlap)
    slice_spacing = slice_thickness - slice_overlap
    if slice_spacing == 0:
        slice_spacing = slice_overlap
    x_spacing_orig, y_spacing_orig, z_spacing_orig = float(xy_spacing_orig[0]), float(xy_spacing_orig[1]), float(
        slice_spacing)
    ISO_for_metrics = 0.5
    """Calculate ISO 0.5 mm"""
    x_iso_valid = ISO_for_metrics / x_spacing_orig
    z_iso_valid = ISO_for_metrics / z_spacing_orig
    SIZE_ISO_VALID = int(nhlbi_dataset.shape[1] / x_iso_valid)
    ZDIM_ISO_VALID = int(nhlbi_dataset.shape[0] / z_iso_valid)
    print('z_spacing_orig, x_spacing_orig: ' + str(z_spacing_orig) + ' ' + str(x_spacing_orig))
    print('z_iso_valid, x_iso_valid: ' + str(z_iso_valid) + ' ' + str(x_iso_valid))
    print('ZDIM_ISO_VALID, SIZE_ISO_VALID: ' + str(ZDIM_ISO_VALID) + ' ' + str(SIZE_ISO_VALID))
    nhlbi_iso = resize(nhlbi_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                       preserve_range=True)

    nhlbi_iso = nhlbi_iso.reshape(
        (nhlbi_iso.shape[0], nhlbi_iso.shape[1] * nhlbi_iso.shape[2]))
    nhlbi_binary_volume = np.zeros(nhlbi_iso.shape, dtype=np.uint)
    nhlbi_binary_volume[nhlbi_iso > -1000] = 1

    p = p_gold[case_num]
    print(p)
    gold_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                gold_dcm_files.append(os.path.join(root, item))
    gold_dataset, _, _, _, _, _ = dcm_reader(gold_dcm_files)

    gold_iso = resize(gold_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                      preserve_range=True)

    gold_iso = gold_iso.reshape(
        (gold_iso.shape[0], gold_iso.shape[1] * gold_iso.shape[2]))
    gold_binary_volume = np.zeros(gold_iso.shape, dtype=np.uint)
    gold_binary_volume[gold_iso > -1000] = 1

    p = p_vitrea[case_num]
    print(p)
    vitrea_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                vitrea_dcm_files.append(os.path.join(root, item))
    vitrea_dataset, _, _, _, _, _ = dcm_reader(vitrea_dcm_files)

    vitrea_iso = resize(vitrea_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                        preserve_range=True)

    vitrea_iso = vitrea_iso.reshape(
        (vitrea_iso.shape[0], vitrea_iso.shape[1] * vitrea_iso.shape[2]))
    vitrea_binary_volume = np.zeros(vitrea_iso.shape, dtype=np.uint)
    vitrea_binary_volume[vitrea_iso > -1000] = 1

    return [gold_binary_volume, nhlbi_binary_volume, vitrea_binary_volume]


def read_seg_by_case_medpy(path, case_num):  # Use this to read data for evaluation
    paths = os.listdir(path)

    p_gold = []
    p_vitrea = []
    p_nhlbi = []

    """
    Get all cases under each folder
    Make sure cases study numbers are matched for p_gold, p_vitrea, p_nhlbi
    """
    for p in paths:
        this_p = path + '/' + p
        all_folds = os.listdir(this_p)
        for fol in all_folds:
            last_p = this_p + '/' + fol
            # If gold volume
            if 'gold' in last_p:
                p_gold.append(last_p)
            # Vitrea volume
            elif 'vitrea' in last_p:
                p_vitrea.append(last_p)
            # NHLBI volume
            elif 'nhlbi' in last_p:
                p_nhlbi.append(last_p)

    """
    Throw error if study numbers in 3 sets are not in order
    """
    print("Checking study number...")
    for item in range(len(p_nhlbi)):
        gold_item = p_gold[item].split("CT")
        gold_num = gold_item[-1][0:10]
        nhlbi_item = p_nhlbi[item].split("CT")
        nhlbi_num = nhlbi_item[-1][0:10]
        vitrea_item = p_vitrea[item].split("CT")
        vitrea_num = vitrea_item[-1][0:10]
        if gold_num == nhlbi_num and nhlbi_num == vitrea_num:
            continue
        else:
            print("Study number in 3 sets are not in order. Debug me!")
            return

    """
    Read DICOM in gold, nhlbi, vitrea folders and convert to binary mask
    """
    p = p_nhlbi[case_num]
    print(p)
    nhlbi_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                nhlbi_dcm_files.append(os.path.join(root, item))
    nhlbi_dataset, xy_spacing_orig, slice_overlap, slice_thickness, _, _ = dcm_reader(nhlbi_dcm_files)

    slice_overlap = float(slice_overlap)
    slice_spacing = slice_thickness - slice_overlap
    if slice_spacing == 0:
        slice_spacing = slice_overlap
    x_spacing_orig, y_spacing_orig, z_spacing_orig = float(xy_spacing_orig[0]), float(xy_spacing_orig[1]), float(
        slice_spacing)
    ISO_for_metrics = 0.5
    """Calculate ISO 0.5 mm"""
    x_iso_valid = ISO_for_metrics / x_spacing_orig
    z_iso_valid = ISO_for_metrics / z_spacing_orig
    SIZE_ISO_VALID = int(nhlbi_dataset.shape[1] / x_iso_valid)
    ZDIM_ISO_VALID = int(nhlbi_dataset.shape[0] / z_iso_valid)
    print('z_spacing_orig, x_spacing_orig: ' + str(z_spacing_orig) + ' ' + str(x_spacing_orig))
    print('z_iso_valid, x_iso_valid: ' + str(z_iso_valid) + ' ' + str(x_iso_valid))
    print('ZDIM_ISO_VALID, SIZE_ISO_VALID: ' + str(ZDIM_ISO_VALID) + ' ' + str(SIZE_ISO_VALID))
    nhlbi_iso = resize(nhlbi_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                       preserve_range=True)

    nhlbi_binary_volume = np.zeros(nhlbi_iso.shape, dtype=np.uint)
    nhlbi_binary_volume[nhlbi_iso > -1000] = 1

    p = p_gold[case_num]
    print(p)
    gold_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                gold_dcm_files.append(os.path.join(root, item))
    gold_dataset, _, _, _, _, _ = dcm_reader(gold_dcm_files)

    gold_iso = resize(gold_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                      preserve_range=True)

    gold_binary_volume = np.zeros(gold_iso.shape, dtype=np.uint)
    gold_binary_volume[gold_iso > -1000] = 1

    p = p_vitrea[case_num]
    print(p)
    vitrea_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                vitrea_dcm_files.append(os.path.join(root, item))
    vitrea_dataset, _, _, _, _, _ = dcm_reader(vitrea_dcm_files)

    vitrea_iso = resize(vitrea_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                        preserve_range=True)

    vitrea_binary_volume = np.zeros(vitrea_iso.shape, dtype=np.uint)
    vitrea_binary_volume[vitrea_iso > -1000] = 1

    return [gold_binary_volume, nhlbi_binary_volume, vitrea_binary_volume]


def read_seg_by_case_for_intensity(path, case_num):  # Use this to read data for evaluation
    paths = os.listdir(path)

    p_lv = []
    p_rv = []
    p_myo = []
    p_rc = []
    p_liver = []
    p_doa = []

    """
    Get all cases under each folder
    Make sure cases study numbers are matched for p_gold, p_vitrea, p_nhlbi
    """
    for p in paths:
        this_p = path + '/' + p
        all_folds = os.listdir(this_p)
        for fol in all_folds:
            last_p = this_p + '/' + fol
            if 'LV' in last_p:
                p_lv.append(last_p)
            elif 'RV' in last_p:
                p_rv.append(last_p)
            elif 'Myo' in last_p:
                p_myo.append(last_p)
            elif 'RC' in last_p:
                p_rc.append(last_p)
            elif 'Liver' in last_p:
                p_liver.append(last_p)
            elif 'DoA' in last_p:
                p_doa.append(last_p)

    """
    Throw error if study numbers in 3 sets are not in order
    """
    print("Checking study number...")
    for item in range(len(p_lv)):
        lv_item = p_lv[item].split("CT")
        lv_num = lv_item[-1][0:10]
        rv_item = p_rv[item].split("CT")
        rv_num = rv_item[-1][0:10]
        myo_item = p_myo[item].split("CT")
        myo_num = myo_item[-1][0:10]
        rc_item = p_rc[item].split("CT")
        rc_num = rc_item[-1][0:10]
        liver_item = p_liver[item].split("CT")
        liver_num = liver_item[-1][0:10]
        doa_item = p_doa[item].split("CT")
        doa_num = doa_item[-1][0:10]
        if lv_num == rv_num == myo_num == rc_num == liver_num == doa_num:
            continue
        else:
            print("Study number in 3 sets are not in order. Debug me!")
            return

    """
    Read DICOM in lv, rv, myo, rc, liver, doa folders
    """
    p = p_lv[case_num]
    print(p)
    lv_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                lv_dcm_files.append(os.path.join(root, item))
    lv_dataset, xy_spacing_orig, slice_overlap, slice_thickness, _, _ = dcm_reader(lv_dcm_files)

    slice_overlap = float(slice_overlap)
    slice_spacing = slice_thickness - slice_overlap
    if slice_spacing == 0:
        slice_spacing = slice_overlap
    x_spacing_orig, y_spacing_orig, z_spacing_orig = float(xy_spacing_orig[0]), float(xy_spacing_orig[1]), float(
        slice_spacing)
    ISO_for_metrics = 0.5
    """Calculate ISO 0.5 mm"""
    x_iso_valid = ISO_for_metrics / x_spacing_orig
    z_iso_valid = ISO_for_metrics / z_spacing_orig
    SIZE_ISO_VALID = int(lv_dataset.shape[1] / x_iso_valid)
    ZDIM_ISO_VALID = int(lv_dataset.shape[0] / z_iso_valid)
    print('ZDIM_ISO_VALID, SIZE_ISO_VALID: ' + str(ZDIM_ISO_VALID) + ' ' + str(SIZE_ISO_VALID))

    lv_iso = resize(lv_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                    preserve_range=True)
    lv_iso = lv_iso.reshape((lv_iso.shape[0], lv_iso.shape[1] * lv_iso.shape[2]))

    """
    RV
    """
    p = p_rv[case_num]
    print(p)
    rv_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                rv_dcm_files.append(os.path.join(root, item))
    rv_dataset, _, _, _, _, _ = dcm_reader(rv_dcm_files)
    rv_iso = resize(rv_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                    preserve_range=True)
    rv_iso = rv_iso.reshape((rv_iso.shape[0], rv_iso.shape[1] * rv_iso.shape[2]))

    """
    Myo
    """
    p = p_myo[case_num]
    print(p)
    myo_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                myo_dcm_files.append(os.path.join(root, item))
    myo_dataset, _, _, _, _, _ = dcm_reader(myo_dcm_files)
    myo_iso = resize(myo_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                     preserve_range=True)
    myo_iso = myo_iso.reshape((myo_iso.shape[0], myo_iso.shape[1] * myo_iso.shape[2]))

    """
    Chest Wall
    """
    p = p_rc[case_num]
    print(p)
    rc_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                rc_dcm_files.append(os.path.join(root, item))
    rc_dataset, _, _, _, _, _ = dcm_reader(rc_dcm_files)
    rc_iso = resize(rc_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                    preserve_range=True)
    rc_iso = rc_iso.reshape((rc_iso.shape[0], rc_iso.shape[1] * rc_iso.shape[2]))

    """
    Liver
    """
    p = p_liver[case_num]
    print(p)
    liver_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                liver_dcm_files.append(os.path.join(root, item))
    liver_dataset, _, _, _, _, _ = dcm_reader(liver_dcm_files)
    liver_iso = resize(liver_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                       preserve_range=True)
    liver_iso = liver_iso.reshape((liver_iso.shape[0], liver_iso.shape[1] * liver_iso.shape[2]))

    """
    DoA
    """
    p = p_doa[case_num]
    print(p)
    doa_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                doa_dcm_files.append(os.path.join(root, item))
    doa_dataset, _, _, _, _, _ = dcm_reader(doa_dcm_files)
    doa_iso = resize(doa_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                     preserve_range=True)
    doa_iso = doa_iso.reshape((doa_iso.shape[0], doa_iso.shape[1] * doa_iso.shape[2]))

    return [lv_iso, rv_iso, myo_iso, rc_iso, liver_iso, doa_iso, p]


def read_seg_by_case_for_color(path, case_num):  # Use this to read data for evaluation
    paths = os.listdir(path)

    p_lv = []
    p_rv = []
    p_myo = []

    """
    Get all cases under each folder
    Make sure cases study numbers are matched for p_gold, p_vitrea, p_nhlbi
    """
    for p in paths:
        this_p = path + '/' + p
        all_folds = os.listdir(this_p)
        for fol in all_folds:
            last_p = this_p + '/' + fol
            if 'nhlbi' in last_p:
                p_lv.append(last_p)
            elif 'gold' in last_p:
                p_rv.append(last_p)
            elif 'vitrea' in last_p:
                p_myo.append(last_p)

    """
    Throw error if study numbers in 3 sets are not in order
    """
    print("Checking study number...")
    for item in range(len(p_lv)):
        lv_item = p_lv[item].split("CT")
        lv_num = lv_item[-1][0:10]
        rv_item = p_rv[item].split("CT")
        rv_num = rv_item[-1][0:10]
        myo_item = p_myo[item].split("CT")
        myo_num = myo_item[-1][0:10]
        if lv_num == rv_num and myo_num == rv_num:
            continue
        else:
            print("Study number in 3 sets are not in order. Debug me!")
            return

    """
    Read DICOM in lv, rv, myo folders
    """
    p = p_lv[case_num]
    print(p)
    lv_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                lv_dcm_files.append(os.path.join(root, item))
    lv_dataset, xy_spacing_orig, slice_overlap, slice_thickness, _, _ = dcm_reader(lv_dcm_files)

    slice_overlap = float(slice_overlap)
    slice_spacing = slice_thickness - slice_overlap
    if slice_spacing == 0:
        slice_spacing = slice_overlap
    x_spacing_orig, y_spacing_orig, z_spacing_orig = float(xy_spacing_orig[0]), float(xy_spacing_orig[1]), float(
        slice_spacing)
    ISO_for_metrics = 0.5
    """Calculate ISO 0.5 mm"""
    x_iso_valid = ISO_for_metrics / x_spacing_orig
    z_iso_valid = ISO_for_metrics / z_spacing_orig
    SIZE_ISO_VALID = int(lv_dataset.shape[1] / x_iso_valid)
    ZDIM_ISO_VALID = int(lv_dataset.shape[0] / z_iso_valid)
    print('ZDIM_ISO_VALID, SIZE_ISO_VALID: ' + str(ZDIM_ISO_VALID) + ' ' + str(SIZE_ISO_VALID))

    lv_iso = resize(lv_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                    preserve_range=True)

    p = p_rv[case_num]
    print(p)
    rv_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                rv_dcm_files.append(os.path.join(root, item))
    rv_dataset, _, _, _, _, _ = dcm_reader(rv_dcm_files)
    rv_iso = resize(rv_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                    preserve_range=True)

    p = p_myo[case_num]
    print(p)
    myo_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                myo_dcm_files.append(os.path.join(root, item))
    myo_dataset, _, _, _, _, _ = dcm_reader(myo_dcm_files)
    myo_iso = resize(myo_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                     preserve_range=True)

    return [lv_iso, rv_iso, myo_iso, p]


def read_orig_by_case_for_grayscale(path, case_num):  # Use this to read data for evaluation
    paths = os.listdir(path)

    p_orig = []

    """
    Get all cases under each folder
    Make sure cases study numbers are matched for p_gold, p_vitrea, p_nhlbi
    """
    for p in paths:
        this_p = path + '/' + p
        all_folds = os.listdir(this_p)
        for fol in all_folds:
            last_p = this_p + '/' + fol
            if 'orig' in last_p:
                p_orig.append(last_p)

    """
    Read DICOM in orig folders
    """
    p = p_orig[case_num]
    print(p)
    orig_dcm_files = []
    for root, dirs, file in os.walk(os.path.join(path, p)):
        if file != [] and file[0].endswith('.dcm'):
            for item in file:
                orig_dcm_files.append(os.path.join(root, item))
        orig_dataset, xy_spacing_orig, slice_overlap, slice_thickness, _, _ = dcm_reader(orig_dcm_files)

    slice_overlap = float(slice_overlap)
    slice_spacing = slice_thickness - slice_overlap
    if slice_spacing == 0:
        slice_spacing = slice_overlap
    x_spacing_orig, y_spacing_orig, z_spacing_orig = float(xy_spacing_orig[0]), float(xy_spacing_orig[1]), float(
        slice_spacing)
    ISO_for_metrics = 0.5
    """Calculate ISO 0.5 mm"""
    x_iso_valid = ISO_for_metrics / x_spacing_orig
    z_iso_valid = ISO_for_metrics / z_spacing_orig
    SIZE_ISO_VALID = int(orig_dataset.shape[1] / x_iso_valid)
    ZDIM_ISO_VALID = int(orig_dataset.shape[0] / z_iso_valid)
    print('ZDIM_ISO_VALID, SIZE_ISO_VALID: ' + str(ZDIM_ISO_VALID) + ' ' + str(SIZE_ISO_VALID))

    orig_iso = resize(orig_dataset, (ZDIM_ISO_VALID, SIZE_ISO_VALID, SIZE_ISO_VALID), order=0, mode='reflect',
                      preserve_range=True)

    return [orig_iso, p]


if __name__ == "__main__":
    path = "D:/TRANL/DATA/SQUEEZ/original data/SQUEEZ-0005-2\\1.2.392.200036.9116.2.2462354099.1572400965.2.1353500002.1"
    read_cases(path)
