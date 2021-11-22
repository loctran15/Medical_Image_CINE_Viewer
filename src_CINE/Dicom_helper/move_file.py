import os
import re
import shutil
from pathlib import Path
import pydicom
import glob

def move_file(curr_dir,des_dir,new_file_name = None):
    if (new_file_name is None):
        shutil.move(curr_dir,des_dir)
    else:
        des_dir = os.path.normpath(des_dir)
        des_dir_split = os.path.split(des_dir)
        des_dir = "/".join(des_dir_split[:-1]) + f"/{new_file_name}"
        shutil.move(curr_dir,des_dir)

def move_file_3d(curr_dirs):

    for curr_dir in curr_dirs:
        #filter our the proceesed dataset
        print(f"current directory: {curr_dir}")

        for root, dirs, files in os.walk(curr_dir):
            for file in files:
                if (file.endswith(".dcm") and os.path.split(os.path.normpath(root))[-1].startswith("Loc")):
                    file_path = os.path.join(root, file)
                    des_path = f"{os.path.dirname(root)}/{file}"
                    move_file(file_path, des_path)
                    shutil.rmtree(root)

        for root, dirs, files in os.walk(curr_dir):
            if (len(files) != 0):
                file_name = files[0]
                phase_dir = root
                study_dir = os.path.dirname(phase_dir)
                dicom_image = pydicom.filereader.dcmread(os.path.join(phase_dir, file_name))
                seriesInstanceUID = dicom_image.SeriesInstanceUID
                image_number = dicom_image.InstanceNumber
                shutil.move(phase_dir, f"{study_dir}/{seriesInstanceUID}.{image_number}")

def move_file_2d(curr_dirs):
    for curr_dir in curr_dirs:
        # filter our the proceesed dataset
        print(f"current directory: {curr_dir}")
        # dest_dir = DEST_DIRS[index]
        for root, dirs, files in os.walk(curr_dir):
            for file in files:
                if (file.endswith(".dcm") and os.path.split(os.path.normpath(root))[-1].startswith("Loc")):
                    file_path = os.path.join(root, file)
                    des_path = f"{os.path.dirname(root)}/{file}"
                    move_file(file_path, des_path)
                    shutil.rmtree(root)

        for root, dirs, files in os.walk(curr_dir):
            if (len(files) != 0):
                file_name = files[0]
                phase_dir = root
                study_dir = os.path.dirname(phase_dir)
                dicom_image = pydicom.filereader.dcmread(os.path.join(phase_dir, file_name))
                seriesInstanceUID = dicom_image.SeriesInstanceUID
                shutil.move(phase_dir, f"{study_dir}/{seriesInstanceUID}")

#move dicom files to it's parent directorie!
if __name__ == "__main__":
    studies_2d = []
    studies_3d =[]
    for study in glob.glob(r"D:\DATA\CTA_CINE\original_CTA_CINE\CT*P"):
        if(len(re.split(r"[_-]",os.path.basename(study))) < 3):
            continue
        else:
            print(study)
            locs_path = glob.glob(f"D:/DATA/CTA_CINE/original_CTA_CINE/{os.path.basename(study)}/*")
            print(locs_path[0])
            if(len(os.listdir(locs_path[0])) == 1):
                studies_3d.append(study)
            else:
                studies_2d.append(study)

    print(studies_2d)
    print(studies_3d)
    move_file_2d(studies_2d)
    #move_file_3d(studies_3d)
