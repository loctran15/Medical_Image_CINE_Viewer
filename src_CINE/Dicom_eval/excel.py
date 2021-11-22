import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import datetime

def dicts_to_excel(dataset_dicts, sheet_names, out_dir = "D:/DATA/TEST/", excel_name='dataset.xlsx', index = "study id"):
    if (out_dir == "D:/DATA/TEST/"):
        DATE = datetime.date.today()
        now = datetime.datetime.now()
        TIME = now.strftime("%H-%M-%S")
        out_dir = out_dir+f"{DATE}/{TIME}"
    assert out_dir != "default", "should not use default directory"
    assert len(dataset_dicts) == len(sheet_names), f"the number of elements in dataset_dicts is {len(dataset_dicts)} while in sheet_names is {len(sheet_names)}"

    if(not excel_name.endswith(".xlsx")):
        excel_name = excel_name + ".xlsx"

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    excel_path = os.path.join(out_dir,excel_name)
    writer = pd.ExcelWriter(excel_path)
    for i in range(len(dataset_dicts)):
        dataset_dict = dataset_dicts[i]
        sheet_name = sheet_names[i]
        column_labels = dataset_dict.keys()
        df = pd.DataFrame.from_dict(dataset_dict)
        df.columns = column_labels
        df.set_index(index, inplace = True)
        df.to_excel(excel_writer=writer,sheet_name=sheet_name)
    writer.save()
    print(f"saved to: {excel_path}")

def matrices_to_excel(matrix_list, sheet_names, column_names, index = None, out_dir = "D:/DATA/TEST/", excel_name='dataset.xlsx'):
    assert out_dir != "default", "should not use default directory"
    assert len(matrix_list) == len(
        sheet_names), f"the number of elements in dataset_dicts is {len(matrix_list)} while in sheet_names is {len(sheet_names)}"

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if (not excel_name.endswith(".xlsx")):
        excel_name = excel_name + ".xlsx"

    excel_path = os.path.join(out_dir, excel_name)
    writer = pd.ExcelWriter(excel_path)
    for i in range(len(matrix_list)):
        matrix = matrix_list[i]
        matrix = np.asarray(matrix)
        sheet_name = sheet_names[i]
        assert matrix.shape[1] == len(column_names), f"the number of matrix columns is {matrix.shape[1]} while the number of column_labels is {len(column_names)}"
        df = pd.DataFrame(data=matrix, index=index, columns=column_names)
        df.to_excel(excel_writer=writer, sheet_name=sheet_name)
    writer.save()
    print(f"saved to: {excel_path}")