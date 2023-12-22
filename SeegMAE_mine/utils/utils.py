import os
from openpyxl import load_workbook
import numpy as np
import pickle

def getFilelist(folder_path, ext=None):
    filelist = []
    for root, dirs, files in os.walk((folder_path)):
        for filename in files:
            if filename.endswith(ext) and not filename.startswith('.'):
                # print("root", root)
                # print("dirs", dirs)
                # print("files", filename)
                filepath = os.path.join(root, filename)
                filelist.append(filepath)

    return filelist

def electrode_region(location_xlsx_file, patient_name):
    ########  获得脑区对应坐标和命名信息   ############
    # exl_path = '/home/wangshuo/Datasets/SIMIT/sEEG/20230906_bipo_co_excel/Qianyusheng.xlsx'
    xlsx_path = location_xlsx_file

    data_frame = load_workbook(xlsx_path)
    # sheet = data_frame['Qianyusheng']
    sheet = data_frame[patient_name]

    co_x = sheet['B']
    co_x = co_x[1:]
    x = np.array([i.value for i in co_x])

    co_y = sheet['C']
    co_y = co_y[1:]
    y = np.array([i.value for i in co_y])

    co_z = sheet['D']
    co_z = co_z[1:]
    z = np.array([i.value for i in co_z])

    br1 = sheet['E']
    br1 = br1[1:]
    br = [i.value for i in br1]

    return br, x, y, z

def dict2binary(binary_file_path, data_dict):
    with open(binary_file_path, 'wb') as binary_file:
        pickle.dump(data_dict, binary_file)

def binary2dict(binary_file_path):
    with open(binary_file_path, 'rb') as binary_file:
        loaded_data_dict = pickle.load(binary_file)
    return loaded_data_dict
