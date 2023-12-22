from utils import *
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np


pkl_path = '/home/jz/Downloads/remove_indicesDC1181WS.pkl'

with open(pkl_path, 'rb') as binary_file:
    loaded_data_dict = json.load(binary_file)

bins = np.arange(0, 5760000 + 40000, 40000)

# 创建直方图


ch = loaded_data_dict['O1-O2']
# chmax = np.max(ch_E13)

plt.figure(figsize=(100, 80))
plt.hist(ch, bins=bins)
plt.show()

print('debug')