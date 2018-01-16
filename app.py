import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import ExcelWriter
import openpyxl
import datetime
from clustering.ghsom_tensorflow import GHSOM
from model.saveTopologyMap import saveTopologyMap



# ---
# input preprocess
# ---
# change data to np array (SOM accept nparray format)
df = pd.read_excel('./data_tmp/data.xlsx', header=None)
input_data = np.array(df.as_matrix())
input_dim = 5
input_num = 10

# ---
# init check tau2 to do first SOM
# ---
ghsom = GHSOM(2, 2, input_data, input_num, input_dim)
tmp_weight_result, tmp_init_som_result = ghsom.check_tau2_condition()

# ---
# check tau1 condition and do horizontal expand
# ---
ghsom.check_tau1_condition(tmp_init_som_result, tmp_weight_result)

# ---
# save first layer result
# ---
# savetopologymap = saveTopologyMap('layer-1', './data_tmp')
# savetopologymap.toCsv(tmp_init_som_result, tmp_weight_result)
