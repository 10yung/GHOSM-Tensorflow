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
ghsom = GHSOM(3, 2, input_data, input_num, input_dim)
tmp_weight_result, tmp_init_som_result, prev_mqe = ghsom.check_tau2_condition()

# ---
# check tau1 condition and do horizontal expand
# ---
satisfy_tau1_or_not, opology_map_size_tf, weight_vector, insert_weight_vector, insert_direction, error_unit_index, dissimilar_neighborbood_index, pivot_point, start_point, filter_map_m, filter_map_n, lower_section_weight_vector, upper_section_weight_vector = ghsom.check_tau1_condition(tmp_init_som_result, tmp_weight_result, prev_mqe)

if satisfy_tau1_or_not == 0:
    if insert_direction == 0:
        print('insert x direction')
        print(pivot_point)
        
        weight_vector = weight_vector.reshape(filter_map_m, filter_map_n, -1)
        print(weight_vector)
        weight_vector = np.vsplit(weight_vector, 2)
        print(weight_vector)
    else:
        print('insert y direction')
        # print(pivot_point)
        # print(weight_vector)
else:
    print('satisfy tau1')

# ---
# save first layer result
# ---
# savetopologymap = saveTopologyMap('layer-1', './data_tmp')
# savetopologymap.toCsv(tmp_init_som_result, tmp_weight_result)
