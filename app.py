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
tmp_weight_result, tmp_init_som_result, prev_mqe = ghsom.check_tau2_condition()

# ---
# check tau1 condition and do horizontal expand
# ---
filter_map_m = 2
filter_map_n = 2

print(prev_mqe)

while True:
    print('-------------------------call check tau1------------------------------')
    satisfy_tau1_or_not, result_filter_map, result_filter_map_m, result_filter_map_n, result_topology_map_size_tf, weight_vector, insert_weight_vector, insert_direction, error_unit_index, dissimilar_neighborbood_index, start_point, start_point, lower_section_weight_vector, upper_section_weight_vector = ghsom.check_tau1_condition(
        tmp_init_som_result, tmp_weight_result, prev_mqe, filter_map_m, filter_map_n)

    if satisfy_tau1_or_not == 1:
        if insert_direction == 0:
            
            # print('insert x direction')
            # print(start_point)
            # print(weight_vector)
            # print('-------------after insertion--------------')
            # print(insert_weight_vector)
            # reshap weight_vector to [0,2,4] [1,3,5] format
            trained_weight_default_index = np.arange(filter_map_m*filter_map_n)
            new_order = []
            
            for i in range(filter_map_n):
                trained_weight_remainder = trained_weight_default_index%filter_map_n
                new_order = np.append(new_order, np.where(trained_weight_remainder == i)[0])
            
            new_order = new_order.astype(int)

            weight_topology_map  = weight_vector[new_order,:][:].reshape(filter_map_n,filter_map_m, -1).astype(np.float32)
            new_weight_topology_map = np.insert(weight_topology_map, start_point, insert_weight_vector, 0)
            
            # TODO: check insert x weight after reshape order
            new_weight_after_insertion = np.swapaxes(new_weight_topology_map,0,1).reshape(-1, input_dim).astype(np.float32)
            # print(new_weight_topology_map)
            # print(new_weight_after_insertion)

            filter_map_n = filter_map_n + 1
            filter_map_m = filter_map_m

        else:
            # print('insert y direction')
            # print(weight_vector)
            # print('------------after insertion---------------')
            tmp = np.append(lower_section_weight_vector, insert_weight_vector, axis=0)
            new_weight_after_insertion = np.append(tmp, upper_section_weight_vector, axis=0)
            # print(new_weight_after_insertion)

            filter_map_m = filter_map_m + 1
            filter_map_n = filter_map_n
        
        # print(filter_map_m)
        # print(filter_map_n)
        # print(new_weight_after_insertion)
        tmp_weight_result, tmp_init_som_result = ghsom.call_som(filter_map_m, filter_map_n, input_data, input_dim, new_weight_after_insertion)

        # print(tmp_weight_result)

    else:
        print('------------satisfy tau1------------')
        break

print(result_filter_map)
print(result_filter_map_m)
print(result_filter_map_n)
print(weight_vector)

# ---
# save first layer result
# ---
# savetopologymap = saveTopologyMap('layer-1', './data_tmp')
# savetopologymap.toCsv(tmp_init_som_result, tmp_weight_result)
