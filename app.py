import numpy as np
import numpy.ma as ma
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
df = pd.read_excel('./data/WPG.xlsx')
print('finish read file')
df_ran = df.sample(frac=1).reset_index(drop=True)

df_noimal = df_ran.ix[:, ['Report Date', 'Customer',
                                 'Item Short Name', 'Brand', 'Sales']]
df_numerical = df_ran.ix[:, ['Actual AWU', 'Avail.',
                                 'BL <= 9WKs', 'Backlog', 'DC OH', 'Hub OH', 'TTL OH', 'FCST M']]

input_data = np.array(df_numerical.as_matrix())
input_dim = input_data.shape[1]
input_num = input_data.shape[0]

# ---
# init check tau2 to do first SOM
# ---
print('-------------------------call check tau2------------------------------')
ghsom = GHSOM(2, 2, input_data, input_num, input_dim)
tmp_weight_result, tmp_init_som_result, prev_mqe = ghsom.check_tau2_condition()

print(tmp_weight_result)
print(tmp_init_som_result)
print(prev_mqe)
# ---
# check tau1 condition and do horizontal expand
# ---
filter_map_m = 2
filter_map_n = 2

# print(prev_mqe)
counter = 1
while True:
    print('-------------------------call check tau1------------------------------')
    satisfy_tau1_or_not, result_filter_map, result_filter_map_m, result_filter_map_n, result_topology_map_size_tf, weight_vector, insert_weight_vector, insert_direction, error_unit_index, dissimilar_neighborbood_index, start_point, start_point, lower_section_weight_vector, upper_section_weight_vector = ghsom.check_tau1_condition(
        tmp_init_som_result, tmp_weight_result, prev_mqe, filter_map_m, filter_map_n)

    if satisfy_tau1_or_not == 1:
        if insert_direction == 0:
            
            # print('insert x direction')
            # print(error_unit_index)
            # print(dissimilar_neighborbood_index)
            # print(weight_vector)
            # print('----------lower weight vector---------')
            # print(lower_section_weight_vector)
            # print('--------upper weight vector---------')
            # print(upper_section_weight_vector)
            # print('-------insert_weight_vector---------')
            # print(insert_weight_vector)
            # reshap weight_vector to [0,2,4] [1,3,5] format
            trained_weight_default_index = np.arange(filter_map_m*filter_map_n)
            new_order = []
            
            for i in range(filter_map_n):
                trained_weight_remainder = trained_weight_default_index%filter_map_n
                new_order = np.append(new_order, np.where(trained_weight_remainder == i)[0])
            
            new_order = new_order.astype(int)

            weight_topology_map  = weight_vector[new_order,:][:].reshape(filter_map_n,filter_map_m, -1).astype(np.float32)
            # print('----------topology map after regroup------------')
            # print(weight_topology_map)
            new_weight_topology_map = np.insert(weight_topology_map, start_point+1, insert_weight_vector, 0)
            # print('---------topology map after insert--------------')
            # print(new_weight_topology_map)

            # TODO: check insert x weight after reshape order
            new_weight_after_insertion = np.swapaxes(new_weight_topology_map,0,1).reshape(-1, input_dim).astype(np.float32)
            # print('-------------flate array after insertion--------------')
            # print(new_weightinput_data[filter_mask,:]_after_insertion)

            filter_map_n = filter_map_n + 1
            filter_map_m = filter_map_m

        else:
            # print('insert y direction')
            # print(weight_vector)
            # print('--------df1----after insertion---------------')
            tmp = np.append(lower_section_weight_vector, insert_weight_vector, axis=0)
            new_weight_after_insertion = np.append(tmp, upper_section_weight_vector, axis=0)
            # print(new_weight_after_insertion)

            filter_map_m = filter_map_m + 1
            filter_map_n = filter_map_n

        if counter%100 == 0:
            print('LoopNo: ' + str(counter) + ' - Time: ' + str(datetime.datetime.now().time()))

        counter += 1
        print(filter_map_m)
        print(filter_map_n)
        print(new_weight_after_insertion)
        tmp_weight_result, tmp_init_som_result = ghsom.call_som(filter_map_m, filter_map_n, input_data, input_dim, new_weight_after_insertion)

        # print(tmp_weight_result)

    else:
        print('------------satisfy tau1------------')
        break

# print(result_filter_map)
# print(result_filter_map_m)
# print(result_filter_map_n)
# print(weight_vector)
# print(result_topology_map_size_tf)
# print(result_filter_map)


# ---
# save first layer result
# ---
# savetopologymap = saveTopologyMap('layer-1', './data_tmp')
# savetopologymap.toCsv(tmp_init_som_result, tmp_weight_result)

result = np.expand_dims(np.arange(input_data.shape[0]), axis=0)
result_header = []

for i in range(result_filter_map.shape[0]):
    each_filter_mask = result_filter_map[i][:][:]

    # compress filter map and chage data type (False => 0, True => 1)
    each_group_result =np.expand_dims(np.all(each_filter_mask, axis=1).astype(int), axis=0)
    result = np.append(result, each_group_result, axis=0)

    # create result header
    each_header = np.array_str(result_topology_map_size_tf[i][:])
    result_header.append(each_header)


result = np.swapaxes(result, 0, 1)
result_df = pd.DataFrame(result[:,1:], index= result[:,0], columns=result_header)
result_df = pd.concat([df_noimal, result_df], axis=1)

result_df.to_excel('./data/1-layer-1.xlsx', index=False)