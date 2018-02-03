import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from pandas import ExcelWriter
import openpyxl
import datetime
from clustering.call_ghsom import call_ghsom


# ---
# input preprocess
# ---
# change data to np array (SOM accept nparray format)
df = pd.read_excel('./data/WPG.xlsx')
print('finish read file')

# select unique item name first
selected_col_name = 'Sales'
selected_col = df[selected_col_name]
selected_col = selected_col.replace(np.nan, 'unknow', regex=True)
selected_col = selected_col.replace('Pinky.Sam', 'Sam,Pinky', regex=True)
selected_col = selected_col.replace('Sam/Pinky', 'Sam,Pinky', regex=True)
selected_col = selected_col.replace('Pinky,', 'Pinky', regex=True)
selected_col = selected_col.replace('pinky,', 'Pinky', regex=True)
selected_col = selected_col.replace('JOY', 'joy', regex=True)
selected_col = selected_col.str.lower()
print(selected_col.unique())

for value in selected_col.unique():
    each_df = df.loc[selected_col == value].reset_index(drop=True)

    df_noimal = each_df.ix[:, ['Report Date', 'Customer',
                            'Item Short Name', 'Brand', 'Sales', 'AWU_Avial_ratio', 'Backlog_Avail_ratio']]
    df_numerical=each_df.ix[:, ['Actual AWU', 'Avail.',
                                    'BL <= 9WKs', 'Backlog', 'DC OH', 'Hub OH', 'TTL OH', 'FCST M']]

    input_data = np.array(df_numerical.as_matrix())
    input_dim = input_data.shape[1]
    input_num = input_data.shape[0]
    init_m = 2
    init_n = 2


    # call ghsom
    result_filter_map, result_topology_map_size_tf = call_ghsom(input_data, input_dim, input_num, init_m, init_n)

    #  save file section
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

    save_path = './data/' + selected_col_name + '_layer_1_' + value + '_test.xlsx'
    result_df.to_excel(save_path, index=False)
