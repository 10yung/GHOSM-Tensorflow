import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from pandas import ExcelWriter
import openpyxl
import datetime
from .ghsom_tensorflow import GHSOM

def call_ghsom(input_data, input_dim, input_num, init_m, init_n):
    # ---
    # init check tau2 to do first SOM
    # ---
    print('-------------------------call check tau2------------------------------')
    ghsom = GHSOM(init_m, init_n, input_data, input_num, input_dim)
    tmp_weight_result, tmp_init_som_result, prev_mqe = ghsom.check_tau2_condition()

    print(tmp_weight_result)
    print(tmp_init_som_result)
    print(prev_mqe)
    # ---
    # check tau1 condition and do horizontal expand
    # ---
    filter_map_m = init_m
    filter_map_n = init_n

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
                trained_weight_default_index = np.arange(
                    filter_map_m * filter_map_n)
                new_order = []

                for i in range(filter_map_n):
                    trained_weight_remainder = trained_weight_default_index % filter_map_n
                    new_order = np.append(new_order, np.where(
                        trained_weight_remainder == i)[0])

                new_order = new_order.astype(int)

                weight_topology_map = weight_vector[new_order, :][:].reshape(
                    filter_map_n, filter_map_m, -1).astype(np.float32)
                # print('----------topology map after regroup------------')
                # print(weight_topology_map)
                new_weight_topology_map = np.insert(
                    weight_topology_map, start_point + 1, insert_weight_vector, 0)
                # print('---------topology map after insert--------------')
                # print(new_weight_topology_map)

                new_weight_after_insertion = np.swapaxes(
                    new_weight_topology_map, 0, 1).reshape(-1, input_dim).astype(np.float32)
                # print('-------------flate array after insertion--------------')
                # print(new_weightinput_data[filter_mask,:]_after_insertion)

                filter_map_n = filter_map_n + 1
                filter_map_m = filter_map_m

            else:
                # print('insert y direction')
                # print(weight_vector)
                # print('--------df1----after insertion---------------')
                tmp = np.append(lower_section_weight_vector,
                                insert_weight_vector, axis=0)
                new_weight_after_insertion = np.append(
                    tmp, upper_section_weight_vector, axis=0)
                # print(new_weight_after_insertion)

                filter_map_m = filter_map_m + 1
                filter_map_n = filter_map_n

            if counter % 100 == 0:
                print('LoopNo: ' + str(counter) + ' - Time: ' +
                    str(datetime.datetime.now().time()))

            counter += 1
            print(filter_map_m)
            print(filter_map_n)
            print(new_weight_after_insertion)
            tmp_weight_result, tmp_init_som_result = ghsom.call_som(
                filter_map_m, filter_map_n, input_data, input_dim, new_weight_after_insertion)

            # print(tmp_weight_result)

        else:
            print('------------satisfy tau1------------')
            return result_filter_map, result_topology_map_size_tf

# print(result_filter_map)
# print(result_filter_map_m)
# print(result_filter_map_n)
# print(weight_vector)
# print(result_topology_map_size_tf)
# print(result_filter_map)
