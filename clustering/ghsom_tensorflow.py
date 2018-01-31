import tensorflow as tf
import numpy as np
from .som import SOM

class GHSOM(object):

    def __init__(self, m, n, input_data, input_num, dim, tau1=0.9, tau2=0.3):
        #Assign required variables first
        self._m = m
        self._n = n
        self._input_data = input_data
        self._input_num = input_num
        self._dim = dim
        self._tau1 = tau1
        self._tau2 = tau2

    def check_tau1_condition(self, init_som_result, weight_vector, prev_mqe, filter_map_m=None, filter_map_n=None):

        # INITIALIZE FILTER MAP FOR LATER USE
        # if m is None:
        #     filter_map_m = self._m
        # if n is None:
        #     filter_map_n = self._n

        filter_map_size = np.array(list(self.som_neuron_locations(filter_map_m, filter_map_n)))

        # INITIALIZE CHECK TAU1  GRAPH
        tau1_check_graph = tf.Graph()

        # POPULATE TAU1 GRAPH
        with tau1_check_graph.as_default():
            
            # initial all tensorflow variable
            input_data_tf = tf.constant(self._input_data, dtype="float32")
            tau1 = tf.constant(self._tau1, dtype="float32")
            init_som_result_tf = tf.constant(init_som_result)
            weight_vector_tf = tf.constant(weight_vector, dtype="float32")
            topology_map_size_tf = tf.constant(filter_map_size)
            mqe0_tf = tf.constant(prev_mqe, dtype="float32")
            # ---
            # check each data belong to which group
            # ---a
            # 1. create filter_map
            filter_map = tf.tile(
                    tf.reshape(
                        tf.reduce_all(
                            tf.equal(
                                tf.stack([topology_map_size_tf for i in range(len(self._input_data))], axis=1),
                                tf.stack([init_som_result_tf for i in range(filter_map_m*filter_map_n)])
                            )
                        , 2)
                    , [-1, self._input_num,1])
                , [1, 1, self._dim])

            # 2. use filter_map to map data to cluster and find the location of the max mqe group
            i = tf.constant(0, dtype="int32")
            n = tf.constant(filter_map_m*filter_map_n)
            all_mqe = tf.TensorArray(tf.float32, n, clear_after_read=False)

            # define stop condition
            def cond(i, all_mqe):
                return i < n

            def body(i, all_mqe):
                # create filter map and find which input data belong which group
                group_data = tf.reshape(
                                tf.boolean_mask(
                                        input_data_tf, 
                                        tf.squeeze(tf.slice(filter_map, [i, 0, 0], [1, self._input_num, self._dim]))
                                    )
                            , [ -1, self._dim])

                # calculate each mqe (one group by a time)
                group_mqe = tf.reduce_mean(
                        tf.sqrt(
                            tf.reduce_sum(
                                tf.pow(
                                    tf.subtract(
                                        group_data,
                                        tf.stack([tf.reduce_mean(group_data, 0)])
                                    )
                                ,2)
                            , 1, keep_dims=True)
                        )
                    , 0)

                # if mqe is Nan then return zero
                each_group_mqe = tf.squeeze(tf.cond(
                                    tf.is_nan(tf.squeeze(group_mqe)),
                                    lambda: tf.constant([0], dtype="float32"),
                                    lambda: group_mqe
                                ))
                
                # write mqe to all_mqe TensorArray (just like np array, it's dynamic tf array)
                all_mqe = all_mqe.write(i, each_group_mqe)

                # return for next step
                return i+1, all_mqe

            i,  all_mqe_tmp = tf.while_loop(cond, body, (i, all_mqe))
                        
            mean_mqe = tf.reduce_mean(all_mqe_tmp.stack())

            # compare to check tau1 condition
            tau1_cond = tf.cond(
                                tf.less(mean_mqe, tf.multiply(mqe0_tf, tau1)),
                                lambda: self.satisfy_tau1_cond(filter_map, init_som_result_tf, weight_vector_tf, mqe0_tf, all_mqe.stack(), input_data_tf, topology_map_size_tf, filter_map_m, filter_map_n), 
                                lambda: self.cal_horizontal_expand_vector(init_som_result_tf, weight_vector_tf, mqe0_tf, all_mqe.stack(), input_data_tf, topology_map_size_tf, filter_map_m, filter_map_n)
                            )

            ##INITIALIZE SESSION
            tau1_sess = tf.Session()
            init_op = tf.global_variables_initializer()

            # run session
            tau1_sess.run(init_op)
            # print('-------------tau1 connd result----------------')
            # print((tau1_sess.run(mean_mqe)
            # print(tau1_sess.run(tf.multiply(mqe0_tf, tau1)))
            # print(tau1_sess.run(tau1_cond))
            return tau1_sess.run(tau1_cond)
    
    # expand once and then return weight vector and map size
    def cal_horizontal_expand_vector(self, init_som_result, weight_vector, mqe0_tf, all_mqe, input_data_tf, topology_map_size_tf, filter_map_m, filter_map_n):
        # find error unit data
        error_unit_index = tf.argmax(all_mqe)

        neighborhood_distance = tf.reduce_sum(tf.squeeze(tf.abs(
                                        tf.subtract(
                                            topology_map_size_tf,    
                                            tf.stack([tf.slice(topology_map_size_tf,  [error_unit_index, 0], [1, 2])])
                                        )))
                                        ,1)
        # find the index which point is 1 distance between error unit 
        neighborhoods_location_index = tf.where(tf.equal(neighborhood_distance, tf.stack(tf.constant(1, dtype="int64"))))
        
        error_unit_weight = tf.slice(weight_vector, [error_unit_index, 0], [1, -1])

        neighborhoods_weight = tf.gather(weight_vector, neighborhoods_location_index)
        
        dissimilar_neighborbood_index = tf.squeeze(tf.gather(
                    neighborhoods_location_index,
                    tf.argmax(tf.reduce_sum(tf.squeeze(tf.pow(
                            tf.subtract(tf.stack([error_unit_weight]), neighborhoods_weight)
                        , 2))
                    , 1))))
        
        insert_direction, insert_weight_vector, error_unit_index, dissimilar_neighborbood_index, pivot_point, start_point, lower_section_weight_vector, upper_section_weight_vector = tf.cond(
                                tf.equal(tf.constant(1, dtype="int64"), tf.abs(tf.subtract(error_unit_index, dissimilar_neighborbood_index))),
                                lambda: self.insert_x_direction(error_unit_index, dissimilar_neighborbood_index, weight_vector, filter_map_m, filter_map_n), 
                                lambda: self.insert_y_direction(error_unit_index, dissimilar_neighborbood_index, weight_vector, filter_map_m, filter_map_n)
                            )

                                    
        return tf.constant(1, dtype="int32"), tf.constant(False, dtype="bool"), filter_map_m, filter_map_n, topology_map_size_tf, weight_vector, insert_weight_vector, insert_direction, error_unit_index, dissimilar_neighborbood_index, pivot_point, start_point, lower_section_weight_vector, upper_section_weight_vector
    
    # if satisfy tau1 condition then return filter map
    def satisfy_tau1_cond(self, filter_map, init_som_result, weight_vector, mqe0_tf, all_mqe, input_data_tf, topology_map_size_tf, filter_map_m, filter_map_n):

        return tf.constant(0, dtype="int32"), filter_map, filter_map_m, filter_map_n, topology_map_size_tf, weight_vector, tf.constant(0, dtype="float32"), tf.constant(0, dtype="int64"), tf.constant(0, dtype="int64"), tf.constant(0, dtype="int64"),tf.constant(0, dtype="int64"), tf.constant(0, dtype="int64"), tf.constant(0, dtype="float32"), tf.constant(0, dtype="float32")

    def insert_y_direction(self, error_unit_index, dissimilar_neighborbood_index, weight_vector, filter_map_m, filter_map_n):
        # find pivot point
        pivot_point = tf.cond(
                    tf.less(error_unit_index, dissimilar_neighborbood_index),
                    lambda: error_unit_index,
                    lambda: dissimilar_neighborbood_index
                )
        # find remaindeprint('satisfy tau2')r as the start point and n(y direction as stride) each step
        start_point = tf.multiply(tf.floordiv(pivot_point, filter_map_n), filter_map_n)
        lower_weight_vector = tf.slice(weight_vector, [start_point,0], [filter_map_n, -1])
        upper_weight_vector =  tf.slice(weight_vector, [tf.add(start_point, tf.constant(filter_map_n, dtype="int64")),0], [filter_map_n, weight_vector.get_shape()[1]])
        insert_weight_vector = tf.div(tf.add(upper_weight_vector, lower_weight_vector), tf.constant(2, dtype="float32"))
        
        # slice to two section
        test= tf.cast(tf.subtract(weight_vector.get_shape()[0], tf.cast(tf.add(start_point, tf.constant(filter_map_n, dtype="int64")), tf.int32)), tf.int32)
        lower_section_weight_vector = tf.slice(weight_vector, [0,0], [tf.cast(tf.add(start_point,tf.constant(filter_map_n, dtype="int64")), tf.int32), -1])
        upper_section_weight_vector = tf.slice(weight_vector, [tf.cast(tf.add(start_point,tf.constant(filter_map_n, dtype="int64")) , tf.int32),0], [test, -1])

        return tf.constant(1, dtype="int64"), insert_weight_vector, error_unit_index, dissimilar_neighborbood_index, pivot_point, start_point, lower_section_weight_vector, upper_section_weight_vector

    def insert_x_direction(self, error_unit_index, dissimilar_neighborbood_index, weight_vector, filter_map_m, filter_map_n):
        # find pivot point
        pivot_point = tf.cond(
                    tf.less(error_unit_index, dissimilar_neighborbood_index),
                    lambda: error_unit_index,
                    lambda: dissimilar_neighborbood_index
                )
        # find remainder as the start point and n(y direction as stride) each step
        start_point = tf.mod(pivot_point, filter_map_n)
        upper_start_point = tf.add(start_point,tf.constant(1, dtype="int64"))
        square_m_n = tf.cast(tf.multiply(tf.subtract(filter_map_m, 1), filter_map_n), tf.int64)
        test = tf.add(upper_start_point, square_m_n)
        end = tf.cast(tf.add(tf.multiply(tf.subtract(filter_map_m, 1), filter_map_n), 1), tf.int64)
        lower_weight_vector = tf.strided_slice(weight_vector, [start_point,0], [tf.add(end, start_point), weight_vector.get_shape()[1]], [filter_map_n, 1])
        upper_weight_vector = tf.strided_slice(weight_vector, [upper_start_point, 0], [tf.add(test, 1), weight_vector.get_shape()[1]], [filter_map_n, 1])
        insert_weight_vector = tf.div(tf.add(upper_weight_vector, lower_weight_vector), tf.constant(2, dtype="float32"))

        # slice to two section

        return tf.constant(0, dtype="int64"), insert_weight_vector, error_unit_index, dissimilar_neighborbood_index, pivot_point, start_point, lower_weight_vector, upper_weight_vector

    # ----------------------------------------------------------------------------------------------------------------------------------------------
    def check_tau2_condition(self, mqe0=None):
    
        ##INITIALIZE CHECK TAU2 GRAPH
        tau2_check_graph = tf.Graph()

        ##POPULATE TAU2 GRAPH
        with tau2_check_graph.as_default():

            # initial tensorflow variable
            input_data_tf = tf.placeholder("float32")
            tau2 = tf.constant(self._tau2, dtype="float32")

            mqe = tf.reduce_mean(
                    tf.sqrt(
                        tf.reduce_sum(
                            tf.pow(
                                tf.subtract(
                                    input_data_tf,
                                    tf.stack([tf.reduce_mean(input_data_tf, 0) for i in range(self._input_num)])
                                )
                            ,2)
                        , 1, keep_dims=True)
                    )
                , 0)
                
            # check if previous  mqe exist
            if mqe0 is None:
                mqe0 = mqe
            else:
                mqe0 = tf.constant(mqe0, dtype="float32")

            # check if mqe less than previous mqe
            tau2_cond = tf.cond(
                                tf.less(mqe[0], tf.multiply(mqe[0], tau2)),
                                lambda: self.satisfy_tau2_cond(), 
                                lambda: self.call_som(self._m, self._n, self._input_data, self._dim)
                            )

            ##INITIALIZE SESSION
            tau2_sess = tf.Session()

            # run session
            ini_trained_weight, ini_som_result = tau2_sess.run(tau2_cond, feed_dict={input_data_tf: self._input_data})
            mqe = tau2_sess.run(mqe[0], feed_dict={input_data_tf: self._input_data})
            
            return ini_trained_weight, ini_som_result, mqe


    # called when tau2 condition is satisfy
    def satisfy_tau2_cond(self): 

        return tf.constant(0, dtype="float32"), tf.constant(0, dtype="int64")


    # do vertical expand (SOM) when tau2 condition not satisfy
    def call_som(self, m, n, input_data, dim, weight_after_insertion=None, alpha=None, sigma=None):
        # get the shape of input data inorder to calc iternumber in SOM
        iter_no = input_data.shape[0]*2
        som = SOM(m, n, dim, weight_after_insertion, n_iterations= iter_no)
        trained_weight = som.train(input_data)
        mapped = som.map_vects(input_data)
        result = np.array(mapped)
        return trained_weight, result

    # create default topology map position by m*n (m cols, n rows)
    def som_neuron_locations(self, m, n):
        for i in range(m):
            for j in range(n):
                yield np.array([i, j])




    def train(self, input_vect):
        print(self._sess.run(self.mqe, feed_dict={self.input_data_tf: input_vect}))
