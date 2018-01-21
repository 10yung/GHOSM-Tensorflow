import tensorflow as tf
import numpy as np
from .som import SOM

class GHSOM(object):

    def __init__(self, m, n, input_data, input_num, dim, tau1=0.1, tau2=0.03):
        #Assign required variables first
        self._m = m
        self._n = n
        self._input_data = input_data
        self._input_num = input_num
        self._dim = dim
        self._tau1 = tau1
        self._tau2 = tau2


    def check_tau1_condition(self, init_som_topology_map, weight_vector):
        
        # INITIALIZE FILTER MAP FOR LATER USE
        filter_map_m = self._m
        filter_map_n = self._n
        filter_map_size = np.array(list(self.som_neuron_locations(filter_map_m, filter_map_n)))

        # INITIALIZE CHECK TAU1  GRAPH
        tau1_check_graph = tf.Graph()

        # POPULATE TAU1 GRAPH
        with tau1_check_graph.as_default():
            
            # initial all tensorflow variable
            input_data_tf = tf.constant(self._input_data, dtype="float32")
            tau1 = tf.constant(self._tau1, dtype="float32")
            init_som_topology_map = tf.constant(init_som_topology_map)
            weight_vector = tf.constant(weight_vector, dtype="float32")
            filter_map_size_tf = tf.Variable(filter_map_size)
            # ---
            # check each data belong to which group
            # ---
            # 1. create filter_map
            filter_map = tf.tile(
                    tf.reshape(
                        tf.reduce_all(
                            tf.equal(
                                tf.stack([filter_map_size_tf for i in range(len(self._input_data))], axis=1),
                                tf.stack([init_som_topology_map for i in range(filter_map_m*filter_map_n)])
                            )
                        , 2)
                    , [-1, 10,1])
                , [1, 1, self._dim])

            # 2. use filter_map to map data to cluster and find the location of the max mqe group
            i = tf.constant(0, dtype="int32")
            n = tf.constant(filter_map_m*filter_map_n)
            all_mqe = tf.TensorArray(tf.float32, n)

            # define stop condition
            def cond(i, all_mqe):
                return i < n

            def body(i, all_mqe):
                # create filter map and find which input data belong which group
                group_data = tf.reshape(
                                tf.boolean_mask(
                                        input_data_tf, 
                                        tf.squeeze(tf.slice(filter_map, [i, 0, 0], [1, 10, 5]))
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

            i,  all_mqe = tf.while_loop(cond, body, (i, all_mqe))

                        
            all_mqe = all_mqe.stack()     


            tau1_sess = tf.Session()
            init_op = tf.global_variables_initializer()
            tau1_sess.run(init_op)
            print(tau1_sess.run(tf.squeeze(tf.slice(filter_map, [0, 0, 0], [1, 10, 5]))))
            print(tau1_sess.run(all_mqe))




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
                                tf.less(mqe[0], tf.multiply(mqe[0], self._tau2)),
                                lambda: self.satisfy_tau2_cond('test'), 
                                lambda: self.call_som(self._m, self._n, self._input_data, self._dim)
                            )

            ##INITIALIZE SESSION
            tau2_sess = tf.Session()

            # run session
            ini_trained_weight, ini_som_result = tau2_sess.run(tau2_cond, feed_dict={input_data_tf: self._input_data})
            
            return ini_trained_weight, ini_som_result


    # called when tau2 condition is satisfy
    def satisfy_tau2_cond(self, test): 
        print('satisfy tau2')
        return tf.constant(1, dtype="float32"), tf.constant(2, dtype="int64")


    # do vertical expand (SOM) when tau2 condition not satisfy
    def call_som(self, m, n, input_data, dim, weight_after_insertion=None, alpha=None, sigma=None):
        # get the shape of input data inorder to calc iternumber in SOM
        iter_no = input_data.shape[0]*10
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
