# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)


class Feature_Extractor:
    '''
    a deep convolutional network
    take into input frame and output feature_representation
    '''
    def __init__(self, config):
        self.config = config

        self.add_placeholder()
        self.get_feature_representation()


    def add_placeholder(self):
        dim1 = self.config.input_frame_height
        dim2 = self.config.input_frame_width
        dim3 = self.config.input_frame_channels

        self.input_frame = tf.placeholder(tf.int32, [None, dim1, dim2, dim3], name="input_frame")


    def get_feature_representation(self):

        conv_filters = self.config.conv_filters
        conv_kernel_sizes = self.config.conv_kernel_sizes 
        conv_strides = self.config.conv_strides
        conv_paddings = self.config.conv_paddings
        conv_activations = self.config.conv_activations
        conv_initializer = self.config.conv_initializer()

        n_denses = self.config.n_denses
        dense_activations = self.config.dense_activations
        dense_initializer = self.config.dense_initializer()
        n_outputs = self.config.n_outputs
        
        hidden_layer = self.input_frame
        name = "Feature_Extractor"

        with tf.variable_scope(name) as scope:

            for filters, kernel_size, strides, padding, activation in zip(
                conv_filters, conv_kernel_sizes, conv_strides, conv_paddings, conv_activations):

                hidden_layer = tf.layers.conv2d(inputs = hidden_layer, filters = filters, 
                                                kernel_size = kernel_size, strides = strides, 
                                                padding = padding, activation = activation,
                                                kernel_initializer = conv_initializer)

            flatten = tf.layers.flatten(hidden_layer)
            hidden_layer = flatten
            for n_dense, activation in zip(n_denses, dense_activations):
                hidden_layer = tf.layers.dense(hidden_layer, n_dense, activation = activation, 
                                               kernel_initializer = dense_initializer)

            self.feature = tf.layers.dense(hidden_layer, n_outputs,
                                           kernel_initializer = dense_initializer)

        # Collect all the variables in this network
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                   scope = scope.name)
        self.params = {param.name[len(scope.name):]: param for param in params}



class STRAW:
    '''
    Strategic Attentive Writer
    '''
    def __init__(self, config):
        self.config = config
        self.feature_extractor = Feature_Extractor(config)

        self.initialize_plans()


    def initialize_plans(self):
        self.T = self.config.max_T
        self.n_actions = self.config.n_actions

        commitment_plan = [0]*self.T
        commitment_plan[0] = 1
        self.commitment_plan = tf.convert_to_tensor(commitment_plan, dtype=tf.float32)
        self.action_plan = tf.zeros(shape=[self.n_actions, self.T], dtype=tf.float32)


    def read(self, attention_params):
        # read operation
        with tf.variable_scope("read"):
            ######## ?
            grid_pos, log_stride, log_var = tf.squeeze(attention_params)

            stride = tf.math.exp(log_stride)
            var = tf.math.exp(log_var)

            K = self.config.K_filters

            mean_locs = np.arange(K, dtype = np.float) - K/2 -0.5
            mean_locs = tf.convert_to_tensor(mean_locs, dtype=tf.float32)
            mean_locs = tf.math.truediv(mean_locs, stride) + grid_pos
            mean_locs = tf.expand_dims(mean_locs, axis = 0)

            # Fx (T, K)  
            Fx = [[i]*K for i in range(self.T)]
            Fx = tf.convert_to_tensor(Fx, dtype=tf.float32)
            Fx = tf.math.exp(-tf.math.truediv(tf.math.square(Fx - mean_locs), 2*var+1e-8))
            self.Fx = tf.math.truediv(Fx, tf.math.reduce_sum(Fx, axis = 0, keepdims = True))

            # action_plan (A, T)
            # beta_t (A, K)  
            beta_t = tf.linalg.matmul(self.action_plan, self.Fx)

        return beta_t


    def intermediate_representation(self, beta_t, z_t):
        # beta_t (A, K)
        # z_t (1, ?)

        # (K, ?) 
        # concat


        # (K, ?) 
        epsilon_t = 0

        return epsilon_t


    def write(self, epsilon_t):
        # write operation
        with tf.variable_scope("write"):
            # (K, A)
            action_patch = tf.layers.dense(epsilon_t, self.n_actions,
                                           kernel_initializer = self.config.dense_initializer(),
                                           name = "f_A")
            # (T, K)*(K, A) = (T, A)
            new_term = tf.linalg.matmul(self.Fx, action_patch)
            # (A, T)
            update_term = tf.linalg.transpose(new_term)

        return update_term

    
    def time_shift(self, X):

        X_shift = tf.roll(X, shift = 1, axis = 1)

        return X_shift



    def plans_update_and_sample(self):
        '''
        action plan update
        '''
        # current state of commitment plan, scalar
        g_t = self.gt_ph = tf.placeholder(tf.int32, shape=[], name="g_t")
        # feature of the currrent frame, (1, ?)
        z_t = self.feature_extractor.feature

        # (1, 3)
        attention_params = tf.layers.dense(z_t, 3,
                                           kernel_initializer = self.config.dense_initializer(),
                                           name = "f_phi")
        # (A, K)
        beta_t = self.read(attention_params)
        # (K, ?) 
        epsilon_t = self.intermediate_representation(beta_t, z_t)
        # (A, T)
        update_term = self.write(epsilon_t)

        new_action_plan = self.time_shift(self.action_plan)

        # To do
        # new_action_plan = tf.mask

        self.action_plan = tf.cond(g_t > 0, new_action_plan + update_term, lambda: new_action_plan)
        
        '''
        commitment plan update
        '''



        '''
        sample an action
        '''
        # action = sample_from(self.action_plan[:,0])
        # return action