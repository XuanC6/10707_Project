# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

'''
Define models and agent
'''

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

        n_denses = self.config.fe_n_denses
        dense_activations = self.config.fe_dense_activations
        dense_initializer = self.config.fe_dense_initializer()
        n_outputs = self.config.fe_n_outputs
        
        hidden_layer = self.input_frame
        with tf.variable_scope("Feature_Extractor") as scope:

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



class Agent_STRAWe:
    '''
    Strategic Attentive Writer
    '''
    def __init__(self, config):
        self.config = config
        self.feature_extractor = Feature_Extractor(config)

        self.initialize_plans()
        self.plans_update_and_sample()


    def initialize_plans(self):
        self.T = self.config.max_T
        self.n_actions = self.config.n_actions

        with tf.variable_scope("init_agent"):
            commitment_plan = [0]*self.T
            commitment_plan[0] = 1
            self.commitment_plan = tf.convert_to_tensor(commitment_plan, dtype=tf.float32)
            self.action_plan = tf.zeros(shape=[self.n_actions, self.T], dtype=tf.float32)

            # for time_shift op
            # (A, T)
            _idx = tf.tile([tf.range(self.T)], [self.n_actions, 1])
            self.mask = tf.math.less(_idx, self.T - 1)
            self.zeros = tf.zeros(shape=[self.n_actions, self.T], dtype=tf.float32)


    def compute_attention_params(self, feature):
        # compute attention parameters from feature
        with tf.variable_scope("compute_attention_params"):
            # (grid position, stride, variance of Gaussian filters)
            attention_params = tf.layers.dense(feature, 3,
                                               kernel_initializer = self.config.linear_initializer(),
                                               name = "f_phi")
        return attention_params


    def read(self, attention_params):
        # read operation
        with tf.variable_scope("read"):
            grid_pos = tf.gather_nd(attention_params, [0, 0])
            log_stride = tf.gather_nd(attention_params, [0, 1])
            log_var = tf.gather_nd(attention_params, [0, 2])

            stride = tf.math.exp(log_stride)
            var = tf.math.exp(log_var)

            self.K = self.config.K_filters

            mean_locs = np.arange(self.K, dtype = np.float) - self.K/2 -0.5
            mean_locs = tf.convert_to_tensor(mean_locs, dtype=tf.float32)
            mean_locs = tf.math.truediv(mean_locs, stride) + grid_pos
            mean_locs = tf.expand_dims(mean_locs, axis = 0)

            # Fx (T, K)  
            Fx = [[i]*self.K for i in range(self.T)]
            Fx = tf.convert_to_tensor(Fx, dtype=tf.float32)
            Fx = tf.math.exp(-tf.math.truediv(tf.math.square(Fx - mean_locs), 2*var+1e-8))
            self.Fx = tf.math.truediv(Fx, tf.math.reduce_sum(Fx, axis = 0, keepdims = True))

            # action_plan (A, T)
            # beta_t (A, K)  
            beta_t = tf.linalg.matmul(self.action_plan, self.Fx)

        return beta_t


    def intermediate_representation(self, beta_t, z_t):
        # a two layer perceptron
        # beta_t (A, K)
        # z_t (1, ?)
        n_hidden = self.config.ir_n_hidden
        activation = self.config.ir_activation
        initializer = self.config.ir_initializer()
        n_epsilon_t = self.config.n_epsilon_t

        with tf.variable_scope("intermediate_representation"):
            hidden_layer = tf.concat([tf.transpose(beta_t), tf.tile(z_t, [self.K, 1])], 
                                     axis = 1)

            hidden_layer = tf.layers.dense(hidden_layer, n_hidden, activation = activation, 
                                           kernel_initializer = initializer)

            epsilon_t = tf.layers.dense(hidden_layer, n_epsilon_t,
                                        kernel_initializer = initializer)

        return epsilon_t


    def write(self, epsilon_t):
        # write operation
        with tf.variable_scope("write"):
            # (K, A)
            action_patch = tf.layers.dense(epsilon_t, self.n_actions,
                                           kernel_initializer = self.config.linear_initializer(),
                                           name = "f_A")
            # (T, K)*(K, A) = (T, A)
            new_term = tf.linalg.matmul(self.Fx, action_patch)
            # (A, T)
            update_term = tf.linalg.transpose(new_term)

        return update_term

    
    def time_shift(self, X):
        # time_shift operation
        with tf.variable_scope("time_shift"):
            # (A, T)
            X_shift = tf.roll(X, shift = 1, axis = 1)
            # mask the last column to 0
            mask_X_shift = tf.where(self.mask, X_shift, self.zeros)
        
        return mask_X_shift


    def new_commitment_plan(self, attention_params, epsilon_t):
        # generate new commitment plan when g_t = 1
        # (1, 3)
        attention_params
        # (K, ?) 
        epsilon_t

        # To do: new commitment_plan


        new_commit_plan = None

        return new_commit_plan


    def plans_update_and_sample(self):
        '''
        action plan update
        '''
        # current state of commitment plan, scalar
        g_t = self.gt_ph = tf.placeholder(tf.int32, shape=[], name="g_t")
        # feature of the currrent frame, (1, ?)
        z_t = self.feature_extractor.feature

        # (1, 3) 
        attention_params = self.compute_attention_params(z_t)
        # (A, K)
        beta_t = self.read(attention_params)
        # (K, ?) 
        epsilon_t = self.intermediate_representation(beta_t, z_t)
        # (A, T)
        update_term = self.write(epsilon_t)

        new_action_plan = self.time_shift(self.action_plan)
        self.action_plan = tf.cond(g_t > 0, 
                                   lambda: new_action_plan + update_term, 
                                   lambda: new_action_plan)
        
        '''
        commitment plan update
        '''
        # To do: new commitment_plan
        self.commitment_plan = tf.cond(g_t > 0, 
                                       lambda: self.new_commitment_plan(attention_params, epsilon_t),
                                       lambda: self.time_shift(self.commitment_plan)) 

        '''
        sample an action
        '''
        # self.action = sample_from(self.action_plan[:,0])