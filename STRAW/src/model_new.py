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

class Feature_Extractor(tf.keras.Model):
    '''
    a deep convolutional network
    take into input frame and output feature_representation
    '''
    def __init__(self, config):
        super(Feature_Extractor, self).__init__()

        conv_filters = config.conv_filters
        conv_kernel_sizes = config.conv_kernel_sizes 
        conv_strides = config.conv_strides
        conv_paddings = config.conv_paddings
        conv_activations = config.conv_activations
        conv_initializer = config.conv_initializer()

        n_denses = config.fe_n_denses
        dense_activations = config.fe_dense_activations
        dense_initializer = config.fe_dense_initializer()
        n_outputs = config.fe_n_outputs

        # define layers
        self.conv_layers = []
        for filters, kernel_size, strides, padding, activation in zip(
            conv_filters, conv_kernel_sizes, conv_strides, conv_paddings, conv_activations):

            self.conv_layers.append(tf.keras.layers.Conv2D(filters = filters, 
                                                            kernel_size = kernel_size, strides = strides, 
                                                            padding = padding, activation = activation,
                                                            kernel_initializer = conv_initializer))

        self.flatten_layer = tf.keras.layers.Flatten()

        self.dense_layers = []
        for n_dense, activation in zip(n_denses, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(n_dense, activation = activation, 
                                                             kernel_initializer = dense_initializer))

        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)

        # define input (shape)
        dim1 = config.input_frame_height
        dim2 = config.input_frame_width
        dim3 = config.input_frame_channels
        input_tensor = tf.keras.layers.Input(shape = [dim1, dim2, dim3])
         
        _ = self.call(input_tensor)

    
    def call(self, inputs):
        '''Run the model'''
        hidden = inputs

        for layer in self.conv_layers:
            hidden = layer(hidden)

        hidden = self.flatten_layer(hidden)

        for layer in self.dense_layers:
            hidden = layer(hidden)

        feature = self.output_layer(hidden)

        return feature



class STRAW:
    '''
    Strategic Attentive Writer
    '''
    def __init__(self, config):
        self.config = config
        self.feature_extractor = Feature_Extractor(config)

        # with tf.variable_scope("Agent_STRAW"):
        #     self.initialize_plans()
        #     self.plans_update()


    def initialize_plans(self):
        self.T = self.config.max_T
        self.n_actions = self.config.n_actions

        # (A + 1, T)
        self.action_plan = tf.zeros(shape=[self.n_actions + 1, self.T], dtype=tf.float32)

        commitment_plan = [0]*self.T
        commitment_plan[0] = 1
        # (1, T)
        self.commitment_plan = tf.convert_to_tensor([commitment_plan], dtype=tf.int32)
        
        # for time_shift op of action plan
        # (A + 1, T)
        _idx_action = tf.manip.tile([tf.range(self.T)], [self.n_actions + 1, 1])
        self.mask_action = tf.math.less(_idx_action, self.T - 1)
        self.zeros_action = tf.zeros(shape=[self.n_actions + 1, self.T], dtype=tf.float32)

        # for time_shift op of commitment plan
        # (1, T)
        _idx_commit = tf.manip.tile([tf.range(self.T)], [1, 1])
        self.mask_commit = tf.math.less(_idx_commit, self.T - 1)
        self.zeros_commit = tf.zeros(shape=[1, self.T], dtype=tf.int32)


    def compute_attention_params(self, feature):
        # compute attention parameters from feature
        # (grid position, stride, variance of Gaussian filters)
        attention_params = tf.layers.dense(feature, 3,
                                            kernel_initializer = self.config.linear_initializer(),
                                            name = "f_phi")
    
        return attention_params


    def read(self, attention_params):
        # read operation
        grid_pos, log_stride, log_var = attention_params[0]

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

        # action_plan (A + 1, T)
        # beta_t (A + 1, K)  
        beta_t = tf.linalg.matmul(self.action_plan, self.Fx)

        return beta_t


    def intermediate_representation(self, beta_t, z_t):
        # a two layer perceptron
        # beta_t (A + 1, K)
        # z_t (1, ?)
        n_hidden = self.config.ir_n_hidden
        activation = self.config.ir_activation
        initializer = self.config.ir_initializer()
        n_epsilon_t = self.config.n_epsilon_t

        # (K, A+?)
        hidden_layer = tf.concat([tf.linalg.transpose(beta_t), tf.manip.tile(z_t, [self.K, 1])], 
                                    axis = 1)

        hidden_layer = tf.layers.dense(hidden_layer, n_hidden, activation = activation, 
                                        kernel_initializer = initializer)

        # (K, ?)
        epsilon_t = tf.layers.dense(hidden_layer, n_epsilon_t,
                                    kernel_initializer = initializer)

        return epsilon_t


    def write(self, epsilon_t):
        # write operation
        # epsilon_t (K, ?)
        # (K, A + 1)
        action_patch = tf.layers.dense(epsilon_t, self.n_actions + 1,
                                        kernel_initializer = self.config.linear_initializer(),
                                        name = "f_A")
        # (T, K)*(K, A + 1) = (T, A + 1)
        # transpose -> (A + 1, T)
        update_term = tf.linalg.transpose(tf.linalg.matmul(self.Fx, action_patch))

        return update_term

    
    def time_shift_action_plan(self, action_plan):
        # time_shift operation of action plan
        # (A + 1, T)
        action_plan_shift = tf.roll(action_plan, shift = 1, axis = 1)
        # mask the last column to 0
        mask_action_plan_shift = tf.where(self.mask_action, action_plan_shift, self.zeros_action)
        
        return mask_action_plan_shift


    def time_shift_commit_plan(self, commit_plan):
        # time_shift operation of commit_plan
        # (1, T)
        commit_plan_shift = tf.roll(commit_plan, shift = 1, axis = 1)
        # mask the last column to 0
        mask_commit_plan_shift = tf.where(self.mask_commit, commit_plan_shift, self.zeros_commit)
        
        return mask_commit_plan_shift


    def generate_new_commit_plan(self, attention_params, epsilon_t):
        # generate new commitment plan when g_t = 1
        '''
        !!! Based on my own understanding
        '''
        # (1, ?)       
        feature = tf.concat([tf.manip.reshape(epsilon_t, [1, -1]), attention_params], axis = 1)

        # (grid position, stride, variance of Gaussian filters)
        attention_params_c = tf.layers.dense(feature, 3,
                                                kernel_initializer = self.config.linear_initializer(),
                                                name = "f_c")

        grid_pos, log_stride, log_var = attention_params_c[0]

        stride = tf.math.exp(log_stride)
        var = tf.math.exp(log_var)

        mean_locs = np.arange(1, dtype = np.float) - 1/2 -0.5
        mean_locs = tf.convert_to_tensor(mean_locs, dtype=tf.float32)
        mean_locs = tf.math.truediv(mean_locs, stride) + grid_pos
        mean_locs = tf.expand_dims(mean_locs, axis = 0)

        # Fx (T, 1)  
        Fx = [[i]*1 for i in range(self.T)]
        Fx = tf.convert_to_tensor(Fx, dtype=tf.float32)
        Fx = tf.math.exp(-tf.math.truediv(tf.math.square(Fx - mean_locs), 2*var+1e-8))
        Fx = tf.math.truediv(Fx, tf.math.reduce_sum(Fx, axis = 0, keepdims = True))

        (1, 1)
        e = tf.convert_to_tensor([[self.config.e]], dtype=tf.float32)

        # (T, 1)*(1, 1)
        # transpose -> (1, T)
        commit_plan_logits = tf.linalg.transpose(tf.linalg.matmul(Fx, e))
        bias = tf.Variable(np.zeros(shape = [1, self.T]), name = "commit_bias")

        commit_plan_logits = commit_plan_logits + bias
        new_commit_plan = tf.math.sigmoid(commit_plan_logits)

        return new_commit_plan


    def plans_update_and_sample(self, inputs):

        # current state of commitment plan, scalar
        commit_probs = tf.concat([1-self.commitment_plan, self.commitment_plan], axis = 0)
        g_t = tf.squeeze(tf.random.multinomial(commit_probs[:, 0], 1))

        # feature of the currrent frame, (1, ?)
        z_t = self.feature_extractor(inputs)

        # (1, 3) 
        attention_params = self.compute_attention_params(z_t)
        # (A + 1, K)
        beta_t = self.read(attention_params)
        # (K, ?) 
        epsilon_t = self.intermediate_representation(beta_t, z_t)
        # (A + 1, T)
        update_term = self.write(epsilon_t)

        new_action_plan = self.time_shift_action_plan(self.action_plan)

        '''
        plans update
        '''
        if g_t > 0:
            self.action_plan = new_action_plan + update_term
            self.commitment_plan = self.generate_new_commit_plan(attention_params, epsilon_t)
        else:
            self.action_plan = new_action_plan
            self.commitment_plan = self.time_shift_commit_plan(self.commitment_plan)

        # sample an action, shape = ()
        action_to_play = tf.squeeze(tf.random.multinomial(tf.log([self.action_plan[:, 0]]), 1))

        return action_to_play, g_t