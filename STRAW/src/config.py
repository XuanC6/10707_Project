# -*- coding: utf-8 -*-
import os
import tensorflow as tf
# from model import #?

'''
All parameters and hyperparameters
'''


class Configuration:

    def __init__(self):

        self.base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        '''
        Feature Extractor
        '''
        self.input_frame_height = None
        self.input_frame_width = None
        self.input_frame_channels = None

        self.conv_filters = None
        self.conv_kernel_sizes  = None
        self.conv_strides = None
        self.conv_paddings = None
        self.conv_activations = None
        self.conv_initializer = None

        self.fe_n_denses = None
        self.fe_dense_activations = None
        self.fe_dense_initializer = None
        self.fe_n_outputs = None

        '''
        Agent
        '''
        self.max_T = None
        self.n_actions = None
        self.linear_initializer = None
        self.K_filters = None

        self.ir_n_hidden = None
        self.ir_activation = None
        self.ir_initializer = None
        self.n_epsilon_t = None

        self.e = None

        '''
        log paths
        '''