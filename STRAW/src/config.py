# -*- coding: utf-8 -*-
import os
import gym
import tensorflow as tf

tf.enable_eager_execution()

'''
All parameters and hyperparameters
'''

class Configuration:

    def __init__(self):

        self.base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

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
        self.K_filters = None

        self.linear_initializer = None
        self.ir_n_hidden = None
        self.ir_activation = None
        self.ir_initializer = None
        self.n_epsilon_t = None

        self.e = None

        '''
        Trainer
        '''
        self.env = 
        self.max_episodes = 500
        self.render_when_train = False
        self.render_when_test = True
        self.gamma = 0.95
        self.commit_lambda = 
        self.N_compute_returns = 10

        self.lr_actor = 
        self.lr_critic = 

        self.optimizer_actor = 
        self.optimizer_critic = 

        self.save_interval = 
        self.test_interval = 

        self.n_test_episodes = 10

        '''
        log paths
        '''
        self.weight_dir = os.path.join(self.base_dir, "weight")
        self.pic_dir = os.path.join(self.base_dir, "pic")

        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir)

        self.weight_path = self.weight_dir + '/saved_weights.ckpt'