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

        self.conv_filters = []
        self.conv_kernel_sizes  = []
        self.conv_strides = []
        self.conv_paddings = []
        self.conv_activations = "relu"
        self.conv_initializer = "he_normal"

        self.fe_n_denses = None
        self.fe_dense_activations = "relu"
        self.fe_dense_initializer = "he_normal"
        self.fe_n_outputs = None

        '''
        Agent
        '''
        self.max_T = None
        self.n_actions = None
        self.K_filters = 12

        self.linear_initializer = "he_normal"
        self.ir_n_hidden = None
        self.ir_activation = "relu"
        self.ir_initializer = "he_normal"
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

        self.lr_actor = 0.001
        self.lr_critic = 0.001

        self.optimizer_actor = tf.keras.optimizers.Adam
        self.optimizer_critic = tf.keras.optimizers.Adam

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