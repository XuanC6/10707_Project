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

        self.env = gym.make("MsPacman-v0")
        # self.env = gym.make("SpaceInvaders-v0")
        # self.env = gym.make("CartPole-v1")

        '''
        Feature Extractor
        '''
        # self.input_frame_height = 88
        # self.input_frame_width = 80
        # self.input_frame_channels = 1

        self.input_frame_height = 170
        self.input_frame_width = 160
        self.input_frame_channels = 3

        # self.conv_filters = [32, 64, 64]
        # self.conv_kernel_sizes  = [(8, 8), (4, 4), (3, 3)]
        # self.conv_strides = [4, 2, 1]

        self.conv_filters = [64, 128, 128, 128]
        self.conv_kernel_sizes  = [(8, 8), (4, 4), (4, 4), (3, 3)]
        self.conv_strides = [4, 2, 2, 1]

        self.conv_paddings = ["SAME"] * len(self.conv_filters)
        self.conv_activations = ["elu"] * len(self.conv_filters)
        self.conv_initializer = "he_normal"

        # self.fe_n_denses = [512]
        self.fe_n_denses = [1024, 1024]
        self.fe_dense_activations = ["elu"] * len(self.fe_n_denses)

        self.fe_dense_initializer = "he_normal"
        self.fe_n_outputs = 128

        '''
        Agent
        '''
        self.n_actions = self.env.action_space.n
        # self.max_T = 100
        # self.K_filters = 10

        # self.linear_initializer = "he_normal"
        # self.ir_n_hidden = 64
        # self.ir_activation = "relu"
        # self.ir_initializer = "he_normal"
        # self.n_epsilon_t = 64

        # self.e = 100

        '''
        Trainer
        '''
        self.max_episodes = 5000
        self.render_when_train = False
        self.render_when_test = False
        self.gamma = 0.99
        # self.commit_lambda = 0.1
        
        self.restore = 1

        self.entropy_coeff = 2e-2
        self.N_compute_returns = 50
        self.lr = 1e-4
        self.momentum = 0.95

        self.optimizer = tf.train.MomentumOptimizer(self.lr, self.momentum, use_nesterov=True)
        # self.optimizer = tf.train.AdamOptimizer(self.lr)

        self.save_interval = 20
        self.test_interval = 20

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

        # self.weight_actor_dir = os.path.join(self.base_dir, "weight_actor")
        # self.weight_critic_dir = os.path.join(self.base_dir, "weight_critic")

        # if not os.path.exists(self.weight_actor_dir):
        #     os.makedirs(self.weight_actor_dir)
        # if not os.path.exists(self.weight_critic_dir):
        #     os.makedirs(self.weight_critic_dir)
        
        # self.weight_actor_path = self.weight_actor_dir + '/saved_weights.ckpt'
        # self.weight_critic_path = self.weight_critic_dir + '/saved_weights.ckpt'