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
        Encoder model
        '''
        self.batch_size = None
        self.timesteps = None
        #for ms-pacman
        self.height = 210
        self.width  = 160
        self.kernel = None
        self.channles = 3
        self.filters = None
        self.conv_dims = None

        '''
        Decoder model
        '''

        #todo

        '''
        Agent 
        '''
        #todo

        '''
        Trainer
        '''
        #todo

        '''
        log paths
        '''
        #todo
        self.weight_dir = os.path.join(self.base_dir, "weight")
        self.pic_dir = os.path.join(self.base_dir, "pic")

        if not os.path.exists(self.weight_dir):
            os.makedirs(self.weight_dir)
        if not os.path.exists(self.pic_dir):
            os.makedirs(self.pic_dir)

        self.encoder_weight_path = self.weight_dir + '/saved_encoder_weights.ckpt'
        self.decoder_weight_path = self.weight_dir + '/saved_decoder_weights.ckpt'
