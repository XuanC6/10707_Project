# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

'''
Define the agent
'''

class A2C(tf.keras.Model):
    '''
    A2C
    '''
    def __init__(self, config):
        super(A2C, self).__init__()

        self.config = config

        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):

        conv_filters = self.config.conv_filters
        conv_kernel_sizes = self.config.conv_kernel_sizes 
        conv_strides = self.config.conv_strides
        conv_paddings = self.config.conv_paddings
        conv_activations = self.config.conv_activations
        conv_initializer = self.config.conv_initializer

        self.conv_layers = []
        for filters, kernel_size, strides, padding, activation in zip(
            conv_filters, conv_kernel_sizes, conv_strides, conv_paddings, conv_activations):

            self.conv_layers.append(tf.keras.layers.Conv2D(filters = filters, 
                                                            kernel_size = kernel_size, strides = strides, 
                                                            padding = padding, activation = activation,
                                                            kernel_initializer = conv_initializer))

        self.flatten_layer = tf.keras.layers.Flatten()

        n_denses = self.config.fe_n_denses
        dense_activations = self.config.fe_dense_activations
        dense_initializer = self.config.fe_dense_initializer
        n_outputs = self.config.n_actions + 1

        self.dense_layers = []
        for n_dense, activation in zip(n_denses, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(n_dense, activation = activation, 
                                                             kernel_initializer = dense_initializer))

        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        # define input (shape)
        dim1 = self.config.input_frame_height
        dim2 = self.config.input_frame_width
        dim3 = self.config.input_frame_channels
        input_tensor = tf.keras.layers.Input(shape = [dim1, dim2, dim3], batch_size = 1)
        
        self.call(input_tensor)


    def call(self, inputs):
        '''
        inputs: frame
        '''
        hidden = inputs
        
        for layer in self.conv_layers:
            hidden = layer(hidden)

        hidden = self.flatten_layer(hidden)

        for layer in self.dense_layers:
            hidden = layer(hidden)

        logits_and_state_value = self.output_layer(hidden)

        action_logits = tf.slice(logits_and_state_value, [0, 0], [1, self.config.n_actions])
        self.actions = tf.math.softmax(action_logits)
        self.action_tensor = tf.squeeze(tf.random.categorical(action_logits, 1))
        self.actions_entropy = -tf.reduce_sum(self.actions * tf.math.log(self.actions))

        self.state_value_tensor = tf.squeeze(tf.slice(logits_and_state_value, [0, self.config.n_actions], [1, 1]))



# class Actor(tf.keras.Model):
#     '''
#     Actor
#     '''
#     def __init__(self, config):
#         super(Actor, self).__init__()

#         self.config = config

#         self.initialize_layers()
#         self.activate_layers()


#     def initialize_layers(self):

#         conv_filters = self.config.conv_filters
#         conv_kernel_sizes = self.config.conv_kernel_sizes 
#         conv_strides = self.config.conv_strides
#         conv_paddings = self.config.conv_paddings
#         conv_activations = self.config.conv_activations
#         conv_initializer = self.config.conv_initializer

#         self.conv_layers = []
#         for filters, kernel_size, strides, padding, activation in zip(
#             conv_filters, conv_kernel_sizes, conv_strides, conv_paddings, conv_activations):

#             self.conv_layers.append(tf.keras.layers.Conv2D(filters = filters, 
#                                                             kernel_size = kernel_size, strides = strides, 
#                                                             padding = padding, activation = activation,
#                                                             kernel_initializer = conv_initializer))

#         self.flatten_layer = tf.keras.layers.Flatten()

#         n_denses = self.config.fe_n_denses
#         dense_activations = self.config.fe_dense_activations
#         dense_initializer = self.config.fe_dense_initializer
#         n_outputs = self.config.n_actions

#         self.dense_layers = []
#         for n_dense, activation in zip(n_denses, dense_activations):
#             self.dense_layers.append(tf.keras.layers.Dense(n_dense, activation = activation, 
#                                                              kernel_initializer = dense_initializer))

#         self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


#     def activate_layers(self):
#         # define input (shape)
#         dim1 = self.config.input_frame_height
#         dim2 = self.config.input_frame_width
#         dim3 = self.config.input_frame_channels
#         input_tensor = tf.keras.layers.Input(shape = [dim1, dim2, dim3], batch_size = 1)
        
#         self.call(input_tensor)


#     def call(self, inputs):
#         '''
#         inputs: frame
#         '''
#         hidden = inputs
        
#         for layer in self.conv_layers:
#             hidden = layer(hidden)

#         hidden = self.flatten_layer(hidden)

#         for layer in self.dense_layers:
#             hidden = layer(hidden)

#         action_logits = self.output_layer(hidden)

#         self.actions = tf.math.softmax(action_logits)
#         self.action_tensor = tf.squeeze(tf.random.categorical(action_logits, 1))
#         self.actions_entropy = -tf.reduce_sum(self.actions * tf.math.log(self.actions))


# class Critic(tf.keras.Model):
#     '''
#     Actor
#     '''
#     def __init__(self, config):
#         super(Critic, self).__init__()

#         self.config = config

#         self.initialize_layers()
#         self.activate_layers()


#     def initialize_layers(self):

#         conv_filters = self.config.conv_filters
#         conv_kernel_sizes = self.config.conv_kernel_sizes 
#         conv_strides = self.config.conv_strides
#         conv_paddings = self.config.conv_paddings
#         conv_activations = self.config.conv_activations
#         conv_initializer = self.config.conv_initializer

#         self.conv_layers = []
#         for filters, kernel_size, strides, padding, activation in zip(
#             conv_filters, conv_kernel_sizes, conv_strides, conv_paddings, conv_activations):

#             self.conv_layers.append(tf.keras.layers.Conv2D(filters = filters, 
#                                                             kernel_size = kernel_size, strides = strides, 
#                                                             padding = padding, activation = activation,
#                                                             kernel_initializer = conv_initializer))

#         self.flatten_layer = tf.keras.layers.Flatten()

#         n_denses = self.config.fe_n_denses
#         dense_activations = self.config.fe_dense_activations
#         dense_initializer = self.config.fe_dense_initializer
#         n_outputs = 1

#         self.dense_layers = []
#         for n_dense, activation in zip(n_denses, dense_activations):
#             self.dense_layers.append(tf.keras.layers.Dense(n_dense, activation = activation, 
#                                                              kernel_initializer = dense_initializer))

#         self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


#     def activate_layers(self):
#         # define input (shape)
#         dim1 = self.config.input_frame_height
#         dim2 = self.config.input_frame_width
#         dim3 = self.config.input_frame_channels
#         input_tensor = tf.keras.layers.Input(shape = [dim1, dim2, dim3], batch_size = 1)
        
#         self.call(input_tensor)


#     def call(self, inputs):
#         '''
#         inputs: frame
#         '''
#         hidden = inputs
        
#         for layer in self.conv_layers:
#             hidden = layer(hidden)

#         hidden = self.flatten_layer(hidden)

#         for layer in self.dense_layers:
#             hidden = layer(hidden)

#         self.state_value_tensor = self.output_layer(hidden)



class Decoder(tf.keras.Model):

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config

        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        pass


    def activate_layers(self):
        # define input (shape)
        dim1 = self.config.input_frame_height
        dim2 = self.config.input_frame_width
        dim3 = self.config.input_frame_channels
        input_tensor = tf.keras.layers.Input(shape = [dim1, dim2, dim3], batch_size = 1)
         
        self.call(input_tensor)


    def call(self, inputs):
        '''
        inputs: option
        '''
        pass