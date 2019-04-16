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
class Option_Encoder(tf.keras.Model):

    def __init__(self, config):
        super(Option_Encoder, self).__init__()

        self.config = config
        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        dense_dims = self.config.dense_dims_OE
        dense_activations = self.config.dense_activations_OE
        dense_initializer = self.config.dense_initializer_OE
        n_outputs = self.config.option_dim

        self.dense_layers = []
        for dim, activation in zip(dense_dims, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation = activation, 
                                                            kernel_initializer = dense_initializer))
        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        input_tensor = tf.keras.layers.Input(shape = [self.config.input_length], batch_size = 1)
        self.call(input_tensor)


    def call(self, inputs):
        hidden = inputs
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        return self.output_layer(hidden)



class Critic(tf.keras.Model):

    def __init__(self, config):
        super(Critic, self).__init__()

        self.config = config
        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        dense_dims = self.config.dense_dims_Cr
        dense_activations = self.config.dense_activations_Cr
        dense_initializer = self.config.dense_initializer_Cr
        n_outputs = 1

        self.dense_layers = []
        for dim, activation in zip(dense_dims, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation = activation, 
                                                            kernel_initializer = dense_initializer))

        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        input_tensor = tf.keras.layers.Input(shape = [self.config.input_length], batch_size = 1)
        self.call(input_tensor)


    def call(self, inputs):
        hidden = inputs
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        return tf.squeeze(self.output_layer(hidden))



class Decoder(tf.keras.Model):

    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config
        self.initialize_layers()
        self.activate_layers()


    def initialize_layers(self):
        self.gru_cell = tf.keras.layers.GRUCell(units = self.config.units)

        dense_dims = self.config.dense_dims_De
        dense_activations = self.config.dense_activations_De
        dense_initializer = self.config.dense_initializer_De
        n_outputs = self.config.output_dim_De

        self.dense_layers = []
        for dim, activation in zip(dense_dims, dense_activations):
            self.dense_layers.append(tf.keras.layers.Dense(dim, activation = activation, 
                                                            kernel_initializer = dense_initializer))

        self.output_layer = tf.keras.layers.Dense(n_outputs, kernel_initializer = dense_initializer)


    def activate_layers(self):
        self.gru_cell.build(input_shape = (None, self.config.output_dim_De))

        input_tensor = tf.keras.layers.Input(shape = [self.config.units], batch_size = 1)
        hidden = input_tensor
        for layer in self.dense_layers:
            hidden = layer(hidden)
        _ =self.output_layer(hidden)


    def call(self, inputs, states):
        '''
        inputs: (1, n_actions)
        states: (1, option_dim)
        '''
        new_states, _ = self.gru_cell(inputs, [states])

        hidden = new_states
        for layer in self.dense_layers:
            hidden = layer(hidden)
        
        self.logits = self.output_layer(hidden)
        self.scores = tf.nn.softmax(self.logits)
        # self.action_tensor = tf.squeeze(tf.random.categorical(logits, 1))
        self.actions_entropy = -tf.reduce_sum(self.scores  * tf.math.log(self.scores))

        return new_states