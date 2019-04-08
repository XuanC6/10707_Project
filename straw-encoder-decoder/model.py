# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import ConvLSTMCell
from .utils import conv2d, flattenallbut0, normc_initializer
tf.enable_eager_execution()

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

class Encoder(tf.keras.Model):
    """"
    tf.contrib.rnn.ConvLSTMCell
    __init__(
    conv_ndims,
    input_shape,
    output_channels,
    kernel_shape,
    use_bias=True,
    skip_connection=False,
    forget_bias=1.0,
    initializers=None,
    name='conv_lstm_cell'
)
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        self.batch_size = self.config.batch_size
        #config for convnet feature extractors
        self.height = self.config.height
        self.width = self.config.width
        self.kernel = self.config.kernel
        self.channles = self.config.channels
        self.filters = self.config.filters
        self.conv_dims = self.config.conv_dims
        self.convnet_kind = self.config.convnet_kind
        #config for lstm nets
        self.timesteps = self.config.timesteps
        self.en_units = self.config.enc_units
        self.gru = tf.keras.layers.GRU(self.en_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        #TODO: whether define a placeholder here or just convert inputs to tensor and feed
        #directly to the model which is a common practice for TF2.0
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, self.timesteps]  + [
            self.height, self.width])
        # self.convLSTM = tf.keras.layers.ConvLSTM2D()


        """
          Input shape:
    - if data_format='channels_first'
        5D tensor with shape:
        `(samples, time, channels, rows, cols)`
    - if data_format='channels_last'
        5D tensor with shape:
        `(samples, time, rows, cols, channels)`

  Output shape:
    - if `return_sequences`
         - if data_format='channels_first'
            5D tensor with shape:
            `(samples, time, filters, output_row, output_col)`
         - if data_format='channels_last'
            5D tensor with shape:
            `(samples, time, output_row, output_col, filters)`
    - else
        """

    def extract_feature(self, x):
        """
        extract feature through conv nets
        :param x: shape should be (BS, Height, Width, Channel) aka "NWHC"
        :return: featur vectors in shape (BS, hidden_dim) hidden_dim is 256 for small
        kind convnet or  512  for large kind convnet
        """
        if self.convnet_kind == 'small':  # from A3C paper
            x = tf.nn.relu(conv2d(x, 16, [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(conv2d(x, 32, [4, 4], [2, 2], pad="VALID"))
            x = flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 256, name='lin',
                                           kernel_initializer=normc_initializer(1.0)))

        elif self.convnet_kind == 'large':  # Nature DQN
            x = tf.nn.relu(conv2d(x, 32, [8, 8], [4, 4], pad="VALID"))
            x = tf.nn.relu(conv2d(x, 64, [4, 4], [2, 2], pad="VALID"))
            x = tf.nn.relu(conv2d(x, 64, [3, 3], [1, 1], pad="VALID"))
            x = flattenallbut0(x)
            x = tf.nn.relu(tf.layers.dense(x, 512, name='lin',
                                           kernel_initializer=normc_initializer(1.0)))
        return x

    def call(self, inputs, hidden):
        x = self.extract_feature(inputs)
        output, state = self.gru(x, initial_state=hidden)
        #output shape (BS, timestes, units) units is the hidden size
        #hidden state shape (BS, units)
        #state is the last output output[:,-1,:] =  state
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, hidden_size)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights