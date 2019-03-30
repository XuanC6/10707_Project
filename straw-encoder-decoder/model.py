# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import ConvLSTMCell
import tensorflow.nn.d
tf.enable_eager_execution()

base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)

class Encoder(object):
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
        self.timesteps = self.config.timesteps
        self.height = self.config.height
        self.width = self.config.width
        self.kernel = self.config.kernel
        self.channles = self.config.channels
        self.filters = self.config.filters
        self.conv_dims = self.config.conv_dims

    def forward(self, input):
        # Create a placeholder for videos.
        k = self.batch_size
        inputs = tf.placeholder(tf.float32, [self.batch_size, self.timesteps] + [] + [channels])


        cell = ConvLSTMCell(self.shape, self.filters, self.kernel)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

        # There's also a ConvGRUCell that is more memory efficient.
        from cell import ConvGRUCell
        cell = ConvGRUCell(shape, filters, kernel)
        outputs, state = tf.nn.dynamic_rnn(cell, inputs, dtype=inputs.dtype)

        # It's also possible to enter 2D input or 4D input instead of 3D.
        shape = [100]
        kernel = [3]
        inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
        cell = ConvLSTMCell(shape, filters, kernel)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)

        shape = [50, 50, 50]
        kernel = [1, 3, 5]
        inputs = tf.placeholder(tf.float32, [batch_size, timesteps] + shape + [channels])
        cell = ConvGRUCell(shape, filters, kernel)
        outputs, state = tf.nn.bidirectional_dynamic_rnn(cell, cell, inputs, dtype=inputs.dtype)
