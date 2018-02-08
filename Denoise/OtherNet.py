import numpy as np
import tensorflow as tf

def OtherNet(_X):
    out = tf.layers.conv2d(
            inputs = _X,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    return out
