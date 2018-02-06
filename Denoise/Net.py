import numpy as np
import tensorflow as tf

# RecNet Model

n_channels = [3,32]
def RecNet(_X,J):
    out = RecNet_level(_X, J, 1)
    return out
'''
def RecNet(_X, J, H, W):
    input_layer = tf.reshape(_X,[-1,H,W,3])
    rec = RecNet_level(input_layer, J, 1)
    out = tf.reshape(rec,[-1,H,W,3])
    return out
'''

def RecNet_level(_X, J, l):
    # J == l
    if l == J:
        out = tf.layers.conv2d(
            inputs = _X,
            filters = n_channels[J-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    else:
        conv1 = tf.layers.conv2d(
            inputs = _X,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        conv2 = conv1 + tf.layers.conv2d(
            inputs = conv1,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        resconv = _X - tf.layers.conv2d(
            inputs = conv2,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu);
        xc= tf.layers.conv2d(
            inputs = resconv,
            filters = n_channels[l],
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)
        ec = RecNet_level(xc,J,l+1)
        convtp1 = conv2 + tf.layers.conv2d_transpose(
            inputs = ec,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)
        out1 = convtp1 + tf.layers.conv2d(
            inputs = convtp1,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        out =  tf.layers.conv2d(
            inputs = out1,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    return out
