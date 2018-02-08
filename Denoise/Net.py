import numpy as np
import tensorflow as tf

# RecNet Model

def RecNet(_X, J):
    ml = [2]
    n_channels = [3]
    for i in range(J):
        n_channels.append(n_channels[i]*2)
        ml.append(ml[i])
    out = RecNet_level(_X, J, 1, ml, n_channels)
    return out
'''
def RecNet(_X, J, H, W):
    input_layer = tf.reshape(_X,[-1,H,W,3])
    rec = RecNet_level(input_layer, J, 1)
    out = tf.reshape(rec,[-1,H,W,3])
    return out
'''

def RecNet_level(_X, J, l, ml, n_channels):
    # J == l
    if l == J:
        out = tf.layers.conv2d(
            inputs = _X,
            filters = n_channels[J-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        for i in range(ml[J-1]-1):
            out = out + tf.layers.conv2d(
                inputs = out,
                filters = n_channels[J-1],
                kernel_size = [5,5],
                padding = "same",
                activation = tf.nn.relu)
    else:
        conv = tf.layers.conv2d(
            inputs = _X,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)

        for i in range(ml[l-1]-1):
            conv = conv + tf.layers.conv2d(
                inputs = conv,
                filters = n_channels[l-1],
                kernel_size = [5,5],
                padding = "same",
                activation = tf.nn.relu)

        resconv = _X - tf.layers.conv2d(
            inputs = conv,
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
        ec = RecNet_level(xc,J,l+1,ml,n_channels)
        convtp = conv + tf.layers.conv2d_transpose(
            inputs = ec,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)

        for i in range(ml[l-1]):
            convtp = convtp + tf.layers.conv2d(
                inputs = convtp,
                filters = n_channels[l-1],
                kernel_size = [5,5],
                padding = "same",
                activation = tf.nn.relu)

        out =  tf.layers.conv2d(
            inputs = convtp,
            filters = n_channels[l-1],
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    return out
