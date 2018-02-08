import numpy as np
import tensorflow as tf

def OtherNet(_X):
    conv1 = tf.layers.conv2d(
            inputs = _X,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    X1 = conv1 + conv2
    resconv = _X - tf.layers.conv2d(
            inputs = X1,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    xc= tf.layers.conv2d(
            inputs = resconv,
            filters = 6,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)
    out1 = tf.layers.conv2d(
            inputs = xc,
            filters = 6,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
       
    convtp1 = X1 + tf.layers.conv2d_transpose(
            inputs = out1,
            filters = 3,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)

    convtp2 = tf.layers.conv2d(
                inputs = convtp1,
                filters = 3,
                kernel_size = [5,5],
                padding = "same",
                activation = tf.nn.relu)
    X2 = convpt2 + convpt1
    out =  tf.layers.conv2d(
            inputs = X2,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    return out
