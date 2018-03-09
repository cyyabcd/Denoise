import numpy as np
import tensorflow as tf

def OtherNet(_X):
    conv1 = tf.layers.conv2d(
            inputs = _X,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    
    resconv = _X - conv1
    xc= tf.layers.conv2d(
            inputs = resconv,
            filters = 6,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same")
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

    out =  tf.layers.conv2d(
            inputs = convtp1,
            filters = 3,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    return out
