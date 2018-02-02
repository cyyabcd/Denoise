import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
#matplotlib inline  
print ("PACKAGES LOADED")


mnist = input_data.read_data_sets('tmp/data/', one_hot=True)
train_X   = mnist.train.images
train_Y = mnist.train.labels
test_X    = mnist.test.images
test_Y  = mnist.test.labels
print ("MNIST ready")

# NETOWRK PARAMETERS
n_input    = 784 
n_output   = 784  

'''
n_hidden_1 = 256 
n_hidden_2 = 256 
'''
# PLACEHOLDERS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")

J = 2
ml = 2
# WEIGHTS

n_channel = 1;
'''
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}
'''
# RecNet Model
def RecNet(_X, J):
    input_layer = tf.reshape(_X,[-1,28,28,1])
    rec = RecNet_level(input_layer, J, 1)
    out = tf.reshape(rec,[-1, n_output])
    return out


def RecNet_level(_X, J, l):
    # J == l
    if l == J:
        out = tf.layers.conv2d(
            inputs = _X,
            filters = n_channel,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    else:
        conv1 = tf.layers.conv2d(
            inputs = _X,
            filters = n_channel,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        conv2 = _X + tf.layers.conv2d(
            inputs = conv1,
            filters = n_channel,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        resconv = _X - tf.layers.conv2d(
            inputs = conv2,
            filters = n_channel,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu);
        xc= tf.layers.conv2d(
            inputs = resconv,
            filters = n_channel,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)
        ec = RecNet_level(xc,J,l+1)
        convtp1 = conv1 + tf.layers.conv2d_transpose(
            inputs = ec,
            filters = n_channel,
            kernel_size = [5,5],
            strides = [2,2],
            padding = "same",
            activation = tf.nn.relu)
        out1 = convtp1 + tf.layers.conv2d(
            inputs = convtp1,
            filters = n_channel,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
        out =  tf.layers.conv2d(
            inputs = out1,
            filters = 1,
            kernel_size = [5,5],
            padding = "same",
            activation = tf.nn.relu)
    return out
# MODEL
def denoise_auto_encoder(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1'])) 
    layer_1out = tf.nn.dropout(layer_1, _keep_prob) 
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2'])) 
    layer_2out = tf.nn.dropout(layer_2, _keep_prob) 
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])

# MODEL AS A FUNCTION
#reconstruction = denoise_auto_encoder(x, weights, biases, dropout_keep_prob)
with tf.device("/device:GPU:4"):
    reconstruction = RecNet(x,J)



# COST
    cost = tf.reduce_mean(tf.pow(reconstruction-y, 2))
# OPTIMIZER
    optm = tf.train.AdamOptimizer(0.01).minimize(cost) 
# INITIALIZER
    init = tf.initialize_all_variables()
print ("FUNCTIONS READY")


savedir = "tmp/"
saver   = tf.train.Saver(max_to_keep=1)
print ("SAVER READY")


TRAIN_FLAG = 1
epochs     = 50
batch_size = 100
disp_step  = 10

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
sess.run(init)
if TRAIN_FLAG:
    print ("START OPTIMIZATION")
    for epoch in range(epochs):
        num_batch  = int(mnist.train.num_examples/batch_size)
        total_cost = 0.
        for i in range(num_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            batch_xs_noisy = batch_xs + 0.3*np.random.randn(batch_size, 784)
            feeds = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: 1.}
            sess.run(optm, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)
        if epoch % disp_step == 0:
            print ("Epoch %02d/%02d average cost: %.6f" 
                   % (epoch, epochs, total_cost/num_batch))
    saver.save(sess, savedir + 'denoise_auto_encoder.ckpt', global_step=epoch)
print ("OPTIMIZATION FINISHED")