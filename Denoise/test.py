import dataset
import matplotlib.pyplot as plt
import tensorflow as tf
import Net
from PIL import Image
J = 2
x = tf.placeholder("float", [None,128,128,3])

y = tf.placeholder("float", [None,128,128,3])
reconstruction = Net.RecNet(x,J)
# COST
cost = tf.reduce_mean(tf.pow(reconstruction-y, 2))
# OPTIMIZER
optm = tf.train.AdamOptimizer(0.01).minimize(cost)

epochs = 50
batch_size = 100
n_example = 14000
disp_step = 10
label, noise = dataset.read_and_decode(["data.tfrecords"])
label_batch, noise_batch = tf.train.shuffle_batch([label, noise],
                                                batch_size=batch_size, capacity=n_example,
                                                min_after_dequeue=1000)

init_op = tf.global_variables_initializer()


with tf.Session() as sess: #开始一个会话
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    total_cost = 0
    for epoch in range(epochs):
        num_batch = int(n_example/batch_size)
        total_cost = 0.

        for i in range(num_batch):
            label_img, noise_img= sess.run([label_batch, noise_batch])
            feeds = {x:noise_img, y:label_img}
            sess.run(optm, feed_dict = feeds)
            total_cost += sess.run(cost,feed_dict=feeds)
        print ("%.6f" %(total_cost/num_batch))
    coord.request_stop()
    coord.join(threads)


