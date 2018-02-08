import dataset
import matplotlib.pyplot as plt
import tensorflow as tf
import Net
import numpy as np
from PIL import Image
J = 3
x = tf.placeholder("float", [None,None,None,3])
y = tf.placeholder("float", [None,None,None,3])
H = tf.placeholder("int32")
W = tf.placeholder("int32")
reconstruction = Net.RecNet(x,J)
print("模型完成")
# COST
cost = tf.reduce_mean(tf.pow(reconstruction-y, 2))
# OPTIMIZER
optm = tf.train.AdamOptimizer(0.01).minimize(cost)

epochs = 10
batch_size = 100
n_example = 14000
disp_step = 10
label, noise = dataset.read_and_decode(["data.tfrecords"])
label_batch, noise_batch = tf.train.shuffle_batch([label, noise],
                                                batch_size=batch_size, capacity=n_example,
                                                min_after_dequeue=1000)

init_op = tf.global_variables_initializer()

testdatapath='data/test/'
outputpath='tmp/test/'

with tf.Session() as sess: #开始一个会话
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    total_cost = 0
    num_batch = int(n_example/batch_size)
    for epoch in range(epochs):
        total_cost = 0.

        for i in range(num_batch):
            label_img, noise_img= sess.run([label_batch, noise_batch])
            label1= label_img[0,:]*255
            noise1= noise_img[0,:]*255
            label1_img = Image.fromarray(label1.astype(np.uint8))
            noise1_img = Image.fromarray(noise1.astype(np.uint8))
            label1_img.show()
            noise1_img.show()
    print("训练完成！")
    
    print("开始测试。。。")
    for i in range(5):
        for sigma in dataset.Image_Noise:
            noise_img_path = testdatapath+"%d_%d.jpg"%(i,sigma)
            noise_img_savepath = outputpath + "%d_%d.jpg"%(i,sigma)
            noise_img = Image.open(noise_img_path)
            width, height = noise_img.size
            noise_img_data = np.array(noise_img)
            data = np.array(noise_img_data,dtype = 'float')/255.
            data = np.reshape(data,(-1, height, width, 3))
            feeds_nosie ={x:data}
            denoise_img = sess.run(reconstruction, feed_dict=feeds_nosie)
            denoise_img = np.reshape(denoise_img,(height, width, 3))
            denoise_img = denoise_img*255
            new_im = Image.fromarray(denoise_img.astype(np.uint8))
            new_im.save(noise_img_savepath)
    coord.request_stop()
    coord.join(threads)



