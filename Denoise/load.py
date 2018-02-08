import dataset
import matplotlib.pyplot as plt
import tensorflow as tf
import Net
import numpy as np
from skimage import io
from PIL import Image
J = 3
x = tf.placeholder("float", [None,None,None,3])
y = tf.placeholder("float", [None,None,None,3])
H = tf.placeholder("int32")
W = tf.placeholder("int32")
reconstruction = Net.RecNet(x,J)
# COST
cost = tf.reduce_mean(tf.square(reconstruction-y))
# OPTIMIZER
optm = tf.train.AdamOptimizer(0.01).minimize(cost)


init_op = tf.global_variables_initializer()

testdatapath='data/noise/'
outputpath='tmp/test2/'

savedir = "tmp/"
saver   = tf.train.Saver(max_to_keep=1)
with tf.Session() as sess: #开始一个会话
    saver.restore(sess, 'tmp/denoise_auto_encoder.ckpt-9')
    for i in range(2000):
        for sigma in dataset.Image_Noise:
            noise_img_path = testdatapath+"%d_%d.jpg"%(i,sigma)
            noise_img_savepath = outputpath + "%d_%d.jpg"%(i,sigma)
            noise_img = io.imread(noise_img_path)
            width, height = noise_img.shape[0], noise_img.shape[1]
            noise_img_data = np.array(noise_img)
            data = np.array(noise_img_data,dtype = 'float')/255.
            data = np.reshape(data,(1, height, width, 3))
            feeds_nosie ={x:data}
            denoise_img = sess.run(reconstruction, feed_dict=feeds_nosie)
            denoise_img = np.reshape(denoise_img,(height, width, 3))
            denoise_img = denoise_img*255
            new_im = Image.fromarray(denoise_img.astype(np.uint8),'RGB')
            new_im.save(noise_img_savepath)





