import os
import numpy as np
import tensorflow as tf
from PIL import Image

ImageSizeW, ImageSizeH = 128,128
N_Image = 2000
Image_Noise = [10,15,20,25,30,50,70]

cwd_label = 'data/label/'
cwd_noise = 'data/noise/'

def createdataset():
    writer = tf.python_io.TFRecordWriter("data.tfrecords")

    for index in range(N_Image):
        label_img_path = cwd_label+"%d.jpg"%(index)
        label_img = Image.open(label_img_path)
        label_img_raw = label_img.tobytes()
        for noise in Image_Noise:
            noise_img_path = cwd_noise+"%d_%d.jpg"%(index,noise)
            noise_img = Image.open(noise_img_path)
            noise_img_raw = noise_img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_img_raw])),
            "noise": tf.train.Feature(bytes_list=tf.train.BytesList(value=[noise_img_raw]))
        }))
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(TFFileName):
    filename_queue = tf.train.string_input_producer(TFFileName)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example,
                                features={
                                'label': tf.FixedLenFeature([],tf.string),
                                'noise': tf.FixedLenFeature([],tf.string)})
    label_img = tf.decode_raw(features['label'],tf.uint8)
    noise_img = tf.decode_raw(features['noise'],tf.uint8)
    label_img = tf.reshape(label_img,[ImageSizeW,ImageSizeH,3])
    noise_img = tf.reshape(noise_img,[ImageSizeW,ImageSizeH,3])
    label_img = tf.cast(label_img,tf.float32)*(1./255)
    noise_img = tf.cast(noise_img,tf.float32)*(1./255)
    return label_img, noise_img
