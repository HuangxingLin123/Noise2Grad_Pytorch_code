#-*- coding=utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread, imsave
from matplotlib.image import imsave
import tensorflow as tf
import os
import cv2
import scipy
import scipy.misc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True



def add_train_noise_tf(x,lam_max):
    chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=5, maxval=lam_max)
    # chi_rng = tf.random_uniform(shape=[1, 1, 1], minval=30, maxval=30)
    # print(chi_rng.shape)
    # out  = tf.random_poisson(chi_rng*(x+0.5), shape=[])/chi_rng - 0.5
    # print(out)
    #chi_rng = lam_max
    return tf.random_poisson(chi_rng*(x), shape=[])/chi_rng


a= os.path.exists('./datasets/train/noisy_train/')
if a:
    pass
else:
    os.mkdir('./datasets/train/noisy_train/')
count = 0

with tf.Session(config = config) as sess:
    path = os.listdir('./datasets/train/ground_truth/')
    for img_name in path:
        img = imread(os.path.join('./datasets/train/ground_truth/',img_name))
        # img = img /255.0
        count += 1
        print(count)
        img_shape = img.shape
        img_input = tf.placeholder(tf.float32,img_shape)

        img_tensor = add_train_noise_tf(img_input,50)
        img_np = sess.run(img_tensor,feed_dict={img_input:img})
        imsave(os.path.join('./datasets/train/noisy_train/',img_name), np.clip(img_np,0.0,1.0))