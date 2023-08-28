# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import cv2

def lossL2(tensor1,tensor2):
    return tf.reduce_mean(tf.square(tensor1 - tensor2))  
    
def GaussianSmooth(img,kernelsize=3,sigma=1.0):
    # inputs: batch, width, height, channel
    f = np.multiply(cv2.getGaussianKernel(kernelsize,sigma), np.transpose(cv2.getGaussianKernel(kernelsize,sigma)))
    kernel = tf.reshape(np.float32(f), [kernelsize, kernelsize, 1, 1], 'kernel')  
    low = tf.nn.conv2d(img, kernel, strides=[1,1,1,1],name='f1', padding='SAME')
    high = img - low
    return high
    
    
    
    
    
    
    
    
    
    
    
