 # -*- coding: utf-8 -*-

import tensorflow as tf

def lrelu(inputs, slope=0.0):
    return tf.maximum(inputs, slope * inputs)

def conv2d(input,name,num_in,num_out):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights',shape=[3,3,num_in,num_out],dtype=tf.float32)
        biases = tf.get_variable('biases',shape=[num_out],dtype=tf.float32)
        return tf.nn.conv2d(input,weights,strides=[1,1,1,1],padding='SAME') + biases
 
def VGG16_Feature_Extractor(input,reuse=None):
    with tf.variable_scope('vgg_16',reuse=reuse):
        with tf.variable_scope('conv1'):
            conv1_1 = conv2d(input,'conv1_1',3,64)
            relu1_1 = lrelu(conv1_1)
            conv1_2 = conv2d(relu1_1,'conv1_2',64,64)
            relu1_2 = lrelu(conv1_2)
			
        maxpool1 = tf.nn.max_pool(relu1_2,[1,2,2,1],[1,2,2,1],'VALID')
		
        with tf.variable_scope('conv2'):
            conv2_1 = conv2d(maxpool1,'conv2_1',64,128)
            relu2_1 = lrelu(conv2_1)
            conv2_2 = conv2d(relu2_1,'conv2_2',128,128)
            relu2_2 = lrelu(conv2_2)
        
        maxpool2 = tf.nn.max_pool(relu2_2,[1,2,2,1],[1,2,2,1],'VALID')
		
        with tf.variable_scope('conv3'):
            conv3_1 = conv2d(maxpool2,'conv3_1',128,256)
            relu3_1 = lrelu(conv3_1)
            conv3_2 = conv2d(relu3_1,'conv3_2',256,256)
            relu3_2 = lrelu(conv3_2)
            conv3_3 = conv2d(relu3_2,'conv3_3',256,256)
            relu3_3 = lrelu(conv3_3)
        
        maxpool3 = tf.nn.max_pool(relu3_3,[1,2,2,1],[1,2,2,1],'VALID')

        with tf.variable_scope('conv4'):
            conv4_1 = conv2d(maxpool3,'conv4_1',256,512)
            relu4_1 = lrelu(conv4_1)
            conv4_2 = conv2d(relu4_1,'conv4_2',512,512)
            relu4_2 = lrelu(conv4_2)
            conv4_3 = conv2d(relu4_2,'conv4_3',512,512)
            relu4_3 = lrelu(conv4_3)		

        maxpool4 = tf.nn.max_pool(relu4_3,[1,2,2,1],[1,2,2,1],'VALID')

        with tf.variable_scope('conv5'):
            conv5_1 = conv2d(maxpool4,'conv5_1',512,512)
            relu5_1 = lrelu(conv5_1)
            conv5_2 = conv2d(relu5_1,'conv5_2',512,512)
            relu5_2 = lrelu(conv5_2)
			
    return conv1_2, conv2_2, conv3_2, conv4_2, conv5_2 
 
	