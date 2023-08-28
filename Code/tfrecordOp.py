# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy import io
import tensorflow as tf
import HddDataOperation as hdd

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def recordGen():
    
    lst = 'Your Training DataSet Path'
    cnt = 0

    if not os.path.exists('./tfrecords'):
        os.mkdir('./tfrecords')
    
    name = './tfrecords/train.tfrecord'
    writer = tf.python_io.TFRecordWriter(name)

    for fileName in lst:

        info = io.loadmat(fileName)
		
        Input = np.float32(info['Input']) # our data is 250*512*512
        Label = np.float32(info['Label'])
        Input[Input<0] = 0
        Label[Label<0] = 0
    
        V, _, _ = Input.shape
        
        for v in range(V):
            cnt = cnt + 1
            img = Input[v]
            ref = Label[v]

            img_raw = img.tobytes()
            lab_raw = ref.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(img_raw),        
            'res': _bytes_feature(lab_raw),
            }))
            writer.write(example.SerializeToString())
				
    writer.close()

def read_and_decode(filename_queue, shape_input,shape_label):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
            serialized_example,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'res': tf.FixedLenFeature([], tf.string),
                })

    image = tf.decode_raw(features['image'], tf.float32, little_endian=True)
    image = tf.reshape(image,[shape_input[0],shape_input[1],shape_input[2]])
    
    res = tf.decode_raw(features['res'], tf.float32, little_endian=True)
    res = tf.reshape(res,[shape_label[0],shape_label[1],shape_label[2]])
    return image, res


def input_pipeline(filenames, batch_size, num_epochs=None, 
                   num_features_input=None,num_features_label=None):
    '''num_features := width * height for 2D image'''
    filename_queue = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=True)
    example, label = read_and_decode(filename_queue, shape_input=num_features_input,
                                     shape_label=num_features_label)
#    label = read_and_decode(filename_queue, num_features)
    # min_after_dequeue defines how big a buffer we will randomly sample
    #   from -- bigger means better shuffling but slower start up and more
    #   memory used.
    # capacity must be larger than min_after_dequeue and the amount larger
    #   determines the maximum we will prefetch.  Recommendation:
    #   min_after_dequeue + (num_threads + a small safety margin) * batch_size
    min_after_dequeue = 4000
    capacity = min_after_dequeue + 10 * batch_size 
    img_batch, res_batch = tf.train.shuffle_batch(
            [example, label], batch_size=batch_size, capacity=capacity,
            min_after_dequeue=min_after_dequeue,num_threads=64,allow_smaller_final_batch=True)
    return img_batch, res_batch


if __name__ == '__main__':
    recordGen()
    print("convert finishes")
 







