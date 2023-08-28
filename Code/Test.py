# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy import io
import tensorflow as tf

import models
import HddDataOperation as hdd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  

low_holder = tf.placeholder(tf.float32,[1, 512, 512, 1])

result, _ = models.BaseModelv1MainAtt(low_holder, False)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
Saver = tf.train.Saver()


chkpt_fname = './SEA-Net/checkpoints/SEA-Net-49'
Saver.restore(sess, chkpt_fname)

lst = glob.glob('Your Test DataSet Path')

for filename in range(len(lst)):

    info = io.loadmat(filename)
    validInput = np.float32(info['Input'])
    validInput[validInput<0] = 0

    V, _, _ = validInput.shape
    SEA_Net = np.zeros_like(validInput, dtype=np.float32)

    for v in range(V):
        valid_input = np.reshape(validInput[v],[1, 512, 512, 1])
        output = sess.run(result,feed_dict={low_holder:valid_input})
        SEA_Net[v] = output[0,:,:,0]
    SEA_Net.astype(np.float32).tofile('%s.raw' % filename)
        












