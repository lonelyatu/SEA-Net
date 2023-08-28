# -*- coding: utf-8 -*-

import os
import glob
import numpy as np
from scipy import io
import tensorflow as tf

import models
import VGG16
import tfrecordOp
import HddDataOperation as hdd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DataSize = 102500
batchSize = 4
row = 512
column = 512
channel = 1
num_epoch = 50
Iter = DataSize // batchSize  

low_holder = tf.placeholder(tf.float32,[None,None,None,channel])
label_holder = tf.placeholder(tf.float32,[None,None,None,channel])
train_holder = tf.placeholder(tf.bool)

learn_rate = tf.placeholder(tf.float32)

result, r2 = models.BaseModelv1MainAtt(low_holder, train_sign=train_holder)
reconloss = hdd.lossL2(result,label_holder)
gradientloss = hdd.lossL2(r2, hdd.GaussianSmooth(label_holder,9,9))


result_inter = tf.tile(result,[1,1,1,3])
labels_inter = tf.tile(label_holder,[1,1,1,3])

logits1_1,logits1_2,logits1_3,logits1_4,logits1_5 = VGG16.VGG16_Feature_Extractor(result_inter)
logits2_1,logits2_2,logits2_3,logits2_4,logits2_5= VGG16.VGG16_Feature_Extractor(labels_inter,True)	
	
HighLevelLoss1 = hdd.lossL2(logits1_1,logits2_1) 
HighLevelLoss2 = hdd.lossL2(logits1_2,logits2_2) 
HighLevelLoss3 = hdd.lossL2(logits1_3,logits2_3) 
HighLevelLoss4 = hdd.lossL2(logits1_4,logits2_4) 
HighLevelLoss5 = hdd.lossL2(logits1_5,logits2_5) 

HighLevelLoss = HighLevelLoss1 + HighLevelLoss2 + HighLevelLoss3 + HighLevelLoss4 + HighLevelLoss5
loss = reconloss + gradientloss + 0.001 * HighLevelLoss

t_vars = tf.trainable_variables()

g_vars = [var for var in t_vars if 'BaseModel' in var.name]
d_vars = [var for var in t_vars if 'vgg_16' in var.name]


trainlossl2 = tf.summary.scalar('train_reconloss',reconloss)
trainlossgradient = tf.summary.scalar('train_gradient',gradientloss)
trainHighLevelLoss = tf.summary.scalar('train_HighLevelLoss',HighLevelLoss)
trainHighLevelLoss1 = tf.summary.scalar('train_HighLevelLoss1',HighLevelLoss1)
trainHighLevelLoss2 = tf.summary.scalar('train_HighLevelLoss2',HighLevelLoss2)
trainHighLevelLoss3 = tf.summary.scalar('train_HighLevelLoss3',HighLevelLoss3)
trainHighLevelLoss4 = tf.summary.scalar('train_HighLevelLoss4',HighLevelLoss4)
trainHighLevelLoss5 = tf.summary.scalar('train_HighLevelLoss5',HighLevelLoss5)
merge_loss = tf.summary.merge([trainlossgradient,trainlossl2,trainHighLevelLoss,trainHighLevelLoss1,trainHighLevelLoss2,trainHighLevelLoss3,trainHighLevelLoss4,trainHighLevelLoss5])

lossvalid = tf.placeholder(tf.float32)
valid_loss = tf.summary.scalar('valid_loss',lossvalid)
merge_validloss  = tf.summary.merge([valid_loss])

optimizer = tf.train.AdamOptimizer(learn_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(loss,var_list=g_vars)
    
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)
GenSaver = tf.train.Saver(max_to_keep=500)
VGGSvaer = tf.train.Saver(var_list=d_vars)

check_dir = './SEA-Net/checkpoints/'    

print("Initialing Network:success \nTraining....")

writer = tf.summary.FileWriter('./SEA-Net/logdir/',sess.graph)
  
filenames = ['./tfrecords/train.tfrecord']
#    
img_batch, label_batch = tfrecordOp.input_pipeline(filenames, batch_size=batchSize,
        num_epochs=None, num_features_input=[row,column,channel],
        num_features_label=[row,column,channel])

if not os.path.exists(check_dir):
    os.makedirs(check_dir)
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)
    VGGSvaer.restore(sess,'./vgg_16.ckpt')

else : 
    chkpt_fname = tf.train.latest_checkpoint(check_dir)
    GenSaver.restore(sess, chkpt_fname) 
    VGGSvaer.restore(sess, './vgg_16.ckpt')
    sess.run(tf.local_variables_initializer())
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

try:
    for epoch in range(0,num_epoch):  
        if epoch < 10:	
            curr_lr = 1e-3
        elif epoch < 20:
            curr_lr = 1e-4
        else:
            curr_lr = 1e-5
        for iter in range(Iter):
            samp_low, samp_label = sess.run([img_batch, label_batch])
			
            mlossl2,gl,High,_,mergeloss =\
                          sess.run([loss, gradientloss,HighLevelLoss,train,merge_loss],
                            feed_dict={low_holder:samp_low,label_holder:samp_label,learn_rate:curr_lr, train_holder:True})
            if (iter % 100==0):
                print("epoch: %3d Iter %7d training loss: %10.5f high level: %10.5f gradientloss: %10.5f" 
                      % (epoch,iter,mlossl2,High,gl))
            if (iter % 100 == 0):
                writer.add_summary(mergeloss,epoch*Iter + iter)
                                
        GenSaver.save(sess, os.path.join(check_dir,"SEA-Net"), global_step = epoch)
        
        validloss = 0
        lst = glob.glob('Your Validation DataSet Path')
            
        # print(lst)
        for l in range(len(lst)):
            info = io.loadmat(lst[l])
            validInput = np.float32(info['Input']) 
            validLabel = np.float32(info['Label'])
            validInput[validInput<0] = 0
            validLabel[validLabel<0] = 0
            
            V,_,_ = validInput.shape
            for v in range(V):
                valid_input = np.reshape(validInput[v],[1,512,512,channel])
                valid_label = np.reshape(validLabel[v],[1,512,512,channel])
                valid_loss = sess.run(loss,feed_dict={low_holder:valid_input,
                                                        label_holder:valid_label, 
                                                        train_holder:False})
                validloss = validloss + valid_loss
        
        validloss = validloss / V / len(lst)
        print(('Epoch %d the average valid: %.5f') % 
              (epoch,validloss))
        validLoss = sess.run(merge_validloss,feed_dict=
                             {lossvalid:validloss})
        writer.add_summary(validLoss,epoch)

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    coord.request_stop()
    coord.join(threads)    
writer.close()












